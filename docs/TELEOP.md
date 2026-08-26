# Leader–Follower Teleoperation

`openarm_teleop` drives a **follower** arm from a **leader** arm of the same design.
One process handles **one arm pair**, so a bimanual setup runs **two** processes
and needs **four CAN FD channels**.

---

## 1. CAN channel budget

```
process 1 (right_arm):   leader can2   ↔   follower can0
process 2 (left_arm) :   leader can3   ↔   follower can1
```

A single PCAN-USB Pro FD gives two channels, so **two adapters are required** for
bimanual. With one adapter you can still run a single arm pair.

### Channel numbering follows attach order

This is the single easiest thing to get wrong. On Linux the `canN` names are
assigned in the order the adapters appear, so the **attach order decides which
adapter is the leader**. Get it backwards and torque goes to the leader while you
push the follower by hand.

Our mapping (verified by unplugging one adapter and watching which BUSID
disappeared — the motor IDs are identical on every channel, `0x01`–`0x08`, so CAN
traffic alone cannot tell them apart):

| usbipd BUSID | attach order | channels | role |
|---|---|---|---|
| `1-7` | **first** | `can0`, `can1` | **follower** |
| `1-8` (or `3-2`) | second | `can2`, `can3` | **leader** |

Label the adapters physically. BUSIDs change if you move the plug to another port.

---

## 2. Bring up the buses

WSL2 only — on native Linux the adapters appear directly and you can skip to
`openarm_can_setup.sh`.

```powershell
Start-Service usbipd                      # admin, only if the service died
usbipd attach --wsl --busid 1-7           # follower FIRST
usbipd attach --wsl --busid 1-8
```

```bash
~/openarm_can_setup.sh
```

Expected:

```
can0   usb=1-1:1.0    FOLLOWER  ERROR-ACTIVE
can1   usb=1-1:1.0    FOLLOWER  ERROR-ACTIVE
can2   usb=1-2:1.0    LEADER    ERROR-ACTIVE
can3   usb=1-2:1.0    LEADER    ERROR-ACTIVE
```

`ERROR-ACTIVE` is the **normal** CAN state, not a fault. All four run
**1 Mbps arbitration / 5 Mbps data, CAN FD**.

`usbipd attach` creates the interfaces **down** — the script is what configures
the bitrate and brings them up.

---

## 3. Check the motors

```bash
OACLI=~/ros2_ws/install/openarm_can/bin/openarm-can-cli
for i in can0 can1 can2 can3; do $OACLI -i $i discover; done
```

Each channel must report **8 motors** (`0x01`–`0x08` = 7 joints + gripper).

- **0 motors** → motor power is off (we hit exactly this)
- **7 motors, `0x01` missing** → run it again. `discover` allows only 2 retries
  with a 30 ms window per motor (`discover_motor_commands.cpp:141`), so detection
  is probabilistic and a slow-booting joint 1 gets missed. If it stays at 7 across
  repeated scans, check the joint 1 connector and bus termination — `openarm_hardware`
  hard-codes IDs `0x01`–`0x07`, so a flaky joint 1 means losing the shoulder mid-motion.

---

## 4. Build the teleop binaries

`openarm_teleop` is a plain CMake project, **not** a ROS 2 package.

```bash
git clone https://github.com/enactic/openarm_teleop.git ~/openarm_teleop
cd ~/openarm_teleop
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/ros2_ws/install/openarm_can;/opt/ros/humble"
cmake --build build -j$(nproc)
```

Dependencies (`orocos_kdl`, `kdl_parser`, `Eigen3`, `urdfdom`, `yaml-cpp`,
`OpenArmCAN`) all come from the ROS 2 desktop install.

Generate the URDF the controller needs — **with the cameras off**, since the
controller builds a KDL chain from it:

```bash
mkdir -p ~/openarm_teleop/urdf
SHARE=$(ros2 pkg prefix openarm_description)/share/openarm_description
xacro "$SHARE/assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro" \
  arm_type:=v10 body_type:=v10 bimanual:=true ros2_control:=false cameras:=false \
  -o ~/openarm_teleop/urdf/leader.urdf
cp ~/openarm_teleop/urdf/leader.urdf ~/openarm_teleop/urdf/follower.urdf
```

> **Do not use `script/launch_*.sh`.** They hard-code `~/openarm_ros2_ws` and
> `src/openarm_description/urdf/robot/v10.urdf.xacro`, neither of which exists in
> a current checkout. Call the binaries directly.

---

## 5. Run

Shut down any ROS 2 node first — `demo.launch.py` and teleop fight over the same
CAN interfaces.

```bash
cd ~/openarm_teleop          # REQUIRED: config/*.yaml is loaded by relative path
                             # (openarm_unilateral_control.cpp:211)

# one-way, leader → follower. Start here.
./build/unilateral_control urdf/leader.urdf urdf/follower.urdf right_arm can2 can0
./build/unilateral_control urdf/leader.urdf urdf/follower.urdf left_arm  can3 can1

# bilateral (force feedback) — only after unilateral is proven
./build/bilateral_control  urdf/leader.urdf urdf/follower.urdf right_arm can2 can0
```

Argument order:

```
<leader_urdf> <follower_urdf> [arm_side] [leader_can] [follower_can]
                                          ^leader      ^follower
```

### Never omit the CAN arguments

The binaries default to `leader=can0, follower=can2`, which is **the reverse of
the mapping above**. Omit them and the leader gets the torque.

```
right arm:  can2 can0          left arm:  can3 can1
```

---

## 6. What normal looks like

**The leader moving on its own is correct.** `unilateral_step()` sends the leader
gravity + friction + Coriolis compensation (`control.cpp:278`):

```cpp
effort = gravity[i] + friction[i] * 0.3 + coriolis[i] * 0.1;
```

with MIT parameters `{kp=0, kd=0, q=0, dq=0, tau=effort}`. Position gain is zero,
so it is not being driven to a pose — it is being made weightless so you can move
it by hand.

- Leader stays put when released, moves easily when pushed → correct
- Leader drives itself toward some pose → **stop immediately** (`Ctrl+C`)

---

## 7. Stopping

**Always `Ctrl+C`.** The SIGINT handler calls `disable_all()` on both arms
(`openarm_unilateral_control.cpp:299-300`). `kill -9` leaves the motors armed.

---

## 8. Known limits

**1 kHz control on WSL2 is not guaranteed.** `FREQUENCY` is 1000 Hz
(`openarm_constants.hpp`), but the controller manager cannot get FIFO real-time
scheduling under WSL (`Could not enable FIFO RT scheduling policy: Operation not
permitted`). Loop jitter shows up as vibration in bilateral mode. Native Linux
plus `rtprio` in `/etc/security/limits.conf` fixes the first half; a `PREEMPT_RT`
kernel is needed for hard real-time.

**Safety limits are compiled in**, not configurable at runtime:

```
effort_limit    20.0 Nm      velocity_limit  8.0 rad/s
ELBOWLIMIT       0.0         (joint 4 lower bound)
```

Gains and friction coefficients live in `config/leader.yaml` and
`config/follower.yaml`.

**Left/right wiring is unverified.** The mapping above assumes each adapter's
first channel is the right arm. If moving the right leader moves the *left*
follower, swap to `can3 can1`.
