# Distributed setup — Linux control PC + Windows planning station

Running the robot from two machines: a **Linux PC wired to the arms** doing nothing
but control, and a **Windows/WSL2 workstation** running MoveIt and RViz over the network.

```
Linux PC  10.2.12.118                    Windows PC (WSL2)  10.2.12.71
├─ robot_state_publisher                 ├─ move_group          planning
├─ ros2_control_node                     └─ RViz + MotionPlanning   UI
├─ 5 controllers
├─ realsense2_camera                        ROS_DOMAIN_ID = 0
└─ CAN x4  (PCAN-USB Pro FD x2)             DDS over Wi-Fi
```

Everything below was found by getting it wrong first; the failure symptom is given
next to each fix so you can recognise it.

---

## 1. Linux PC — control only

```bash
# CAN up
~/openarm_can_setup.sh

# verify motors (read-only)
~/ros2_ws/install/openarm_can/bin/openarm-can-cli -i can0 discover

# control layer — keep this terminal open
cd ~/ros2_ws && source install/setup.bash
ros2 launch openarm_bringup openarm.bimanual.launch.py \
  arm_type:=v10 use_fake_hardware:=false \
  right_can_interface:=can2 left_can_interface:=can3
```

`openarm_bringup` also tries to start its own RViz. Over SSH there is no display,
so it dies with `exit code -6`. That is harmless — control is unaffected.

### Channel mapping is per-machine

`canN` numbering follows the order the adapters are detected, so **the mapping from
the WSL setup does not carry over**. On this Linux PC it came out reversed:

| USB port | channels | role |
|---|---|---|
| `1-1.2` | `can0`, `can1` | **leader** |
| `1-4` | `can2`, `can3` | **follower** |

Motor IDs are identical (`0x01`–`0x08`) on every channel, so CAN traffic cannot
tell the arms apart. Determine it by unplugging one adapter and seeing which
interfaces disappear:

```bash
lsusb | grep -ci 0c72          # adapters present
ls /sys/class/net/ | grep can  # channels present
```

---

## 2. Windows PC — planning and UI

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
unset ROS_LOCALHOST_ONLY

ros2 launch openarm_bimanual_moveit_config move_group.launch.py arm_type:=openarm_v1.0 &
```

RViz needs the robot model **generated locally** (see §5):

```bash
SHARE=$(ros2 pkg prefix openarm_description)/share/openarm_description
MC=$(ros2 pkg prefix openarm_bimanual_moveit_config)/share/openarm_bimanual_moveit_config

xacro "$SHARE/assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro" \
      arm_type:=v10 bimanual:=true ros2_control:=true > /tmp/local.urdf

python3 - <<'PY'
import yaml, pathlib, subprocess, os
mc = subprocess.check_output(['ros2','pkg','prefix','openarm_bimanual_moveit_config']).decode().strip()
mc = os.path.join(mc, 'share', 'openarm_bimanual_moveit_config')
p = lambda f: pathlib.Path(mc, 'config/openarm_v1.0', f)
yaml.safe_dump({'/**': {'ros__parameters': {
    'robot_description':            pathlib.Path('/tmp/local.urdf').read_text(),
    'robot_description_semantic':   p('openarm_bimanual.srdf').read_text(),
    'robot_description_kinematics': yaml.safe_load(p('kinematics.yaml').read_text()),
    'robot_description_planning':   yaml.safe_load(p('joint_limits.yaml').read_text()),
}}}, open('/tmp/rviz_params.yaml','w'), allow_unicode=True)
PY

rviz2 -d "$MC/config/openarm_v1.0/moveit.rviz" \
      --ros-args --params-file /tmp/rviz_params.yaml
```

Then: *MotionPlanning* → Planning Group `both_arms` → Goal State → **Plan** → **Execute**.

---

## 3. WSL2 networking — mirrored mode is required

Default WSL2 NAT lets WSL reach the robot but **not the reverse**, so DDS never
completes. Symptom: `ros2 node list` on Windows shows nothing from the robot.

`%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
firewall=false
guiApplications=true
```

Then `wsl --shutdown` and reopen. WSL now shares the Windows LAN address.

Open the DDS ports on the Windows side (admin PowerShell, once):

```powershell
New-NetFirewallRule -DisplayName "ROS2 DDS" -Direction Inbound `
  -Protocol UDP -LocalPort 7400-7700 -Action Allow -Profile Any
```

> ICMP stays blocked, so `ping` from the robot to the Windows box still fails even
> when DDS works. Do not use ping to test this — use `ros2 node list`.

---

## 4. `firewall=true` breaks WSLg (blank RViz window)

**Symptom:** `rviz2` starts, logs `OpenGl version: 4.2`, gets a taskbar button and a
real window handle — and paints nothing. The window title carries `[WARN:COPY MODE]`.

That warning is WSLg's VAIL path falling back from shared surfaces to a copy path.
With `firewall=true` under mirrored networking it never recovers and the window is
never composited onto the Windows desktop.

**Fix:** `firewall=false` in `.wslconfig`, then `wsl --shutdown`.

Trade-off: this disables the Hyper-V firewall for WSL only — the Windows host
firewall still applies — but WSL is directly exposed on the LAN. Use it on a
network you trust.

If a window is merely off-screen rather than blank, it can sit at negative
coordinates (`X=-26`) or the minimised sentinel (`-32000,-32000`). Check with:

```bash
xdotool search --class . | while read id; do
  echo "$(xdotool getwindowname $id): $(xdotool getwindowgeometry --shell $id | tr '\n' ' ')"
done
```

---

## 5. `robot_description` does not survive Wi-Fi DDS

**Symptom:**

```
[rviz]: Could not find parameter robot_description and did not receive
        robot_description via std_msgs::msg::String subscription within 10s
[planning_scene_monitor]: Robot model not loaded
```

`ros2 topic info /robot_description` reports publishers (3 of them), but
`ros2 topic echo /robot_description --once` returns **nothing**. Discovery works;
the payload does not arrive.

The URDF is ~51 KB, which DDS fragments across many UDP datagrams. Over Wi-Fi that
reassembly fails often enough to never complete.

**Fix: do not ship the model over the network.** Both machines have the same
workspace, so generate the URDF locally and pass it to RViz as a parameter (§2).
Only small messages — joint states, TF, planning requests — then cross the link.

---

## 6. MoveIt config layout vs `MoveItConfigsBuilder`

**Symptom:** `File .../config/pilz_cartesian_limits.yaml doesn't exist`, or RViz
opens with only a Grid and `Fixed Frame [map] does not exist`.

`move_group.launch.py` and `moveit_rviz.launch.py` build the config with
`MoveItConfigsBuilder`, which auto-discovers files at `config/<name>`. The OpenArm
package keeps them one level down in `config/openarm_v1.0/`. `demo.launch.py`
computes paths itself, which is why it works and the other two do not.

**Fix** — symlink the v1.0 files into the config root:

```bash
cd ~/ros2_ws/src/openarm_ros2/openarm_bimanual_moveit_config/config
for f in pilz_cartesian_limits.yaml moveit_controllers.yaml joint_limits.yaml \
         kinematics.yaml sensors_3d.yaml moveit.rviz \
         openarm_bimanual.srdf openarm_bimanual.urdf.xacro ros2_controllers.yaml; do
  [ -e "$f" ] || ln -s "openarm_v1.0/$f" "$f"
done
cd ~/ros2_ws && colcon build --packages-select openarm_bimanual_moveit_config --symlink-install
```

Also note `arm_type` differs between launch files: `demo.launch.py` takes `v10`,
while `move_group.launch.py` / `moveit_rviz.launch.py` use the value as a directory
name and need `openarm_v1.0`.

---

## 7. SSH access to the control PC

```bash
ssh-keygen -t ed25519 -f ~/.ssh/openarm_ros -N ''
ssh-copy-id -i ~/.ssh/openarm_ros.pub seongmin@10.2.12.118
ssh -i ~/.ssh/openarm_ros seongmin@10.2.12.118
```

A laptop used as the control PC will suspend when the lid closes and drop every
session. Disable it:

```bash
sudo tee /etc/systemd/logind.conf.d/99-nosleep.conf <<'EOF'
[Login]
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
IdleAction=ignore
EOF
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
sudo systemctl kill -s HUP systemd-logind
```

Verify it actually took — writing the file through `sudo tee` while piping the
password into `sudo -S` produces an **empty file**, because sudo consumes the piped
stdin as the password:

```bash
busctl get-property org.freedesktop.login1 /org/freedesktop/login1 \
  org.freedesktop.login1.Manager HandleLidSwitch     # must say "ignore"
```
