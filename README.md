# OpenArm v1.0 — Bimanual Setup, Cameras, and Leader–Follower Teleoperation

Working configuration for a **bimanual OpenArm v1.0** pair (one leader, one follower)
on **ROS 2 Humble**, including RealSense camera integration in the URDF and
leader–follower teleoperation over CAN FD.

Everything here was derived and verified on real hardware, not from documentation
alone. Where a value came from measurement or CAD, that is stated.

---

## What this repository contains

| Path | Contents |
|---|---|
| `docs/STARTUP.md` | Full power-on → run procedure, failure modes, recovery |
| `docs/TELEOP.md` | Leader–follower setup, CAN channel mapping, safety |
| `docs/DISTRIBUTED.md` | Linux control PC + Windows/WSL2 planning station over DDS |
| `scripts/openarm_can_setup.sh` | Brings up all four CAN FD channels, prints the role mapping |
| `scripts/openarm_wave.py` | Bimanual named-pose sequence player (RViz preview + real execution) |
| `overlay/` | Modified upstream files — cameras in the URDF, `both_arms` planning group |

The `overlay/` tree mirrors the upstream package layout. Copy the files over a
normal upstream checkout (see *Install*).

---

## Hardware this was built against

- **2 × OpenArm v1.0 bimanual arms** (one leader, one follower)
- **2 × PEAK PCAN-USB Pro FD** (2 CAN FD channels each → 4 total, which is the minimum for bimanual teleop)
- **RealSense D435** on the chest, **RealSense D405 × 2** on the wrists
- Host: Windows 11 + WSL2 (Ubuntu 22.04) — see the caveat below

> **WSL2 caveat.** CAN and cameras reach WSL2 through `usbipd`, which proved
> unreliable under load: the USB forwarding dropped **three times in 55 minutes**
> (once a WSL VM restart, twice a `usbipd` service crash, exit code 1067). Each
> drop kills all four CAN channels at once — during teleoperation that is an
> immediate loss of control. The CAN links themselves were always clean
> (error counters 0). **For real use, boot native Linux.** See `docs/STARTUP.md` §6.

---

## Install

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/enactic/openarm_description.git
git clone https://github.com/enactic/openarm_ros2.git
vcs import . < openarm_ros2/openarm.repos      # pulls openarm_can
```

Two dependencies are **not** resolvable with `rosdep` and must be installed by hand:

```bash
sudo apt install ros-humble-zed-description   # openarm_description needs it, no rosdep rule
sudo apt install libcli11-dev                 # openarm_can needs CLI11, build aborts without it
```

> **`rosdep install` can fail silently.** It shells out to `sudo -H apt-get install`.
> On a machine where sudo needs a password, and in a session without a tty (SSH,
> scripted install), those calls fail and rosdep still exits 0. The build then breaks
> far away from the cause:
>
> - missing `ros2_control` → `openarm_hardware`: *Could not find hardware_interface*
> - missing MoveIt → `ModuleNotFoundError: No module named 'moveit_configs_utils'`
>
> Check with `rosdep check --from-paths src --ignore-src --rosdistro humble`; it must
> say *All system dependencies have been satisfied*. If not, install explicitly:
>
> ```bash
> sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers \
>   ros-humble-moveit ros-humble-moveit-configs-utils ros-humble-ros-gz
> ```

Then the rest, plus RealSense:

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
sudo apt install ros-humble-realsense2-camera ros-humble-realsense2-description \
                 ros-humble-librealsense2 can-utils v4l-utils
```

Apply this repository's overlay and build:

```bash
git clone https://github.com/SeongminJaden/openarm-bimanual-setup.git ~/openarm-bimanual-setup
cp -r ~/openarm-bimanual-setup/overlay/openarm_description/*  ~/ros2_ws/src/openarm_description/
cp -r ~/openarm-bimanual-setup/overlay/openarm_bimanual_moveit_config/* \
      ~/ros2_ws/src/openarm_ros2/openarm_bimanual_moveit_config/
cp ~/openarm-bimanual-setup/scripts/openarm_wave.py ~/ros2_ws/
cp ~/openarm-bimanual-setup/scripts/openarm_can_setup.sh ~/

cd ~/ros2_ws && colcon build --symlink-install
```

---

## Run

### Visualise (no robot, no CAN)

```bash
cd ~/ros2_ws && source install/setup.bash
ros2 launch openarm_bimanual_moveit_config demo.launch.py arm_type:=v10
```

`arm_type:=v10` matters — without it the **v2.0** model loads.

RViz → *MotionPlanning* → *Planning Group* → **`both_arms`** plans all 14 joints
together, with inter-arm collision checking. Named goal states: `home`,
`hands_up`, `wave_ready`, `wave_out`, `wave_in`.

### Drive the real robot

```bash
ros2 launch openarm_bimanual_moveit_config demo.launch.py \
  arm_type:=v10 use_fake_hardware:=false \
  right_can_interface:=can0 left_can_interface:=can1
```

⚠️ Controllers go active immediately — **all 14 joints are energised before you
press Plan.** Support the arms and keep a power cut-off within reach.

### Wave sequence

```bash
cd ~/ros2_ws && source install/setup.bash

python3 openarm_wave.py --list                 # available poses
python3 openarm_wave.py --cycles 2 --dry-run   # numbers only, publishes nothing
python3 openarm_wave.py --cycles 2 --preview   # animates in RViz, robot does not move
python3 openarm_wave.py --cycles 2             # executes
```

Default sequence: `wave_ready → wave_in → wave_out → home`, repeated.

Both arms share one timeline, so they reach every pose at the same instant.
Every point is checked against the real joint limits read from
`/robot_description`; if any point is out of range the trajectory is **not sent**.

Useful flags: `--speed` (rad/s), `--min-segment` (s), `--hold` (s),
`--arms left|right|both`, `--sequence`, `--preview-dt`.

> `--preview` resamples the path at 0.05 s to match RViz's *State Display Time*.
> Without that, RViz replays a sparse trajectory at a fixed step and the motion
> looks ~40× too fast — the preview speed is a display setting, not the real speed.

### Teleoperation

See **[docs/TELEOP.md](docs/TELEOP.md)**.

---

## Cameras

Three RealSense cameras are in the URDF, positioned from the official CAD assembly
(`OpenArm_v1.1_follower.STEP` with the camera attachments fitted). Alignment was
done by **matching bolt-hole patterns**, not by eye:

| Interface | Hole pitch | Axis |
|---|---|---|
| D405 ↔ wrist housing | 20 mm | `(0, 0.6, 0.8)` |
| D435 ↔ chest bracket | 45 mm | `(−0.707, 0.707, 0)` |
| chest bracket ↔ post | 30 mm | 60×60 extrusion T-slot |

Resulting frames (zero pose, world mm), verified against CAD to **0.00 mm**:

```
openarm_right_camera_link   [ 68.48, -153.50, 150.10]   optical axis [-0.600, 0, -0.800]
openarm_left_camera_link    [ 68.48,  153.50, 150.10]   optical axis [-0.600, 0, -0.800]
openarm_chest_camera_link   [ 55.44,    0.00, 634.11]   optical axis [ 0.707, 0, -0.707]
```

Wrist cameras look back-and-down at the gripper; the chest camera looks
forward-and-down at the workspace.

Physical mounts are included as meshes (`*_camera_mount`, `*_camera_body` links),
so the brackets and housings show up in RViz rather than a floating camera.

`cameras:=false` builds the bare robot.

### Two conversions that are easy to get wrong

**1. The CAD part frame is not the camera frame.** The CAD modules have **+Z**
as the optical axis (mounting holes on the −Z face); `realsense2_description`
uses **+X**. Copying the CAD rotation straight into the URDF points the cameras
sideways. The overlay applies the mapping `X_urdf→Z_cad, Y_urdf→−X_cad, Z_urdf→−Y_cad`.

**2. Use arm-relative coordinates, not CAD-absolute.** The CAD places each arm at
|Y| = 152.5 mm, the URDF at 153.498 mm. Transplanting absolute CAD coordinates
carries that ~1 mm discrepancy into the mounts and pushes both cameras inboard of
their own arm. Referencing the hand link instead makes the two sides identical.

---

## Command safety reference

Verified by reading the source, not inferred from the names:

| Command | Motor state | Notes |
|---|---|---|
| `candump can0` | receive only | safe |
| `openarm-can-cli … discover` | not enabled | safe; does change the interface bitrate while scanning, and restores it |
| `openarm-can-cli … monitor` | **armed** | calls `enable_all()` (`monitor_motor_status_commands.cpp:91`); torque command is 0 |
| `openarm-can-cli … enable` | **armed** | torque on |

---

## Licence and attribution

Original work in this repository (`scripts/`, `docs/`, and the modifications in
`overlay/`) is released under **Apache-2.0**, matching upstream.

Files in `overlay/` are **derivatives of Enactic, Inc. work**:

- `openarm_description`, `openarm_ros2` — Apache-2.0, © 2025–2026 Enactic, Inc.
- The `.stl` meshes under `overlay/.../sensor/realsense/meshes/` are exported from
  the **OpenArm hardware CAD**, which is licensed **CERN-OHL-S-2.0**
  (strongly reciprocal). They are redistributed here under that same licence.
  Source CAD: <https://github.com/enactic/openarm_hardware>

Intel RealSense CAD models are **not** redistributed here. Download them from
<https://dev.realsenseai.com/docs/cad-files/> if you need them.

See [`NOTICE`](NOTICE) for details.
