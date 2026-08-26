# OpenArm 시작 절차서

> 하드웨어: **OpenArm v1.0 양팔** (리더 1대 + 팔로워 1대)
> 환경: Windows 11 + WSL2 Ubuntu 22.04 + ROS 2 Humble
> 최종 갱신: 2026-08-25

---

## 0. 전원 켜는 순서

1. **컴퓨터 부팅**
2. **PCAN 어댑터 2대를 USB에 연결** — 항상 같은 포트를 쓸 것
   (포트가 바뀌면 BUSID가 바뀌어 아래 절차가 어긋납니다)
3. **로봇 모터 전원 ON** — 리더·팔로워 둘 다
4. 팔이 안전한 자세인지, 작업 반경에 사람·물건이 없는지 확인
5. **비상 시 전원 차단 수단을 손 닿는 곳에** 확보

> 모터 전원을 안 켜면 CAN 링크는 올라와도 `discover`에서 0개가 나옵니다.
> 실제로 겪은 적 있음 — 그때는 전원 문제였습니다.

---

## 1. Windows — USB를 WSL로 넘기기

**관리자 PowerShell** (서비스 시작만 관리자 필요):

```powershell
Start-Service usbipd
```

**일반 PowerShell로 충분**:

```powershell
usbipd attach --wsl --busid 1-7
usbipd attach --wsl --busid 1-8
```

### 🔴 순서가 절대적으로 중요합니다

`can` 번호는 **attach 순서**로 정해집니다. `1-7`을 먼저 붙여야 합니다.

| BUSID | 역할 | 채널 |
|---|---|---|
| `1-7` | **팔로워** | `can0`, `can1` |
| `1-8` | **리더** | `can2`, `can3` |

순서가 뒤바뀌면 리더와 팔로워가 통째로 뒤집혀서, **리더에 토크가 걸리고 팔로워를 손으로 밀어야 하는** 상황이 됩니다.

확인:
```powershell
usbipd list      # 1-7, 1-8 이 Attached 인지
```

`bind`는 영구적이라 다시 할 필요 없습니다. 다만 초기화됐다면:
```powershell
usbipd bind --busid 1-7     # 관리자 필요
usbipd bind --busid 1-8
```

---

## 2. WSL — CAN 4채널 올리기

```bash
~/openarm_can_setup.sh
```

이 출력이 나와야 정상입니다:

```
can0   usb=1-1:1.0    FOLLOWER  ERROR-ACTIVE
can1   usb=1-1:1.0    FOLLOWER  ERROR-ACTIVE
can2   usb=1-2:1.0    LEADER    ERROR-ACTIVE
can3   usb=1-2:1.0    LEADER    ERROR-ACTIVE
All four channels ready.
```

- `ERROR-ACTIVE`는 **정상 상태**입니다 (이름과 달리 에러가 아님)
- `FOLLOWER`/`LEADER` 라벨이 뒤바뀌어 있으면 attach 순서가 틀린 것 → 둘 다 detach 후 재시도
- 1 Mbps / 5 Mbps CAN FD 로 설정됩니다

> `usbipd attach`만으로는 인터페이스가 `DOWN` 상태로 생깁니다.
> 반드시 이 스크립트로 비트레이트를 잡아줘야 합니다.

---

## 3. 모터 확인 (선택, 권장)

```bash
OACLI=~/ros2_ws/install/openarm_can/bin/openarm-can-cli
$OACLI -i can0 discover
```

각 채널에서 **8개**(`0x01`~`0x08` = 7관절 + 그리퍼)가 나와야 합니다.

- 0개 → 모터 전원 확인
- 7개 → 한 번 더 실행 (부팅 타이밍 문제로 놓칠 수 있음, 실제로 겪음)

### ⚠️ 명령별 안전성

| 명령 | 모터 상태 | 비고 |
|---|---|---|
| `candump can0` | 수신만 | 완전 안전 |
| `discover` | enable 안 함 | 안전 (단 비트레이트를 바꿨다 복원함) |
| **`monitor`** | **armed** | `enable_all()` 호출함. 토크 지령은 0 |
| `enable` | **armed** | 위험 |

---

## 4-A. MoveIt / RViz 로 쓰기

```bash
cd ~/ros2_ws && source install/setup.bash

# 시뮬레이션 (로봇 안 움직임) - 먼저 이걸로 확인할 것
ros2 launch openarm_bimanual_moveit_config demo.launch.py arm_type:=v10

# 실제 로봇
ros2 launch openarm_bimanual_moveit_config demo.launch.py \
  arm_type:=v10 \
  use_fake_hardware:=false \
  right_can_interface:=can0 \
  left_can_interface:=can1
```

- **`arm_type:=v10`** 을 빼면 v2.0 모델이 로드됩니다 (우리 로봇은 v1.0)
- **`use_fake_hardware:=false`** 를 빼면 가짜 하드웨어로 돕니다 — 실물이 안 움직임
- 띄우는 즉시 컨트롤러가 active 되어 **14축에 토크가 걸립니다**
- 이 터미널을 닫거나 `Ctrl+C` 하면 전체가 내려갑니다

RViz → MotionPlanning → Planning Group에서 `both_arms` 선택 시
양팔 14축을 한 번에 계획할 수 있습니다 (직접 추가한 그룹).

Goal State 목록: `home`, `hands_up`, `wave_ready`, `wave_out`, `wave_in`

---

## 4-B. 손 흔들기 스크립트

MoveIt(4-A)을 띄운 상태에서 **다른 터미널**:

```bash
cd ~/ros2_ws && source install/setup.bash

python3 openarm_wave.py --list                 # 자세 목록
python3 openarm_wave.py --cycles 2 --dry-run   # 숫자만 출력, 전송 없음
python3 openarm_wave.py --cycles 2 --preview   # RViz에서 재생, 로봇 안 움직임
python3 openarm_wave.py --cycles 2             # 실제 실행
```

기본 시퀀스: `wave_ready → wave_in → wave_out → home` 반복

주요 옵션:

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--cycles` | 3 | 반복 횟수 |
| `--speed` | 0.6 | rad/s, 구간 시간 산정 |
| `--min-segment` | 1.0 | 구간 최소 시간(초) |
| `--hold` | 0.0 | 각 자세에서 멈춤(초) |
| `--arms` | both | `left` / `right` |
| `--sequence` | 위 4단계 | 쉼표로 자유 구성 |

- 전송 전에 모든 점을 실제 관절 한계와 대조하고, 벗어나면 **중단**합니다
- 좌우가 공통 타임라인을 쓰므로 두 팔이 정확히 같이 움직입니다
- `--speed`만 낮추면 짧은 구간은 안 느려집니다 → `--min-segment`도 같이 올릴 것

---

## 4-C. 리더-팔로워 텔레오퍼레이션

**MoveIt 등 ROS 노드를 먼저 전부 종료하세요.** 같은 CAN을 두고 충돌합니다.

```bash
cd ~/openarm_teleop        # ← 필수. YAML을 상대경로로 찾습니다

# 단방향 (리더 → 팔로워). 먼저 이것부터.
./build/unilateral_control urdf/leader.urdf urdf/follower.urdf right_arm can2 can0
./build/unilateral_control urdf/leader.urdf urdf/follower.urdf left_arm  can3 can1

# 양방향 (힘 피드백). 단방향 검증 후에만.
./build/bilateral_control  urdf/leader.urdf urdf/follower.urdf right_arm can2 can0
```

인자 순서:
```
<leader_urdf> <follower_urdf> [arm_side] [leader_can] [follower_can]
                                          ^리더 먼저   ^팔로워 나중
```

### 🔴 CAN 인자를 생략하지 마세요

바이너리 기본값은 `leader=can0, follower=can2` 인데 **우리 배선과 정반대**입니다.

```
오른팔:  can2 can0          왼팔:  can3 can1
```

### 정상 동작 참고

- **리더가 스스로 살짝 뜨는 것은 정상**입니다. 중력·마찰·코리올리 보상 토크를
  걸어 손으로 가볍게 움직이도록 만드는 설계입니다 (`control.cpp:278`).
  위치 게인은 0이라 특정 자세로 끌고 가지 않습니다.
- 리더를 놓았을 때 그 자리에 떠 있고, 밀면 가볍게 따라오면 정상
- 스스로 특정 자세로 이동하려 하면 비정상 → 즉시 `Ctrl+C`

### 종료

**반드시 `Ctrl+C`** 로 종료하세요. SIGINT 핸들러가 `disable_all()`을 호출해
모터를 안전하게 해제합니다. 강제 종료(`kill -9`)하면 **모터가 armed로 남습니다.**

---

## 5. 종료 절차

1. teleop / ROS 노드를 `Ctrl+C` 로 종료 (강제 종료 금지)
2. 프로세스가 남았는지 확인
   ```bash
   ps -eo pid,cmd --no-headers | grep -E 'unilateral|bilateral|/opt/ros/' | grep -v grep
   ```
3. 로봇 모터 전원 OFF
4. (선택) CAN 링크 내리기
   ```bash
   for i in can0 can1 can2 can3; do sudo ip link set $i down; done
   ```

---

## 6. 🔴 자주 발생하는 장애 — usbip 단절

**55분 동안 3번 발생했습니다.** 증상은 항상 동일: **CAN 4채널이 한꺼번에 사라짐.**

| 원인 | 판별법 |
|---|---|
| WSL VM 재시작 | `uptime` 이 `up 0 min`, eth0 MAC 변경 |
| usbipd 서비스 크래시 | `Get-Service usbipd` → `Stopped` (exit 1067) |

**제어 중 발생하면 즉시 제어 상실입니다.** 팔이 들려 있으면 내려앉을 수 있습니다.

### 복구

```powershell
Start-Service usbipd                      # 관리자, Stopped 일 때만
usbipd attach --wsl --busid 1-7           # 순서 중요
usbipd attach --wsl --busid 1-8
```
```bash
~/openarm_can_setup.sh
```

### 진단 명령

```powershell
Get-Service usbipd
usbipd list
```
```bash
ip -br link show | grep can
lsusb | grep -i 0c72
uptime
```

### 근본 해결

세 번 모두 **usbip 계층**에서 발생했고, CAN 링크와 모터는 매번 정상이었습니다.
**로봇·어댑터 문제가 아니라 Windows↔WSL USB 포워딩 문제**입니다.

양방향 제어는 통신량이 배로 늘어 더 자주 끊길 가능성이 높습니다.
**네이티브 리눅스로 부팅하면 usbip 계층이 사라져 이 문제가 근본적으로 없어집니다.**
1kHz 실시간 스케줄링(`FIFO RT scheduling` 경고)도 함께 해결됩니다.
지금까지 만든 것은 전부 리눅스에서 그대로 씁니다.

---

## 7. 파일 위치

| 경로 | 내용 |
|---|---|
| `~/openarm_can_setup.sh` | CAN 4채널 복구 |
| `~/ros2_ws/` | ROS 2 워크스페이스 (6패키지 빌드됨) |
| `~/ros2_ws/openarm_wave.py` | 손 흔들기 시퀀스 플레이어 |
| `~/openarm_teleop/` | 텔레오퍼레이션 (순수 CMake) |
| `~/openarm_teleop/build/` | `unilateral_control`, `bilateral_control`, `comm_test`, `gravity_comp` |
| `~/openarm_teleop/urdf/` | teleop용 URDF (카메라 제외) |
| `~/openarm_teleop/config/` | `leader.yaml`, `follower.yaml` (게인·마찰 계수) |

### 직접 수정한 upstream 파일 (백업 있음)

| 파일 | 변경 | 백업 |
|---|---|---|
| `.../openarm_v1.0/urdf/openarm_v10.urdf.xacro` | ZED 카메라 3대 추가 (`cameras:=true` 기본) | `/tmp/openarm_v10_xacro.bak` |
| `.../config/openarm_v1.0/openarm_bimanual.srdf` | `both_arms` 그룹 + 자세 5개 | `/tmp/openarm_v10_srdf.bak` |
| `.../config/openarm_v2.0/openarm_bimanual.srdf` | 동일 (v2.0용, 미사용) | `/tmp/openarm_bimanual.srdf.bak` |

`/tmp` 백업은 **재부팅하면 사라집니다.** 오래 보관하려면 옮겨두세요.
`git pull` 시 충돌할 수 있습니다.

---

## 8. 아직 확인 안 된 것

- **좌우 배선** — 각 어댑터의 첫 채널(`can0`/`can2`)이 오른팔이라고 가정한 상태입니다.
  반대라면 좌우가 뒤바뀐 채 동작합니다. 오른쪽 리더를 움직였는데 왼쪽 팔로워가
  따라가면 그것이니, `can2 can0` 대신 `can3 can1`로 바꿔 확인하세요.
- **카메라 장착 좌표** — URDF의 ZED 3대 위치는 추정값입니다.
  `openarm_v10.urdf.xacro` 맨 아래 `mount_origin` 에서 조정하세요.
- **WSL에서 1kHz 제어 유지 여부** — `FIFO RT scheduling` 을 못 씁니다.
  양방향 제어에서 루프가 밀리면 진동이 생길 수 있습니다.
