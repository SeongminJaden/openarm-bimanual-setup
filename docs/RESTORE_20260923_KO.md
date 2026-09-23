# 2026-09-23 노트북 보정 작업 백업 및 복원

이 커밋은 Ubuntu 22.04 / ROS 2 Humble 노트북에서 시험한 상태를 보존한다.
Ubuntu 24.04 / Isaac ROS 5.0으로의 이식이 완료됐다는 의미는 아니다.
설치 후 ROS 버전과 의존성을 맞추고 다시 빌드·검증해야 한다.

## 보존한 내용

- `overlay/openarm_description`: 노트북의 D455 가슴 카메라 설정, 보정된 장착값,
  사용하던 브래킷/커버 및 설명 파일 수정본.
- `overlay/openarm_vision_pick`: 기존 비전 패키지 전체 소스와 추가한 보정·관절각 읽기 노드,
  설정, 테스트, ID 10 마커 인쇄 파일.
- `calibration/2026-09-23`: 원본 관측 및 제외 후 재계산 결과. 이전 실패 결과도 구별해 보존.
- 기존 저장소의 MoveIt 설정과 운용 문서는 그대로 유지.

최종 적용 결과는 다음 폴더에 있다.

`calibration/2026-09-23/20260923_113311_888298/without_samples_13_29_20260923_113934/`

원본 31개 중 사용자가 지정한 13번·29번을 제외했고, 원래 검증 표본
5, 10, 15, 20, 25, 30번은 그대로 유지했다.

| 최대 잔차 | 계산용 23개 | 검증용 6개 |
|---|---:|---:|
| 위치 | 4.562 mm | 3.703 mm |
| 방향 | 1.705° | 1.177° |

통과 조건은 최대 10 mm 및 2°이며 절대 정확도를 보장하는 인증 기준은 아니다.
통과값을 소스 URDF와 실행 중 TF에 반영해 일치를 확인했다.

```xml
<xacro:arg name="chest_camera_xyz" default="0.044070223 0.004944888 0.625958005" />
<xacro:arg name="chest_camera_rpy" default="0.023831869 0.778021855 0.035976233" />
```

위치는 m, 각도는 rad이고 `openarm_body_link0` →
`openarm_chest_camera_bottom_screw_frame`의 절대 장착값이다.
동일한 실물 장착 상태에만 사용한다. 다른 OpenArm에 일반 보정값으로 적용하지 않는다.

## 소스 복원

새 환경에 공식 OpenArm 의존 저장소를 준비한 다음 이 저장소의 오버레이를 적용한다.
아래 명령은 파일 복원만 하며 로봇을 구동하지 않는다.
기존 작업 폴더가 있다면 덮어쓰기 전에 따로 백업한다.

```bash
cd ~/openarm-bimanual-setup
mkdir -p ~/ros2_ws/src/openarm_vision_pick
cp -a overlay/openarm_vision_pick/. ~/ros2_ws/src/openarm_vision_pick/
cp -a overlay/openarm_description/. ~/ros2_ws/src/openarm_description/
cp -a overlay/openarm_bimanual_moveit_config/. \
  ~/ros2_ws/src/openarm_ros2/openarm_bimanual_moveit_config/
mkdir -p ~/chest_calibration
cp -a calibration/2026-09-23/. ~/chest_calibration/
```

시험 당시 의존 저장소의 기준 커밋:

| 저장소 | 커밋 |
|---|---|
| enactic/openarm_description | `1fba2cbc05001f05b4514120b70130b4ac06f409` |
| enactic/openarm_ros2 | `4e837e1d0dae692ff67b560b69d8d281d7a8d4ed` |
| enactic/openarm_can | `a30364622ca939b8c8a741167c317b3e95e38a49` |

패키지 의존성과 ROS 환경을 준비한 후의 빌드 대상은
`openarm_description`, `openarm_vision_pick`이다. Ubuntu 24.04에서
`/opt/ros/humble/setup.bash`를 그대로 사용하지 말고 실제 설치한 ROS 환경을 사용한다.
Humble용 기존 하드웨어/MoveIt 패키지의 Lyrical 호환성은 아직 검증하지 않았다.

## 수동 보정 운용

원래 노트북에서는 오른팔 `can0`, 왼팔 `can1`, CAN-FD 1 Mbps / 5 Mbps를 사용했다.
재설치 후 USB 채널 이름과 실제 좌우 팔 대응을 다시 확인한다.
마커는 왼팔의 고정 손 링크에 붙인 ArUco `DICT_4X4_50`, ID 10, 검정 사각형 60 mm다.
인쇄 후 실측 길이를 설정에 반영한다.

ROS 환경을 불러온 터미널마다 `export ROS_LOCALHOST_ONLY=1`을 동일하게 설정한다.
각 노드는 별도 터미널에서 한 번씩만 실행한다.

```bash
ros2 run openarm_vision_pick joint_state_readonly --ros-args \
  -p right_can_interface:=can0 -p left_can_interface:=can1
```

이 노드는 팔 7개와 그리퍼의 상태 조회만 전송한다. 모터 활성화/비활성화,
위치·토크 지령, 영점 변경은 하지 않는다. 비활성 상태의 실제 응답만 발행하고
누락된 관절각을 0이나 과거 값으로 대체하지 않는다.
평행 그리퍼 변환은 기존 하드웨어 코드의 근사값 `0.044 / -1.0472` m/rad이며
`gripper_m_per_rad`로 조정할 수 있다. 다른 그리퍼 형태에는 재검토가 필요하다.

```bash
xacro ~/ros2_ws/src/openarm_description/assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro \
  ros2_control:=false > /tmp/chest_calibration_robot.urdf
ros2 run robot_state_publisher robot_state_publisher /tmp/chest_calibration_robot.urdf
```

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=chest camera_name:=openarm_chest_camera \
  serial_no:=_419122302733 enable_depth:=false enable_infra1:=false enable_infra2:=false \
  rgb_camera.color_profile:=640x480x30
```

```bash
ros2 run openarm_vision_pick chest_calibration --ros-args \
  --params-file ~/ros2_ws/src/openarm_vision_pick/config/chest_calibration.yaml
```

팔을 손으로 옮긴 뒤 정지 상태에서 수집한다. 새 노드는 새 세션을 만든다.
원본 관측의 제외 작업은 오프라인 사본에서 수행했으며 실행 중 노드에서 삭제한 것이 아니다.

```bash
ros2 service call /chest_calibration/capture std_srvs/srv/Trigger '{}'
ros2 service call /chest_calibration/solve std_srvs/srv/Trigger '{}'
```

일반 bringup은 활성화 시 모터를 켜고 영점 자세로 이동할 수 있으므로 이 수동 보정 절차와
혼용하지 않는다. 카메라·관절각·TF 노드를 중복 실행하지 않는다.

## 확인한 검사

```bash
cd ~/ros2_ws/src/openarm_vision_pick
python3 test/test_joint_state_readonly.py
python3 test/test_chest_calibration.py --ros
```

실물에서 양팔 약 20 Hz 관절각, 그리퍼 및 팔 TF, D455f 640×480 RGB 영상,
촬영 시각 TF 조회와 보정값 적용을 확인했다.
USB-CAN 연결이 끊기면 장치를 재연결하고 CAN 설정 후 읽기 노드를 다시 시작한다.
TF 시간을 강제로 최신값으로 바꿔 오래된 관절 상태를 사용하지 않는다.

## 별도 전체 소스 백업

Git에는 보정 작업 관련 파일을 올렸다. 이와 별도로 노트북의 `ros2_ws/src` 전체와
보정 기록을 다음 파일에 보관했다. `.git`, 빌드 결과, 캐시는 포함하지 않는다.

- 파일: `openarm_full_src_before_ubuntu24_20260923.tar.gz`
- 현재 Windows PC: `C:/Users/pa/OpenArmBackup/20260923/`
- SHA-256: `d52810c73674853ce240b204c8c338a384280c8f19b9aa3360bd28ada7980ecb`

이는 노트북 전체 디스크 백업이 아니다. 홈 폴더의 다른 프로젝트·문서·인증키·설정은
이 백업 범위에 포함되지 않는다.
