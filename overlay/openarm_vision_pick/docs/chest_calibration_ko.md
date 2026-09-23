# 가슴 D435 카메라 장착 오차 보정 예제

목적은 **실제 가슴 카메라의 몸통 기준 위치·각도**를 측정해 URDF의
`chest_camera_xyz`, `chest_camera_rpy`를 바꾸는 것입니다.
카메라는 몸통에 고정하고, 마커를 고정한 팔을 움직입니다.
마커와 팔 사이의 정확한 위치·각도는 몰라도 됩니다. 함께 추정합니다.
프로그램은 로봇 이동 명령, TF 발행, 기존 URDF 수정은 하지 않습니다.

## 준비

- 인쇄용 파일: `assets/aruco_4x4_50_id10_60mm.svg`.
  검정 사각형 60 × 60 mm, 흰 여백 사방 10 mm로 전체 80 × 80 mm입니다.
  SVG를 **실제 크기 / 100%**로 인쇄하고 페이지에 맞추기를 끕니다.
  `aruco_4x4_50_id10_preview.png`는 미리보기이며 인쇄 크기 정보가 없습니다.
  무광 종이를 평평하게 붙이고 검정 테두리와 흰 여백을 가리지 마세요.
- 평평한 ArUco 마커를 출력 중인 지그에 단단히 붙입니다. 예제 기본값은
  `DICT_4X4_50`, ID `10`입니다. 다른 사전/ID를 사용했다면 설정을 맞춥니다.
- **검정 사각형 외곽 한 변의 실제 길이**를 재서 `marker_size_m`에 미터로 넣습니다.
  흰 여백과 3D 프린트 지그 크기는 포함하지 않습니다. 인쇄 배율 오차도 측정에 반영합니다.
  기본값 `0.0`은 측정 없이 잘못된 보정을 실행하지 않도록 의도적으로 실행을 막습니다.
- 지그가 손 링크에 고정돼 있으면 `openarm_right_hand`를 사용합니다.
  왼손이면 `openarm_left_hand`, 전완에 고정했다면 **실제로 지그가 고정된 링크**를
  `hand_frame`으로 설정합니다. 중간에 움직이는 관절이 있으면 안 됩니다.
- 실제 관절 상태를 사용하는 `robot_state_publisher`와 가슴 카메라를 실행합니다.
  가짜 관절 상태나 지령 관절각으로 실제 팔 자세를 대체하면 안 됩니다.
  팔의 관절 영점/링크 모델이 정확하다는 전제이며, 그 오차는 별도 보정 대상입니다.
- RealSense 드라이버의 카메라 내부 TF가 필요합니다. 같은 내부 TF를 URDF의
  nominal extrinsics와 드라이버가 중복 발행하지 않도록 기존 구성을 확인합니다.
  실제 장치가 연결된 환경에서 사용하고, `use_sim_time`은 실제 시간 설정에 맞춥니다.

마커가 아직 없다면 Ubuntu의 기본 OpenCV로 생성할 수 있습니다.
인쇄 후 **실제 검정 사각형 크기를 다시 측정**하세요.

```bash
/usr/bin/python3 - <<'PY'
import cv2
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
image = (cv2.aruco.generateImageMarker(d, 10, 600)
         if hasattr(cv2.aruco, 'generateImageMarker') else cv2.aruco.drawMarker(d, 10, 600))
image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
assert cv2.imwrite('arm_marker_10.png', image)
PY
```

## 빌드와 실행

아래 명령은 Ubuntu/WSL 터미널 기준입니다.

```bash
cd /home/pa/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select openarm_vision_pick --symlink-install
source install/setup.bash
```

`src/openarm_vision_pick/config/chest_calibration.yaml`에서 마커 크기와 링크를 설정합니다.
토픽은 예제값이므로 `ros2 topic list`로 실제 가슴 카메라의 RGB 토픽과
해당 `camera_info` 토픽을 확인합니다. 손목 카메라를 선택하지 마세요.
`image_raw`와 그 영상에 맞는 CameraInfo를 사용합니다. 보정 영상에 원본 왜곡값을
다시 적용하면 안 됩니다. 영상과 CameraInfo의 frame_id는 `optical_frame`과 같아야 합니다.

```bash
ros2 run openarm_vision_pick chest_calibration --ros-args \
  --params-file src/openarm_vision_pick/config/chest_calibration.yaml
```

예를 들어 **실측값이 60 mm인 경우에만** 설정 파일 대신 크기를 덮어쓸 수 있습니다.

```bash
ros2 run openarm_vision_pick chest_calibration --ros-args \
  --params-file src/openarm_vision_pick/config/chest_calibration.yaml \
  -p marker_size_m:=0.060
```

## 자세 수집

1. 기존 로봇 조작 방법으로 마커가 가슴 영상에 보이는 자세를 만듭니다.
2. 팔을 완전히 멈추고 약 1초 기다립니다.
3. 다른 터미널에서 아래 서비스를 호출합니다.
4. 위치와 회전 방향을 바꿔 20~30회 반복합니다. 최소 15개가 필요합니다.
   적어도 두 개의 서로 다른 축으로 충분히 기울여야 합니다.
   이동만 하거나 손목 한 축만 돌리는 데이터로는 보정하지 않습니다.
   매 다섯 번째 자세는 검증용이므로 이 자세들도 다양하게 수집합니다.

```bash
source /opt/ros/humble/setup.bash
source /home/pa/ros2_ws/install/setup.bash
ros2 service call /chest_calibration/capture std_srvs/srv/Trigger '{}'
```

반환값 `success: true`와 수집 개수를 확인합니다. 흐림, 가림, 너무 작게 보이는 마커는
피합니다. 정면에 가까운 평면 마커는 방향 해가 모호할 수 있어 약간 기울여 관측합니다.
감지가 안 되면 RGB 영상에서 사전/ID, 여백, 크기를 확인합니다.

수집기는 영상 시각의 팔 TF를 사용합니다. 이전 0.5초와 비교해 팔 위치가 1 mm 또는
각도가 0.5도보다 달라지면 거절합니다. 이 검사는 전체 구간의 정지를 보장하지 않으므로
반드시 실제로 멈춘 상태에서 수집하세요. TF 시각 오류가 나면 실제 관절 상태가 지속해서
발행되는지 확인하고, 최신 TF로 강제로 대체하지 마세요.

## 계산과 결과

```bash
ros2 service call /chest_calibration/solve std_srvs/srv/Trigger '{}'
```

출력은 `~/chest_calibration/날짜_시간/`에 저장됩니다.

- `samples.json`: 설정, 영상 시각, 팔/마커 변환, 검출 코너, 내부 파라미터 및 기존 장착 TF.
  매 수집마다 저장합니다. 새 실행은 새 폴더를 사용하며 이전 파일을 덮어쓰지 않습니다.
- `result.json`: 추정 카메라/마커/장착점 변환, 기존 URDF와의 차이, 학습·검증 오차.
- `chest_camera_origin.xacro.txt`: 검증을 통과한 경우에만 생성하는 교체용 두 줄.

결과는 마지막으로 `solve`한 시점의 데이터에 해당합니다. 추가 수집 후에는 반드시
다시 `solve`하고 `result.json`의 `sample_count`를 확인합니다.

매 다섯 번째 관측은 계산에 넣지 않고 검증에만 사용합니다. 기본 통과 조건은 학습 및
검증 데이터 모두 최대 위치 잔차 10 mm, 최대 방향 잔차 2도 이하입니다.
이는 예제용 품질 기준이며 **실제 절대 정확도 보증이 아닙니다**. 실패하면 먼저 마커 크기,
지그 흔들림, 팔 영점, 회전 다양성, 렌즈 파라미터를 확인합니다. 통과시키기 위해 임계값만
느슨하게 하지 마세요. 표본이 잘못됐다면 새 실행에서 다시 수집합니다.

## URDF 반영

실물 검증 전 기존 값을 기록해 둡니다. 현재 소스의 대상 파일은:

`src/openarm_description/assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro`

이 파일의 `chest_camera_xyz`와 `chest_camera_rpy` 기본값 두 줄을 생성된 두 줄로
교체하는 방식입니다. 위치 단위는 m, RPY는 rad이며, 기존 값에 더하는 보정량이 아니라
**새 절대 장착값**입니다. 실행 launch가 같은 인자를 덮어쓰는지 확인하고 해당 값도
일치시켜야 합니다. description 패키지를 빌드한 후 새 환경을 불러오고
`robot_state_publisher`를 포함한 로봇 설명 소비 노드를 재시작합니다.
설치본과 소스가 다를 수 있으므로 실행 중 로봇이 실제로 이 v1.0 모델을 쓰는지도 확인합니다.

**광학 좌표계에서 얻은 위치를 URDF에 그대로 붙여 넣지 않습니다.**
D435 매크로의 origin은 `openarm_chest_camera_bottom_screw_frame`을 지정합니다.
예제가 `mount_frame → optical_frame` 내부 변환을 제거해 이 장착점 기준으로 출력합니다.
이 내부 변환은 카메라 드라이버/모델 값이 정확하다는 전제입니다.
기존 카메라 장착 TF와 경쟁하는 별도 static TF를 추가하지 마세요.

반영 후 새로운 팔 자세 여러 개에서 마커가 예상한 위치와 맞는지 재확인합니다.
카메라를 물리적으로 옮기거나 지그가 흔들렸다면 새 데이터로 다시 보정합니다.
지그는 한 수집 세션 동안 같은 장착 상태를 유지해야 합니다.

## 하드웨어 없이 계산 확인 / 저장 데이터 재계산

```bash
cd /home/pa/ros2_ws/src/openarm_vision_pick
/usr/bin/python3 test/test_chest_calibration.py
```

ROS 메시지·TF 처리까지 가짜 데이터로 검사하려면 ROS 환경을 불러온 후
`/usr/bin/python3 test/test_chest_calibration.py --ros`를 실행합니다.
이 검사는 콜백을 직접 호출하며 실제 로봇이나 카메라를 사용하지 않습니다.
계산, 장착점 변환, 별도 관측 검증, ArUco 검출, TF 조회, 파일 저장을 검사했습니다.
현재 WSL에서 별도 프로세스 사이의 DDS 통신 검사는 요청 전달이 되지 않아 완료하지
못했습니다. 실제 환경에서는 카메라 영상 수신 로그와 `capture` 서비스 응답부터
확인해야 하며, 실물 정확도와 전체 통신 경로는 아직 검증되지 않았습니다.

저장한 관측을 다시 계산하려면 위 디렉터리에서 경로를 바꿔 실행합니다.
이 명령은 화면에 결과를 출력하며 URDF는 바꾸지 않습니다.

```python
import json
from openarm_vision_pick.chest_calibration import solve
with open('/home/pa/chest_calibration/세션폴더/samples.json') as f:
    data = json.load(f)
p = data['parameters']
result = solve(data['samples'], data['mount_camera'],
               p['max_position_error_m'], p['max_angle_error_deg'])
print(json.dumps(result, indent=2))
```

계산식은 `base_hand × hand_marker = base_camera × camera_marker`입니다.
OpenCV `calibrateHandEye`에 **팔 TF의 역변환**을 전달해 eye-to-hand 구성으로 풀고,
팔에 고정된 마커 변환은 학습 관측에서 평균합니다.
참고: [OpenCV hand-eye 공식 문서](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
