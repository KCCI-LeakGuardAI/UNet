# Jetson Nano 실시간 누수 추론

이 폴더만 Jetson Nano로 복사해서 실행할 수 있도록 모델 추론, 마스크 후처리,
누출 분석, 화면 표시, 카메라 입력과 상태 로그를 분리해 구성했습니다.

## 화면에 표시되는 정보

- 누출 면적: Monitoring ROI 전체 픽셀 대비 누출 픽셀 백분율
- 4단계 상태: `NORMAL`, `SMALL_LEAK`, `WARNING`, `DANGER`
- 순간 면적과 이동평균 면적
- 연속 프레임 기반 누출 확정 여부
- 면적 증가 기반 확산 여부와 증가량
- Danger ROI 침범 여부와 중첩 픽셀 수
- 분리된 누출 덩어리 수
- U-Net 추론 시간, 전체 처리 시간, FPS, GPU/CPU
- Monitoring ROI, Danger ROI, 누출 overlay와 이진 마스크 미리보기

키보드:

- `Q` 또는 `Esc`: 종료
- `S`: 현재 화면을 `snapshots/`에 저장
- `R`: 이동평균·연속 프레임·확산 판정 상태 초기화

상태는 기본적으로 `logs/realtime_status.csv`에 1초 주기로 기록되며 경고 단계가
바뀌면 즉시 기록됩니다.

## 1. Jetson으로 복사

다음 폴더 전체를 Jetson Nano로 복사합니다.

```text
Jetson Nano/
├─ run_realtime.py
├─ config.yaml
├─ requirements-jetson.txt
├─ leakguard/
├─ models/
│  └─ best_model.h5
└─ tests/
```

현재 PC에서는 학습된 모델이 `models/best_model.h5`에 복사되어 있습니다.
Git에는 용량 때문에 모델을 올리지 않으므로 GitHub에서 코드를 받은 경우 다음
원본 모델을 별도로 복사해야 합니다.

```text
artifacts/training/models/best_model.h5
→ Jetson Nano/models/best_model.h5
```

## 2. Jetson 환경 확인

JetPack에 맞는 NVIDIA TensorFlow 2.4.1이 이미 설치된 환경을 사용합니다.
일반 PyPI의 TensorFlow를 새로 설치하면 Jetson GPU 빌드가 덮어써질 수 있습니다.

```bash
cd "Jetson Nano"
python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
python3 -c "import cv2; print(cv2.__version__)"
python3 -m pip install -r requirements-jetson.txt
```

OpenCV가 없다면 JetPack의 시스템 패키지를 우선 사용합니다.

```bash
sudo apt update
sudo apt install python3-opencv
```

## 3. USB 웹캠 확인

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

필요하면 도구를 설치합니다.

```bash
sudo apt install v4l-utils
```

PLEOMAX USB 웹캠이 `/dev/video0`이라면 기본 설정 그대로 실행할 수 있습니다.

## 4. 실시간 실행

```bash
cd "Jetson Nano"
python3 run_realtime.py
```

다른 카메라 번호:

```bash
python3 run_realtime.py --camera 1
```

모니터 없이 SSH에서 실행:

```bash
python3 run_realtime.py --headless
```

영상 또는 이미지로 먼저 검사:

```bash
python3 run_realtime.py --video sample.mp4
python3 run_realtime.py --image sample.jpg
python3 run_realtime.py --image sample.jpg --headless
```

CSI 카메라는 GStreamer pipeline을 직접 전달할 수 있습니다.

```bash
python3 run_realtime.py --gstreamer "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
```

## 5. 경고 단계

기본 기준은 `config.yaml`에서 수정할 수 있습니다.

```text
면적 < 0.3%                  → NORMAL
0.3% 이상, 1.5% 미만        → SMALL_LEAK
1.5% 이상, 3.5% 미만        → WARNING
3.5% 이상                   → DANGER
Danger ROI 20픽셀 이상 침범 → 즉시 DANGER
```

일반 누출은 최근 3프레임 중 2프레임 이상 검출돼야 확정됩니다. Danger ROI 침범과
면적 3.5% 이상은 안전을 위해 즉시 DANGER로 표시됩니다. 확산 상태가 충분히
지속되면 설정에 따라 한 단계 상승합니다.

현재 임계값은 프로젝트 초기 예시값이므로 실제 카메라에서 새로운 검증 영상을
수집한 뒤 재조정해야 합니다.

## 6. ROI 조정

`config.yaml`의 좌표는 640×480 기준이며 실제 카메라 해상도에 맞춰 자동
스케일됩니다.

```yaml
roi:
  monitoring:
    x: 80
    y: 60
    width: 480
    height: 360
  danger:
    x: 430
    y: 260
    width: 100
    height: 100
```

현재 모델은 촬영 당시와 동일한 전체 화면 구도를 입력받도록
`crop_to_monitoring_roi: false`가 기본입니다. 이 상태에서도 면적과 경고 판정은
Monitoring ROI 내부만 사용합니다. 이 값을 `true`로 바꾸면 모델 입력 구도가
달라지므로 ROI crop으로 다시 만든 데이터로 재학습한 뒤 사용하는 것을 권장합니다.

## 7. 코드 검사

Jetson 또는 PC에서 다음 명령으로 TensorFlow 모델을 열지 않고 분석 로직만 검사할
수 있습니다.

```bash
cd "Jetson Nano"
python3 -m pytest -q
```

실제 모델 포함 전체 검사:

```bash
python3 run_realtime.py --image sample.jpg --headless
```

## 주의사항

- 이 모델은 흰색 폼보드와 갈색 시험액 환경으로 학습됐습니다.
- 실제 설치 전 Jetson에 연결한 동일 웹캠으로 외부 검증 데이터를 촬영해야 합니다.
- 카메라 위치·각도·폼보드가 바뀌면 ROI 좌표를 다시 맞춰야 합니다.
- 화면 표시가 필요하면 Jetson 데스크톱에서 실행하거나 올바른 `DISPLAY` 환경이
  연결돼 있어야 합니다.
- UART와 TCP 통신은 이번 실시간 추론 코드의 범위에 포함하지 않았습니다.
