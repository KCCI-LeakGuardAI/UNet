# U-Net 기반 색상 액체 누출 감지

고정 웹캠으로 촬영한 영상에서 갈색 액체 영역을 이진 분할하는 프로젝트입니다.
PC에서는 TensorFlow 2.4.1로 U-Net을 학습하고, 이후 Jetson Nano에서 실시간
추론할 수 있도록 입력·출력 형식과 TensorFlow 버전을 맞춥니다.

## 현재 구현 범위

- 웹캠 촬영 및 노출 제어 가능 여부 진단
- HSV 기반 초기 이진 마스크 생성과 수동 검수 자료 생성
- 이미지/마스크 무결성 검사
- 원본 촬영 단위 Train/Validation/Test 분리
- Train 전용 동기화 데이터 증강
- TensorFlow `tf.data` 입력 파이프라인
- 약 195만 파라미터의 경량 U-Net
- BCE + soft Dice loss, 이진 Dice/IoU/Precision/Recall 평가
- 체크포인트, 조기 종료, 학습률 감소, 학습 곡선 저장
- Test 평가 CSV와 원본/정답/예측/오버레이 이미지 생성

## 폴더 구조

```text
configs/                  데이터, 증강, 모델, ROI 설정
dataset/
  images/                 검수 완료 원본 이미지
  masks/                  검수 완료 이진 마스크
  prepared/               split 및 Train 증강 결과
  metadata/               검수·split·증강 기록 CSV
scripts/                  촬영, 마스크, 검수, split, 증강 도구
src/
  data/                   전처리, pair 탐색, tf.data loader
  models/                 U-Net, loss, metric
  monitoring/             학습 곡선 생성
  utils/                  설정 및 로그 공통 코드
tests/                    데이터 파이프라인과 모델 단위 테스트
artifacts/                학습 모델, 로그, 그래프, 예측 결과
train.py                  U-Net 학습 진입점
evaluate.py               고정 Test 세트 평가 진입점
```

`dataset/images`, `dataset/masks`, `dataset/prepared`, `artifacts`는 용량과 데이터
보호를 위해 Git에서 제외합니다.

## 학습 환경

이 프로젝트의 PC 학습 환경은 Python 3.8, TensorFlow 2.4.1, CUDA 11,
cuDNN 8 조합입니다.

```bash
conda env create -f environment-tf241.yml
conda activate unet-tf241
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

환경을 수동으로 만들 때는 다음 명령을 사용합니다.

```bash
conda create -n unet-tf241 python=3.8 pip -y
conda activate unet-tf241
python -m pip install -r requirements-pc-tf241.txt
```

Jetson의 Linux/ARM64 전용 wheel과 TensorRT 패키지는 Windows에 그대로 설치할
수 없습니다. Python, TensorFlow, NumPy 및 HDF5 계열의 호환 버전을 맞추고,
배포 전에 Jetson에서 모델 재로딩과 출력 shape를 다시 확인합니다.

## 데이터 준비

### 1. HSV 마스크 생성

```bash
python scripts/generate_hsv_masks.py --config configs/dataset.yaml
```

### 2. 마스크 검수 자료 생성

```bash
python scripts/review_masks.py
```

검수 보고서는 `dataset/metadata/mask_review.csv`, 원본·마스크·오버레이
contact sheet는 `artifacts/mask_review`에 생성됩니다.

### 3. 데이터 무결성 검사

```bash
python scripts/validate_dataset.py --images dataset/images --masks dataset/masks
```

### 4. Train/Validation/Test 분리

```bash
python scripts/split_dataset.py --config configs/augmentation.yaml
```

같은 촬영 계열이 서로 다른 split으로 들어가지 않도록 원본 촬영 단위로
70/15/15 비율에 가깝게 분리합니다.

### 5. Train 데이터 증강

```bash
python scripts/augment_dataset.py --config configs/augmentation.yaml
```

기하 변환은 이미지와 마스크에 동일하게 적용하고, 마스크에는 nearest-neighbor
보간만 사용합니다. 밝기·대비·색조·blur·noise는 이미지에만 적용합니다.
Validation과 Test에는 증강을 적용하지 않습니다.

## U-Net 구조

입력은 `256 x 256 x 3`, 출력은 `256 x 256 x 1` sigmoid 확률 맵입니다.
Encoder 채널은 `16 → 32 → 64 → 128`, bottleneck은 256 채널이며 Decoder는
transpose convolution과 skip connection으로 원래 해상도를 복원합니다.
자세한 설정은 `configs/model.yaml`, 모델 코드는 `src/models/unet.py`에 있습니다.

학습 loss는 미세한 누수 픽셀의 클래스 불균형을 보완하도록
`Binary Cross Entropy + soft Dice loss`를 사용합니다. 모델 선택용 Dice와
IoU는 실제 추론 마스크와 동일하게 확률 0.5에서 이진화해 계산합니다.

## 학습과 평가

```bash
conda activate unet-tf241
python train.py
python evaluate.py
```

주요 산출물:

```text
artifacts/training/models/best_model.h5
artifacts/training/models/final_model.h5
artifacts/training/models/saved_model/
artifacts/training/logs/training_log.csv
artifacts/training/logs/training_summary.json
artifacts/training/logs/test_results.csv
artifacts/training/logs/test_summary.json
artifacts/training/plots/
artifacts/training/predictions/
```

### 현재 기준 결과

검수 원본 301장 중 Train 원본 210장에 증강 420장을 더해 630쌍으로 학습했고,
Validation 49장과 Test 42장은 증강하지 않았습니다.

| 항목 | 결과 |
|---|---:|
| 모델 파라미터 | 1,945,521 |
| 최적 epoch | 8 |
| Validation Dice | 0.9818 |
| Validation IoU | 0.9650 |
| Test Dice (이미지별 평균) | 0.9827 |
| Test IoU (이미지별 평균) | 0.9664 |
| Test Dice (전체 픽셀 집계) | 0.9803 |
| Test IoU (전체 픽셀 집계) | 0.9613 |
| Test Precision | 0.9822 |
| Test Recall | 0.9783 |
| 정상 Test 오검출 | 0/10장 |

가장 작은 2방울 클래스의 이미지별 평균 Dice도 0.9722입니다. 정상 Test
10장은 모두 예측 픽셀이 0이었고, 확산·ROI 샘플의 평균 Dice는 0.9759입니다.

## 테스트

```bash
python -m pytest -q
python -m compileall -q src scripts train.py evaluate.py tests
```

## 카메라 노출 진단

```bash
python scripts/test_camera_exposure.py --camera 0 --exposures -4 -6 -8 --show
```

PLEOMAX 웹캠 드라이버가 수동 노출이나 후광 보정 변경을 허용하지 않으면 원본
카메라 출력을 사용합니다. 현재 데이터는 흰색 폼보드를 배경으로 사용해 자동
노출 변화가 작아지도록 촬영했습니다.

## Jetson Nano 실시간 추론

Jetson Nano로 그대로 복사할 수 있는 실시간 카메라 추론 코드는
`Jetson Nano/`에 있습니다.

```bash
cd "Jetson Nano"
python3 run_realtime.py
```

누출 면적, 4단계 경고, 확산, Danger ROI, FPS와 추론시간을 화면에 표시하며
자세한 설치·설정 방법은 `Jetson Nano/README.md`를 참고합니다.
