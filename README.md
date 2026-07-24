# U-Net 기반 색상 액체 누출 감지 시스템

고정 카메라로 촬영한 색상 액체의 누출 영역을 수집하고, HSV 기반 후보 마스크를
생성·검증·분석하기 위한 1단계 파이프라인입니다. 실제 U-Net 학습, Jetson Nano
실시간 추론, UART/TCP 통합은 데이터 품질과 임계값을 검증한 다음 단계에서
구현하도록 분리했습니다.

## 현재 구현 범위

- `configs/`: 카메라, HSV, ROI, 모델, 판정 임계값, 통신 설정
- `scripts/capture_dataset.py`: 고정 카메라 촬영 및 메타데이터 CSV 기록
- `scripts/generate_hsv_masks.py`: HSV 임계값, morphology, 작은 성분 제거, overlay 생성
- `scripts/validate_dataset.py`: 이미지/마스크 1:1 대응, 크기, 채널, 이진값 검증
- `scripts/analyze_dataset.py`: Monitoring ROI 내 누출 면적 및 0·2·5·8방울 그룹 통계
- `src/data/`: 재사용 가능한 ROI와 마스크 처리 로직
- `tests/`: 핵심 마스크 처리 및 ROI 단위 테스트

Danger ROI는 원본 이미지에 칠하지 않으며 [configs/roi.yaml](configs/roi.yaml)의
좌표로만 관리합니다. 면적 계산은 Monitoring ROI 밖의 픽셀을 제외합니다.

## 설치

Python 3.10 이상을 권장합니다.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Jetson TensorFlow 2.4.1 호환 학습 환경

Jetson의 전체 `pip freeze`에는 Windows에서 설치할 수 없는 Ubuntu/ARM64 전용
패키지가 포함되므로 그대로 사용하지 않습니다. 모델 호환성에 필요한 Python,
TensorFlow, NumPy, HDF5 계열 버전만 맞춘 환경은 다음과 같이 생성합니다.

```bash
conda env create -f environment-tf241.yml
conda activate unet-tf241
python -c "import tensorflow as tf; print(tf.__version__)"
```

기존 환경을 수동으로 구성하려면 다음 명령을 사용합니다.

```bash
conda create -n unet-tf241 python=3.8 pip -y
conda activate unet-tf241
python -m pip install -r requirements-pc-tf241.txt
```

PC의 CUDA/cuDNN과 Jetson의 TensorRT 패키지는 플랫폼별 설치 항목이므로 동일한
wheel을 공유하지 않습니다. 배포 전에는 Jetson에서 저장한 모델을 다시 로드하여
동일 입력에 대한 출력 shape과 수치 오차를 확인해야 합니다.

## 1. 데이터 촬영

카메라를 640×480으로 고정하고 ROI 좌표를 먼저 조정합니다.

```bash
python scripts/capture_dataset.py --config configs/dataset.yaml --roi-config configs/roi.yaml --session S01
```

키:

- `0`, `2`, `5`, `8`: 해당 방울 그룹을 선택하고 즉시 한 장 저장
- `S`: 현재 그룹으로 한 장 추가 저장
- `Q` 또는 `Esc`: 정상 종료

파일명은 `S01_D02_0001.jpg` 형식이며 촬영 정보는
`dataset/metadata/dataset_metadata.csv`에 누적됩니다. 카메라 열기/프레임 읽기
실패 시 오류 코드와 원인을 출력하고 장치를 정리합니다.

## 2. HSV 후보 마스크 생성

```bash
python scripts/generate_hsv_masks.py --config configs/dataset.yaml --input dataset/raw/session_01 --output-images dataset/images_unreviewed --output-masks dataset/masks_unreviewed --output-overlays dataset/overlays
```

마스크는 원본과 같은 stem의 단일 채널 PNG이며 값은 0 또는 255입니다.
실험 액체에 맞춰 `configs/dataset.yaml`의 HSV 범위를 반드시 보정하십시오.

## 3. 데이터 검증

```bash
python scripts/validate_dataset.py --images dataset/images_unreviewed --masks dataset/masks_unreviewed
```

파일 누락, 손상, 크기 불일치, 다채널/비이진 마스크를 파일명과 함께 보고합니다.

## 4. 면적 통계

```bash
python scripts/analyze_dataset.py --metadata dataset/metadata/dataset_metadata.csv --masks dataset/masks_unreviewed --roi-config configs/roi.yaml --output artifacts/dataset_analysis.csv
```

각 샘플의 `leak_pixels`, `roi_pixels`, `leak_ratio`를 CSV로 저장하고 그룹별
최소·평균·중앙값·표준편차·최댓값을 출력합니다. 이 분포를 확인한 뒤
`configs/thresholds.yaml`의 실제 판정 임계값을 결정해야 합니다.

## 검사

```bash
python -m pytest
python -m compileall -q src scripts
```

카메라가 OpenCV의 수동 노출 제어를 지원하는지는 다음 명령으로 진단할 수 있습니다.

```bash
python scripts/test_camera_exposure.py --camera 0 --exposures -4 -6 -8
```

각 노출 설정의 `set()` 결과, 카메라가 다시 보고한 값, 평균 밝기를
`artifacts/camera_exposure_test.json`에 기록합니다. 흰 물체를 화면에 넣고 빼면서
자동 노출 반응을 눈으로 확인하려면 `--show`를 추가합니다. DirectShow 장치에서
수동 모드 값은 흔히 `0.25`이지만 장치에 따라 다르므로
`--manual-auto-value`로 변경할 수 있습니다.

## 후속 구현 경계

데이터 검수 및 세션 단위 train/validation/test 분리가 완료된 후 다음 모듈을
추가합니다: Dataset Loader/Augmentation, 경량 U-Net, BCE+Dice loss와 평가 지표,
실시간 predictor/postprocess/leak analyzer, UART packet/sender, 비동기 TCP server,
event logger. 실제 데이터가 없는 현재 단계에서는 학습·평가 수치를 생성하거나
성능을 주장하지 않습니다.
