# Small Drone Detection & Failure Analysis

DUT Anti-UAV 공개 데이터셋에서 소형 UAV 탐지 실패 원인을 분석하고, 입력 해상도와 학습 Epoch를 단계적으로 변경해 성능을 개선한 프로젝트입니다.

> 데이터 분석 → 서브셋 구성 → Baseline 실패 분석 → Input Size 증가(V1) → Epoch 증가(V2) → Validation 기반 최종 모델 선정 → Test 최종 확인

## 1. Results

### Experiment

| Model | Input Size | Epoch | 변경 목적 |
|---|---:|---:|---|
| Baseline | 640 | 30 | 초기 성능 및 실패 유형 확인 |
| Improved V1 | 960 | 30 | 극소형 UAV 특징 손실 완화 |
| Improved V2 | 960 | 100 | 추가 학습 효과 확인 |

48시간 내 반복 학습과 실패 분석이 가능한 조건, 소형 객체 비중, 사용 가능한 GPU 환경을 고려해 YOLO26s 사전학습 모델을 Baseline으로 선정했습니다. 각 단계에서는 다른 주요 조건을 유지하고 핵심 조건 하나만 변경했습니다.

### Validation

| Model | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9416 | 0.8462 | 0.8914 | 0.8937 | 0.6230 | 0.5561 | 123 | 99 |
| Improved V1 | 0.9603 | 0.8847 | 0.9209 | **0.9362** | 0.7225 | 0.6202 | 80 | 75 |
| Improved V2 | **0.9658** | **0.9032** | **0.9334** | 0.9354 | **0.7703** | **0.6500** | **76** | **51** |

가장 작은 객체군 Q1 Recall은 `0.7489 → 0.8265 → 0.8311`로 개선됐습니다. V2는 V1보다 mAP50이 0.0008 낮았지만, Recall·F1·mAP75·mAP50-95가 상승하고 전체 FN·FP가 감소했습니다. 이를 종합해 Improved V2를 최종 모델로 선정했습니다.

세부 실패 유형별 분석 결과와 최종 모델 선정 근거는 아래 **Improvement Flow**에서 설명합니다.

### Final Test

| Model | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9483 | 0.8838 | 0.9149 | 0.9340 | 0.7187 | 0.6269 |
| Improved V1 | 0.9661 | 0.9252 | 0.9452 | **0.9687** | 0.7624 | 0.6685 |
| Improved V2 | **0.9704** | **0.9372** | **0.9535** | 0.9584 | **0.7954** | **0.6956** |

모델 선택과 개선 방향은 Validation 결과만으로 결정했으며, Test Set은 마지막 성능 확인에만 사용했습니다.

## 2. Dataset

- Dataset: [DUT Anti-UAV Detection](https://github.com/wangdongdut/DUT-Anti-UAV)
- Task: Single-class UAV Detection
- Annotation conversion: Pascal VOC XML → YOLO TXT
- Experiment data: 공식 Train·Validation·Test Split 내부에서 각각 약 1/3 추출

### Full Dataset

| Split | Images | Objects | bbox 면적 비율 중앙값 | bbox 면적 1% 미만 |
|---|---:|---:|---:|---:|
| Train | 5,200 | 5,243 | 0.0472% | 88.33% |
| Validation | 2,600 | 2,621 | 0.0459% | 88.55% |
| Test | 2,200 | 2,245 | 0.0911% | 75.86% |

Train·Validation 객체의 약 88%가 이미지 면적의 1% 미만입니다. bbox 크기 중앙값은 약 43×23 px이며, Input Size 640으로 변환하면 약 15×8 px가 되어 특징 손실 가능성이 높다고 판단했습니다.

### Subset

| Split | Images | Objects |
|---|---:|---:|
| Train | 1,733 | 1,776 |
| Validation | 867 | 876 |
| Test | 733 | 749 |

서브셋 구성 기준은 다음과 같습니다.

- 공식 Split 간 데이터 이동 없음
- 단일 객체 이미지를 bbox 면적 기준 5개 구간으로 나누어 균형 추출
- Train의 배경 이미지 3장과 다중 객체 이미지 29장 전체 보존
- Validation·Test의 배경 및 다중 객체 이미지 비율 유지
- Seed 42 고정

## 3. Improvement Flow

### Baseline — Failure Analysis

Baseline Validation의 FN 123개와 FP 99개를 확인했습니다. 대표 FN 15건과 FP 20건을 직접 검토한 결과, 극소형·복잡 배경 UAV 미검출, 낮은 신뢰도, bbox 위치 부정확, 조류·비행기·건물 구조물 오탐이 주요 실패 유형이었습니다.

### V1 — Input Size 640 → 960

Input Size 1280은 CUDA OOM이 발생해 반복 학습이 가능한 960으로 조정했습니다. Input Size만 변경한 결과 Q1 Recall은 0.7489에서 0.8265, 전체 Recall은 0.8462에서 0.8847, mAP75는 0.6230에서 0.7225로 상승했습니다.

### V2 — Epoch 30 → 100

V1은 30 Epoch 종료 시점에도 Train·Validation loss가 감소하고 Recall·mAP50-95가 상승했으며, `best.pt`도 마지막 30 Epoch에서 생성됐습니다. 추가 학습 여지가 있다고 판단해 Input Size 960을 유지하고 Epoch만 100으로 늘렸습니다. V2에서는 Recall 0.9032, mAP75 0.7703, mAP50-95 0.6500을 기록했습니다.

## 4. Project Structure

```text
small-drone-detection/
├── src/
│   ├── 01_download_dataset.py
│   ├── 02_make_dataset.py
│   ├── 03_train.py
│   ├── 04_make_subset.py
│   ├── 05_validation.py
│   └── 06_test.py
├── analysis/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_subset_analysis.ipynb
│   ├── 03_baseline_failure_analysis.ipynb
│   ├── 04_v1_failure_analysis.ipynb
│   ├── 05_v2_failure_analysis.ipynb
│   └── 06_final_test.ipynb
├── data/                         # 실행 시 생성, Git 제외
├── pretrained/                   # 사전학습 가중치 자동 다운로드, Git 제외
├── models/
│   ├── baseline/
│   ├── improved_v1/
│   └── improved_v2/
├── requirements.txt
└── README.md
```

파일 번호는 작성 순서를 유지해 실제 실행 순서와 일부 다릅니다. 재현할 때는 아래 순서를 사용합니다.

## 5. Environment

원 실험 환경은 다음과 같습니다.

| Item | Setting |
|---|---|
| OS | Windows 10 |
| GPU | NVIDIA GeForce RTX 3090 |
| Python | 3.11.16 |
| PyTorch | 2.13.0+cu126 |
| CUDA Runtime | 12.6 |
| Ultralytics | 8.4.138 |
| Batch Size | 16 |
| Seed | 42 |
| Workers | 4 |

다른 CUDA GPU에서도 실행할 수 있으나, 하드웨어와 CUDA 환경 차이로 추론 시간 및 소수점 수준의 metric 차이가 발생할 수 있습니다. `batch=16`에서 CUDA OOM이 발생하면 `src/03_train.py`, `src/05_validation.py`, `src/06_test.py`의 `BATCH_SIZE`를 줄여야 합니다.

## 6. Reproduction

### 6.1 Repository Clone

```bash
git clone https://github.com/sswwd95/small-drone-detection.git
cd small-drone-detection
```

### 6.2 Environment Setup

Conda 환경을 생성하고 저장된 패키지 버전을 설치합니다.

```bash
conda create -n small-drone python=3.11.16 -y
conda activate small-drone
python -m pip install --upgrade pip
pip install -r requirements.txt
```

설치 및 GPU 인식 확인:

```bash
python -c "import torch, ultralytics; print('torch:', torch.__version__); print('ultralytics:', ultralytics.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

이 프로젝트의 학습·평가 코드는 `DEVICE=0`으로 설정되어 있어 CUDA GPU가 필요합니다. CPU로 실행하려면 각 실행 파일의 `DEVICE`를 `"cpu"`로 바꿔야 하며 학습 시간이 크게 증가합니다.

### 6.3 Dataset Download and Conversion

다음 명령은 데이터를 Google Drive에서 내려받고, Pascal VOC XML을 YOLO 형식으로 변환한 뒤, Seed 42로 실험용 서브셋을 생성합니다.

```bash
python src/01_download_dataset.py
python src/02_make_dataset.py
python src/04_make_subset.py
```

완료 후 다음 파일이 있어야 합니다.

```text
data/yolo/data.yaml
data/yolo_subset/data.yaml
data/yolo_subset/images/{train,val,test}/
data/yolo_subset/labels/{train,val,test}/
```

정상 생성 수량:

| Split | Full images | Subset images | Subset objects |
|---|---:|---:|---:|
| Train | 5,200 | 1,733 | 1,776 |
| Validation | 2,600 | 867 | 876 |
| Test | 2,200 | 733 | 749 |

### 6.4 Evaluate Included Weights

저장소에는 세 모델의 `best.pt`가 포함되어 있습니다. 전체 재학습 없이 공개 결과를 다시 평가하려면 먼저 데이터 준비를 완료한 뒤 `src/05_validation.py` 또는 `src/06_test.py` 상단의 `RUN_NAME`과 `IMG_SIZE`를 아래 표에 맞춰 변경하고 실행합니다.

| Model | `RUN_NAME` | `IMG_SIZE` |
|---|---|---:|
| Baseline | `"baseline"` | 640 |
| Improved V1 | `"improved_v1"` | 960 |
| Improved V2 | `"improved_v2"` | 960 |

Validation 실행:

```bash
python src/05_validation.py
```

Test 실행:

```bash
python src/06_test.py
```

실행 결과는 다음 위치에 새 폴더로 저장됩니다.

```text
models/<model_name>/val_<YYMMDD_HHMM>/val_metrics.csv
models/<model_name>/test_<YYMMDD_HHMM>/test_metrics.csv
```

세 모델을 모두 비교하려면 설정 변경과 실행을 모델별로 한 번씩, 총 3회 수행합니다. `analysis/06_final_test.ipynb`는 각 모델 폴더에서 가장 최근의 `test_metrics.csv`를 자동으로 읽어 비교합니다.

### 6.5 Retrain All Models

`src/03_train.py` 상단의 `VERSION`을 아래 값으로 바꾸고 모델별로 한 번씩 실행합니다. 나머지 실험값은 코드의 `EXPERIMENTS`에 정의되어 있습니다.

| Model | `VERSION` | Input Size | Epoch | Output directory |
|---|---|---:|---:|---|
| Baseline | `"baseline"` | 640 | 30 | `models/baseline/` |
| Improved V1 | `"v1"` | 960 | 30 | `models/improved_v1/` |
| Improved V2 | `"v2"` | 960 | 100 | `models/improved_v2/` |

각 설정에서 실행:

```bash
python src/03_train.py
```

처음 실행할 때 Ultralytics가 `yolo26s.pt`를 내려받아 `pretrained/`에 저장합니다. 학습 결과는 같은 모델 폴더에 저장되며 `exist_ok=True`이므로, 저장소에 포함된 결과를 보존하려면 기존 `models/`를 먼저 별도로 복사해야 합니다.

### 6.6 Failure Analysis

Notebook 실행에 필요한 JupyterLab을 설치하고 프로젝트 루트에서 시작합니다.

```bash
pip install jupyterlab
jupyter lab
```

권장 실행 순서:

```text
analysis/01_dataset_analysis.ipynb
analysis/02_subset_analysis.ipynb
analysis/03_baseline_failure_analysis.ipynb
analysis/04_v1_failure_analysis.ipynb
analysis/05_v2_failure_analysis.ipynb
analysis/06_final_test.ipynb
```

저장소에 포함된 Validation 결과를 사용할 때는 실패 분석 Notebook의 `VAL_NAME`을 바꿀 필요가 없습니다. 새 Validation을 실행한 경우에만 각 Notebook 첫 코드 셀의 `VAL_NAME`을 새로 생성된 `val_<YYMMDD_HHMM>` 폴더명으로 변경합니다.

공통 실패 판정 설정:

| Setting | Value |
|---|---:|
| Confidence threshold | 0.25 |
| Match IoU | 0.5 |
| Near IoU | 0.1 |

| FN Type | Rule |
|---|---|
| 낮은 신뢰도 | IoU 0.5 이상인 예측이 있으나 confidence 0.25 미만 |
| 위치 부정확 | GT와 예측의 최고 IoU가 0.1 이상 0.5 미만 |
| 미검출 | 모든 예측과의 IoU가 0.1 미만 |

## 7. Quick Reproduction

아래 명령은 환경 구성부터 서브셋 생성까지 수행합니다.

```bash
git clone https://github.com/sswwd95/small-drone-detection.git
cd small-drone-detection

conda create -n small-drone python=3.11.16 -y
conda activate small-drone
python -m pip install --upgrade pip
pip install -r requirements.txt

python src/01_download_dataset.py
python src/02_make_dataset.py
python src/04_make_subset.py
```

이후 목적에 따라 다음 중 하나를 수행합니다.

- 포함된 가중치 평가: `src/05_validation.py` 또는 `src/06_test.py`의 모델 설정 후 실행
- 전체 재학습: `src/03_train.py`의 `VERSION`을 `baseline`, `v1`, `v2`로 변경하며 총 3회 실행
- 실패 분석: `pip install jupyterlab` 후 `jupyter lab` 실행

## 8. Limitations

- 전체 데이터가 아닌 약 1/3 서브셋 기반 실험
- 동일 데이터셋 내 유사 촬영 환경 또는 연속 장면 포함 가능성
- IoU 기반 FN 유형 분류만으로 실제 실패 원인을 완전히 분리하기 어려움
- 공식 Test 외 새로운 촬영 환경에 대한 일반화 검증 부족

## References

- [DUT Anti-UAV](https://github.com/wangdongdut/DUT-Anti-UAV)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
