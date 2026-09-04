# Small Drone Detection & Failure Analysis

DUT Anti-UAV 공개 데이터셋을 이용해 **소형 UAV 탐지 실패 원인을 분석하고, 분석 결과에 근거해 모델을 단계적으로 개선한 프로젝트**입니다.

> **데이터 분석 → 서브셋 검증 → Baseline 실패 분석 → Input Size 증가(V1) → Epoch 증가(V2) → 최종 모델 선정 → Test 확인**

---

## 1. Results

### Experiment

| Model | Input Size | Epoch | 실험 목적 |
|---|---:|---:|---|
| Baseline | 640 | 30 | 초기 성능 및 실패 유형 확인 |
| Improved V1 | 960 | 30 | 극소형 UAV 특징 손실 완화 |
| Improved V2 | 960 | 100 | 추가 학습 효과 확인 |

48시간 내 반복 학습과 실패 분석이 가능한 조건, 소형 객체 비중, 사용 가능한 GPU 환경을 고려해 **YOLO26s 사전학습 모델**을 Baseline으로 선정했습니다.

각 단계에서는 다른 주요 조건을 유지하고 **핵심 조건 하나만 변경**했습니다.

### Validation

| Model | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9416 | 0.8462 | 0.8914 | 0.8937 | 0.6230 | 0.5561 | 123 | 99 |
| Improved V1 | 0.9603 | 0.8847 | 0.9209 | **0.9362** | 0.7225 | 0.6202 | 80 | 75 |
| Improved V2 | **0.9658** | **0.9032** | **0.9334** | 0.9354 | **0.7703** | **0.6500** | **76** | **51** |

가장 작은 객체군 Q1 Recall:

**0.7489 → 0.8265 → 0.8311**

V2는 V1보다 mAP50이 0.0008 낮고 완전 미검출은 증가했지만, Recall·F1·mAP75·mAP50-95가 상승하고 전체 FN·FP와 위치 부정확 오류가 감소했습니다. 이를 종합해 **Improved V2를 최종 모델로 선정**했습니다.

### Final Test

| Model | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9483 | 0.8838 | 0.9149 | 0.9340 | 0.7187 | 0.6269 |
| Improved V1 | 0.9661 | 0.9252 | 0.9452 | **0.9687** | 0.7624 | 0.6685 |
| Improved V2 | **0.9704** | **0.9372** | **0.9535** | 0.9584 | **0.7954** | **0.6956** |

모델 선택과 개선 방향은 Validation 결과를 기준으로 결정했으며, Test Set은 최종 확인에만 사용했습니다.

`analysis/06_final_test.ipynb`에서 세 모델을 동일한 조건으로 비교했습니다.

---

## 2. Dataset

- **DUT Anti-UAV Detection**

- Single-class UAV Detection

- Pascal VOC XML → YOLO format

- Official Dataset: https\://github.com/wangdongdut/DUT-Anti-UAV

### Full Dataset

| Split | Images | Objects | bbox 면적 비율 중앙값 | bbox 면적 1% 미만 |
|---|---:|---:|---:|---:|
| Train | 5,200 | 5,243 | 0.0472% | 88.33% |
| Validation | 2,600 | 2,621 | 0.0459% | 88.55% |
| Test | 2,200 | 2,245 | 0.0911% | 75.86% |

Train·Validation 객체의 약 88%가 이미지 면적의 1% 미만이며, bbox 크기 중앙값도 약 **43×23 px** 수준이었습니다. Input Size 640 기준으로는 중앙값이 약 **15×8 px**로 줄어들어 작은 객체의 정보 손실 가능성이 높다고 판단했습니다.

48시간 내 반복 실험이 가능하도록 공식 Split을 유지하면서 각 Split의 약 1/3을 사용했습니다.

### Subset

| Split | Images | Objects |
|---|---:|---:|
| Train | 1,733 | 1,776 |
| Validation | 867 | 876 |
| Test | 733 | 749 |

서브셋 구성 기준:

- 공식 Train / Validation / Test 간 데이터 이동 없음

- 단일 객체 이미지를 bbox 크기 기준 5개 구간에서 균형 추출

- Train 배경 이미지 3장과 다중 객체 이미지 29장 전체 보존

- Validation / Test의 희소 사례 비율 유지

- `Seed=42` 고정

- 원본과 서브셋의 bbox 중앙값 및 소형 객체 비율 비교로 대표성 확인

---

## 3. Improvement Flow

### Baseline — Failure Analysis

Baseline Validation에서 **FN 123개, FP 99개**를 확인했습니다.

대표 **FN 15건과 FP 20건**을 직접 검토한 결과 주요 실패 유형은 다음과 같았습니다.

- 극소형·복잡 배경 UAV 미검출

- 작은 객체의 낮은 신뢰도

- 작은 bbox의 위치 부정확

- 조류·비행기·건물 구조물 등 UAV 유사 객체 오탐

가장 작은 객체군 Q1 Recall이 **0.7489**로 가장 낮아, 첫 번째 개선 대상으로 작은 UAV의 특징 손실을 선정했습니다.

### V1 — Input Size 640 → 960

처음에는 Input Size 1280을 검토했지만 CUDA OOM이 발생해 반복 학습이 가능한 **960**으로 조정했습니다.

V1에서는 Input Size만 변경했습니다.

- Q1 Recall: **0.7489 → 0.8265**

- Recall: **0.8462 → 0.8847**

- mAP75: **0.6230 → 0.7225**

- FN: **123 → 80**

- FP: **99 → 75**

작은 UAV의 Recall과 전체 오류 수가 함께 개선되어 Input Size 증가 효과를 확인했습니다.

### V2 — Epoch 30 → 100

V1 학습 종료 시점까지 Train·Validation loss가 감소하고 Recall·mAP50-95가 상승했으며, `best.pt`가 마지막 30 Epoch에서 생성됐습니다.

추가 학습 여지가 있다고 판단해 Input Size 960을 유지하고 Epoch만 100으로 증가했습니다.

- Recall: **0.8847 → 0.9032**

- mAP75: **0.7225 → 0.7703**

- mAP50-95: **0.6202 → 0.6500**

- 위치 부정확 FN: **23 → 10**

- FP: **75 → 51**

- 완전 미검출: **33 → 41**

추가 학습은 극소형 객체를 새롭게 찾는 것보다 **bbox 정밀도와 전반적인 판별 안정성 개선**에 더 효과가 있었습니다.

상세 실패 사례와 실험 해석은 기술보고서와 분석 Notebook에 정리했습니다.

---

## 4. Project Structure

```text

small-drone-detection/

├── src/

│   ├── 01_download_dataset.py

│   ├── 02_make_dataset.py

│   ├── 03_train.py

│   ├── 04_make_subset.py

│   ├── 05_validation.py

│   └── 06_test.py

│

├── analysis/

│   ├── 01_dataset_analysis.ipynb

│   ├── 02_subset_analysis.ipynb

│   ├── 03_baseline_failure_analysis.ipynb

│   ├── 04_v1_failure_analysis.ipynb

│   ├── 05_v2_failure_analysis.ipynb

│   └── 06_final_test.ipynb

│

├── data/                  # Git 제외

├── pretrained/

├── models/

│   ├── baseline/

│   ├── improved_v1/

│   └── improved_v2/

├── requirements.txt

└── README.md

```

파일 번호는 작성 순서를 유지하고 있어 실제 실행 순서와 일부 다릅니다.

재현 시 아래 **Reproduction** 순서를 기준으로 실행합니다.

---

## 5. Environment

실험 환경:

| Item | Setting |
|---|---|
| OS | Windows 10 |
| GPU | NVIDIA GeForce RTX 3090 |
| Python | 3.11.16 |
| Model | YOLO26s |
| Batch Size | 16 |
| Seed | 42 |
| Workers | 4 |

다른 CUDA GPU에서도 실행할 수 있으나 환경 차이에 따라 추론 시간과 소수점 수준의 metric 차이가 발생할 수 있습니다.

---

## 6. Reproduction

프로젝트를 Clone한 뒤 아래 순서대로 실행합니다.

### 1) Repository Clone

```bash

git clone https\://github.com/sswwd95/small-drone-detection.git

cd small-drone-detection

```

### 2) Environment Setup

```bash

conda create -n small-drone python=3.11.16 -y

conda activate small-drone

pip install -r requirements.txt

```

설치 확인:

```bash

python -c "import torch, ultralytics; print(torch.__version__); print(ultralytics.__version__); print(torch.cuda.is_available())"

```

### 3) Dataset Download

```bash

python src/01_download_dataset.py

```

DUT Anti-UAV 공식 Train / Validation / Test 데이터를 `data/raw/`에 다운로드합니다.

### 4) Pascal VOC → YOLO

```bash

python src/02_make_dataset.py

```

결과:

```text

data/yolo/

├── images/{train,val,test}/

├── labels/{train,val,test}/

└── data.yaml

```

### 5) Subset 생성

`src/04_make_subset.py`는 학습보다 먼저 실행합니다.

```bash

python src/04_make_subset.py

```

결과:

```text

data/yolo_subset/

├── images/{train,val,test}/

├── labels/{train,val,test}/

└── data.yaml

```

동일한 `Seed=42`와 선정 규칙을 사용해 동일한 서브셋을 재생성할 수 있습니다.

### 6) Train

세 모델 모두 `src/03_train.py`를 사용합니다.

| Version | `VERSION` | `IMG_SIZE` | `EPOCHS` |
|---|---|---:|---:|
| Baseline | `"baseline"` | 640 | 30 |
| V1 | `"v1"` | 960 | 30 |
| V2 | `"v2"` | 960 | 100 |

모델별 설정을 변경한 뒤 각각 실행합니다.

```bash

python src/03_train.py

```

공통 학습 조건:

```text

Model   : yolo26s.pt

Dataset : yolo_subset

Batch   : 16

Seed    : 42

Workers : 4

```

최종 모델 경로:

```text

models/baseline/weights/best.pt

models/improved_v1/weights/best.pt

models/improved_v2/weights/best.pt

```

### 7) Validation

`src/05_validation.py`의 `RUN_NAME`, `IMG_SIZE`를 모델에 맞게 변경합니다.

```text

baseline    : RUN_NAME="baseline",    IMG_SIZE=640

improved_v1 : RUN_NAME="improved_v1", IMG_SIZE=960

improved_v2 : RUN_NAME="improved_v2", IMG_SIZE=960

```

각 모델마다 실행합니다.

```bash

python src/05_validation.py

```

결과:

```text

models/\<model_name>/val_\<timestamp>/val_metrics.csv

```

### 8) Failure Analysis

```bash

jupyter lab

```

다음 순서로 실행합니다.

```text

analysis/03_baseline_failure_analysis.ipynb

analysis/04_v1_failure_analysis.ipynb

analysis/05_v2_failure_analysis.ipynb

```

새로운 Validation 결과를 사용할 경우 각 Notebook 첫 코드 셀의 `VAL_NAME`을 해당 `val_\<timestamp>` 폴더명으로 변경합니다.

공통 실패 판정 기준:

| FN Type | 기준 |
|---|---|
| 낮은 신뢰도 | IoU ≥ 0.5 예측이 있으나 confidence < 0.25 |
| 위치 부정확 | GT와 가장 높은 IoU가 0.1 이상 0.5 미만 |
| 미검출 | 모든 예측과 IoU < 0.1 |

공통 설정:

```text

Confidence threshold = 0.25

Match IoU            = 0.5

Near IoU             = 0.1

```

### 9) Final Test

모든 개선 방향과 최종 모델을 Validation에서 결정한 뒤 실행합니다.

```text

analysis/06_final_test.ipynb

```

Test 결과를 이용한 추가 튜닝은 수행하지 않았습니다.

---

## 7. Limitations & Next Step

### Limitations

- 전체 데이터가 아닌 약 1/3 서브셋 기반 실험

- 동일 데이터셋 내 유사 촬영 환경 또는 연속 장면이 포함될 가능성

- IoU 기반 FN 유형 분류만으로 실제 실패 원인을 완전히 분리하기 어려움

- 공식 Test 외 새로운 촬영 환경에 대한 일반화 검증 부족

### Next Step

- 극소형 UAV: 고해상도 입력 또는 Tiling 적용 검토

- 완전 미검출: 복잡한 배경의 UAV 사례 추가 학습

- 배경 오탐: 조류·비행기·건물·바위·그림자 등 Hard Negative 보강

- 낮은 신뢰도 FN: Validation 기준 confidence threshold 재검토

- bbox 위치 오차: GT annotation 일관성 점검

- 일반화 성능: 외부 UAV 영상 추가 검증

---

## 8. Notebook Guide

| Notebook | 역할 |
|---|---|
| `01_dataset_analysis.ipynb` | 전체 데이터 구조 및 소형 객체 특성 분석 |
| `02_subset_analysis.ipynb` | 원본 대비 서브셋 대표성 검증 |
| `03_baseline_failure_analysis.ipynb` | Baseline 실패 분석 및 V1 개선 근거 |
| `04_v1_failure_analysis.ipynb` | V1 개선 효과 및 V2 개선 근거 |
| `05_v2_failure_analysis.ipynb` | V2 성능, 잔여 오류 및 학습 종료 판단 |
| `06_final_test.ipynb` | Baseline·V1·V2 최종 Test 비교 |

---

## 9. Quick Reproduction

```bash

# Repository

git clone https\://github.com/sswwd95/small-drone-detection.git

cd small-drone-detection

# Environment

conda create -n small-drone python=3.11.16 -y

conda activate small-drone

pip install -r requirements.txt

# Dataset

python src/01_download_dataset.py

python src/02_make_dataset.py

python src/04_make_subset.py

# Baseline → V1 → V2 설정 변경 후 각각 실행

python src/03_train.py

# 모델별 설정 변경 후 각각 실행

python src/05_validation.py

# Analysis

jupyter lab

```

Notebook 실행 순서:

```text

01_dataset_analysis.ipynb

02_subset_analysis.ipynb

03_baseline_failure_analysis.ipynb

04_v1_failure_analysis.ipynb

05_v2_failure_analysis.ipynb

06_final_test.ipynb

```

---

## References

- DUT Anti-UAV: https\://github.com/wangdongdut/DUT-Anti-UAV

- Ultralytics: https\://github.com/ultralytics/ultralytics
