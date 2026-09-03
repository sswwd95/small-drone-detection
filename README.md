# Small Drone Detection & Failure Analysis

DUT Anti-UAV 공개 데이터셋을 이용해 **소형 UAV 탐지 실패 원인을 분석하고, 분석 결과에 근거해 모델을 단계적으로 개선한 프로젝트**입니다.

> **데이터 분석 → 서브셋 검증 → Baseline 실패 분석 → Input Size 증가(V1) → Epoch 증가(V2) → Validation에서 모델 고정 → Test 최종 확인**

---

## 1. Results

### Experiment

| Model | Input Size | Epoch | 변경 목적 |
|---|---:|---:|---|
| Baseline | 640 | 30 | 초기 성능 및 실패 유형 확인 |
| Improved V1 | 960 | 30 | 극소형 UAV 특징 손실 완화 |
| Improved V2 | 960 | 100 | 추가 학습 효과 확인 |

각 단계에서는 다른 조건을 유지하고 **주요 조건 하나만 변경**했습니다.

### Validation

| Model | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9416 | 0.8462 | 0.8914 | 0.8937 | 0.6230 | 0.5561 | 123 | 99 |
| Improved V1 | 0.9603 | 0.8847 | 0.9209 | **0.9362** | 0.7225 | 0.6202 | 80 | 75 |
| Improved V2 | **0.9658** | **0.9032** | **0.9334** | 0.9354 | **0.7703** | **0.6500** | **76** | **51** |

가장 작은 객체군 Q1 Recall:

**0.7489 → 0.8265 → 0.8311**

Validation 결과를 기준으로 **Improved V2를 최종 모델로 고정**했습니다.

### Final Test

| Model | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9483 | 0.8838 | 0.9149 | 0.9143 | 0.7051 | 0.6139 |
| Improved V1 | 0.9661 | 0.9252 | 0.9452 | 0.9405 | 0.7500 | 0.6561 |
| Improved V2 | **0.9704** | **0.9372** | **0.9535** | **0.9419** | **0.7801** | **0.6852** |

`analysis/06_final_test.ipynb`에서 세 모델에 동일하게 `conf=0.25`를 적용한 비교 결과입니다.

---

## 2. Dataset

- **DUT Anti-UAV Detection**
- Single-class UAV Detection
- Pascal VOC XML → YOLO format
- Official dataset: https://github.com/wangdongdut/DUT-Anti-UAV

### Full Dataset

| Split | Images | Objects | bbox 면적 1% 미만 |
|---|---:|---:|---:|
| Train | 5,200 | 5,243 | 88.33% |
| Validation | 2,600 | 2,621 | 88.55% |
| Test | 2,200 | 2,245 | 75.86% |

대부분의 UAV가 작은 객체이므로, 48시간 내 반복 실험이 가능하도록 공식 Split을 유지한 채 각 Split의 약 1/3을 사용했습니다.

### Subset

| Split | Images | Objects |
|---|---:|---:|
| Train | 1,733 | 1,776 |
| Validation | 867 | 876 |
| Test | 733 | 749 |

서브셋 구성 기준:

- 공식 Train / Validation / Test 간 데이터 이동 없음
- 단일 객체 이미지를 bbox 크기순 5개 구간에서 균형 추출
- Train의 희소한 배경·다중 객체 사례 전체 보존
- Validation / Test의 희소 사례 비율 유지
- `Seed=42` 고정
- 원본과 서브셋의 bbox 분포 비교 후 대표성 확인

---

## 3. Project Structure

```text
small-drone-detection/
├── src/
│   ├── 01_download_dataset.py
│   ├── 02_make_dataset.py
│   ├── 03_train.py
│   ├── 04_make_subset.py
│   ├── 05_validation.py
│   └── 06_test.py
│
├── analysis/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_subset_analysis.ipynb
│   ├── 03_baseline_failure_analysis.ipynb
│   ├── 04_v1_failure_analysis.ipynb
│   ├── 05_v2_failure_analysis.ipynb
│   └── 06_final_test.ipynb
│
├── data/                  # Git 제외
├── pretrained/
├── models/
│   ├── baseline/
│   ├── improved_v1/
│   └── improved_v2/
├── requirements.txt
└── README.md
```

> 파일 번호는 작성 순서를 유지하고 있어 실제 실행 순서와 일부 다릅니다.  
> 재현 시 아래 **Reproduction** 순서를 기준으로 실행합니다.

---

## 4. Environment

| Item | Setting |
|---|---|
| Python | 3.11.16 |
| Model | YOLO26s |
| Batch Size | 16 |
| Seed | 42 |
| Workers | 4 |
| Device | CUDA GPU (`device=0`) |

```bash
conda create -n small-drone python=3.11.16 -y
conda activate small-drone
pip install -r requirements.txt
```

설치 확인:

```bash
python -c "import torch, ultralytics; print(torch.__version__); print(ultralytics.__version__); print(torch.cuda.is_available())"
```

---

## 5. Reproduction

모든 명령어는 프로젝트 루트에서 실행합니다.

### 1) Dataset Download

```bash
python src/01_download_dataset.py
```

DUT Anti-UAV 공식 Train / Validation / Test 데이터를 `data/raw/`에 다운로드합니다.

### 2) Pascal VOC → YOLO

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

### 3) Subset 생성

`03_train.py`보다 먼저 실행합니다.

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

동일한 `Seed=42`와 선정 규칙을 사용하므로 동일 서브셋 재생성이 가능합니다.

### 4) Train

세 모델 모두 `src/03_train.py` 사용:

| Version | `VERSION` | `IMG_SIZE` | `EPOCHS` |
|---|---|---:|---:|
| Baseline | `"baseline"` | 640 | 30 |
| V1 | `"v1"` | 960 | 30 |
| V2 | `"v2"` | 960 | 100 |

설정을 변경한 뒤 각각 실행:

```bash
python src/03_train.py
```

공통 학습 조건:

```text
Model   : yolo26s.pt
Dataset : yolo_subset
Batch   : 16
Seed    : 42
Workers : 4
```

학습 결과는 실행 조건과 timestamp가 포함된 폴더에 저장됩니다.

```text
models/baseline_subset_yolo26s_640_ep30_<timestamp>/
models/v1_subset_yolo26s_960_ep30_<timestamp>/
models/v2_subset_yolo26s_960_ep100_<timestamp>/
```

Validation 및 분석 노트북에서 사용하는 경로에 맞춰 최종 학습 폴더를 다음 이름으로 정리합니다.

```text
baseline_subset_... → models/baseline/
v1_subset_...       → models/improved_v1/
v2_subset_...       → models/improved_v2/
```

최종 모델 경로:

```text
models/baseline/weights/best.pt
models/improved_v1/weights/best.pt
models/improved_v2/weights/best.pt
```

### 5) Validation

`src/05_validation.py`의 `RUN_NAME`, `IMG_SIZE`를 모델별로 변경합니다.

```text
baseline    : RUN_NAME="baseline",    IMG_SIZE=640
improved_v1 : RUN_NAME="improved_v1", IMG_SIZE=960
improved_v2 : RUN_NAME="improved_v2", IMG_SIZE=960
```

각 모델마다 실행:

```bash
python src/05_validation.py
```

결과:

```text
models/<model_name>/val_<timestamp>/val_metrics.csv
```

### 6) Failure Analysis

```bash
jupyter lab
```

다음 순서로 실행합니다.

```text
analysis/03_baseline_failure_analysis.ipynb
analysis/04_v1_failure_analysis.ipynb
analysis/05_v2_failure_analysis.ipynb
```

새로 Validation을 실행한 경우 각 노트북 첫 코드 셀의 `VAL_NAME`을 새 `val_<timestamp>` 폴더명으로 변경합니다.

공통 실패 판정 기준:

```text
Confidence threshold = 0.25
Match IoU            = 0.5
Near IoU             = 0.1
```

| FN Type | 기준 |
|---|---|
| 낮은 신뢰도 | IoU ≥ 0.5 예측이 있으나 confidence < 0.25 |
| 위치 부정확 | GT와 가장 높은 IoU가 0.1 이상 0.5 미만 |
| 미검출 | 모든 예측과 IoU < 0.1 |

### 7) Final Test

모든 개선 의사결정이 끝난 뒤 실행:

```text
analysis/06_final_test.ipynb
```

```python
MODELS = {
    "baseline": 640,
    "improved_v1": 960,
    "improved_v2": 960,
}
```

Test Set은 최종 확인용으로만 사용합니다.

---

## 6. Improvement Flow

### Baseline

- Recall: 0.8462
- Q1 Recall: 0.7489
- FN: 123
- FP: 99
- mAP75: 0.6230

가장 작은 객체군의 낮은 Recall을 핵심 문제로 확인했습니다.

### V1 — Input Size 640 → 960

- Q1 Recall: **0.7489 → 0.8265**
- Recall: **0.8462 → 0.8847**
- FN: **123 → 80**
- FP: **99 → 75**
- mAP75: **0.6230 → 0.7225**

Input Size 증가가 극소형 UAV 검출에 효과가 있음을 확인했습니다.

### V2 — Epoch 30 → 100

- Recall: **0.8847 → 0.9032**
- mAP75: **0.7225 → 0.7703**
- mAP50-95: **0.6202 → 0.6500**
- FN: **80 → 76**
- FP: **75 → 51**

전체 성능은 추가 개선됐지만 극소형·저화질 UAV 미검출과 배경 오탐은 일부 남았습니다.

따라서 추가 Epoch보다 **실제 오탐 배경과 어려운 UAV 사례를 학습 데이터에 추가하는 방향**을 다음 개선안으로 판단했습니다.

---

## 7. Limitations

- 사람도 구별하기 어려운 극소형·저화질 UAV 미검출
- 조류·비행기·건물 구조물·바위·그림자 등 배경 객체 오탐
- 복잡한 수목 배경에서 실제 UAV 미검출
- 일부 위치 오차에서 타이트한 GT annotation 영향 가능성
- 다른 촬영 환경에 대한 추가 일반화 검증 필요

---

## 8. Notebook Guide

| Notebook | 역할 |
|---|---|
| `01_dataset_analysis.ipynb` | 전체 데이터 특성 분석 |
| `02_subset_analysis.ipynb` | 서브셋 대표성 검증 |
| `03_baseline_failure_analysis.ipynb` | Baseline 실패 분석 및 V1 근거 |
| `04_v1_failure_analysis.ipynb` | V1 효과 검증 및 V2 근거 |
| `05_v2_failure_analysis.ipynb` | V2 효과 및 학습 종료 판단 |
| `06_final_test.ipynb` | 최종 Test 확인 |

---

## 9. Quick Reproduction

```bash
conda create -n small-drone python=3.11.16 -y
conda activate small-drone
pip install -r requirements.txt

python src/01_download_dataset.py
python src/02_make_dataset.py
python src/04_make_subset.py

# src/03_train.py 설정을 Baseline → V1 → V2로 변경하며 각각 실행
python src/03_train.py

# src/05_validation.py 설정을 각 모델에 맞게 변경하며 각각 실행
python src/05_validation.py

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

GPU, CUDA, PyTorch, OS 차이에 따라 추론 시간과 소수점 수준의 metric 차이가 발생할 수 있습니다.

---

## References

- DUT Anti-UAV: https://github.com/wangdongdut/DUT-Anti-UAV
- Ultralytics: https://github.com/ultralytics/ultralytics
