# Small Drone Detection

YOLO26s 기반의 소형 UAV 탐지 프로젝트입니다.  
DUT Anti-UAV Detection 데이터셋을 YOLO 형식으로 변환하고, 데이터 특성과 실패 사례를 분석한 뒤 입력 해상도와 학습 Epoch를 단계적으로 변경하여 성능을 개선했습니다.

최종적으로 내부 Test Set과 별도의 Anti-UAV300 RGB 영상에서 Baseline, V1, V2를 비교하여 V2를 최종 모델로 선정했습니다.

---

## 1. Experiment Summary

| Model | Input Size | Epoch | Purpose |
|---|---:|---:|---|
| Baseline | 640 | 30 | 기준 성능 확인 |
| V1 | 960 | 30 | 극소형 객체 탐지 개선 |
| V2 | 960 | 100 | 추가 학습 효과 검증 |

### Final Test Results

| Metric | Baseline | V2 | Change |
|---|---:|---:|---:|
| Precision | 0.9483 | **0.9704** | +0.0221 |
| Recall | 0.8838 | **0.9372** | +0.0534 |
| F1 | 0.9149 | **0.9535** | +0.0386 |
| mAP50 | 0.9340 | **0.9584** | +0.0244 |
| mAP75 | 0.7187 | **0.7954** | +0.0767 |
| mAP50-95 | 0.6269 | **0.6956** | +0.0687 |
| Inference Time | **2.2016 ms** | 3.1860 ms | +0.9844 ms |

추가 실패 분석 결과:

- FN: 63 → **42**
- FP: 81 → **45**
- Q1 Recall: 0.8723 → **0.9468**
- Low-confidence FN: 22 → **12**
- Localization FN: 21 → **16**
- Miss FN: 20 → **14**

---

## 2. Project Structure

```text
small-drone-detection/
├── src/
│   ├── 01_download_dataset.py
│   ├── 02_make_dataset.py
│   ├── 03_train.py
│   ├── 04_make_subset.py
│   ├── 05_test.py
│   └── 06_video_test.py
├── data/
│   ├── raw/
│   ├── yolo/
│   ├── yolo_subset/
│   └── external/
├── models/
├── analysis/
├── requirements.txt
└── README.md
```

---

## 3. Environment

### Training Environment

| Item | Environment |
|---|---|
| OS | Windows 10 |
| GPU | NVIDIA GeForce RTX 3090 |
| Device | GPU 0 |
| PyTorch | 2.13.0+cu126 |
| CUDA | 12.6 |
| Ultralytics | 8.4.138 |
| Model | YOLO26s |

필수 Python 패키지:

```text
ultralytics
torch
numpy
pandas
opencv-python
gdown
kaggle
```

설치:

```bash
git clone https://github.com/sswwd95/small-drone-detection.git
cd small-drone-detection

conda create -n ardet python=3.11.16 -y
conda activate ardet

pip install -r requirements.txt
```


---

## 4. Dataset

### Training / Validation / Test

[DUT Anti-UAV Detection](https://github.com/wangdongdut/DUT-Anti-UAV)

공식 Detection 데이터의 Train / Val / Test split을 그대로 사용합니다.

```bash
python src/01_download_dataset.py
```

실행 후:

```text
data/raw/
├── train/
├── val/
└── test/
```

### YOLO Format Conversion

원본 XML annotation을 YOLO 형식으로 변환합니다.

```bash
python src/02_make_dataset.py
```

생성 구조:

```text
data/yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

YOLO label format:

```text
class x_center y_center width height
```

본 프로젝트는 단일 클래스 `UAV`를 사용합니다.

---

## 5. Reproducible Subset

전체 학습 시간 제약으로 공식 Train / Val / Test 각각에서 약 1/3을 추출하여 실험했습니다.

```bash
python src/04_make_subset.py
```

Subset 생성 기준:

- Seed: `42`
- 각 공식 split에서 약 `1/3` 추출
- bbox 크기 분포를 5개 구간으로 나누어 단일 객체 이미지 샘플링
- Train의 희소한 background 이미지와 multi-object 이미지는 모두 포함
- Val / Test는 원본의 object-count 구성 비율 유지
- 동일 seed 사용 시 동일 subset 재생성

---

## 6. Training

```bash
python src/03_train.py
```

공통 설정:

```text
Model         : YOLO26s
Dataset       : yolo_subset
Seed          : 42
Deterministic : True
Device        : GPU 0
```

실험별 변경값:

| Version | IMG_SIZE | EPOCHS |
|---|---:|---:|
| Baseline | 640 | 30 |
| V1 | 960 | 30 |
| V2 | 960 | 100 |

`03_train.py` 상단의 `VERSION`, `IMG_SIZE`, `EPOCHS`를 위 표에 맞춰 실행합니다.

학습 결과는 실행 조건과 시간이 포함된 폴더명으로 `models/`에 저장됩니다.

```text
models/
└── <run_name>/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.csv
    ├── results.png
    ├── confusion_matrix.png
    └── training_info.txt
```

---

## 7. Test Evaluation

```bash
python src/05_test.py
```

평가 지표:

- Precision
- Recall
- F1
- mAP50
- mAP75
- mAP50-95
- Inference Time

> `05_test.py`의 `RUN_NAME`은 평가하려는 학습 결과 폴더명으로 지정해야 합니다.

---

## 8. External RGB Video Evaluation

내부 Test Set 이외의 환경에서도 개선 효과를 확인하기 위해 공개 [Anti-UAV300](https://github.com/zsx060/Anti-UAV-datasets) RGB 영상을 추가로 사용했습니다.

외부 영상은 학습에 사용하지 않았으며 최종 모델 선택을 위한 보조 일반화 검증으로 사용했습니다.

로컬 구조:

```text
data/external/
└── <sequence_name>/
    ├── visible.mp4
    └── visible.json
```

평가:

```bash
python src/06_video_test.py
```

영상 출력:

- Green box: Ground Truth
- Red box: Prediction
- Prediction confidence
- Frame-level IoU

저장 결과:

```text
models/<model>/external_test/
├── <sequence>_result.mp4
└── <sequence>_metrics.txt
```

### External Test Example

| Sequence | Model | Precision | Recall | F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| 20190925_111757_1_5 | Baseline | 0.8589 | 0.9070 | 0.8823 | 149 | 93 |
|  | V1 | 0.9645 | **0.9780** | 0.9712 | 36 | **22** |
|  | V2 | **0.9869** | **0.9780** | **0.9824** | **13** | **22** |
| 20190925_131530_1_2 | Baseline | 0.9068 | 0.2920 | 0.4418 | **30** | 708 |
|  | V1 | 0.6197 | 0.4660 | 0.5320 | 286 | 534 |
|  | V2 | **0.7662** | **0.5440** | **0.6363** | 166 | **456** |

---

## 9. Full Reproduction Order

```bash
# 1. Dataset download
python src/01_download_dataset.py

# 2. XML -> YOLO conversion
python src/02_make_dataset.py

# 3. Reproducible subset
python src/04_make_subset.py

# 4. Training
# Baseline / V1 / V2 설정 후 각각 실행
python src/03_train.py

# 5. Test evaluation
# RUN_NAME을 평가할 모델 폴더명으로 지정
python src/05_test.py

# 6. Optional external RGB video evaluation
python src/06_video_test.py
```

핵심 파이프라인:

```text
Public Dataset
      ↓
YOLO Conversion
      ↓
Reproducible Subset
      ↓
Baseline
      ↓
Failure Analysis
      ↓
V1: Input Size 640 → 960
      ↓
V2: Epoch 30 → 100
      ↓
Test + External Video Evaluation
      ↓
Final Model Selection
```

---

## 10. Reproducibility

실험 재현을 위해 다음 조건을 고정했습니다.

- Public dataset 사용
- 공식 Train / Val / Test split 유지
- Subset seed `42`
- Training seed `42`
- `deterministic=True`
- 데이터 변환 및 subset 생성 코드 공개
- 학습 조건 및 결과를 `training_info.txt`에 저장
- Test 결과를 CSV로 저장
- 외부 영상 평가 결과를 TXT로 저장

GPU, CUDA, PyTorch 및 Ultralytics 버전에 따라 소수점 수준의 결과 차이는 발생할 수 있습니다.

---

## 11. Limitations

- 시간 제약으로 전체 데이터가 아닌 subset 기반 학습 수행
- 일부 외부 영상만 활용한 일반화 성능 검증
- 원본 데이터에 유사한 연속 프레임이 포함될 가능성
- 작은 UAV에서 localization error와 miss가 잔존
- UAV와 유사한 배경 패턴에서 false positive가 잔존
- Input Size 증가에 따른 inference time 증가

향후에는 Hard Positive / Hard Negative 보강, 영상 단위 독립 split, 외부 데이터 확대, 입력 크기 및 추론 최적화를 추가 검증할 예정입니다.

---

## 12. Dataset Sources

- DUT Anti-UAV Detection  
  https://github.com/wangdongdut/DUT-Anti-UAV

- Anti-UAV300  
  https://github.com/zsx060/Anti-UAV-datasets

학습 및 외부 검증 데이터는 원본 저장소의 공개 데이터를 사용하며, 원본 데이터 자체는 본 GitHub 저장소에 재배포하지 않습니다.
