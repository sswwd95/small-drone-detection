# Small Drone Detection

YOLO26s 기반의 소형 UAV 탐지 프로젝트입니다.

DUT Anti-UAV Detection 데이터셋을 YOLO 형식으로 변환하고, 데이터 특성과 Baseline 실패 사례를 분석한 뒤 **Input Size**와 **Epoch**를 단계적으로 변경하여 성능 변화를 검증했습니다.

- **Baseline**: YOLO26s, 640, 30 Epoch
- **V1**: Input Size 640 → 960
- **V2**: Epoch 30 → 100
- **Final model**: V2 `best.pt`

> 이 README는 제3자가 학습·평가를 재현하는 데 필요한 절차를 중심으로 작성했습니다.  
> 상세한 데이터 분석, 실패 사례 시각화, 개선 근거는 `analysis/` 에서 확인할 수 있습니다.

---

## 1. Key Results

### Internal Test Set

동일한 Test subset 733장을 모든 모델에 고정하여 비교했습니다.

| Metric | Baseline | V1 | V2 |
|---|---:|---:|---:|
| Precision | 0.9483 | 0.9661 | **0.9704** |
| Recall | 0.8838 | 0.9252 | **0.9372** |
| F1 | 0.9149 | 0.9452 | **0.9535** |
| mAP50 | 0.9340 | **0.9687** | 0.9584 |
| mAP75 | 0.7187 | 0.7624 | **0.7954** |
| mAP50-95 | 0.6269 | 0.6685 | **0.6956** |
| Inference Time | **2.2016 ms** | 3.2384 ms | 3.1860 ms |

Failure analysis:

| Metric | Baseline | V2 |
|---|---:|---:|
| FN | 63 | **42** |
| FP | 81 | **45** |
| Q1 Recall | 0.8723 | **0.9468** |
| Low-confidence FN | 22 | **12** |
| Localization FN | 21 | **16** |
| Miss FN | 20 | **14** |

V2는 V1보다 `mAP50`이 0.0103 낮지만, `Recall`, `mAP75`, `mAP50-95`, Q1 Recall과 외부 영상 결과를 함께 고려하여 최종 모델로 선정했습니다.

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
│
├── analysis/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_subset_analysis.ipynb
│   ├── 03_baseline_failure_analysis.ipynb
│   ├── 04_v1_failure_analysis.ipynb
│   ├── 05_v2_failure_analysis.ipynb
│   └── 06_test_video.ipynb
│
├── data/
│   ├── raw/
│   ├── yolo/
│   ├── yolo_subset/
│   └── external/
│
├── models/
├── requirements.txt
└── README.md
```

`analysis/` 노트북은 **기본 학습·평가 실행에 필수는 아닙니다.**  
보고서의 데이터 분석, FN·FP 유형화, 객체 크기별 Recall, 외부 영상 비교 결과를 다시 확인할 때 사용합니다.

---

## 3. Environment

실험 환경:

| Item | Environment |
|---|---|
| OS | Windows 10 |
| GPU | NVIDIA GeForce RTX 3090 |
| Python | 3.11.16 |
| PyTorch | 2.13.0+cu126 |
| CUDA | 12.6 |
| Ultralytics | 8.4.138 |
| Model | YOLO26s |

설치:

```bash
git clone https://github.com/sswwd95/small-drone-detection.git
cd small-drone-detection

conda create -n ardet python=3.11.16 -y
conda activate ardet

pip install -r requirements.txt
```

GPU, CUDA, PyTorch 및 Ultralytics 버전에 따라 소수점 수준의 결과 차이가 발생할 수 있습니다.

---

## 4. Core Reproduction

기본 재현은 아래 순서만 실행하면 됩니다.

> 파일 번호는 생성 순서를 유지하고 있어 실제 실행 순서는 `01 → 02 → 04 → 03 → 05`입니다.

### Step 1. Dataset Download

[DUT Anti-UAV Detection](https://github.com/wangdongdut/DUT-Anti-UAV)의 공식 Detection Train / Validation / Test split을 사용합니다.

```bash
python src/01_download_dataset.py
```

```text
data/raw/
├── train/
├── val/
└── test/
```

### Step 2. XML → YOLO Conversion

Pascal VOC XML annotation을 YOLO 형식으로 변환합니다.

```bash
python src/02_make_dataset.py
```

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

단일 클래스 `UAV`를 사용합니다.

### Step 3. Reproducible Subset

48시간 내 반복 실험을 위해 각 공식 split에서 약 1/3을 추출합니다.

```bash
python src/04_make_subset.py
```

고정 조건:

- `Seed = 42`
- Train / Validation / Test 간 이동 없음
- 단일 객체 이미지는 bbox 크기순 5등분 후 구간별 균등 추출
- Train의 background 3장과 multi-object 29장은 모두 보존
- Validation / Test의 희소 사례는 원본 비율에 맞춰 추출
- 동일 seed 사용 시 동일 subset 재생성

생성 결과:

| Split | Original | Subset |
|---|---:|---:|
| Train | 5,200 | 1,733 |
| Validation | 2,600 | 867 |
| Test | 2,200 | 733 |

Validation의 multi-object 이미지는 13장 중 4장, Test는 33장 중 11장이 선택되어 원본 비율을 유지합니다.

### Step 4. Training

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

실험별 설정:

| Version | IMG_SIZE | EPOCHS |
|---|---:|---:|
| Baseline | 640 | 30 |
| V1 | 960 | 30 |
| V2 | 960 | 100 |

`src/03_train.py` 상단의 `VERSION`, `IMG_SIZE`, `EPOCHS`를 위 표에 맞춰 각각 실행합니다.

학습 결과 예시:

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

최종 평가는 마지막 Epoch의 `last.pt`가 아니라 Validation 성능 기준 `best.pt`를 사용합니다.

### Step 5. Test Evaluation

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

`src/05_test.py` 상단의 `RUN_NAME`을 평가할 학습 결과 폴더명으로 지정합니다.

---

## 5. Optional: External RGB Video Evaluation

내부 Test 이외의 환경에서 모델을 추가 비교하기 위해 [Anti-UAV300](https://github.com/zsx060/Anti-UAV-datasets) RGB 영상을 사용했습니다.

최종 비교 영상:

- `20190925_111757_1_5`
- `20190925_131530_1_2`

외부 영상은 학습에 사용하지 않았으며, 제한된 2개 영상에 대한 **보조 일반화 검증**으로만 사용했습니다.

```bash
python src/06_video_test.py
```

예상 입력:

```text
data/external/
└── <sequence_name>/
    ├── visible.mp4
    └── visible.json
```

예상 출력:

```text
models/<run_name>/external_test/
├── <sequence>_result.mp4
└── <sequence>_metrics.txt
```

### External Test Results

| Sequence | Model | Precision | Recall | F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| 20190925_111757_1_5 | Baseline | 0.8589 | 0.9070 | 0.8823 | 149 | 93 |
|  | V1 | 0.9645 | **0.9780** | 0.9712 | 36 | **22** |
|  | V2 | **0.9869** | **0.9780** | **0.9824** | **13** | **22** |
| 20190925_131530_1_2 | Baseline | **0.9068** | 0.2920 | 0.4418 | **30** | 708 |
|  | V1 | 0.6197 | 0.4660 | 0.5320 | 286 | 534 |
|  | V2 | 0.7662 | **0.5440** | **0.6363** | 166 | **456** |

첫 번째 영상에서는 V2가 V1과 동일한 Recall을 유지하면서 FP를 36 → 13으로 줄였습니다.

두 번째 고난도 영상에서는 V2가 Baseline 대비 Recall과 F1을 개선했지만, Precision은 0.9068 → 0.7662로 낮아지고 FP는 30 → 166으로 증가했습니다. 따라서 이 결과는 **미탐 감소와 오탐 증가의 trade-off**로 해석했습니다.

---

## 6. Optional: Reproduce Report Analysis

`analysis/` 노트북은 보고서의 판단 과정과 실패 분석을 확인하기 위한 파일입니다.  
**모델 학습과 기본 Test 재현만 목적이라면 실행할 필요가 없습니다.**

| Notebook | Purpose | Prerequisite |
|---|---|---|
| `01_dataset_analysis.ipynb` | 데이터 구조, bbox 크기, 밝기·대비, 희소 사례, Split 특성 확인 | `data/raw/` |
| `02_subset_analysis.ipynb` | 원본과 subset의 bbox·희소 사례 분포 비교 | `data/yolo/` |
| `03_baseline_failure_analysis.ipynb` | Baseline FN·FP, 원인 유형, 크기별 Recall 분석 | Baseline 학습 + Test |
| `04_v1_failure_analysis.ipynb` | V1 개선 효과와 잔여 실패 분석 | V1 학습 + Test |
| `05_v2_failure_analysis.ipynb` | V2 성능, 학습곡선, 최종 실패 분석 | V2 학습 + Test |
| `06_test_video.ipynb` | 외부 영상 선정 과정 및 모델별 결과 비교 | External Test 결과 |

### Failure-analysis notebook 경로 설정

`03_baseline_failure_analysis.ipynb`, `04_v1_failure_analysis.ipynb`, `05_v2_failure_analysis.ipynb`는 첫 설정 셀의 `RUN_NAME`과 `TEST_NAME`을 실제 생성된 폴더명으로 맞춘 뒤 실행합니다.

예:

```python
RUN_NAME = "<training_run_name>"
TEST_NAME = "<test_run_name>"
```

### External comparison notebook 경로 설정

`06_test_video.ipynb`의 `MODEL_DIRS`를 실제 `external_test` 결과 폴더에 맞춰 수정한 뒤 비교 셀을 실행합니다.

분석 노트북에서 사용하는 FN·FP 기준:

```text
TP               : confidence >= 0.25 and IoU >= 0.5
FP               : confidence >= 0.25, unmatched prediction
Low-confidence FN: matching location but confidence < 0.25
Localization FN  : nearest prediction IoU 0.1 ~ 0.5
Miss FN          : nearest prediction IoU < 0.1
```

`confidence = 0.25`는 FN·FP 사례 분석 기준이며, mAP 계산 기준과 동일한 의미로 사용하지 않습니다.

---

## 7. Reproducibility Notes

재현성을 위해 다음 조건을 고정했습니다.

- 공개 데이터 사용
- 공식 Train / Validation / Test split 유지
- Subset seed `42`
- Training seed `42`
- `deterministic=True`
- 동일 Test subset 733장 고정
- 모델별 Validation 기준 `best.pt` 사용
- 데이터 변환 및 subset 생성 코드 공개
- 학습 조건과 실행 시간을 `training_info.txt`에 저장
- Test 결과 저장
- 외부 영상 평가 결과 저장

### Important Evaluation Limitation

Baseline 실패 사례를 확인한 뒤 **동일한 Test subset 733장으로 V1과 V2를 반복 비교**했습니다.

따라서 이 Test 결과는 완전히 untouched된 최종 평가셋의 성능으로 해석하지 않았습니다. 후속 실험에서는 Validation에서 실패 분석과 모델 선택을 완료한 뒤, 영상 단위의 독립 Test를 최종 1회 평가에 사용하는 것이 더 엄밀합니다.

---

## 8. Limitations

- 48시간 제약으로 전체 데이터가 아닌 약 1/3 subset 기반 학습
- 동일 Test subset을 개선 방향 결정과 반복 비교에 사용
- 원본 데이터의 유사 연속 프레임 가능성
- 외부 검증 영상이 2개로 제한됨
- 작은 UAV에서 localization error와 miss가 잔존
- UAV와 유사한 배경 패턴에서 false positive가 잔존
- 두 번째 외부 영상에서 Recall 개선과 함께 Precision 하락·FP 증가 확인
- Input Size 증가에 따른 inference time 증가

향후에는 Hard Positive / Hard Negative 보강, Validation 기반 모델 선택, 영상 단위 독립 Test, 외부 데이터 확대, 입력 크기 및 추론 설정 최적화를 추가 검증할 예정입니다.

---

## 9. Dataset Sources

- [DUT Anti-UAV Detection](https://github.com/wangdongdut/DUT-Anti-UAV)
- [Anti-UAV300](https://github.com/zsx060/Anti-UAV-datasets)

학습 및 외부 검증 데이터는 원본 공개 데이터에서 내려받아 사용하며, 원본 데이터 자체는 본 저장소에 재배포하지 않습니다.
