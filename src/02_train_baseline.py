from datetime import datetime
from pathlib import Path
import shutil
import time

import torch
from ultralytics import YOLO


# --------------------------------------------------
# 1. 학습 설정
# --------------------------------------------------

# 사전학습 모델
# 소형 객체 대응 학습 구조와 노트북 연산 환경을 고려한 YOLO26s 선정
MODEL_NAME = "yolo26s.pt"

# YOLO 데이터 설정 파일
DATA_YAML = "data/yolo/data.yaml"

# 입력 이미지 크기
IMG_SIZE = 640
# 학습 epoch 수
EPOCHS = 1
# 학습 batch 크기
BATCH_SIZE = 8
# GPU 번호
DEVICE = 0
# 공통 시드
SEED = 42
# DataLoader worker 수
WORKERS = 4
# 학습 결과 경로
PROJECT = "runs/train"
# 모델 저장 경로
MODEL_ROOT = Path("models")
# 실행 시각
RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
# 실행 이름
RUN_NAME = f"baseline_{MODEL_NAME.replace('.pt', '')}_{IMG_SIZE}_{RUN_TIME}"
# 실행별 모델 폴더
MODEL_DIR = MODEL_ROOT / RUN_NAME

# --------------------------------------------------
# 2. 모델 로드
# --------------------------------------------------

# COCO 사전학습 가중치
model = YOLO(MODEL_NAME)


# --------------------------------------------------
# 3. Baseline 학습
# --------------------------------------------------

# 학습 시작 시각
start_datetime = datetime.now()

# 학습 시간 측정 시작
start_time = time.perf_counter()

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    seed=SEED,
    deterministic=True, # 재학습 시 연산 결과 변동 최소화를 위한 결정론적 연산 사용
    workers=WORKERS,
    plots=True,   # 학습 및 검증 분석 그래프 저장
    project=PROJECT,
    name=RUN_NAME
)

# 학습 소요 시간
elapsed_seconds = time.perf_counter() - start_time

# 학습 종료 시각
end_datetime = datetime.now()


# --------------------------------------------------
# 4. 모델 저장
# --------------------------------------------------

# 실행별 모델 폴더 생성
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# best.pt 경로
best_weight = Path(results.save_dir) / "weights" / "best.pt"

# 최종 모델 경로
saved_model = MODEL_DIR / f"{RUN_NAME}_best.pt"

# 최종 모델 복사
shutil.copy2(best_weight, saved_model)


# --------------------------------------------------
# 5. 학습 정보 저장
# --------------------------------------------------

# GPU 정보
gpu_name = torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else "CPU"

# 학습 정보
training_info = f"""Baseline 학습 결과
============================================================
실행 이름       : {RUN_NAME}
모델            : {MODEL_NAME}
입력 크기       : {IMG_SIZE}
Epoch           : {EPOCHS}
Batch           : {BATCH_SIZE}
Seed            : {SEED}
Workers         : {WORKERS}
GPU             : {gpu_name}

학습 시작 시각  : {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}
학습 종료 시각  : {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}
학습 시간(초)   : {elapsed_seconds:.2f}
학습 시간(분)   : {elapsed_seconds / 60:.2f}

학습 결과 경로  : {results.save_dir}
최종 모델 경로  : {saved_model}
"""

# 터미널 출력
print(training_info)

# 학습 정보 TXT 경로
log_path = MODEL_DIR / f"{RUN_NAME}_training_info.txt"

# 학습 정보 TXT 저장
log_path.write_text(training_info, encoding="utf-8")

print("학습 정보 경로  :", log_path)
