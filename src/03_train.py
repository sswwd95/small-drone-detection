from contextlib import chdir
from datetime import datetime
from pathlib import Path
import time

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

VERSION = "v1" #baseline, v1,
PRETRAINED_MODEL = "yolo26s.pt"
PRETRAINED_DIR = ROOT / "pretrained"
PRETRAINED_PATH = PRETRAINED_DIR / PRETRAINED_MODEL
MODEL_ROOT = ROOT / "models"

USE_SUBSET = True
DATA_NAME = "yolo_subset" if USE_SUBSET else "yolo"
DATA_DIR = ROOT / "data" / DATA_NAME
DATA_YAML = DATA_DIR / "data.yaml"

# baseline : 640
# v1 : 1280
IMG_SIZE = 1280
EPOCHS = 30
BATCH_SIZE = 16
DEVICE = 0
SEED = 42
WORKERS = 4


def main():
    """모델 학습 및 결과 저장."""

    if not DATA_DIR.exists():
        guide = "python src/04_make_subset.py" if USE_SUBSET else "python src/02_prepare_dataset.py"
        raise FileNotFoundError(f"데이터셋 없음: {DATA_DIR}\n먼저 {guide} 실행 필요")

    yaml_text = (
        f"path: {DATA_DIR.resolve().as_posix()}\n\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: UAV\n"
    )
    DATA_YAML.write_text(yaml_text, encoding="utf-8")

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    if PRETRAINED_PATH.exists():
        model = YOLO(str(PRETRAINED_PATH))
    else:
        with chdir(PRETRAINED_DIR):
            model = YOLO(PRETRAINED_MODEL)

    data_type = "subset" if USE_SUBSET else "full"
    run_name = (
        f"{VERSION}_{data_type}_{Path(PRETRAINED_MODEL).stem}_"
        f"{IMG_SIZE}_ep{EPOCHS}_{datetime.now():%y%m%d_%H%M}"
    )

    start_datetime = datetime.now()
    start_time = time.perf_counter()

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        seed=SEED,
        deterministic=True,
        workers=WORKERS,
        plots=True,
        amp=False,
        project=str(MODEL_ROOT),
        name=run_name,
    )

    end_datetime = datetime.now()
    elapsed_minutes = (time.perf_counter() - start_time) / 60

    model_dir = Path(results.save_dir)
    best_model = model_dir / "weights" / "best.pt"
    gpu_name = torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else "CPU"

    training_info = f"""학습 결과
============================================================
버전            : {VERSION}
실행 이름       : {run_name}
데이터          : {data_type}
사전학습 모델   : {PRETRAINED_MODEL}
입력 크기       : {IMG_SIZE}
Epoch           : {EPOCHS}
Batch           : {BATCH_SIZE}
Seed            : {SEED}
Workers         : {WORKERS}
GPU             : {gpu_name}
PyTorch         : {torch.__version__}
CUDA            : {torch.version.cuda}

학습 시작 시각  : {start_datetime:%Y-%m-%d %H:%M:%S}
학습 종료 시각  : {end_datetime:%Y-%m-%d %H:%M:%S}
학습 시간(분)   : {elapsed_minutes:.2f}

학습 결과 경로  : {model_dir}
최종 모델 경로  : {best_model}
"""

    print(training_info)
    (model_dir / "training_info.txt").write_text(training_info, encoding="utf-8")


if __name__ == "__main__":
    main()