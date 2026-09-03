from datetime import datetime
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

RUN_NAME = "improved_v2" # baseline, improved_v1, improved_v2

MODEL_DIR = ROOT / "models" / RUN_NAME
MODEL_PATH = MODEL_DIR / "weights" / "best.pt"
DATA_YAML = ROOT / "data" / "yolo_subset" / "data.yaml"

# baseline IMG_SIZE = 640
# improved_v1, improved_v2 IMG_SIZE = 960
IMG_SIZE = 960
BATCH_SIZE = 16
DEVICE = 0
SEED = 42


def main():
    """Validation 성능 및 추론시간 평가."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 없음: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    val_name = f"val_{datetime.now():%y%m%d_%H%M}"

    results = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        seed=SEED,
        plots=True,
        project=str(MODEL_DIR),
        name=val_name,
    )

    precision = results.box.mp
    recall = results.box.mr
    f1 = 2 * precision * recall / (precision + recall)

    metrics = pd.DataFrame([{
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "mAP50": results.box.map50,
        "mAP75": results.box.map75,
        "mAP50-95": results.box.map,
        "Inference(ms)": results.speed["inference"],
    }])

    save_path = Path(results.save_dir) / "val_metrics.csv"
    metrics.to_csv(save_path, index=False)

    print("\nValidation 결과")
    print("=" * 90)
    print(metrics.round(4).to_string(index=False))
    print(f"\n저장 경로: {save_path}")


if __name__ == "__main__":
    main()