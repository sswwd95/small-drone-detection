from pathlib import Path

import pandas as pd
from ultralytics import YOLO


# 설정
ROOT = Path(__file__).resolve().parents[1]

RUN_NAME = "baseline_subset_yolo26s_640_ep30_260902_2050"

MODEL_DIR = ROOT / "models" / RUN_NAME
MODEL_PATH = MODEL_DIR / "weights" / "best.pt"
DATA_YAML = ROOT / "data" / "yolo_subset" / "data.yaml"

IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0


def main():
    """Baseline 모델 Test 성능 평가."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 없음: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
        project=str(MODEL_DIR),
        name="test",
        exist_ok=True,
    )

    # Test 결과 표
    metrics = pd.DataFrame([{
        "Precision": results.box.mp,
        "Recall": results.box.mr,
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
    }])

    save_path = Path(results.save_dir) / "test_metrics.csv"
    metrics.to_csv(save_path, index=False)

    print("\nBaseline Test 결과")
    print("=" * 50)
    print(metrics.round(4).to_string(index=False))
    print(f"\n저장 경로: {save_path}")


if __name__ == "__main__":
    main()