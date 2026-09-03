from pathlib import Path
import json
import subprocess

import cv2
from ultralytics import YOLO


# Anti-UAV300
# Source: https://github.com/ZhaoJ9014/Anti-UAV
# Kaggle: https://www.kaggle.com/datasets/mm1991/anti-uav-dataset-300
# Project License: MIT

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external"

MODEL_NAME = "improved_v2"  # baseline, improved_v1, improved_v2
MODEL_DIR = ROOT / "models" / MODEL_NAME
MODEL_PATH = MODEL_DIR / "weights" / "best.pt"
OUTPUT_DIR = MODEL_DIR / "external_test"

DATASET = "mm1991/anti-uav-dataset-300"
SEQUENCES = ["20190925_111757_1_5", "20190925_131530_1_2"]

IMG_SIZE = 960
CONF = 0.25
MATCH_IOU = 0.5


# 평가 영상과 GT 다운로드
def download_data(sequence):
    folder = DATA_DIR / sequence
    folder.mkdir(parents=True, exist_ok=True)

    for name in ["visible.mp4", "visible.json"]:
        save_path = folder / name
        if save_path.exists():
            continue

        for split in ["test", "train"]:
            file = f"{split}/{sequence}/{name}"
            result = subprocess.run(
                ["kaggle", "datasets", "download", DATASET,
                 "-f", file, "-p", str(folder), "--unzip"],
                capture_output=True
            )

            found = list(folder.rglob(name))
            if result.returncode == 0 and found:
                found[0].replace(save_path)
                break

        if not save_path.exists():
            raise FileNotFoundError(f"다운로드 실패: {sequence}/{name}")

    return folder


# IoU 계산
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])

    overlap = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])

    return overlap / (area_a + area_b - overlap) if area_a + area_b else 0


# 영상 평가
def evaluate(model, folder):
    sequence = folder.name
    labels = json.loads(
        (folder / "visible.json").read_text(encoding="utf-8-sig")
    )

    cap = cv2.VideoCapture(str(folder / "visible.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_DIR / f"{sequence}_result.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )

    tp = fp = fn = 0
    iou_sum = time_sum = 0

    for frame_num, (gt, exists) in enumerate(
        zip(labels["gt_rect"], labels["exist"]), 1
    ):
        ok, frame = cap.read()
        if not ok:
            break

        gt_box = None
        if exists and len(gt) == 4:
            x, y, w, h = gt
            gt_box = [x, y, x+w, y+h]

        result = model.predict(
            frame, imgsz=IMG_SIZE, conf=CONF, verbose=False
        )[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        time_sum += result.speed["inference"]

        best_iou = 0

        if gt_box:
            x1, y1, x2, y2 = map(int, gt_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame, f"{conf:.2f}", (x1, max(20, y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )

            if gt_box:
                best_iou = max(best_iou, iou(gt_box, box))

        if gt_box and best_iou >= MATCH_IOU:
            tp += 1
            fp += max(0, len(boxes)-1)
            iou_sum += best_iou
        elif gt_box:
            fn += 1
            fp += len(boxes)
        else:
            fp += len(boxes)

        writer.write(frame)

    cap.release()
    writer.release()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2*precision*recall / (precision+recall) if precision+recall else 0
    avg_iou = iou_sum / tp if tp else 0
    avg_time = time_sum / frame_num if frame_num else 0

    report = f"""영상: {sequence}
TP: {tp}
FP: {fp}
FN: {fn}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1: {f1:.4f}
평균 TP IoU: {avg_iou:.4f}
평균 추론시간: {avg_time:.2f} ms
추론 FPS: {1000 / avg_time:.2f}
"""

    (OUTPUT_DIR / f"{sequence}_metrics.txt").write_text(
        report, encoding="utf-8"
    )
    print(report)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)

    for sequence in SEQUENCES:
        print(f"\n평가: {sequence}")
        evaluate(model, download_data(sequence))


if __name__ == "__main__":
    main()