from pathlib import Path
import json
import subprocess

import cv2
from ultralytics import YOLO


# 경로 및 설정
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external"
MODEL_DIR = ROOT / "models" / "improved_v2" # baseline, improved_v1, improved_v2
MODEL_PATH = MODEL_DIR / "weights" / "best.pt"
OUTPUT_DIR = MODEL_DIR / "external_test"

DATASET = "mm1991/anti-uav-dataset-300"

IMG_SIZE = 960
CONF = 0.25
MATCH_IOU = 0.5


# 없는 GT만 다운로드
def prepare_gt(folder):
    gt_path = folder / "visible.json"

    if gt_path.exists():
        return gt_path

    for split in ["test", "train"]:
        file_name = f"{split}/{folder.name}/visible.json"

        result = subprocess.run([
            "kaggle", "datasets", "download", DATASET,
            "-f", file_name,
            "-p", str(folder),
            "--unzip"
        ], capture_output=True, text=True)

        found = list(folder.rglob("visible.json"))
        if result.returncode == 0 and found:
            if found[0] != gt_path:
                found[0].replace(gt_path)

            print(f"GT 다운로드 완료: {folder.name}")
            return gt_path

    raise FileNotFoundError(f"GT 없음: {folder.name}")


# IoU 계산
def calc_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

    overlap = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - overlap

    return overlap / union if union else 0


# 영상 추론 및 평가
def evaluate_video(model, folder):
    sequence = folder.name
    video_path = folder / "visible.mp4"
    gt_path = prepare_gt(folder)

    labels = json.loads(gt_path.read_text(encoding="utf-8-sig"))
    gt_list = labels["gt_rect"]
    exist_list = labels["exist"]

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_out = OUTPUT_DIR / f"{sequence}_result.mp4"
    txt_out = OUTPUT_DIR / f"{sequence}_metrics.txt"

    writer = cv2.VideoWriter(
        str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    tp, fp, fn = 0, 0, 0
    iou_sum, time_sum, count = 0, 0, 0

    for gt, exists in zip(gt_list, exist_list):
        ok, frame = cap.read()
        if not ok:
            break

        gt_box = None
        if exists and len(gt) == 4:
            x, y, bw, bh = gt
            gt_box = [x, y, x+bw, y+bh]

        result = model.predict(
            frame, imgsz=IMG_SIZE, conf=CONF, verbose=False
        )[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        time_sum += result.speed["inference"]

        # GT 표시
        if gt_box:
            x1, y1, x2, y2 = map(int, gt_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 예측 표시
        best_iou = 0

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, max(20, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if gt_box:
                best_iou = max(best_iou, calc_iou(gt_box, box))

        # TP / FP / FN
        if gt_box and best_iou >= MATCH_IOU:
            tp += 1
            fp += max(0, len(boxes)-1)
            iou_sum += best_iou
        elif gt_box:
            fn += 1
            fp += len(boxes)
        else:
            fp += len(boxes)

        count += 1
        cv2.putText(frame, f"Frame {count} | IoU: {best_iou:.2f}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()

    # 성능 계산
    precision = tp / (tp+fp) if tp+fp else 0
    recall = tp / (tp+fn) if tp+fn else 0
    f1 = 2*precision*recall / (precision+recall) if precision+recall else 0
    avg_iou = iou_sum / tp if tp else 0
    avg_time = time_sum / count if count else 0
    infer_fps = 1000 / avg_time if avg_time else 0

    report = f"""Anti-UAV300 외부 평가
========================================
영상             : {sequence}
전체 프레임       : {count}
Input Size       : {IMG_SIZE}
Confidence       : {CONF}
Match IoU        : {MATCH_IOU}

TP               : {tp}
FP               : {fp}
FN               : {fn}
Precision        : {precision:.4f}
Recall           : {recall:.4f}
F1               : {f1:.4f}
평균 TP IoU       : {avg_iou:.4f}
평균 추론시간      : {avg_time:.2f} ms
추론 FPS          : {infer_fps:.2f}
"""

    txt_out.write_text(report, encoding="utf-8")
    print(report)


# 저장된 영상 전체 평가
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)

    folders = [
        folder for folder in DATA_DIR.iterdir()
        if folder.is_dir() and (folder / "visible.mp4").exists()
    ]

    print(f"평가 영상: {len(folders)}개")

    for folder in sorted(folders):
        print(f"\n평가 시작: {folder.name}")
        evaluate_video(model, folder)


if __name__ == "__main__":
    main()