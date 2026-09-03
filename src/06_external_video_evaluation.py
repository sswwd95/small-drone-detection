from pathlib import Path
import json

import cv2
from kaggle import api
from ultralytics import YOLO


# https://github.com/ZhaoJ9014/Anti-UAV

# 경로 및 설정
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external"
MODEL_DIR = ROOT / "models" / "baseline"
MODEL_PATH = MODEL_DIR / "weights" / "best.pt"
OUTPUT_DIR = MODEL_DIR / "external_test"

DATASET = "mm1991/anti-uav-dataset-300"

IMG_SIZE = 960
CONF = 0.25
MATCH_IOU = 0.5


# 컬러 영상과 GT 파일 찾기
def find_data():
    response = api.dataset_list_files(DATASET)
    files = [file.name.replace("\\", "/") for file in response.dataset_files]

    for video in files:
        if not video.endswith("visible.mp4"):
            continue

        gt_file = video.replace("visible.mp4", "visible.json")

        if gt_file in files:
            print(f"선택 영상: {video}")
            return video, gt_file

    raise FileNotFoundError("visible.mp4와 visible.json을 찾지 못했습니다.")


# 영상 1개 + GT 1개만 다운로드
def download_data():
    video_file, gt_file = find_data()

    sequence = Path(video_file).parent.name
    save_dir = DATA_DIR / sequence
    save_dir.mkdir(parents=True, exist_ok=True)

    video_path = save_dir / "visible.mp4"
    gt_path = save_dir / "visible.json"

    if not video_path.exists():
        print("영상 다운로드")
        api.dataset_download_file(DATASET, video_file, str(save_dir))

    if not gt_path.exists():
        print("GT 다운로드")
        api.dataset_download_file(DATASET, gt_file, str(save_dir))

    # 실제 다운로드 위치 확인
    if not video_path.exists():
        found = list(save_dir.rglob("visible.mp4"))
        if found:
            video_path = found[0]

    if not gt_path.exists():
        found = list(save_dir.rglob("visible.json"))
        if found:
            gt_path = found[0]

    if not video_path.exists() or not gt_path.exists():
        raise FileNotFoundError("영상 또는 GT 다운로드 실패")

    print(f"영상: {video_path}")
    print(f"GT  : {gt_path}")

    return sequence, video_path, gt_path


# IoU 계산
def calc_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    overlap = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - overlap

    return overlap / union if union > 0 else 0


# 외부 영상 추론 및 평가
def evaluate_video():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 없음: {MODEL_PATH}")

    sequence, video_path, gt_path = download_data()

    labels = json.loads(gt_path.read_text(encoding="utf-8-sig"))
    gt_list = labels["gt_rect"]
    exist_list = labels["exist"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_video = OUTPUT_DIR / f"{sequence}_result.mp4"
    output_report = OUTPUT_DIR / f"{sequence}_metrics.txt"

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w == 0 or h == 0:
        raise RuntimeError("영상 파일을 열지 못했습니다.")

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    tp, fp, fn = 0, 0, 0
    iou_sum, time_sum, frame_count = 0, 0, 0

    while frame_count < len(gt_list):
        ok, frame = cap.read()

        if not ok:
            break

        # GT 변환
        gt = gt_list[frame_count]
        gt_box = None

        if exist_list[frame_count] and len(gt) == 4:
            x, y, bw, bh = gt
            gt_box = [x, y, x+bw, y+bh]

        # 모델 추론
        result = model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF,
            verbose=False
        )[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        time_sum += result.speed["inference"]

        # GT 표시
        if gt_box:
            x1, y1, x2, y2 = map(int, gt_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 예측 표시 및 최고 IoU 확인
        best_iou = 0

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (x1, max(20, y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

            if gt_box:
                score = calc_iou(gt_box, box)
                if score > best_iou:
                    best_iou = score

        # TP / FP / FN 계산
        if gt_box and best_iou >= MATCH_IOU:
            tp += 1
            fp += max(0, len(boxes)-1)
            iou_sum += best_iou

        elif gt_box:
            fn += 1
            fp += len(boxes)

        else:
            fp += len(boxes)

        cv2.putText(
            frame,
            f"Frame {frame_count+1} | IoU: {best_iou:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    # 성능 계산
    precision = tp / (tp+fp) if tp+fp else 0
    recall = tp / (tp+fn) if tp+fn else 0
    f1 = 2*precision*recall / (precision+recall) if precision+recall else 0
    avg_iou = iou_sum / tp if tp else 0
    avg_time = time_sum / frame_count if frame_count else 0
    infer_fps = 1000 / avg_time if avg_time else 0

    # 결과 저장
    report = f"""Anti-UAV300 외부 평가
========================================
영상             : {sequence}
전체 프레임       : {frame_count}
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

    output_report.write_text(report, encoding="utf-8")

    print(report)
    print(f"영상 저장: {output_video}")
    print(f"결과 저장: {output_report}")


if __name__ == "__main__":
    evaluate_video()