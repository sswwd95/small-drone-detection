from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


# 설정
ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
YOLO_ROOT = ROOT / "data" / "yolo"
SPLITS = ("train", "val", "test")


def convert_box(width, height, xmin, ymin, xmax, ymax):
    """Pascal VOC bbox를 YOLO 좌표로 변환."""
    box_w = xmax - xmin
    box_h = ymax - ymin

    x_center = (xmin + box_w / 2) / width
    y_center = (ymin + box_h / 2) / height

    return (
        x_center,
        y_center,
        box_w / width,
        box_h / height,
    )


def convert_xml(xml_path, save_path):
    """XML 라벨을 YOLO TXT로 변환."""
    root = ET.parse(xml_path).getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    labels = []

    for obj in root.findall("object"):
        if obj.find("name").text != "UAV":
            continue

        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        x, y, w, h = convert_box(
            width,
            height,
            xmin,
            ymin,
            xmax,
            ymax,
        )

        labels.append(
            f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        )

    save_path.write_text(
        "\n".join(labels),
        encoding="utf-8",
    )


def prepare_split(split):
    """split별 이미지 복사 및 YOLO 라벨 생성."""
    split_path = RAW_ROOT / split

    # train/train 형태 대응
    if (split_path / split).exists():
        split_path /= split

    images = {
        p.stem: p
        for p in split_path.rglob("*.jpg")
    }

    xmls = {
        p.stem: p
        for p in split_path.rglob("*.xml")
    }

    matched = sorted(
        set(images) & set(xmls)
    )

    image_dir = YOLO_ROOT / "images" / split
    label_dir = YOLO_ROOT / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for name in matched:
        shutil.copy2(
            images[name],
            image_dir / images[name].name,
        )

        convert_xml(
            xmls[name],
            label_dir / f"{name}.txt",
        )

    print(f"\n[{split}]")
    print("원본 이미지 수 :", len(images))
    print("원본 XML 수    :", len(xmls))
    print("변환 완료 수   :", len(matched))

    if len(images) != len(xmls):
        print("주의: 이미지 수와 XML 수 불일치")

    return len(matched)


def make_data_yaml():
    """Ultralytics 학습용 data.yaml 생성."""
    dataset_path = YOLO_ROOT.resolve().as_posix()

    yaml_text = (
        f"path: {dataset_path}\n\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: UAV\n"
    )

    (YOLO_ROOT / "data.yaml").write_text(
        yaml_text,
        encoding="utf-8",
    )


def main():
    """DUT Anti-UAV 데이터셋 YOLO 변환."""
    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"원본 데이터 경로 없음: {RAW_ROOT}\n"
            "먼저 python src/01_download_dataset.py 실행 필요"
        )

    print("DUT Anti-UAV -> YOLO 변환 시작")

    total = sum(
        prepare_split(split)
        for split in SPLITS
    )

    make_data_yaml()

    print("\n변환 완료")
    print("전체 변환 이미지 수:", total)
    print("저장 경로:", YOLO_ROOT)


if __name__ == "__main__":
    main()