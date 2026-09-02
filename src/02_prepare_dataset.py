from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


# --------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------

# DUT 원본 데이터 경로
RAW_ROOT = Path("data/raw")

# YOLO 변환 데이터 저장 경로
YOLO_ROOT = Path("data/yolo")

# 데이터 분할 목록
SPLITS = ["train", "val", "test"]


# --------------------------------------------------
# 2. Pascal VOC bbox의 YOLO 좌표 변환
# --------------------------------------------------

def convert_box(image_width, image_height, xmin, ymin, xmax, ymax):
    """Pascal VOC bbox의 YOLO 정규화 좌표 변환"""

    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    x_center = xmin + bbox_width / 2
    y_center = ymin + bbox_height / 2

    # 이미지 크기 기준 0~1 정규화
    x_center /= image_width
    y_center /= image_height
    bbox_width /= image_width
    bbox_height /= image_height

    return x_center, y_center, bbox_width, bbox_height


# --------------------------------------------------
# 3. XML 단일 파일의 YOLO TXT 변환
# --------------------------------------------------

def convert_xml(xml_path, save_path):
    """XML 단일 파일의 YOLO TXT 라벨 변환"""

    root = ET.parse(xml_path).getroot()

    # XML 이미지 크기 정보
    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)

    yolo_labels = []

    # 이미지 내 전체 객체 순회
    for obj in root.findall("object"):

        class_name = obj.find("name").text

        # UAV 클래스 대상
        if class_name != "UAV":
            continue

        box = obj.find("bndbox")

        # Pascal VOC bbox 좌표
        xmin = int(float(box.find("xmin").text))
        ymin = int(float(box.find("ymin").text))
        xmax = int(float(box.find("xmax").text))
        ymax = int(float(box.find("ymax").text))

        # YOLO 정규화 좌표
        x_center, y_center, bbox_width, bbox_height = convert_box(
            image_width,
            image_height,
            xmin,
            ymin,
            xmax,
            ymax
        )

        # UAV 클래스 ID 0
        line = (
            f"0 "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{bbox_width:.6f} "
            f"{bbox_height:.6f}"
        )

        yolo_labels.append(line)

    # YOLO TXT 라벨 저장
    save_path.write_text("\n".join(yolo_labels), encoding="utf-8")


# --------------------------------------------------
# 4. split별 데이터 변환
# --------------------------------------------------

def prepare_split(split):
    """train / val / test 단위 YOLO 데이터 변환"""

    split_path = RAW_ROOT / split

    # train/train 형태 중첩 폴더 대응
    if (split_path / split).exists():
        split_path = split_path / split

    # JPG 이미지 탐색
    images = sorted(split_path.rglob("*.jpg"))

    # XML 라벨 탐색
    xmls = sorted(split_path.rglob("*.xml"))

    # 파일명 기준 이미지 매핑
    image_dict = {p.stem: p for p in images}

    # 파일명 기준 XML 매핑
    xml_dict = {p.stem: p for p in xmls}

    # 이미지와 XML 공통 파일명
    matched_names = sorted(set(image_dict) & set(xml_dict))

    # 이미지 저장 경로
    image_save_dir = YOLO_ROOT / "images" / split

    # 라벨 저장 경로
    label_save_dir = YOLO_ROOT / "labels" / split

    # 저장 폴더 생성
    image_save_dir.mkdir(parents=True, exist_ok=True)
    label_save_dir.mkdir(parents=True, exist_ok=True)

    # 매칭 파일 순회
    for name in matched_names:

        image_path = image_dict[name]
        xml_path = xml_dict[name]

        # 원본 이미지 복사
        shutil.copy2(
            image_path,
            image_save_dir / image_path.name
        )

        # XML 라벨의 YOLO TXT 변환
        convert_xml(
            xml_path,
            label_save_dir / f"{name}.txt"
        )

    # split별 변환 결과
    print(f"\n[{split}]")
    print("원본 이미지 수 :", len(images))
    print("원본 XML 수    :", len(xmls))
    print("변환 완료 수   :", len(matched_names))

    # 이미지와 XML 수 불일치 경고
    if len(images) != len(xmls):
        print("주의: 이미지 수와 XML 수 불일치")

    return len(matched_names)


# --------------------------------------------------
# 5. data.yaml 생성
# --------------------------------------------------

def make_data_yaml():
    """Ultralytics YOLO 학습용 data.yaml 생성"""

    yaml_text = (
        "path: data/yolo\n\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: UAV\n"
    )

    yaml_path = YOLO_ROOT / "data.yaml"

    # data.yaml 저장
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print("\ndata.yaml 경로:", yaml_path)


# --------------------------------------------------
# 6. 메인 실행
# --------------------------------------------------

if __name__ == "__main__":

    print("DUT Anti-UAV -> YOLO 변환 시작")

    total = 0

    # 전체 split 순차 변환
    for split in SPLITS:
        total += prepare_split(split)

    # YOLO 데이터 설정 파일 생성
    make_data_yaml()

    # 전체 변환 결과
    print("\n변환 완료")
    print("전체 변환 이미지 수:", total)
    print("저장 경로:", YOLO_ROOT)
