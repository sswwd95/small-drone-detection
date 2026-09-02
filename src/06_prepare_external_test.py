from pathlib import Path
import random
import shutil
import zipfile

from huggingface_hub import hf_hub_download


# 설정
ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "lgrzybowski/seraphim-drone-detection-dataset"

DOWNLOAD_DIR = ROOT / "data" / "_seraphim"
TEST_ROOT = ROOT / "data" / "external_test"

TEST_SIZE = 300
SEED = 42


def download_and_extract():
    """Seraphim Test 이미지와 라벨 다운로드 및 압축 해제."""

    files = [
        "test/images/batch_001.zip",
        "test/labels/batch_001.zip",
    ]

    for filename in files:
        zip_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=DOWNLOAD_DIR,
            )
        )

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(zip_path.parent)

        print(f"압축 해제: {filename}")


def main():
    """외부 Test 300장 생성."""

    download_and_extract()

    image_dir = DOWNLOAD_DIR / "test" / "images"
    label_dir = DOWNLOAD_DIR / "test" / "labels"

    images = {
        p.stem: p
        for p in image_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    labels = {
        p.stem: p
        for p in label_dir.rglob("*.txt")
    }

    # 이미지와 라벨이 모두 존재하는 데이터
    matched = sorted(set(images) & set(labels))

    if not matched:
        raise FileNotFoundError("매칭된 이미지/라벨 없음")

    # 항상 같은 300장 선택
    rng = random.Random(SEED)
    selected = rng.sample(
        matched,
        min(TEST_SIZE, len(matched)),
    )

    # 기존 external_test 초기화
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)

    save_images = TEST_ROOT / "images" / "test"
    save_labels = TEST_ROOT / "labels" / "test"

    save_images.mkdir(parents=True)
    save_labels.mkdir(parents=True)

    # 이미지와 라벨 복사
    for name in selected:
        shutil.copy2(
            images[name],
            save_images / images[name].name,
        )

        shutil.copy2(
            labels[name],
            save_labels / labels[name].name,
        )

    # YOLO용 data.yaml
    dataset_path = TEST_ROOT.resolve().as_posix()

    yaml_text = (
        f"path: {dataset_path}\n\n"
        "train: images/test\n"
        "val: images/test\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: UAV\n"
    )

    (TEST_ROOT / "data.yaml").write_text(
        yaml_text,
        encoding="utf-8",
    )

    # 출처 기록
    source_text = (
        "Dataset: Seraphim Drone Detection Dataset\n"
        f"Source: https://huggingface.co/datasets/{REPO_ID}\n"
        "License: CC BY 4.0\n"
        "Original split: test\n"
        f"Selected images: {len(selected)} / {len(matched)}\n"
        f"Seed: {SEED}\n"
    )

    (TEST_ROOT / "SOURCE.txt").write_text(
        source_text,
        encoding="utf-8",
    )

    print("\nExternal Test 생성 완료")
    print("=" * 50)
    print(f"원본 Test       : {len(matched)}")
    print(f"사용 Test       : {len(selected)}")
    print(f"Seed            : {SEED}")
    print(f"저장 경로       : {TEST_ROOT}")
    print(f"data.yaml       : {TEST_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()