from pathlib import Path
import random
import shutil

import numpy as np


# 설정
ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "yolo"
SUBSET_ROOT = ROOT / "data" / "yolo_subset"

SUBSET_RATIO = 1 / 3
SEED = 42
AREA_BINS = 5
SPLITS = ("train", "val", "test")


def load_records(split):
    """이미지별 bbox 크기와 객체 수 확인."""
    image_dir = SOURCE_ROOT / "images" / split
    label_dir = SOURCE_ROOT / "labels" / split
    records = []

    for image in sorted(image_dir.glob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label = label_dir / f"{image.stem}.txt"
        areas = []

        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()

                # YOLO label: class x y width height
                if len(parts) >= 5:
                    width = float(parts[3])
                    height = float(parts[4])
                    areas.append(width * height)

        records.append({
            "image": image,
            "label": label,
            "areas": areas,
            "count": len(areas),
        })

    return records


def sample_single(records, size, rng):
    """단일 객체 이미지를 bbox 크기별로 골고루 추출."""

    # bbox 크기순 정렬
    records = sorted(records, key=lambda r: r["areas"][0])

    # 작은 bbox부터 큰 bbox까지 5개 구간으로 분할
    groups = np.array_split(records, AREA_BINS)

    # 각 구간에서 비슷한 수량 추출
    base, extra = divmod(size, AREA_BINS)
    selected = []

    for i, group in enumerate(groups):
        take = base

        if i < extra:
            take += 1

        selected.extend(
            rng.sample(list(group), take)
        )

    return selected


def select_subset(records, split, size, rng):
    """split 목적에 맞게 subset 선정."""

    # 객체 수 기준 분류
    empty = [r for r in records if r["count"] == 0]
    single = [r for r in records if r["count"] == 1]
    multi = [r for r in records if r["count"] >= 2]

    if split == "train":
        # 희소한 배경과 다중 객체는 전부 포함
        selected = empty + multi

        # 나머지는 단일 객체에서 추출
        single_size = size - len(selected)
        selected += sample_single(single, single_size, rng)

    else:
        # Val/Test는 원본 객체 수 비율 유지
        empty_size = round(len(empty) / len(records) * size)
        multi_size = round(len(multi) / len(records) * size)
        single_size = size - empty_size - multi_size

        selected = sample_single(single, single_size, rng)
        selected += rng.sample(empty, empty_size)
        selected += rng.sample(multi, multi_size)

    return selected


def copy_subset(split, records):
    """선정 이미지와 라벨 복사."""
    image_dir = SUBSET_ROOT / "images" / split
    label_dir = SUBSET_ROOT / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        image = record["image"]
        label = record["label"]

        shutil.copy2(image, image_dir / image.name)

        if label.exists():
            shutil.copy2(label, label_dir / label.name)
        else:
            # 배경 이미지용 빈 라벨
            (label_dir / f"{image.stem}.txt").touch()


def main():
    """각 공식 split에서 약 1/3 추출."""

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            "YOLO 데이터 없음\n"
            "먼저 python src/02_prepare_dataset.py 실행 필요"
        )

    # 기존 subset 제거
    if SUBSET_ROOT.exists():
        shutil.rmtree(SUBSET_ROOT)

    for i, split in enumerate(SPLITS):
        records = load_records(split)

        # 각 공식 split의 1/3
        size = round(len(records) * SUBSET_RATIO)

        # split마다 고정된 seed 사용
        rng = random.Random(SEED + i)

        selected = select_subset(
            records,
            split,
            size,
            rng,
        )

        # 중복 이미지 확인
        names = [r["image"].name for r in selected]
        assert len(names) == len(set(names))

        copy_subset(split, selected)

        empty = sum(r["count"] == 0 for r in selected)
        multi = sum(r["count"] >= 2 for r in selected)

        print(
            f"{split:5} | {len(records)} -> {len(selected)} | "
            f"empty={empty} | multi={multi}"
        )

    # 현재 PC의 실제 경로로 data.yaml 생성
    dataset_path = SUBSET_ROOT.resolve().as_posix()

    yaml_text = (
        f"path: {dataset_path}\n\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: UAV\n"
    )

    (SUBSET_ROOT / "data.yaml").write_text(
        yaml_text,
        encoding="utf-8",
    )

    print(f"\n완료: {SUBSET_ROOT}")


if __name__ == "__main__":
    main()