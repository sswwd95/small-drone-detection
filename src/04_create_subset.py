from collections import defaultdict
from pathlib import Path
import random
import shutil

import numpy as np


# --------------------------------------------------
# 설정
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "yolo"
SUBSET_ROOT = ROOT / "data" / "yolo_subset"

TARGET_SIZES = {
    "train": 1500,
    "val": 500,
    "test": 500,
}

SEED = 42
AREA_BINS = 10
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def read_areas(label_path):
    """YOLO 라벨의 bbox 면적 비율 목록."""
    areas = []

    if not label_path.exists():
        return areas

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            w, h = float(parts[3]), float(parts[4])
            areas.append(w * h)

    return areas


def load_split(split):
    """이미지와 라벨 정보 로드."""
    image_dir = SOURCE_ROOT / "images" / split
    label_dir = SOURCE_ROOT / "labels" / split

    images = {
        p.stem: p
        for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    }

    records = []

    for stem, image_path in sorted(images.items()):
        label_path = label_dir / f"{stem}.txt"
        areas = read_areas(label_path)

        records.append({
            "stem": stem,
            "image": image_path,
            "label": label_path,
            "areas": areas,
            "median_area": float(np.median(areas)) if areas else 0.0,
        })

    return records


def add_strata(records):
    """bbox 크기 분포 기준 계층 생성."""
    positive = [r["median_area"] for r in records if r["areas"]]

    if not positive:
        for r in records:
            r["stratum"] = "empty"
        return

    edges = np.unique(
        np.quantile(positive, np.linspace(0, 1, AREA_BINS + 1))
    )

    for r in records:
        if not r["areas"]:
            r["stratum"] = "empty"
        else:
            idx = np.searchsorted(
                edges[1:-1],
                r["median_area"],
                side="right",
            )
            r["stratum"] = f"area_{idx:02d}"


def stratified_sample(records, target_size, seed):
    """원본 bbox 크기 비율을 유지하는 계층 추출."""
    if target_size >= len(records):
        return records

    groups = defaultdict(list)

    for r in records:
        groups[r["stratum"]].append(r)

    quotas = {
        key: len(items) * target_size / len(records)
        for key, items in groups.items()
    }

    counts = {
        key: min(len(groups[key]), int(quotas[key]))
        for key in groups
    }

    remaining = target_size - sum(counts.values())

    order = sorted(
        groups,
        key=lambda key: quotas[key] - counts[key],
        reverse=True,
    )

    for key in order:
        if remaining == 0:
            break
        if counts[key] < len(groups[key]):
            counts[key] += 1
            remaining -= 1

    rng = random.Random(seed)
    selected = []

    for key in sorted(groups):
        selected.extend(rng.sample(groups[key], counts[key]))

    return sorted(selected, key=lambda r: r["stem"])


def copy_split(split, selected):
    """선택 데이터 복사."""
    image_out = SUBSET_ROOT / "images" / split
    label_out = SUBSET_ROOT / "labels" / split

    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for r in selected:
        shutil.copy2(r["image"], image_out / r["image"].name)

        if r["label"].exists():
            shutil.copy2(r["label"], label_out / r["label"].name)
        else:
            (label_out / f"{r['stem']}.txt").touch()

    list_path = SUBSET_ROOT / f"{split}_files.txt"
    list_path.write_text(
        "\n".join(r["image"].name for r in selected),
        encoding="utf-8",
    )


def stats(records):
    """분포 요약."""
    areas = [area for r in records for area in r["areas"]]

    if not areas:
        return len(records), 0, 0.0, 0.0, 0.0

    arr = np.array(areas)

    return (
        len(records),
        len(arr),
        float(np.median(arr) * 100),
        float((arr < 0.01).mean() * 100),
        float((arr < 0.05).mean() * 100),
    )


def main():
    """분포 유지 서브셋 생성."""
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            "data/yolo 없음\n"
            "먼저 python src/02_prepare_dataset.py 실행 필요"
        )

    if SUBSET_ROOT.exists():
        shutil.rmtree(SUBSET_ROOT)

    summary = []

    for i, (split, target_size) in enumerate(TARGET_SIZES.items()):
        records = load_split(split)
        add_strata(records)

        selected = stratified_sample(
            records,
            target_size,
            SEED + i,
        )

        copy_split(split, selected)

        original_stats = stats(records)
        subset_stats = stats(selected)

        summary.append(
            f"[{split}]\n"
            f"원본   images={original_stats[0]}, objects={original_stats[1]}, "
            f"bbox 중앙값={original_stats[2]:.4f}%, "
            f"<1%={original_stats[3]:.2f}%, <5%={original_stats[4]:.2f}%\n"
            f"subset images={subset_stats[0]}, objects={subset_stats[1]}, "
            f"bbox 중앙값={subset_stats[2]:.4f}%, "
            f"<1%={subset_stats[3]:.2f}%, <5%={subset_stats[4]:.2f}%"
        )

    yaml_text = """path: data/yolo_subset

train: images/train
val: images/val
test: images/test

names:
  0: UAV
"""
    (SUBSET_ROOT / "data.yaml").write_text(
        yaml_text,
        encoding="utf-8",
    )

    summary_text = "\n\n".join(summary)
    (SUBSET_ROOT / "subset_summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    print(summary_text)
    print(f"\n저장 경로: {SUBSET_ROOT}")


if __name__ == "__main__":
    main()
