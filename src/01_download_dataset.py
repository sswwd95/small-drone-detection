"""
DUT Anti-UAV Detection 데이터셋 다운로드 및 압축 해제

생성 구조:

small-drone-detection/
├── data/
│   └── raw/
│       ├── _archives/
│       │   ├── train.zip
│       │   ├── val.zip
│       │   └── test.zip
│       ├── train/
│       ├── val/
│       └── test/
│
└── src/
    └── 01_download_dataset.py
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


# ============================================================
# 1. 데이터셋 Google Drive ID
# ============================================================

DATASETS = {
    "train": "1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj",
    "val": "1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n",
    "test": "1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0",
}


# ============================================================
# 2. 프로젝트 경로
# ============================================================

# 현재 파일:
# small-drone-detection/src/01_download_dataset.py
#
# parents[1]:
# small-drone-detection/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARCHIVE_DIR = RAW_DIR / "_archives"


# ============================================================
# 3. gdown 설치 확인
# ============================================================

def ensure_gdown() -> None:
    """gdown 설치 확인."""

    if importlib.util.find_spec("gdown") is None:
        print("[설치] gdown")

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "gdown",
            ]
        )


# ============================================================
# 4. ZIP 파일 정상 여부 확인
# ============================================================

def is_valid_zip(path: Path) -> bool:
    """ZIP 파일 무결성 확인."""

    if not path.exists():
        return False

    if path.stat().st_size == 0:
        return False

    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None

    except zipfile.BadZipFile:
        return False


# ============================================================
# 5. Google Drive 다운로드
# ============================================================

def download_file(file_id: str, output_path: Path) -> None:
    """Google Drive 파일 다운로드."""

    import gdown

    # 정상 ZIP 파일이 이미 존재하는 경우 재다운로드 방지
    if is_valid_zip(output_path):
        print(f"[건너뜀] 기존 압축 파일: {output_path.name}")
        return

    # 이전 다운로드 실패 파일 제거
    if output_path.exists():
        print(f"[삭제] 손상된 파일: {output_path.name}")
        output_path.unlink()

    url = f"https://drive.google.com/uc?id={file_id}"

    print()
    print(f"[다운로드 시작] {output_path.name}")

    try:
        result = gdown.download(
            url=url,
            output=str(output_path),
            quiet=False,
            resume=True,
        )

    except TypeError:
        # 구버전 gdown 대응
        result = gdown.download(
            url,
            str(output_path),
            quiet=False,
        )

    if result is None:
        raise RuntimeError(
            f"다운로드 실패: {output_path.name}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"파일 생성 실패: {output_path.name}"
        )

    if not is_valid_zip(output_path):
        raise RuntimeError(
            f"정상적인 ZIP 파일이 아님: {output_path.name}"
        )

    print(f"[다운로드 완료] {output_path.name}")


# ============================================================
# 6. 중첩 폴더 제거
# ============================================================

def flatten_single_root(directory: Path) -> None:
    """
    train/train/... 형태의 단일 중첩 폴더 제거.
    """

    items = [
        item
        for item in directory.iterdir()
        if item.name not in {"__MACOSX", ".download_complete"}
    ]

    if len(items) != 1:
        return

    inner = items[0]

    if not inner.is_dir():
        return

    temp_dir = (
        directory.parent
        / f".{directory.name}_flatten_temp"
    )

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    inner.rename(temp_dir)

    shutil.rmtree(directory)

    temp_dir.rename(directory)


# ============================================================
# 7. ZIP 압축 해제
# ============================================================

def extract_zip(
    zip_path: Path,
    target_dir: Path,
) -> None:
    """ZIP 압축 해제."""

    marker = target_dir / ".download_complete"

    # 이미 정상적으로 압축 해제된 경우
    if marker.exists():
        print(
            f"[건너뜀] 이미 압축 해제됨: "
            f"{target_dir.relative_to(PROJECT_ROOT)}"
        )
        return

    # 이전 실패 폴더 제거
    if target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[압축 해제] "
        f"{zip_path.name} -> "
        f"{target_dir.relative_to(PROJECT_ROOT)}"
    )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

    except zipfile.BadZipFile as exc:
        shutil.rmtree(
            target_dir,
            ignore_errors=True,
        )

        raise RuntimeError(
            f"ZIP 압축 해제 실패: {zip_path}"
        ) from exc

    # train/train 형태 중첩 구조 제거
    flatten_single_root(target_dir)

    # 완료 표시
    marker = target_dir / ".download_complete"
    marker.touch()

    print(
        f"[압축 해제 완료] "
        f"{target_dir.relative_to(PROJECT_ROOT)}"
    )


# ============================================================
# 8. 데이터 파일 개수 확인
# ============================================================

def print_dataset_info(split: str) -> None:
    """split별 이미지 및 XML 개수 출력."""

    split_dir = RAW_DIR / split

    images = (
        list(split_dir.rglob("*.jpg"))
        + list(split_dir.rglob("*.jpeg"))
        + list(split_dir.rglob("*.png"))
    )

    xmls = list(split_dir.rglob("*.xml"))

    print(
        f"{split:5} | "
        f"images: {len(images):6} | "
        f"xml: {len(xmls):6}"
    )


# ============================================================
# 9. 실행
# ============================================================

def main() -> None:
    """데이터셋 다운로드 실행."""

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ensure_gdown()

    print("=" * 70)
    print("DUT Anti-UAV Detection Dataset")
    print("=" * 70)

    print(f"프로젝트 : {PROJECT_ROOT}")
    print(f"데이터   : {RAW_DIR}")

    # --------------------------------------------------------
    # train / val / test 다운로드
    # --------------------------------------------------------

    for split, file_id in DATASETS.items():

        print()
        print("-" * 70)
        print(f"[{split.upper()}]")
        print("-" * 70)

        zip_path = (
            ARCHIVE_DIR
            / f"{split}.zip"
        )

        target_dir = (
            RAW_DIR
            / split
        )

        download_file(
            file_id=file_id,
            output_path=zip_path,
        )

        extract_zip(
            zip_path=zip_path,
            target_dir=target_dir,
        )

    # --------------------------------------------------------
    # 결과 확인
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("데이터셋 준비 완료")
    print("=" * 70)

    for split in DATASETS:
        print_dataset_info(split)

    print()
    print("[저장 구조]")
    print(f"train : {RAW_DIR / 'train'}")
    print(f"val   : {RAW_DIR / 'val'}")
    print(f"test  : {RAW_DIR / 'test'}")
    print(f"zip   : {ARCHIVE_DIR}")


if __name__ == "__main__":
    main()