from pathlib import Path
import importlib.util
import shutil
import subprocess
import sys
import zipfile


# 설정
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
ARCHIVE_DIR = RAW_DIR / "_archives"

DATASETS = {
    "train": "1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj",
    "val": "1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n",
    "test": "1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0",
}


def ensure_gdown():
    """gdown 설치 확인."""
    if importlib.util.find_spec("gdown") is None:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "gdown",
        ])


def is_valid_zip(path):
    """ZIP 파일 정상 여부 확인."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def download_file(file_id, zip_path):
    """Google Drive ZIP 다운로드."""
    import gdown

    if is_valid_zip(zip_path):
        print(f"[건너뜀] {zip_path.name}")
        return

    if zip_path.exists():
        zip_path.unlink()

    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"[다운로드] {zip_path.name}")

    result = gdown.download(
        url,
        str(zip_path),
        quiet=False,
    )

    if result is None or not is_valid_zip(zip_path):
        raise RuntimeError(f"다운로드 실패: {zip_path.name}")


def flatten_folder(folder):
    """train/train 형태의 중첩 폴더 제거."""
    items = [
        p for p in folder.iterdir()
        if p.name not in {"__MACOSX", ".download_complete"}
    ]

    if len(items) != 1 or not items[0].is_dir():
        return

    inner = items[0]
    temp = folder.parent / f".{folder.name}_temp"

    if temp.exists():
        shutil.rmtree(temp)

    inner.rename(temp)
    shutil.rmtree(folder)
    temp.rename(folder)


def extract_zip(zip_path, target_dir):
    """ZIP 압축 해제."""
    marker = target_dir / ".download_complete"

    if marker.exists():
        print(f"[건너뜀] {target_dir.relative_to(ROOT)}")
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True)

    print(f"[압축 해제] {zip_path.name}")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)

    flatten_folder(target_dir)
    marker.touch()


def print_info(split):
    """split별 이미지와 XML 수 출력."""
    split_dir = RAW_DIR / split

    images = [
        p for p in split_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    xmls = list(split_dir.rglob("*.xml"))

    print(
        f"{split:5} | "
        f"images={len(images):4} | "
        f"xml={len(xmls):4}"
    )


def main():
    """DUT Anti-UAV 데이터셋 다운로드."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    ensure_gdown()

    print("DUT Anti-UAV 다운로드 시작\n")

    for split, file_id in DATASETS.items():
        zip_path = ARCHIVE_DIR / f"{split}.zip"
        target_dir = RAW_DIR / split

        download_file(file_id, zip_path)
        extract_zip(zip_path, target_dir)

    print("\n데이터셋 준비 완료")

    for split in DATASETS:
        print_info(split)

    print(f"\n저장 경로: {RAW_DIR}")


if __name__ == "__main__":
    main()