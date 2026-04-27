"""Download CIFAR-10 and CIFAR-10H assets used by the project."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torchvision import datasets

from config import (
    CIFAR10_DOWNLOAD_SUBDIR,
    CIFAR10H_COUNTS_FILENAME,
    CIFAR10H_COUNTS_URL,
    CIFAR10H_DOWNLOAD_SUBDIR,
    RAW_DATA_DIR,
    ensure_project_dirs,
    seed_everything,
)


def download_cifar10(data_root: Path) -> Path:
    """Download both CIFAR-10 train and test splits.

    Args:
        data_root: Root directory where torchvision should place the CIFAR-10 files.

    Returns:
        Path to the CIFAR-10 download directory.
    """

    cifar10_dir = data_root / CIFAR10_DOWNLOAD_SUBDIR
    cifar10_dir.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR10(root=str(cifar10_dir), train=True, download=True)
    datasets.CIFAR10(root=str(cifar10_dir), train=False, download=True)
    return cifar10_dir


def download_cifar10h_counts(data_root: Path, force: bool = False) -> Path:
    """Download the CIFAR-10H annotator count matrix.

    Args:
        data_root: Root directory where the CIFAR-10H file should be stored.
        force: If True, re-download even when the file already exists.

    Returns:
        Path to the downloaded `.npy` file.
    """

    cifar10h_dir = data_root / CIFAR10H_DOWNLOAD_SUBDIR
    cifar10h_dir.mkdir(parents=True, exist_ok=True)
    destination = cifar10h_dir / CIFAR10H_COUNTS_FILENAME

    if destination.exists() and not force:
        return destination

    urllib.request.urlretrieve(CIFAR10H_COUNTS_URL, destination)
    return destination


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the download script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the CIFAR-10H counts file even if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for downloading all project datasets."""

    args = parse_args()
    seed_everything()
    ensure_project_dirs()

    cifar10_dir = download_cifar10(RAW_DATA_DIR)
    counts_path = download_cifar10h_counts(RAW_DATA_DIR, force=args.force)

    print("Downloaded data assets:")
    print(f"  CIFAR-10 directory : {cifar10_dir}")
    print(f"  CIFAR-10H counts   : {counts_path}")


if __name__ == "__main__":
    main()
