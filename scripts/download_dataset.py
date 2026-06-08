"""Download the Smoke-Fire YOLO dataset (analysis only, not used by the service).

Importable from notebooks, or run as a CLI:
``python scripts/download_dataset.py [DEST_DIR] [--force]``. The public Kaggle
endpoint needs no authentication.
"""

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

KAGGLE_SLUG = "sayedgamal99/smoke-fire-detection-yolo"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_SLUG}"
_CHUNK = 1 << 16


def _find_dataset_root(search_dir: Path) -> Path | None:
    """Find the dataset root (the folder holding ``data.yaml``) under ``search_dir``.

    Args:
        search_dir: Directory to search recursively.

    Returns:
        The folder containing ``data.yaml``, or ``None`` if not found.
    """
    matches = sorted(search_dir.rglob("data.yaml"))
    return matches[0].parent if matches else None


def download_dataset(dest_dir: str | Path = "data", *, force: bool = False) -> Path:
    """Download and unzip the dataset, reusing an existing copy when possible.

    Args:
        dest_dir: Directory to download into and extract under.
        force: Re-download even if an extracted copy already exists.

    Returns:
        The dataset root (folder containing ``data.yaml``).
    """
    dest_dir = Path(dest_dir).expanduser().resolve()
    extract_dir = dest_dir / "smoke-fire-detection-yolo"

    if not force and (root := _find_dataset_root(extract_dir)) is not None:
        return root

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "smoke-fire-detection-yolo.zip"

    with requests.get(DOWNLOAD_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            open(zip_path, "wb") as fh,
            tqdm(total=total, unit="iB", unit_scale=True, desc="Downloading") as bar,
        ):
            for chunk in response.iter_content(chunk_size=_CHUNK):
                bar.update(fh.write(chunk))

    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    finally:
        zip_path.unlink(missing_ok=True)

    root = _find_dataset_root(extract_dir)
    if root is None:
        raise RuntimeError(f"data.yaml not found after extracting to {extract_dir}")
    return root


def main() -> None:
    """Run the dataset download from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "dest_dir", nargs="?", default="data", help="download/extract directory"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if already present"
    )
    args = parser.parse_args()
    root = download_dataset(args.dest_dir, force=args.force)
    print(f"Dataset ready at: {root}")


if __name__ == "__main__":
    main()
