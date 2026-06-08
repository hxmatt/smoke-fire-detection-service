"""Dataset discovery and YOLO-label parsing for the analysis notebook.

Analysis only; not imported by the FastAPI service. These functions just walk
the dataset on disk and return tidy tables. The derived features (box area,
distance from center, class names) are added later, in the notebook.
"""

from pathlib import Path

import pandas as pd

BOX_COLUMNS = [
    "split",
    "image",
    "class_id",
    "x_center",
    "y_center",
    "width",
    "height",
]


def discover_splits(root: Path) -> dict[str, dict[str, Path | None]]:
    """Find the dataset splits under ``root``.

    Every ``labels`` folder becomes a split named after its parent, paired with
    the sibling ``images`` folder when there is one. Copes with the slightly
    different layouts Kaggle YOLO exports ship in.

    Args:
        root: Dataset root to search recursively.

    Returns:
        ``{split_name: {"labels": Path, "images": Path | None}}``.
    """
    splits: dict[str, dict[str, Path | None]] = {}
    for labels_dir in sorted(root.rglob("labels")):
        if not labels_dir.is_dir():
            continue
        images_dir = labels_dir.parent / "images"
        split = labels_dir.parent.name or "dataset"
        splits[split] = {
            "labels": labels_dir,
            "images": images_dir if images_dir.is_dir() else None,
        }
    return splits


def parse_yolo_labels(
    splits: dict[str, dict[str, Path | None]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse YOLO label files into a per-box table and a per-image inventory.

    Background images (empty label files) are kept with ``n_objects == 0``;
    zero-area boxes are skipped and don't count.

    Args:
        splits: Mapping produced by :func:`discover_splits`.

    Returns:
        ``(boxes, images)``: per-box rows (columns :data:`BOX_COLUMNS`) and
        per-image rows (``split``, ``image``, ``n_objects``).
    """
    box_rows: list[tuple] = []
    img_rows: list[tuple] = []
    for split, dirs in splits.items():
        for txt in dirs["labels"].glob("*.txt"):
            n_objects = 0
            for line in txt.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid, xc, yc, w, h = parts
                w, h = float(w), float(h)
                if w <= 0 or h <= 0:
                    continue
                box_rows.append(
                    (split, txt.stem, int(float(cid)), float(xc), float(yc), w, h)
                )
                n_objects += 1
            img_rows.append((split, txt.stem, n_objects))

    boxes = pd.DataFrame(box_rows, columns=BOX_COLUMNS)
    images = pd.DataFrame(img_rows, columns=["split", "image", "n_objects"])
    return boxes, images
