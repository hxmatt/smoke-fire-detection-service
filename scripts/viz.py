"""Plotting helpers for the dataset-characterization notebook (analysis only).

Callers pass the image directory and the rows to draw, so these don't reach
into notebook globals.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_image(images_dir: Path | None, stem: str) -> Path | None:
    """Find the image for a label-file ``stem``, trying common extensions.

    Args:
        images_dir: Directory holding the split's images, or ``None``.
        stem: Image file stem (the label file's name without extension).

    Returns:
        The matching image path, or ``None`` if the directory or file is absent.
    """
    if images_dir is None:
        return None
    for ext in _IMAGE_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def draw_sample(
    ax: Axes,
    img_path: Path | None,
    boxes: pd.DataFrame,
    *,
    class_colors: dict[int, str],
    title: str | None = None,
) -> None:
    """Draw an image with its YOLO boxes on ``ax``, colored by class.

    Args:
        ax: Target matplotlib axes.
        img_path: Path to the image, or ``None`` if unavailable.
        boxes: Normalized YOLO rows for this image (``x_center``, ``y_center``,
            ``width``, ``height``, ``class_id``).
        class_colors: Edge color per class id.
        title: Optional axes title.
    """
    if img_path is None:
        ax.set_title("(image unavailable)", fontsize=8)
        ax.axis("off")
        return

    img = plt.imread(img_path)
    height, width = img.shape[0], img.shape[1]
    ax.imshow(img)
    for _, b in boxes.iterrows():
        x = (b["x_center"] - b["width"] / 2) * width
        y = (b["y_center"] - b["height"] / 2) * height
        ax.add_patch(
            Rectangle(
                (x, y),
                b["width"] * width,
                b["height"] * height,
                fill=False,
                edgecolor=class_colors.get(b["class_id"], "white"),
                lw=1.5,
            )
        )
    ax.set_title(title or "", fontsize=9)
    ax.axis("off")
