"""YOLO-to-torchvision dataset adapter for Faster R-CNN (analysis only).

Converts YOLO-format labels (normalized ``cx cy w h``) into torchvisions
pixel ``xyxy`` targets, shifting class ids so 0 stays free for background.
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

from src.models.fasterrcnn import LABEL_OFFSET

SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class YoloDataset(Dataset):
    """One YOLO split as torchvision (image, target) pairs."""

    def __init__(
        self,
        root: Path,
        train: bool,
        imgsz: int,
        fraction: float = 1.0,
        seed: int = 0,
    ) -> None:
        """Index the (image, label) pairs in a split and build the transform.

        Args:
            root: Split folder holding sibling ``images`` and ``labels`` dirs.
            train: Whether this is the training split (enables augmentation).
            imgsz: Target size for the shorter image side.
            fraction: Fraction of the split to keep, sampled at random.
            seed: Seed for the fractional sample.
        """
        images, labels = root / "images", root / "labels"
        pairs = [
            (img, lbl)
            for lbl in sorted(labels.glob("*.txt"))
            if (img := self._image_for(lbl, images))
        ]
        if fraction < 1.0:
            # random sample, not a head slice: ~46% of images are background-only
            # and a leading slice can land entirely on them, leaving mAP undefined
            k = max(1, int(len(pairs) * fraction))
            pairs = random.Random(seed).sample(pairs, k)
        self.pairs = pairs

        self.tf = T.Compose(
            [
                T.ToImage(),
                T.Resize(imgsz, max_size=imgsz + 64),
                T.ToDtype(torch.float32, scale=True),
                *([T.RandomHorizontalFlip(0.5)] if train else []),
            ]
        )

    @staticmethod
    def _image_for(label: Path, images_dir: Path) -> Path | None:
        """Find the image whose stem matches the label, trying each suffix."""
        return next(
            (p for s in SUFFIXES if (p := images_dir / f"{label.stem}{s}").exists()),
            None,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        """Load one sample as a transformed ``(image, target)`` pair.

        Args:
            i: Index into the split.

        Returns:
            ``(image, target)`` where target holds ``boxes`` and ``labels``,
            with zero-area boxes dropped after the resize.
        """
        image_path, label_path = self.pairs[i]
        image = Image.open(image_path).convert("RGB")
        w, h = image.width, image.height
        boxes, labels = self._parse(label_path, w, h)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w)),
            "labels": labels,
        }
        image, target = self.tf(image, target)

        # drop boxes that collapsed to zero area under resize
        b = target["boxes"]
        keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
        target["boxes"] = target["boxes"][keep]
        target["labels"] = target["labels"][keep]

        return image, target

    @staticmethod
    def _parse(path: Path, w: int, h: int) -> tuple[torch.Tensor, torch.Tensor]:
        """YOLO cx,cy,w,h rows -> absolute xyxy boxes + background-shifted labels.

        Args:
            path: Label ``.txt`` with one normalized ``cls cx cy w h`` row per box.
            w: Image width in pixels.
            h: Image height in pixels.

        Returns:
            ``(boxes, labels)`` tensors; empty tensors when the file has no boxes.
        """
        boxes, labels = [], []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid, cx, cy, bw, bh = map(float, parts)
            if bw <= 0 or bh <= 0:
                continue
            boxes.append(
                [
                    (cx - bw / 2) * w,
                    (cy - bh / 2) * h,
                    (cx + bw / 2) * w,
                    (cy + bh / 2) * h,
                ]
            )
            labels.append(int(cid) + LABEL_OFFSET)

        if not boxes:
            return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)

        return torch.tensor(boxes), torch.tensor(labels, dtype=torch.int64)


def collate_fn(batch: list) -> tuple:
    # detection needs variable-size targets, so transpose without stacking
    return tuple(zip(*batch, strict=True))


def loader(
    data_root: Path,
    split: str,
    train: bool,
    imgsz: int,
    fraction: float = 1.0,
    seed: int = 0,
    batch_size: int = 4,
    num_workers: int = 4,
) -> DataLoader:
    """Build a DataLoader over a YOLO split under ``data_root/data/<split>``.

    Args:
        data_root: Dataset root holding the ``data`` directory.
        split: Split name (``train``, ``val``, or ``test``).
        train: Whether to shuffle and augment (training split).
        imgsz: Target size for the shorter image side.
        fraction: Fraction of the split to keep, sampled at random.
        seed: Seed for the fractional sample.
        batch_size: Images per batch.
        num_workers: DataLoader worker processes.

    Returns:
        A ``DataLoader`` yielding ``(images, targets)`` tuples.
    """
    ds = YoloDataset(data_root / "data" / split, train, imgsz, fraction, seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
