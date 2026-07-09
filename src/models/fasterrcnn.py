"""Faster R-CNN architecture for smoke/fire detection.

Shared by training and the service so a saved ``state_dict`` loads back into
the identical architecture.
"""

from pathlib import Path
from typing import cast

import torch
from PIL.Image import Image
from torch import nn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.v2 import functional as F

from src.models.base import CLASS_NAMES, Detection, ModelInfo, ModelWrapper

LABEL_OFFSET = 1  # class 0 is background; raw ids 0/1 shift to 1/2
NUM_CLASSES = 3  # smoke, fire, background


def build_model(pretrained: bool = True, freeze_backbone: bool = True):
    """Faster R-CNN (MobileNetV3-L FPN) with a 3-class head.

    Args:
        pretrained: load COCO-pretrained weights (True for fine-tuning;
            False when about to load a fine-tuned ``state_dict``).
        freeze_backbone: freeze the backbone body so only the FPN and heads
            train.
    """
    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    predictor = cast(FastRCNNPredictor, model.roi_heads.box_predictor)
    in_features = predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    if freeze_backbone:
        for p in cast(nn.Module, model.backbone.body).parameters():
            p.requires_grad = False

    return model


class FasterRCNNWrapper(ModelWrapper):
    """Two-stage detector (torchvision Faster R-CNN, MobileNetV3-L FPN)."""

    id = "fasterrcnn"
    name = "Faster R-CNN"
    description = "Two-stage CNN detector, MobileNetV3-L FPN backbone."

    def __init__(self, weights: Path, device: str | None = None):
        super().__init__(weights)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model: nn.Module | None = None

    def load(self) -> None:
        model = build_model(pretrained=False, freeze_backbone=False)
        state = torch.load(self.weights, map_location=self.device)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: Image, conf: float = 0.25) -> list[Detection]:
        assert self.model is not None
        tensor = F.to_dtype(F.to_image(image), torch.float32, scale=True)
        output = self.model([tensor.to(self.device)])[0]

        detections = []
        for box, label, score in zip(
            output["boxes"], output["labels"], output["scores"], strict=True
        ):
            if score < conf:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append(
                Detection(
                    label=CLASS_NAMES[int(label) - LABEL_OFFSET],
                    confidence=float(score),
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                )
            )

        return detections

    def get_info(self) -> ModelInfo:
        assert self.model is not None
        params = sum(p.numel() for p in self.model.parameters())

        return ModelInfo(
            id=self.id,
            name=self.name,
            description=self.description,
            size_mb=self.weights.stat().st_size / 1e6,
            parameters=params,
        )
