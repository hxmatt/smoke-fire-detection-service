"""YOLOv11 Nano wrapper (Ultralytics)."""

from pathlib import Path

from PIL.Image import Image
from torch import nn
from ultralytics import YOLO
from ultralytics.engine.model import Model

from src.models.base import CLASS_NAMES, Detection, ModelInfo, ModelWrapper


class YoloWrapper(ModelWrapper):
    """Single-stage CNN detector (Ultralytics YOLOv11 Nano)."""

    id = "yolo11n"
    name = "YOLOv11 Nano"
    description = "One-stage anchor-free CNN detector."

    def __init__(self, weights: Path):
        super().__init__(weights)
        self.model: Model | None = None

    def load(self) -> None:
        self.model = YOLO(str(self.weights))

    def predict(self, image: Image, conf: float = 0.25) -> list[Detection]:
        assert self.model is not None
        result = self.model.predict(image, conf=conf, verbose=False)[0]

        detections = []
        for box in result.boxes or []:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    label=CLASS_NAMES[int(box.cls)],
                    confidence=float(box.conf),
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                )
            )

        return detections

    def get_info(self) -> ModelInfo:
        assert self.model is not None
        assert isinstance(self.model.model, nn.Module)
        params = sum(p.numel() for p in self.model.model.parameters())

        return ModelInfo(
            id=self.id,
            name=self.name,
            description=self.description,
            size_mb=self.weights.stat().st_size / 1e6,
            parameters=params,
        )
