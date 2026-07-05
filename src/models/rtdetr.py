"""RT-DETR wrapper (Ultralytics): reuses YoloWrapper, overrides only load()."""

from ultralytics import RTDETR

from src.models.yolo import YoloWrapper


class RTDETRWrapper(YoloWrapper):
    """Transformer-based detector (Ultralytics RT-DETR)."""

    id = "rtdetr"
    name = "RT-DETR"
    description = "End-to-end transformer detector (DETR)."

    def load(self) -> None:
        self.model = RTDETR(str(self.weights))
