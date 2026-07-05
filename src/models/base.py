"""Model abstraction: one interface and output shape for every detector.

A new model is a new wrapper; the routes never change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from PIL.Image import Image

CLASS_NAMES = {0: "smoke", 1: "fire"}


@dataclass
class Detection:
    """One predicted box.

    Attributes:
        label: Class name ("smoke" or "fire").
        confidence: Detection score in [0, 1].
        bbox: Pixel box as [x, y, width, height], top-left origin.
    """

    label: str
    confidence: float
    bbox: list[float]


@dataclass
class ModelInfo:
    """Static facts about a model, shown by GET /models."""

    id: str
    name: str
    description: str
    size_mb: float
    parameters: int


class ModelWrapper(ABC):
    """Common interface for every detector the service can run."""

    # Metadata the registry reads without loading weights.
    id: str = ""
    name: str = ""
    description: str = ""

    def __init__(self, weights: Path):
        self.weights = Path(weights)

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory. Called once before the first prediction."""

    @abstractmethod
    def predict(self, image: Image, conf: float = 0.25) -> list[Detection]:
        """Detect smoke and fire in an image.

        Args:
            image: Input image (RGB).
            conf: Minimum score to keep a detection.

        Returns:
            Detections above the confidence threshold.
        """

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Return the model's metadata for the API."""
