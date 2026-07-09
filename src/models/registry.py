"""Model registry: maps a model id to its wrapper, loaded lazily on first use."""

import os
from functools import lru_cache
from pathlib import Path

from src.models.base import ModelInfo, ModelWrapper
from src.models.fasterrcnn import FasterRCNNWrapper
from src.models.rtdetr import RTDETRWrapper
from src.models.yolo import YoloWrapper

WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", "weights"))

# id -> (wrapper class, weights filename)
_WRAPPERS: dict[str, tuple[type[ModelWrapper], str]] = {
    "yolo11n": (YoloWrapper, "yolo11n_best.pt"),
    "rtdetr": (RTDETRWrapper, "rtdetr_best.pt"),
    "fasterrcnn": (FasterRCNNWrapper, "fasterrcnn_best.pt"),
}


class ModelRegistry:
    """Holds the wrappers and loads them on demand."""

    def __init__(self):
        self._loaded: dict[str, ModelWrapper] = {}

    @property
    def ids(self) -> list[str]:
        """Ids of every model the service can serve."""
        return list(_WRAPPERS)

    def get(self, model_id: str) -> ModelWrapper:
        """Return a ready-to-use wrapper, loading its weights on first call.

        Args:
            model_id: One of the registered ids (see ``ids``).

        Returns:
            The loaded wrapper for that id.

        Raises:
            KeyError: If the id is not registered.
        """
        if model_id not in _WRAPPERS:
            raise KeyError(model_id)

        if model_id not in self._loaded:
            wrapper_cls, weights = _WRAPPERS[model_id]
            wrapper = wrapper_cls(WEIGHTS_DIR / weights)
            wrapper.load()
            self._loaded[model_id] = wrapper

        return self._loaded[model_id]

    def list_info(self) -> list[ModelInfo]:
        """Metadata for every model, loading each once to read its details."""
        return [self.get(mid).get_info() for mid in self.ids]


@lru_cache
def get_registry() -> ModelRegistry:
    """Return the process-wide registry, created once."""
    return ModelRegistry()
