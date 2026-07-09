"""Request and response models for the API."""

from pydantic import BaseModel


class DetectionOut(BaseModel):
    """A single detected box in the response."""

    label: str
    confidence: float
    bbox: list[float]  # [x, y, width, height] in pixels


class ModelInfoOut(BaseModel):
    """An entry in the GET /models list."""

    id: str
    name: str
    description: str
    size_mb: float
    parameters: int


class ModelsResponse(BaseModel):
    available_models: list[ModelInfoOut]


class PredictResponse(BaseModel):
    model_used: str
    detections: list[DetectionOut]
    inference_time_ms: float
