"""API routes: list models and run a prediction."""

import io
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.api.schemas import (
    DetectionOut,
    ModelInfoOut,
    ModelsResponse,
    PredictResponse,
)
from src.models.registry import ModelRegistry, get_registry

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def list_models(
    registry: ModelRegistry = Depends(get_registry),
) -> ModelsResponse:
    """List the available models and their metadata."""
    return ModelsResponse(
        available_models=[
            ModelInfoOut(
                id=i.id,
                name=i.name,
                description=i.description,
                size_mb=i.size_mb,
                parameters=i.parameters,
            )
            for i in registry.list_info()
        ]
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    model_id: str = Form(...),
    conf: float = Form(0.25),
    registry: ModelRegistry = Depends(get_registry),
) -> PredictResponse:
    """Detect smoke and fire in an uploaded image.

    Args:
        image: The uploaded image file.
        model_id: Which model to run (see GET /models).
        conf: Minimum confidence to keep a detection.

    Returns:
        The model used, its detections, and the inference time.

    Raises:
        HTTPException: 400 if the model id is unknown or the file is not an image.
    """
    try:
        wrapper = registry.get(model_id)
    except KeyError:
        raise HTTPException(400, f"Unknown model_id: {model_id}") from None

    try:
        pil_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Uploaded file is not a valid image") from None

    start = time.perf_counter()
    detections = wrapper.predict(pil_image, conf=conf)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        model_used=model_id,
        detections=[
            DetectionOut(label=d.label, confidence=d.confidence, bbox=d.bbox)
            for d in detections
        ],
        inference_time_ms=round(elapsed_ms, 1),
    )
