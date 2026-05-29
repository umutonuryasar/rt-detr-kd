"""Pydantic request/response schemas for the RT-DETR inference server."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image: str = Field(
        ...,
        description="Base64-encoded image (JPEG/PNG). No data-URI prefix needed.",
    )
    score_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Override the server default score threshold for this request.",
    )


class Detection(BaseModel):
    box: list[float] = Field(
        ...,
        description="Bounding box in COCO format: [x, y, width, height] (absolute pixels, top-left origin).",
    )
    score: float = Field(..., description="Confidence score in [0, 1].")
    label: int = Field(..., description="Zero-indexed class ID (0–79 for COCO).")
    label_name: str = Field(..., description="Human-readable COCO category name.")


class PredictResponse(BaseModel):
    detections: list[Detection]
    count: int = Field(..., description="Number of detections returned.")
    image_size: list[int] = Field(
        ...,
        description="Original image dimensions [width, height] before any resizing.",
    )


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    checkpoint: str
    img_size: int
    score_threshold: float
