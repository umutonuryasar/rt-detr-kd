"""FastAPI inference server for RT-DETR Knowledge Distillation.

Endpoints
---------
GET  /health   — liveness check, returns model metadata
POST /predict  — accepts a base64 image, returns COCO-format detections

Start with:
    uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from serve.predictor import RTDETRPredictor
from serve.schemas import HealthResponse, PredictRequest, PredictResponse, Detection

# Predictor is loaded once at startup and shared across all requests.
_predictor: RTDETRPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = RTDETRPredictor()
    yield
    _predictor = None


app = FastAPI(
    title="RT-DETR Inference Server",
    description="Runs a trained RT-DETR student checkpoint for object detection.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_predictor() -> RTDETRPredictor:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return _predictor


@app.get("/health", response_model=HealthResponse, summary="Liveness and model info")
def health() -> HealthResponse:
    """Returns model metadata and server status."""
    p = _get_predictor()
    return HealthResponse(
        status="ok",
        model=p.model_name,
        device=p.device,
        checkpoint=p.checkpoint,
        img_size=p.img_size,
        score_threshold=p.score_thresh,
    )


@app.post("/predict", response_model=PredictResponse, summary="Detect objects in an image")
def predict(request: PredictRequest) -> PredictResponse:
    """Accepts a base64-encoded image and returns object detections.

    The `image` field should be a plain base64 string (JPEG or PNG).
    Strip any `data:image/...;base64,` prefix before sending.

    Boxes are returned in COCO format: `[x, y, width, height]` in absolute
    pixels relative to the **original** image size (before server-side resizing).
    """
    p = _get_predictor()
    try:
        detections_raw, (orig_w, orig_h) = p.predict(
            request.image,
            score_threshold=request.score_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Inference failed: {exc}") from exc

    detections = [Detection(**d) for d in detections_raw]
    return PredictResponse(
        detections=detections,
        count=len(detections),
        image_size=[orig_w, orig_h],
    )
