# RT-DETR Inference Server

FastAPI server that loads a trained RT-DETR student checkpoint and serves
object detection over HTTP.

---

## Deploy in 5 steps

### 1. Place your checkpoint

```bash
mkdir -p weights
cp runs/run08_feature_l1.0/checkpoint_best.pth weights/checkpoint_best.pth
```

Any checkpoint saved by `tools/train_kd.py` works. The server accepts both
the full trainer dict (`{"model_state_dict": ...}`) and bare state-dicts.

---

### 2. Build the Docker image

```bash
docker build -t rtdetr-serve .
```

---

### 3. Run the container

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/weights:/weights:ro \
  -e MODEL_PATH=/weights/checkpoint_best.pth \
  rtdetr-serve
```

Or with docker-compose (mounts `./weights` automatically):

```bash
docker-compose up
```

Optional environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | — **(required)** | Path to `.pth` checkpoint inside the container |
| `MODEL_CFG` | `configs/rtdetr_r18vd_coco.yml` | Student YAML config |
| `SCORE_THRESH` | `0.3` | Default confidence threshold |
| `IMG_SIZE` | `640` | Inference resolution (square) |

---

### 4. Check the server is up

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "model": "RT-DETR-S (ResNet-18 student)",
  "device": "cpu",
  "checkpoint": "/weights/checkpoint_best.pth",
  "img_size": 640,
  "score_threshold": 0.3
}
```

---

### 5. Run a detection

Encode any JPEG or PNG as base64 and POST it to `/predict`:

```bash
IMAGE_B64=$(base64 -w 0 /path/to/photo.jpg)

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_B64\", \"score_threshold\": 0.4}" \
  | python3 -m json.tool
```

Example response:

```json
{
  "detections": [
    {
      "box": [142.3, 87.6, 210.4, 310.9],
      "score": 0.9123,
      "label": 0,
      "label_name": "person"
    },
    {
      "box": [34.1, 201.5, 89.2, 73.4],
      "score": 0.7841,
      "label": 2,
      "label_name": "car"
    }
  ],
  "count": 2,
  "image_size": [640, 480]
}
```

`box` is `[x, y, width, height]` in absolute pixels (COCO format, top-left origin).
`image_size` is the original image size **before** server-side resizing.

---

## API reference

Interactive docs are available at `http://localhost:8000/docs` (Swagger UI)
once the server is running.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check + model metadata |
| `/predict` | POST | Object detection on a base64 image |

---

## GPU inference

The server automatically uses CUDA if available. To pass a GPU into the
container:

```bash
docker run --gpus all --rm -p 8000:8000 \
  -v $(pwd)/weights:/weights:ro \
  -e MODEL_PATH=/weights/checkpoint_best.pth \
  rtdetr-serve
```
