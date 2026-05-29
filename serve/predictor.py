"""Model loading and inference logic for the RT-DETR serving layer.

Environment variables
---------------------
MODEL_PATH   (required) Path to a .pth checkpoint saved by trainer_kd.py.
MODEL_CFG    (optional) Path to the student YAML config. Defaults to
             configs/rtdetr_r18vd_coco.yml relative to PYTHONPATH root.
SCORE_THRESH (optional) Default score filter threshold. Default: 0.3.
IMG_SIZE     (optional) Inference resolution (square). Default: 640.
"""

import base64
import io
import os
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

from src.models.rtdetr import build_rtdetr

# COCO 80-class names, zero-indexed to match model output
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _topk_decode(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    top_k: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DETR-style top-k decoding over the flattened (Q × C) score tensor.

    Mirrors src.trainer_kd._topk_decode exactly so serving uses the same
    protocol as training evaluation.
    """
    B, Q, C = pred_logits.shape
    prob = pred_logits.sigmoid()
    k = min(top_k, Q * C)
    topk_scores, topk_idx = prob.flatten(1).topk(k, dim=1)   # [B, K]
    labels    = topk_idx % C                                   # [B, K]
    query_idx = topk_idx // C                                  # [B, K]
    boxes = pred_boxes[torch.arange(B, device=pred_boxes.device).unsqueeze(1), query_idx]
    return topk_scores, labels, boxes                          # all [B, K] / [B, K, 4]


class RTDETRPredictor:
    """Loads an RT-DETR student checkpoint and runs single-image inference."""

    def __init__(self) -> None:
        model_path = os.environ.get("MODEL_PATH")
        if not model_path:
            raise RuntimeError("MODEL_PATH environment variable is required.")

        cfg_path = os.environ.get(
            "MODEL_CFG",
            str(Path(__file__).parents[1] / "configs" / "rtdetr_r18vd_coco.yml"),
        )
        self.score_thresh = float(os.environ.get("SCORE_THRESH", "0.3"))
        self.img_size     = int(os.environ.get("IMG_SIZE", "640"))
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint   = model_path

        # Build model from config
        cfg: dict = {}
        if Path(cfg_path).exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}

        self.model = build_rtdetr(cfg)

        # Load weights — supports both bare state-dicts and trainer_kd checkpoints
        ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[predictor] Missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"[predictor] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

        self.model.to(self.device)
        self.model.eval()

        backbone = cfg.get("model", cfg).get("backbone", "resnet18")
        print(f"[predictor] Loaded {backbone} checkpoint from {model_path} on {self.device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image_b64: str,
        score_threshold: float | None = None,
    ) -> tuple[list[dict], tuple[int, int]]:
        """Run inference on a base64-encoded image.

        Args:
            image_b64:        Base64-encoded JPEG or PNG (no data-URI prefix).
            score_threshold:  Per-request override; falls back to self.score_thresh.

        Returns:
            (detections, (orig_w, orig_h))
            detections is a list of dicts with keys: box, score, label, label_name.
        """
        threshold = score_threshold if score_threshold is not None else self.score_thresh

        img, (orig_w, orig_h) = self._decode_image(image_b64)
        tensor = self._preprocess(img)

        with torch.no_grad():
            outputs = self.model(tensor)

        detections = self._postprocess(outputs, orig_w, orig_h, threshold)
        return detections, (orig_w, orig_h)

    @property
    def model_name(self) -> str:
        return "RT-DETR-S (ResNet-18 student)"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_image(self, image_b64: str) -> tuple[Image.Image, tuple[int, int]]:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img, img.size  # (width, height)

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        img = TF.resize(img, [self.img_size, self.img_size])
        tensor = TF.to_tensor(img)                              # [3, H, W], float32 [0,1]
        tensor = TF.normalize(tensor, _IMAGENET_MEAN, _IMAGENET_STD)
        return tensor.unsqueeze(0).to(self.device)              # [1, 3, H, W]

    def _postprocess(
        self,
        outputs: dict,
        orig_w: int,
        orig_h: int,
        threshold: float,
    ) -> list[dict]:
        pred_logits = outputs["pred_logits"]  # [1, Q, C]
        pred_boxes  = outputs["pred_boxes"]   # [1, Q, 4] — (cx, cy, w, h) normalized

        scores, labels, boxes = _topk_decode(pred_logits, pred_boxes, top_k=100)

        # Squeeze batch dimension
        scores = scores[0]  # [K]
        labels = labels[0]  # [K]
        boxes  = boxes[0]   # [K, 4]

        # Filter by score threshold
        keep = scores >= threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes  = boxes[keep]

        detections = []
        for score, label, box in zip(
            scores.cpu().tolist(),
            labels.cpu().tolist(),
            boxes.cpu().tolist(),
        ):
            cx, cy, w, h = box
            # Convert normalized (cx,cy,w,h) → absolute pixel [x1,y1,w_px,h_px]
            x1 = (cx - w / 2) * orig_w
            y1 = (cy - h / 2) * orig_h
            w_px = w * orig_w
            h_px = h * orig_h
            label_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else str(label)
            detections.append({
                "box":        [round(x1, 2), round(y1, 2), round(w_px, 2), round(h_px, 2)],
                "score":      round(score, 4),
                "label":      label,
                "label_name": label_name,
            })

        # Sort descending by score for readability
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections
