# RT-DETR Knowledge Distillation

Knowledge distillation for efficient real-time detection transformers. This work systematically studies logit-level, feature-level, combined, and partial KD strategies to compress RT-DETR-L (ResNet-50, 32M) into RT-DETR-S (ResNet-18, 17M) on COCO, with edge deployment analysis via TensorRT INT8.

> **Status:** Training in progress — arXiv preprint coming soon.

---

## Overview

Standard model compression of RT-DETR yields significant mAP degradation. This work investigates whether knowledge distillation can recover that gap while maintaining real-time inference on constrained hardware (RTX 3050, 4GB VRAM).

**Research questions:**
1. Does feature-level KD (encoder MSE + cross-attention cosine) outperform logit-level KD on transformer-based detectors?
2. Which KD component contributes more — encoder distillation or attention distillation?
3. What is the latency-accuracy trade-off after TensorRT INT8 quantization?
4. How does teacher model capacity affect student performance?

---

## Models

| Model | Backbone | Params | mAP@[.5:.95] | FPS (T4) |
|-------|----------|--------|--------------|----------|
| RT-DETR-S (student) | ResNet-18 | 17M | 48.9 | ~120 |
| RT-DETR-M (teacher) | ResNet-34 | 25M | 51.3 | ~117 |
| RT-DETR-L (teacher) | ResNet-50 | 32M | 53.1 | ~114 |

---

## Distillation methods

### Logit-KD
KL divergence between temperature-scaled teacher and student classification logits:

$$\mathcal{L}_{\text{logit}} = T^2 \cdot \text{KL}\left(\sigma\!\left(\frac{t}{T}\right) \,\middle\|\, \sigma\!\left(\frac{s}{T}\right)\right)$$

Temperature $T \in \{2, 4, 8\}$, applied to classification head only.

### Feature-KD
Two complementary components:

$$\mathcal{L}_{\text{feat}} = \text{MSE}\!\left(\text{proj}(s_{\text{enc}}),\, t_{\text{enc}}\right)$$

$$\mathcal{L}_{\text{attn}} = 1 - \text{cos\_sim}(s_{\text{attn}},\, t_{\text{attn}})$$

$$\mathcal{L}_{\text{KD}} = w_f \cdot \mathcal{L}_{\text{feat}} + \alpha \cdot \mathcal{L}_{\text{attn}}$$

### Combined KD
Logit-KD and Feature-KD applied simultaneously with tunable weights.

### Total loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \lambda \cdot \mathcal{L}_{\text{KD}}$$

---

## Ablation grid (14 runs)

| Run | KD type | λ | T | Notes |
|-----|---------|---|---|-------|
| 00 | Baseline | — | — | No KD |
| 01 | Logit | 0.5 | 2 | |
| 02 | Logit | 0.5 | 4 | |
| 03 | Logit | 0.5 | 8 | |
| 04 | Logit | 1.0 | 2 | |
| 05 | Logit | 1.0 | 4 | |
| 06 | Logit | 1.0 | 8 | |
| 07 | Feature | 0.5 | — | |
| 08 | Feature | 1.0 | — | Best projected ★ |
| 09 | Combined | 1.0 | 4 | Logit + Feature |
| 10 | Encoder-only | 1.0 | — | Partial ablation |
| 11 | Attention-only | 1.0 | — | Partial ablation |
| 12 | Feature (teacher=R34) | 1.0 | — | Capacity analysis |
| 13 | Feature (teacher=R50) | 1.0 | — | Capacity upper bound |

---

## Setup

```bash
git clone https://github.com/umutonuryasar/rt-detr-kd
cd rt-detr-kd
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Download COCO subset (30K training images)

```bash
bash scripts/download_coco_subset.sh /data
```

### Pretrained weights

Download RT-DETR pretrained weights from [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr) and place under `weights/`.

---

## Training

```bash
# Single run (feature-KD, λ=1.0)
python tools/train_kd.py \
  --student-cfg configs/rtdetr_r18vd_coco.yml \
  --teacher-cfg configs/rtdetr_r50vd_coco.yml \
  --kd-type feature \
  --kd-lambda 1.0 \
  --epochs 36 \
  --batch-size 4 \
  --output-dir runs/feature_kd_l1.0

# All 14 ablation runs
bash scripts/run_ablation.sh /data/coco runs
```

### Training details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| LR (backbone) | 1e-4 |
| LR (transformer head) | 1e-3 |
| Weight decay | 1e-4 |
| LR schedule | Cosine + 500-iter warmup |
| Batch size | 4 (RTX 3050) / 16 (A100) |
| Epochs | 36 |
| Image size | 640×640 |
| AMP | fp16 |

---

## Evaluation

```bash
python tools/eval.py \
  --cfg configs/rtdetr_r18vd_coco.yml \
  --weights runs/feature_kd_l1.0/checkpoint_best.pth \
  --coco-val /data/coco/val2017 \
  --val-ann /data/coco/annotations/instances_val2017.json
```

---

## FPS benchmarking

```bash
python tools/benchmark_fps.py \
  --cfg configs/rtdetr_r18vd_coco.yml \
  --weights runs/feature_kd_l1.0/checkpoint_best.pth \
  --input-size 640 \
  --warmup 50 \
  --iters 500 \
  --device cuda
```

Protocol: batch=1, fp32, single-stream, 50-iter warmup, 500-iter measurement.

---

## Repository structure

```
rt-detr-kd/
├── configs/
│   ├── rtdetr_r18vd_coco.yml       # Student config
│   ├── rtdetr_r50vd_coco.yml       # Teacher config (R50)
│   ├── rtdetr_r34vd_coco.yml       # Teacher config (R34)
│   └── kd/
│       ├── logit_kd.yml
│       ├── feature_kd.yml
│       ├── combined_kd.yml
│       ├── encoder_only_kd.yml
│       └── attention_only_kd.yml
├── src/
│   ├── distillation/               # KD loss modules
│   ├── models/                     # RT-DETR architecture
│   ├── data/                       # COCO dataset & transforms
│   ├── losses/                     # Detection loss & matcher
│   └── trainer_kd.py
├── tools/
│   ├── train_kd.py
│   ├── eval.py
│   ├── benchmark_fps.py
│   └── export_trt.py               # ONNX → TensorRT INT8
├── notebooks/
│   ├── ablation_analysis.ipynb
│   └── visualize_attention.ipynb
└── scripts/
    ├── download_coco_subset.sh
    └── run_ablation.sh
```

---

## Hardware

- **Local:** Ubuntu 24.04 · RTX 3050 4GB · Ryzen 5800H · 16GB RAM
- **Training:** Google Colab Pro+ · A100 40GB

---

## Author

**Umut Onur Yasar** — Applied AI Researcher  
[GitHub](https://github.com/umutonuryasar) · [LinkedIn](https://linkedin.com/in/umutonuryasar) · [umutonuryasar.com](https://umutonuryasar.com)
