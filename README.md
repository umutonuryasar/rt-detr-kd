# RT-DETR Knowledge Distillation

Knowledge distillation for efficient real-time detection transformers. This work systematically studies logit-level, feature-level, combined, partial, and novel RT-DETR-specific KD strategies to compress RT-DETR-L (ResNet-50, 32M) into RT-DETR-S (ResNet-18, 17M) on COCO. Includes comparisons against CWD and MGD baselines and edge deployment analysis via TensorRT INT8.

> **Status:** Training in progress — arXiv preprint coming soon.

---

## Overview

Standard model compression of RT-DETR yields significant mAP degradation. This work investigates whether knowledge distillation can recover that gap while maintaining real-time inference on constrained hardware (RTX 3050, 4GB VRAM).

**Research questions:**
1. Does feature-level KD (encoder MSE + cross-attention cosine) outperform logit-level KD on transformer-based detectors?
2. Which KD component contributes more — encoder distillation or attention distillation?
3. How do established KD methods (CWD, MGD) compare against our feature-level KD on RT-DETR?
4. Does query-level distillation — targeting RT-DETR's decoder object queries — provide complementary gains?
5. Does stage-adaptive KD weighting (curriculum shift from feature to logit) outperform static weighting?
6. What is the latency-accuracy trade-off after TensorRT INT8 quantization?
7. How does teacher model capacity affect student performance?

---

## Models

This repository pairs a **canonical teacher** loaded from the official
[lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) PyTorch release with a
**simplified custom student** trained from scratch. The two architectures are
intentionally different (see §3.2 below); KD therefore operates
cross-architecture. The teacher mAPs cited below come from the upstream
repository; the student mAP is what this codebase produces and is reported in
all paper tables.

| Role | Backbone | Source | Params | mAP@[.5:.95] | FPS (T4) |
|------|----------|--------|--------|--------------|----------|
| Student (this repo, simplified) | ResNet-18 | trained here | 17M | TBD (Phase 2D) | ~120 |
| Teacher (canonical RT-DETR-M) | ResNet-34 | lyuwenyu/RT-DETR | 25M | 51.3¹ | ~117 |
| Teacher (canonical RT-DETR-L) | ResNet-50 | lyuwenyu/RT-DETR | 32M | 53.1¹ | ~114 |

¹ Cited verbatim from lyuwenyu/RT-DETR's published checkpoints. Verified at
training start via the teacher mAP sanity gate (`tools/train_kd.py`
`--teacher-min-map`).

### 3.2 Implementation differences from canonical RT-DETR

The student differs from the teacher in four ways. Every simplification is
imposed by the 4 GB RTX 3050 VRAM budget for dual-model forward passes during
KD; canonical RT-DETR will not fit. These differences apply to the **student
only**; the teacher is the canonical architecture loaded with published
weights.

| Component | Canonical RT-DETR | This student | Reason |
|-----------|-------------------|--------------|--------|
| Object queries | 300 | 100 | OOMs on 4 GB with teacher+student fp16 forward |
| Decoder layers | 6 | 3 | OOMs at 6 layers with dual forward pass |
| Cross-attention | Multi-scale deformable | Vanilla MHA | Deformable kernel doubles backward memory |
| Encoder memory | C3 + C4 + C5 | C4 + C5 only | C3 token count alone (6400 @ 640²) saturates VRAM |

The Phase 2D final runs are executed on Colab A100, where these constraints
do not apply; however we keep the same simplified student for consistency
across ablation and final phases. Reviewers should interpret the absolute
student mAP as belonging to this simplified architecture; the relative
*KD-method* ranking, which is what the paper measures, is what transfers.

---

## Distillation methods

### Logit-KD
KL divergence between temperature-scaled teacher and student classification logits:

$$\mathcal{L}_{\text{logit}} = T^2 \cdot \text{KL}\left(\sigma\!\left(\frac{t}{T}\right) \,\middle\|\, \sigma\!\left(\frac{s}{T}\right)\right)$$

Temperature $T \in \{2, 4, 8\}$, applied to classification head only.

### Feature-KD
Two complementary components:

$$\mathcal{L}_{\text{feat}} = \text{MSE}\!\left(\text{proj}(s_{\text{enc}}),\, t_{\text{enc}}\right)$$

$$\mathcal{L}_{\text{attn}} = 1 - \frac{s_{\text{attn}} \cdot t_{\text{attn}}}{\|s_{\text{attn}}\| \|t_{\text{attn}}\|}$$

$$\mathcal{L}_{\text{KD}} = w_f \cdot \mathcal{L}_{\text{feat}} + \alpha \cdot \mathcal{L}_{\text{attn}}$$

### Combined KD
Logit-KD and Feature-KD applied simultaneously with tunable weights.

### CWD — Channel-Wise Distillation (baseline)
Distills spatially-normalized channel distributions of feature maps via KL divergence (ICCV'21):

$$\mathcal{L}_{\text{CWD}} = \sum_{c=1}^{C} \text{KL}\!\left(\tilde{t}_c \,\middle\|\, \tilde{s}_c\right)$$

where $\tilde{t}_c = \text{softmax}(t_c / \tau)$ over spatial dimensions.

### MGD — Masked Generative Distillation (baseline)
Randomly masks student features and trains a lightweight generator to reconstruct teacher features, enforcing holistic alignment (ECCV'22):

$$\mathcal{L}_{\text{MGD}} = \left\| \mathcal{G}\!\left(\mathbf{M} \odot s_{\text{feat}}\right) - t_{\text{feat}} \right\|_2^2$$

where $\mathbf{M}$ is a random binary mask and $\mathcal{G}$ is a small convolutional generator.

### Query-KD (novel)
Distills RT-DETR's decoder object queries directly — a transformer-specific component not exploited by CNN-detector KD methods:

$$\mathcal{L}_{\text{query}} = \text{MSE}(q_s,\, q_t)$$

Combined with cross-attention pattern alignment between decoder queries:

$$\mathcal{L}_{\text{query-attn}} = 1 - \frac{A_s^{\text{dec}} \cdot A_t^{\text{dec}}}{\|A_s^{\text{dec}}\| \|A_t^{\text{dec}}\|}$$

**Differentiation from prior work.** DETRDistill (ICLR'23) targets DETR's
*matched* query-to-prediction pairs after Hungarian assignment, requiring
joint matching of teacher and student. Our formulation aligns the *post-norm*
decoder embeddings of the first `min(Q_s, Q_t)` queries directly, with no
shared matcher — simpler, robust to teacher/student query count mismatch
(here 300 vs 100), and combinable with the cross-attention cosine term that
DETRDistill does not consider. The closest precedent on the attention side is
MimicDet (ECCV'20) which mimics RPN-style attention in two-stage detectors;
we instead align decoder cross-attention against encoder memory, which is the
RT-DETR-specific signal that CNN-detector KD cannot use.

### Stage-Adaptive KD (novel)
Curriculum weighting that shifts from feature-heavy (structural alignment) to logit-heavy (semantic refinement) across training:

$$w_f(e) = \cos\!\left(\frac{\pi e}{2E}\right), \qquad w_l(e) = 1 - w_f(e)$$

$$\mathcal{L}_{\text{KD}}^{\text{SA}}(e) = w_f(e)\cdot\mathcal{L}_{\text{feat}} + w_l(e)\cdot\mathcal{L}_{\text{logit}}$$

where $e$ is the current epoch and $E$ is total epochs.

### Total loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \lambda \cdot \mathcal{L}_{\text{KD}}$$

---

## Training strategy

Two-phase design to balance experimental rigor against compute budget:

| Phase | Dataset | Epochs | Runs | Purpose |
|-------|---------|--------|------|---------|
| **2A — Ablation** | COCO 30K subset | 36 | 18 | Hyperparameter search, method selection |
| **2D — Final** | Full COCO 118K | 72 | ~8 | Paper numbers, SOTA comparison |
| **2E — Reliability** | Full COCO 118K | 72 | 3 seeds | Mean ± std for best method |

Phase 2A identifies top-performing configurations; Phase 2D re-trains only those for publishable results.

---

## Ablation grid

### Phase 2A — COCO 30K subset, 36 epochs (23 runs)

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
| 14 | CWD | 1.0 | — | Baseline comparison |
| 15 | MGD | 1.0 | — | Baseline comparison |
| 16 | Query-KD | 1.0 | — | Novel: decoder query distillation |
| 17 | Stage-Adaptive, cosine | 1.0 | — | Novel: curriculum weighting |
| 18 | Stage-Adaptive, linear | 1.0 | — | Schedule ablation |
| 19 | Stage-Adaptive, step | 1.0 | — | Schedule ablation |
| 20 | Stage-Adaptive, sigmoid | 1.0 | — | Schedule ablation |
| 21 | Stage-Adaptive, inverse cosine | 1.0 | — | **Curriculum-direction control** |
| 22 | Baseline (72 epochs, 2×) | — | — | **Reviewer control**: "does KD beat training longer?" |

### Phase 2D — Full COCO, 72 epochs (~8 runs)

Top-5 configurations from Phase 2A re-trained alongside Baseline, CWD, and MGD for final paper results.

### Phase 2E — Statistical reliability

Best method × 3 random seeds on full COCO, 72 epochs. All paper results report mean ± std.

---

## Setup

```bash
git clone --recurse-submodules https://github.com/umutonuryasar/rt-detr-kd
cd rt-detr-kd
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# If you cloned without --recurse-submodules:
git submodule update --init --recursive
```

The `third_party/RT-DETR` submodule pins
[lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) — the official PyTorch
implementation from the original authors. It is used as the canonical KD
teacher (see §3.2).

### Download COCO

```bash
# 30K subset (ablation phase)
bash scripts/download_coco_subset.sh /data

# Full COCO (final phase)
bash scripts/download_coco_full.sh /data
```

### Pretrained weights

Download RT-DETR pretrained weights from [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr) and place under `weights/`.

---

## Training

```bash
# Single run (feature-KD, λ=1.0, ablation phase)
python tools/train_kd.py \
  --student-cfg configs/rtdetr_r18vd_coco.yml \
  --teacher-cfg configs/rtdetr_r50vd_coco.yml \
  --kd-cfg configs/kd/feature_kd.yml \
  --epochs 36 \
  --batch-size 4 \
  --output-dir runs/feature_kd_l1.0

# All 18 ablation runs (Phase 2A)
bash scripts/run_ablation.sh /data/coco runs

# Final paper runs (Phase 2D, full COCO, 72 epochs)
bash scripts/run_final.sh /data/coco runs
```

### Training details

| Hyperparameter | Ablation (2A) | Final (2D) |
|----------------|--------------|-----------|
| Dataset | COCO 30K | Full COCO 118K |
| Epochs | 36 | 72 |
| Optimizer | AdamW | AdamW |
| LR (backbone) | 1e-4 | 1e-4 |
| LR (transformer head) | 1e-3 | 1e-3 |
| Weight decay | 1e-4 | 1e-4 |
| LR schedule | Cosine + 500-iter warmup | Cosine + 500-iter warmup |
| Batch size | 4 (RTX 3050) / 16 (A100) | 16 (A100) |
| Grad accumulation | 2 (RTX) / 1 (A100) | 1 (A100) |
| Image size | 640×640 | 640×640 |
| AMP | fp16 | fp16 |
| Teacher | Frozen, eval mode | Frozen, eval mode |
| Seeds | 1 | 3 (mean ± std) |

---

## Evaluation

```bash
python tools/eval.py \
  --cfg configs/rtdetr_r18vd_coco.yml \
  --weights runs/feature_kd_l1.0/checkpoint_best.pth \
  --coco-val /data/coco/val2017 \
  --val-ann /data/coco/annotations/instances_val2017.json
```

**Decoding protocol.** All evaluation uses DETR-style top-k decoding over
the flattened `(num_queries × num_classes)` score tensor (k=100), not
per-query argmax. A single query may legitimately surface under multiple
classes; top-k recovers those predictions and matches the standard DETR /
Deformable-DETR / RT-DETR evaluation protocol. Argmax decoding
under-reports mAP by 1–3 points and is *not* used anywhere in this
repository (see `src.trainer_kd._topk_decode`).

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
│   ├── rtdetr_r18vd_coco.yml           # Student config
│   ├── rtdetr_r50vd_coco.yml           # Teacher config (R50)
│   ├── rtdetr_r34vd_coco.yml           # Teacher config (R34)
│   └── kd/
│       ├── logit_kd.yml
│       ├── feature_kd.yml
│       ├── combined_kd.yml
│       ├── encoder_only_kd.yml
│       ├── attention_only_kd.yml
│       ├── cwd_kd.yml                  # CWD baseline
│       ├── mgd_kd.yml                  # MGD baseline
│       ├── query_kd.yml                # Novel: query distillation
│       └── stage_adaptive_kd.yml       # Novel: curriculum weighting
├── src/
│   ├── distillation/
│   │   ├── logit_kd.py
│   │   ├── feature_kd.py
│   │   ├── cwd.py                      # CWD implementation
│   │   ├── mgd.py                      # MGD implementation
│   │   ├── query_kd.py                 # Query-KD implementation
│   │   ├── stage_adaptive_kd.py        # Stage-adaptive implementation
│   │   ├── kd_loss.py                  # Unified loss wrapper
│   │   └── __init__.py
│   ├── models/
│   │   ├── rtdetr.py
│   │   ├── rtdetr_kd.py
│   │   ├── backbone.py
│   │   ├── encoder.py
│   │   └── decoder.py
│   ├── data/
│   │   ├── coco_dataset.py
│   │   └── transforms.py
│   ├── losses/
│   │   ├── detection_loss.py
│   │   └── matcher.py
│   └── trainer_kd.py
├── tools/
│   ├── train_kd.py
│   ├── eval.py
│   ├── benchmark_fps.py
│   ├── export_trt.py                   # ONNX → TensorRT FP32/FP16/INT8 + benchmark
│   ├── verify_teacher_kd.py            # Cross-architecture KD smoke test
│   └── aggregate_results.py            # Walk runs/ → CSV + Markdown table
├── tests/                              # pytest smoke tests (CI on every push)
│   ├── test_kd_losses.py
│   ├── test_models.py
│   └── conftest.py
├── third_party/
│   └── RT-DETR/                        # lyuwenyu/RT-DETR submodule — canonical teacher
├── notebooks/
│   ├── ablation_analysis.ipynb
│   ├── visualize_attention.ipynb
│   └── colab_training.ipynb
├── scripts/
│   ├── download_coco_subset.sh
│   ├── download_coco_full.sh
│   ├── run_ablation.sh                  # Phase 2A: 23 runs
│   └── run_final.sh                     # Phase 2D: final paper runs
└── .github/workflows/
    └── ci.yml                           # pytest on every push
```

---

## Hardware

- **Local:** Ubuntu 24.04 · RTX 3050 4GB · Ryzen 5800H · 16GB RAM
- **Training:** Google Colab Pro+ · A100 40GB

---

## Author

**Umut Onur Yasar** — Applied AI Researcher  
[GitHub](https://github.com/umutonuryasar) · [LinkedIn](https://linkedin.com/in/umutonuryasar) · [umutonuryasar.com](https://umutonuryasar.com)
