# RT-DETR Knowledge Distillation

**Systematic knowledge distillation study for real-time detection transformers — 5 KD methods compared, 2 novel contributions, TensorRT INT8 edge deployment on a 4 GB GPU.**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/umutonuryasar/rt-detr-kd/actions/workflows/ci.yml/badge.svg)

> **Status:** Phase 2A ablation in progress — arXiv preprint forthcoming.

---

## Motivation

RT-DETR achieves state-of-the-art detection accuracy but its 32M-parameter ResNet-50 backbone is ill-suited to edge hardware: a direct swap to ResNet-18 (17M params) costs several mAP points with no principled recovery strategy. Knowledge distillation transfers structural and semantic signal from a frozen teacher to a lightweight student, but most KD literature targets CNN detectors — it is unclear how logit-level versus feature-level versus query-level distillation interact with the transformer encoder-decoder architecture that RT-DETR uses. This work runs a controlled ablation across five KD methods on a fixed 4 GB RTX 3050 budget, introduces two transformer-specific techniques (Query-KD and Stage-Adaptive KD), and carries the best configuration through TensorRT INT8 quantization to a deployable FastAPI server.

---

## What I Built

- **6-run ablation:** Baseline, Logit-KD, Feature-KD, CWD (ICCV'21), and two novel methods — isolated and reproducible
- **Feature-KD** with encoder MSE + decoder cross-attention cosine alignment, projecting student features to teacher channel width
- **CWD** (Yang et al., ICCV'21) — channel-wise softmax KL baseline for fair literature comparison
- **Query-KD** *(novel)* — distils RT-DETR's 100/300-dim decoder object queries directly; no shared Hungarian matcher required, robust to teacher/student query-count mismatch
- **Stage-Adaptive KD** *(novel)* — cosine curriculum that shifts weight from feature distillation (structural alignment, early training) to logit distillation (semantic refinement, late training)
- **Cross-architecture teacher adapter** (`src/models/rtdetr_teacher.py`) loading canonical [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) weights with a mAP sanity gate at training start
- **TensorRT INT8 export** with entropy calibration, FP32/FP16/INT8 latency sweep, and a latency-vs-accuracy table (`tools/export_trt.py`)
- **FastAPI inference server** for single-image and batch detection endpoints
- **Automated results aggregation** (`tools/aggregate_results.py`) producing CSV + Markdown tables and mean ± std across seeds

---

## Results

> Ablation training in progress. Table will be populated after Phase 2A (COCO 30K, 36 epochs).

| Method | mAP@\[.5:.95\] | ΔmAP | FPS (RTX 3050) | Params |
|--------|---------------|------|----------------|--------|
| Baseline (no KD) | — | — | — | 17M |
| Logit-KD (λ=1.0, T=4) | — | — | — | 17M |
| Feature-KD (λ=1.0) | — | — | — | 17M |
| CWD — Yang et al. ICCV'21 | — | — | — | 17M |
| **Query-KD** *(novel)* | — | — | — | 17M |
| **Stage-Adaptive KD, cosine** *(novel)* | — | — | — | 17M |
| Teacher RT-DETR-L (R50) | 53.1 | ref | ~114 | 32M |

Final paper numbers (full COCO 118K, 72 epochs, 3 seeds) will be reported as mean ± std.

---

## Architecture

### Student vs. teacher

The repository pairs a **canonical teacher** (loaded from the official lyuwenyu/RT-DETR PyTorch release) with a **simplified custom student** trained from scratch. KD operates cross-architecture. The teacher mAPs below are from the upstream repository; all paper tables report student mAP.

| Role | Backbone | Source | Params | mAP@\[.5:.95\] |
|------|----------|--------|--------|---------------|
| Student (simplified, this repo) | ResNet-18 | trained here | 17M | TBD |
| Teacher RT-DETR-M | ResNet-34 | lyuwenyu/RT-DETR | 25M | 51.3 |
| Teacher RT-DETR-L | ResNet-50 | lyuwenyu/RT-DETR | 32M | 53.1 |

### Implementation differences (student only)

Every simplification is forced by the 4 GB RTX 3050 VRAM budget for dual-model (teacher + student) forward passes. The teacher is the unmodified canonical architecture.

| Component | Canonical RT-DETR | This student | Reason |
|-----------|-------------------|--------------|--------|
| Object queries | 300 | 100 | OOMs at 300 with dual fp16 forward |
| Decoder layers | 6 | 3 | OOMs at 6 layers with dual forward pass |
| Cross-attention | Multi-scale deformable | Vanilla MHA | Deformable kernel doubles backward memory |
| Encoder memory | C3 + C4 + C5 | C4 + C5 only | C3 token count (6400 @ 640²) saturates VRAM |

Phase 2D (final runs) executes on a Colab A100 but deliberately keeps the same student architecture for consistency across ablation and final phases. The paper measures *relative KD-method ranking*, which transfers independently of these simplifications.

---

## Quickstart

```bash
# Clone with canonical teacher submodule
git clone --recurse-submodules https://github.com/umutonuryasar/rt-detr-kd
cd rt-detr-kd
pip install -r requirements.txt

# Run inference server (Docker)
docker pull ghcr.io/umutonuryasar/rt-detr-kd:latest
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/weights:/weights \
    ghcr.io/umutonuryasar/rt-detr-kd serve \
    --weights /weights/checkpoint_best.pth

# Single-image detection
curl -X POST http://localhost:8000/detect \
    -F "image=@photo.jpg" | python -m json.tool
```

---

## Project Structure

```
rt-detr-kd/
├── configs/          # YAML configs: student, teacher, all 5 KD methods
│   └── kd/           # Active KD configs (cwd, query, stage_adaptive)
├── src/              # Core library: models, distillation losses, data, trainer
├── tools/            # train_kd, eval, benchmark_fps, export_trt, serve, aggregate_results
├── tests/            # pytest smoke tests — runs on every push (CPU-only CI)
├── scripts/          # run_ablation.sh (6-run Phase 2A), run_final.sh (Phase 2D)
├── notebooks/        # ablation_analysis, visualize_attention, colab_training
├── third_party/      # lyuwenyu/RT-DETR submodule — canonical teacher weights + config
└── .github/          # CI: pytest on push
```

---

## Distillation Methods

Total loss for all methods: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \lambda \cdot \mathcal{L}_{\text{KD}}$

### Logit-KD

KL divergence between temperature-scaled classification logits:

$$\mathcal{L}_{\text{logit}} = T^2 \cdot \mathrm{KL}\!\left(\sigma\!\left(\tfrac{t}{T}\right) \,\Big\|\, \sigma\!\left(\tfrac{s}{T}\right)\right)$$

$T \in \{2, 4, 8\}$. Applied to the classification head only.

### Feature-KD

Encoder MSE with a learned projection layer + decoder cross-attention cosine alignment:

$$\mathcal{L}_{\text{feat}} = \mathrm{MSE}\!\left(\mathrm{proj}(s_{\text{enc}}),\, t_{\text{enc}}\right)$$

$$\mathcal{L}_{\text{attn}} = 1 - \cos\!\left(s_{\text{attn}},\, t_{\text{attn}}\right)$$

$$\mathcal{L}_{\text{KD}} = w_f \cdot \mathcal{L}_{\text{feat}} + \alpha \cdot \mathcal{L}_{\text{attn}}$$

### CWD — Channel-Wise Distillation (Yang et al., ICCV'21)

Spatially-normalized channel distributions aligned via KL divergence:

$$\mathcal{L}_{\text{CWD}} = \sum_{c=1}^{C} \mathrm{KL}\!\left(\tilde{t}_c \,\Big\|\, \tilde{s}_c\right), \qquad \tilde{t}_c = \mathrm{softmax}\!\left(\tfrac{t_c}{\tau}\right)_{\text{spatial}}$$

### Query-KD *(novel)*

Distils RT-DETR's decoder object queries — a transformer-specific signal unavailable to CNN-detector KD methods. No shared Hungarian matcher is required; alignment is over the first $\min(Q_s, Q_t)$ queries, which is robust to the 100 vs. 300 query-count mismatch between student and teacher.

$$\mathcal{L}_{\text{query}} = \mathrm{MSE}(q_s, q_t) + \alpha \cdot \left(1 - \cos\!\left(A_s^{\text{dec}},\, A_t^{\text{dec}}\right)\right)$$

**Distinction from prior work.** DETRDistill (ICLR'23) aligns *matched* query-prediction pairs after joint Hungarian assignment, which breaks when teacher and student query counts differ. MimicDet (ECCV'20) mimics RPN attention in two-stage detectors; the decoder cross-attention term here is specific to RT-DETR's encoder-memory interaction, which has no CNN analogue.

### Stage-Adaptive KD *(novel)*

Cosine curriculum shifting from feature distillation (structural alignment) to logit distillation (semantic refinement) as training progresses:

$$w_f(e) = \cos\!\left(\frac{\pi e}{2E}\right), \qquad w_l(e) = 1 - w_f(e)$$

$$\mathcal{L}_{\text{KD}}^{\text{SA}}(e) = w_f(e)\cdot\mathcal{L}_{\text{feat}} + w_l(e)\cdot\mathcal{L}_{\text{logit}}$$

where $e$ is the current epoch and $E$ is total epochs. The schedule shape (cosine / linear / step / sigmoid / inverse-cosine) is configurable.

---

## Roadmap

**Done**
- [x] Full distillation pipeline — 5 KD methods, unified loss wrapper, config-driven
- [x] Cross-architecture teacher adapter with mAP sanity gate
- [x] TensorRT FP32 / FP16 / INT8 export with entropy calibration (`tools/export_trt.py`)
- [x] FastAPI inference server with single-image and batch endpoints
- [x] DETR-style top-k decoding (fixes ~2 mAP vs. per-query argmax)
- [x] Automated results aggregation — CSV + Markdown + mean ± std (`tools/aggregate_results.py`)
- [x] CI test suite: pytest smoke tests on every push (CPU-only)

**In progress**
- [ ] Phase 2A ablation runs — 6 configs on COCO 30K subset, 36 epochs
- [ ] Attention visualization notebook (teacher vs. student cross-attention maps)

**Next**
- [ ] Phase 2D — top configurations on full COCO 118K, 72 epochs
- [ ] Phase 2E — best method × 3 seeds, mean ± std
- [ ] arXiv preprint (cs.CV)

---

## Author

**Umut Onur Yasar** — Applied AI Research Engineer  
[GitHub](https://github.com/umutonuryasar) · [LinkedIn](https://linkedin.com/in/umutonuryasar)
