# RT-DETR Knowledge Distillation

## Project overview

**Title:** Knowledge Distillation for Efficient Real-Time Detection Transformers  
**Author:** Umut Onur Yaşar  
**Target:** arXiv cs.CV (1–2 months)  
**Objective:** Systematic KD study on RT-DETR — logit, feature, combined, partial, CWD/MGD baselines, and novel RT-DETR-specific methods — with edge deployment analysis on RTX 3050.

---

## Repository structure

```
rt_detr_kd/
├── CLAUDE.md
├── configs/
│   ├── rtdetr_r18vd_coco.yml
│   ├── rtdetr_r50vd_coco.yml
│   ├── rtdetr_r34vd_coco.yml
│   └── kd/
│       ├── logit_kd.yml
│       ├── feature_kd.yml
│       ├── combined_kd.yml
│       ├── encoder_only_kd.yml
│       ├── attention_only_kd.yml
│       ├── cwd_kd.yml                  # [NEW] CWD baseline
│       ├── mgd_kd.yml                  # [NEW] MGD baseline
│       ├── query_kd.yml                # [NEW] Novel: query distillation
│       └── stage_adaptive_kd.yml       # [NEW] Novel: curriculum weighting
├── src/
│   ├── distillation/
│   │   ├── logit_kd.py
│   │   ├── feature_kd.py
│   │   ├── cwd.py                      # [NEW]
│   │   ├── mgd.py                      # [NEW]
│   │   ├── query_kd.py                 # [NEW]
│   │   ├── stage_adaptive_kd.py        # [NEW]
│   │   ├── kd_loss.py
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
│   └── export_trt.py
├── notebooks/
│   ├── ablation_analysis.ipynb
│   ├── visualize_attention.ipynb
│   └── colab_training.ipynb
├── scripts/
│   ├── download_coco_subset.sh
│   ├── download_coco_full.sh           # [NEW]
│   ├── run_ablation.sh                 # Phase 2A: 18 runs
│   └── run_final.sh                    # [NEW] Phase 2D: final paper runs
└── runs/
    └── run00_baseline/
```

---

## Training strategy (two-phase)

| Phase | Dataset | Epochs | Runs | Purpose |
|-------|---------|--------|------|---------|
| **2A** | COCO 30K subset | 36 | 18 | Hyperparameter search, method selection |
| **2D** | Full COCO 118K | 72 | ~8 | Final paper numbers |
| **2E** | Full COCO 118K | 72 | 3 seeds | Statistical reliability (mean ± std) |

After Phase 2A completes, the top 5 configurations are selected and re-trained in Phase 2D on full COCO for 72 epochs. Baseline + CWD + MGD are always included in Phase 2D.

---

## Phase 1 — Code & infra tasks

### 1. lr_scheduler bug fix ✅
`optimizer.step()` before `lr_scheduler.step()` — code was already correct, no change needed.

---

### 2. Combined KD config ✅
**File:** `configs/kd/combined_kd.yml`

```yaml
kd_type: combined
kd_lambda: 1.0
logit_weight: 0.5
feature_weight: 0.5
temperature: 4
alpha: 0.5
```

---

### 3. Partial KD configs ✅
**File:** `configs/kd/encoder_only_kd.yml`
```yaml
kd_type: feature
kd_lambda: 1.0
alpha: 0.0
```

**File:** `configs/kd/attention_only_kd.yml`
```yaml
kd_type: feature
kd_lambda: 1.0
alpha: 1.0
feat_weight: 0.0
```

`feat_weight` parameter added to `feature_kd.py`.

---

### 4. Teacher capacity config ✅
**File:** `configs/rtdetr_r34vd_coco.yml` — copied from R50 config with backbone changed to ResNet-34.

---

### 5. TensorRT export script skeleton ✅
**File:** `tools/export_trt.py` — to be completed in Phase 3.

---

### 6. run_ablation.sh update ✅
Updated to 14 runs.

---

### 7. Colab notebook skeleton ✅
**File:** `notebooks/colab_training.ipynb`

---

### 8. CWD implementation [NEW]
**File:** `src/distillation/cwd.py`

Channel-Wise Distillation (Yang et al., ICCV'21). Normalizes encoder feature maps channel-wise with softmax and applies KL divergence.

```python
class CWDLoss(nn.Module):
    def __init__(self, student_channels, teacher_channels, tau=1.0):
        super().__init__()
        self.tau = tau
        self.align = nn.Conv2d(student_channels, teacher_channels, 1)

    def forward(self, s_feat, t_feat):
        s_feat = self.align(s_feat)
        N, C, H, W = s_feat.shape
        s_norm = F.softmax(s_feat.view(N, C, -1) / self.tau, dim=-1)
        t_norm = F.softmax(t_feat.view(N, C, -1) / self.tau, dim=-1)
        loss = F.kl_div(s_norm.log(), t_norm, reduction='batchmean')
        return loss
```

**Config:** `configs/kd/cwd_kd.yml`
```yaml
kd_type: cwd
kd_lambda: 1.0
tau: 1.0
```

**run14** will use this config.

---

### 9. MGD implementation [NEW]
**File:** `src/distillation/mgd.py`

Masked Generative Distillation (Yang et al., ECCV'22). Randomly masks student features and trains a lightweight generator to reconstruct teacher features.

```python
class MGDLoss(nn.Module):
    def __init__(self, student_channels, teacher_channels, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.align = nn.Conv2d(student_channels, teacher_channels, 1)
        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(teacher_channels, teacher_channels, 3, padding=1),
        )

    def forward(self, s_feat, t_feat):
        s_feat = self.align(s_feat)
        N, C, H, W = s_feat.shape
        mask = torch.rand(N, 1, H, W, device=s_feat.device) > self.mask_ratio
        masked = s_feat * mask.float()
        generated = self.generation(masked)
        loss = F.mse_loss(generated, t_feat)
        return loss
```

**Config:** `configs/kd/mgd_kd.yml`
```yaml
kd_type: mgd
kd_lambda: 1.0
mask_ratio: 0.75
```

**run15** will use this config.

---

### 10. Query-KD implementation [NEW]
**File:** `src/distillation/query_kd.py`

Novel RT-DETR-specific contribution. Distills the decoder object queries (300×256 tensor) between teacher and student. Two components:
1. Query embedding MSE
2. Decoder cross-attention pattern cosine similarity

```python
class QueryKDLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, s_queries, t_queries, s_dec_attn=None, t_dec_attn=None):
        loss_query = F.mse_loss(s_queries, t_queries)
        loss = loss_query
        if s_dec_attn is not None and t_dec_attn is not None:
            cos_sim = F.cosine_similarity(
                s_dec_attn.flatten(2), t_dec_attn.flatten(2), dim=-1
            )
            loss_attn = (1 - cos_sim).mean()
            loss = loss_query + self.alpha * loss_attn
        return loss
```

**Note:** `rtdetr_kd.py` needs to expose decoder query and attention outputs.

**Config:** `configs/kd/query_kd.yml`
```yaml
kd_type: query
kd_lambda: 1.0
alpha: 0.5          # decoder cross-attention weight
```

**run16** will use this config.

---

### 11. Stage-Adaptive KD implementation [NEW]
**File:** `src/distillation/stage_adaptive_kd.py`

Novel contribution: automatically transitions from feature-KD to logit-KD across training via a cosine schedule.

```python
class StageAdaptiveKDLoss(nn.Module):
    def __init__(self, feature_loss, logit_loss, total_epochs):
        super().__init__()
        self.feature_loss = feature_loss
        self.logit_loss = logit_loss
        self.total_epochs = total_epochs

    def forward(self, epoch, *args, **kwargs):
        w_feat = math.cos(math.pi * epoch / (2 * self.total_epochs))
        w_logit = 1.0 - w_feat
        l_feat = self.feature_loss(*args, **kwargs)
        l_logit = self.logit_loss(*args, **kwargs)
        return w_feat * l_feat + w_logit * l_logit
```

Called in `trainer_kd.py` as `kd_loss.forward(epoch=current_epoch, ...)`.

**Config:** `configs/kd/stage_adaptive_kd.yml`
```yaml
kd_type: stage_adaptive
kd_lambda: 1.0
temperature: 4       # for the logit component
alpha: 0.5           # attention weight within the feature component
```

**run17** will use this config.

---

### 12. run_final.sh [NEW]
**File:** `scripts/run_final.sh`

To be written after Phase 2A completes. Runs the top 5 configs + Baseline + CWD + MGD on full COCO for 72 epochs with 3 seeds.

---

### 13. download_coco_full.sh [NEW]
**File:** `scripts/download_coco_full.sh`

Download script for full COCO (118K train + 5K val).

---

## Full run plan

### Phase 2A — COCO 30K subset, 36 epochs (18 runs)

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
| 16 | Query-KD | 1.0 | — | Novel: query distillation |
| 17 | Stage-Adaptive | 1.0 | — | Novel: curriculum weighting |

Estimated A100 time for Phase 2A: ~50–60 hours.

### Phase 2D — Full COCO, 72 epochs (~8 runs)

Selected after Phase 2A. Expected: run00 (baseline), run08 (best feature), run14 (CWD), run15 (MGD), best logit, run16 or run17 (best novel).

Estimated A100 time for Phase 2D: ~80–100 hours.

### Phase 2E — Statistical reliability

Best method × 3 seeds, full COCO, 72 epochs.  
Estimated A100 time for Phase 2E: ~30–40 hours.

**Total estimated A100 time: ~160–200 hours.**

---

## Paper tables (target outputs)

**Table 1 — Main ablation (run00–09):**
KD Type | λ | T | mAP@[.5:.95] | ΔmAP vs baseline | FPS | VRAM (GB)

**Table 2 — Partial KD contribution analysis (run10–11):**
Method | L_feat | L_attn | mAP | ΔmAP vs Feature-KD

**Table 3 — SOTA KD comparison (run08, 14, 15 — full COCO):**
Method | Venue | mAP | ΔmAP vs baseline | FPS

**Table 4 — Novel contributions (run16, 17 — full COCO):**
Method | Component | mAP | ΔmAP vs Feature-KD | Overhead

**Table 5 — Teacher capacity (run12, 13):**
Teacher backbone | Teacher mAP | Student mAP (KD) | mAP gap

**Table 6 — Edge deployment:**
Model | Precision | Latency (ms) | Throughput (FPS) | mAP

---

## Distillation formulation

```
L_total = L_det + λ · L_KD

Logit-KD:        L_KD = T² · KL( softmax(t/T) ‖ softmax(s/T) )
Feature-KD:      L_KD = feat_weight · MSE(proj(s_enc), t_enc) + α · (1 - cos_sim(s_attn, t_attn))
Combined-KD:     L_KD = logit_weight · L_logit + feature_weight · L_feature
CWD:             L_KD = Σ_c KL( softmax_spatial(t_c) ‖ softmax_spatial(s_c) )
MGD:             L_KD = ||G(mask(s_feat)) - t_feat||²
Query-KD:        L_KD = MSE(q_s, q_t) + α · (1 - cos_sim(A_s^dec, A_t^dec))
Stage-Adaptive:  L_KD = w_f(e) · L_feat + w_l(e) · L_logit,  w_f(e) = cos(πe/2E)
```

---

## Models

| Model | Backbone | Params | mAP (COCO) | FPS (T4) |
|-------|----------|--------|------------|----------|
| RT-DETR-S (student) | ResNet-18 | 17M | 48.9 | ~120 |
| RT-DETR-M (teacher) | ResNet-34 | 25M | 51.3 | ~117 |
| RT-DETR-L (teacher) | ResNet-50 | 32M | 53.1 | ~114 |

Pretrained weights: [PaddleDetection RT-DETR](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr)

---

## Training details

| Hyperparameter | Ablation (2A) | Final (2D/2E) |
|----------------|--------------|--------------|
| Dataset | COCO 30K | Full COCO 118K |
| Epochs | 36 | 72 |
| Optimizer | AdamW | AdamW |
| LR backbone | 1e-4 | 1e-4 |
| LR transformer head | 1e-3 | 1e-3 |
| Weight decay | 1e-4 | 1e-4 |
| LR schedule | Cosine + 500-iter warmup | Cosine + 500-iter warmup |
| Batch size | 4 (RTX 3050) / 16 (A100) | 16 (A100) |
| Grad accumulation | 2 (RTX) / 1 (A100) | 1 (A100) |
| Image size | 640×640 | 640×640 |
| AMP | fp16 | fp16 |
| Teacher | Frozen, eval mode | Frozen, eval mode |
| Seeds | 1 | 3 (mean ± std) |

---

## Environment

- **Local:** Ubuntu 24.04.3 · RTX 3050 4GB · Ryzen 5800H · 16GB RAM
- **Colab:** Pro+ · A100 40GB
- **Python:** 3.12
- **PyTorch:** 2.x + CUDA 12.x

---

## Phase checklist

### Phase 1 — Code & infra
- [x] lr_scheduler bug fix
- [x] `configs/kd/combined_kd.yml`
- [x] `configs/kd/encoder_only_kd.yml`
- [x] `configs/kd/attention_only_kd.yml`
- [x] `feature_kd.py` → `feat_weight` parameter
- [x] `configs/rtdetr_r34vd_coco.yml`
- [x] `tools/export_trt.py` skeleton
- [x] `scripts/run_ablation.sh` 14 runs
- [x] `notebooks/colab_training.ipynb` skeleton
- [x] `src/distillation/cwd.py` implement
- [x] `configs/kd/cwd_kd.yml`
- [x] `src/distillation/mgd.py` implement
- [x] `configs/kd/mgd_kd.yml`
- [x] `src/distillation/query_kd.py` implement
- [x] `configs/kd/query_kd.yml`
- [x] expose decoder query/attention outputs in `rtdetr_kd.py`
- [x] `src/distillation/stage_adaptive_kd.py` implement
- [x] `configs/kd/stage_adaptive_kd.yml`
- [x] `kd_loss.py` → integrate CWD, MGD, Query-KD, Stage-Adaptive
- [x] `scripts/run_ablation.sh` update to 18 runs
- [x] `scripts/download_coco_full.sh`
- [x] `scripts/run_final.sh` skeleton

### Phase 2A — Ablation (30K subset, 36 epochs)
- [ ] run00 baseline
- [ ] run01–06 logit-KD
- [ ] run07–08 feature-KD
- [ ] run09 combined-KD
- [ ] run10 encoder-only
- [ ] run11 attention-only
- [ ] run12 teacher=R34
- [ ] run13 teacher=R50
- [ ] run14 CWD
- [ ] run15 MGD
- [ ] run16 Query-KD
- [ ] run17 Stage-Adaptive
- [ ] Save attention maps for each run
- [ ] Select top 5 configs → prepare Phase 2D list

### Phase 2D — Final (Full COCO, 72 epochs)
- [ ] Complete `scripts/run_final.sh` (after Phase 2A)
- [ ] Run selected ~8 configurations
- [ ] Compare results against baseline

### Phase 2E — Statistical reliability
- [ ] Run best method × 3 seeds
- [ ] Compute mean ± std

### Phase 3 — Analysis & deployment
- [ ] Complete `tools/export_trt.py` (ONNX → TRT INT8)
- [ ] FP32 / FP16 / INT8 latency table (RTX 3050 + T4)
- [ ] COCO category-level AP analysis (small/medium/large breakdown)
- [ ] `visualize_attention.ipynb`: teacher vs student attention comparison (4–5 figures)
- [ ] Teacher capacity curve plot
- [ ] `ablation_analysis.ipynb`: all tables + Pareto plot (paper-ready)

### Phase 4 — Paper & submission
- [ ] Write paper (LaTeX / Overleaf, arxiv.sty)
- [ ] Correctly cite CWD and MGD references
- [ ] GitHub cleanup: description, topics, README figures, LICENSE
- [ ] arXiv submit
- [ ] Update portfolio + LinkedIn
