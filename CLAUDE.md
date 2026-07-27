# RT-DETR Knowledge Distillation

## Project overview

**Title:** Knowledge Distillation for Efficient Real-Time Detection Transformers
**Author:** Umut Onur Yaşar
**Deliverable:** Tech report — an honest Results section in `README.md`, a deployed
demo, and a 3-part blog series. **No arXiv paper.** See `TECH_REPORT_PLAN.md`,
which is the authoritative plan; this file summarises it.
**Objective:** A controlled KD study on RT-DETR — logit, feature, CWD, and two
novel RT-DETR-specific methods — plus edge deployment analysis on an RTX 3050.

**Scope decision (July 2026):** the earlier 18/23-run, full-COCO, 3-seed plan
(Phase 2D/2E, `run14`–`run17`, MGD in the matrix) is **superseded**. The
campaign is nine runs on the COCO 30K subset at a single seed. Anything in the
repo still describing the old plan is stale, not a to-do — `scripts/run_final.sh`
is explicitly out of current scope and is kept only as a starting point if the
project is ever extended to full COCO.

**Architecture decision (2026-05-23):** the main ablation is **own-architecture**
KD — an R50 teacher trained in this repo distilling into the simplified R18
student in `src/models/rtdetr.py` (100 queries, 3 decoder layers, vanilla MHA,
C4+C5 encoder memory), forced by the 4 GB RTX 3050 VRAM budget. Query-KD and the
cross-attention terms need post-norm decoder queries and dense `[Q, N]` attention
maps, which the canonical deformable teacher does not expose. The canonical
`lyuwenyu/RT-DETR` adapter (`src/models/rtdetr_teacher.py`, git submodule under
`third_party/RT-DETR`) is retained for an **optional** cross-architecture
comparison only. Implementation differences are documented in README §3.2.

---

## Repository structure

```
rt-detr-kd/
├── CLAUDE.md
├── TECH_REPORT_PLAN.md          # authoritative plan
├── AUDIT.md                     # pre-campaign audit: findings, fixes, limits
├── configs/
│   ├── rtdetr_r18vd_coco.yml    # student
│   ├── rtdetr_r50vd_coco.yml    # own teacher
│   ├── rtdetr_r34vd_coco.yml    # unused in the 9-run matrix
│   └── kd/
│       ├── cwd_kd.yml           # active
│       ├── query_kd.yml         # active
│       ├── stage_adaptive_kd.yml# active
│       └── archive/             # out-of-scope configs (mgd, combined,
│                                # partial-KD, schedule-shape variants)
├── src/
│   ├── distillation/            # logit, feature, cwd, mgd, query,
│   │                            # stage_adaptive, kd_loss (wrapper)
│   ├── models/                  # rtdetr, rtdetr_kd, rtdetr_teacher,
│   │                            # backbone, encoder, decoder
│   ├── data/                    # coco_dataset, transforms
│   ├── losses/                  # detection_loss, matcher
│   └── trainer_kd.py
├── tools/                       # train_kd, eval, benchmark_fps, export_trt,
│                                # aggregate_results, make_select_split,
│                                # verify_teacher_kd
├── serve/                       # FastAPI inference server
├── tests/                       # pytest, CPU-only, runs in CI
├── scripts/
│   ├── download_coco_subset.sh
│   ├── download_coco_full.sh
│   ├── run_ablation.sh          # THE campaign: 9 runs
│   └── run_final.sh             # OUT OF SCOPE — superseded full-COCO plan
└── runs/
```

---

## Campaign

| Stage | Data | Epochs | Runs | Purpose |
|-------|------|--------|------|---------|
| Teacher | Full COCO 118K | 36 | 1 | Own R50 teacher, ~11 h on A100. Reused by all KD runs. |
| Phase 2A | COCO 30K subset (27.5K train / 2.5K selection) | 36 | 9 | The ablation. ~3.5–4.5 h per run on A100. |

Single seed (**42**) throughout, reported openly as a limitation — no
statistical-significance claims. Identical splits, identical teacher weights and
identical settings across all nine runs; the only thing that varies is the run's
one intended variable.

**Model selection:** per-epoch evaluation and best-checkpoint selection use a
2,500-image selection split carved *from the training pool* by
`tools/make_select_split.py` (seed 42, generated once). `val2017` is evaluated
once at the end of each run and never influences checkpoint choice.

### The nine runs (`scripts/run_ablation.sh`)

| Run | KD type | Varies | Notes |
|-----|---------|--------|-------|
| 0 | none | — | Baseline |
| 1 | logit | `--logit-mode binary` | Sigmoid-matched default formulation |
| 2 | logit | `--logit-mode softmax` | Formulation ablation (Hinton categorical KL) |
| 3 | feature | — | Encoder MSE + decoder cross-attention cosine |
| 4 | cwd | — | Literature baseline (Shu et al., ICCV'21) |
| 5 | query | `--query-matching hungarian` | Novel #1 |
| 6 | query | `--query-matching index` | Matching-contribution ablation |
| 7 | stage_adaptive | `--schedule cosine` | Novel #2 |
| 8 | stage_adaptive | `--schedule inverse_cosine` | Curriculum-**direction** control |

λ=1.0 and T=4 for every KD run. Run 9 (MGD) exists commented out in the script as
an optional extra baseline and is not part of the reported matrix.

A failed run records itself in `$OUTPUT_ROOT/failures.txt` and the ablation
continues; the script exits non-zero at the end if that file is non-empty.
Completed runs are skipped on re-invocation (checkpoint + `eval.log` present), so
a Colab session drop costs only the run that was in flight.

---

## Deliberate decisions — do not "fix" these

- **Logit-KD defaults to per-class binary KL** (`logit_mode="binary"`), not
  Hinton softmax KL, because RT-DETR trains classification with sigmoid focal
  loss. Softmax mode is retained deliberately as run 2.
- **The two logit modes are not λ-matched** (binary averages over classes,
  softmax sums over them). Both are canonical for their own formulation; this is
  deferred to a λ-calibration pass once the teacher exists — see AUDIT.md P-3.
- **Query-KD defaults to prediction-space Hungarian matching.** Index-wise
  truncation is retained deliberately as run 6.
- **`query_matching` is intentionally absent from `configs/kd/query_kd.yml`** so
  the CLI flag governs it. A `--kd-cfg` key that contradicts an explicitly passed
  CLI flag now raises rather than silently winning.
- **Feature-KD/CWD align encoder tokens per scale in 2-D**, pairing scales from
  the coarsest end and dropping any extra teacher fine scale.
- **The student is deliberately simplified** (100 queries, 3 decoder layers,
  vanilla MHA, C4+C5 encoder memory) for the 4 GB budget.
- **`capture_attn` is enabled only for KD types that consume attention**
  (`feature`, `combined`, `query`, `stage_adaptive`) and is threaded into the
  teacher config as well as the student's — one setting, not two.
- **Model-selection-split logic in the trainer, and the rolling
  `checkpoint_latest.pth`**, are intentional.

---

## Distillation formulation

```
L_total = L_det + λ · L_KD

Logit-KD (binary):  L_KD = T² · mean_{b,q,c} KL_bin( σ(t/T) ‖ σ(s/T) )
Logit-KD (softmax): L_KD = T² · KL( softmax(t/T) ‖ softmax(s/T) )     [ablation]
Feature-KD:         L_KD = feat_weight · MSE(proj(s_enc), t_enc) + α · (1 - cos(s_attn, t_attn))
CWD:                L_KD = (τ²/C) · Σ_c KL( softmax_spatial(t_c/τ) ‖ softmax_spatial(s_c/τ) )
Query-KD:           L_KD = MSE(q_s^i, q_t^π(i)) + α · (1 - cos(A_s^dec, A_t^dec))
Stage-Adaptive:     L_KD = w_f(e) · L_feat + w_l(e) · L_logit,  w_f(e) = cos(πe/2E)
```

---

## Models

| Role | Backbone | Params (exact) | Source |
|------|----------|----------------|--------|
| Student (simplified) | ResNet-18 | 15,948,564 | trained here |
| Teacher (own, main ablation) | ResNet-50 | 28,869,908 | trained here |
| RT-DETR-L (cross-arch option) | ResNet-50 | 32M | lyuwenyu/RT-DETR, 53.1 mAP |

Both trained-here counts are ~0.5M below earlier drafts: the encoder's C3 fusion
branch produced a tensor nothing consumed, so 477,824 student / 576,128 teacher
parameters never received a gradient. Removed pre-campaign; encoder outputs are
bit-identical (AUDIT.md, fix P-1).

---

## Training details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| LR backbone / head | 1e-4 / 1e-3 |
| Weight decay | 1e-4 |
| LR schedule | Cosine + 500-iter warmup, stepped per optimizer step |
| Epochs | 36 |
| Batch size | 4 (RTX 3050) / 16 (A100) |
| Grad accumulation | 2 (RTX 3050) / 1 (A100) |
| Image size | 512 (640 OOMs on 4 GB with teacher + student) |
| AMP | fp16 |
| Teacher | Frozen, eval mode, `no_grad`, outputs detached |
| Seed | 42 (single) |

Data order and augmentation are a pure function of `(seed, epoch)`: the loaders
carry an explicit seeded `torch.Generator` and the trainer re-seeds it per epoch,
so a resumed run continues the stream an uninterrupted one would have followed.

---

## Environment

- **Local:** Ubuntu 24.04 · RTX 3050 4 GB · Ryzen 5800H · 16 GB RAM
- **Colab:** Pro+ · A100 40 GB
- **Python:** 3.12 · **PyTorch:** 2.5.1+cu121
- **Dependency management:** `uv`. Run everything as `uv run …`.
- **Tests:** `uv run pytest tests/ -q` — must stay green.

---

## Status

### Done
- [x] Full pipeline — 5 KD methods in the matrix (+ MGD/combined/partial archived),
      unified loss wrapper, config-driven
- [x] Methodology fix batch (`FIXES.md`) — prediction-space Hungarian query
      matching, sigmoid-matched binary logit KL, per-scale 2-D feature alignment,
      mosaic single-normalization, leakage-free checkpoint selection
- [x] Pre-campaign audit (`AUDIT.md`) — CWD reduction, cross-run data-order
      divergence, silent resume-restart, CLI/YAML precedence, LR-schedule
      overshoot, RNG persistence, dead C3 branch, CWD τ², teacher `capture_attn`,
      ablation failure isolation
- [x] Cross-architecture teacher adapter with mAP sanity gate
- [x] TensorRT FP32/FP16/INT8 export with entropy calibration
- [x] FastAPI inference server; DETR-style top-k decoding; results aggregation
- [x] CI: pytest on every push (CPU-only)

### Next
- [ ] Own R50 teacher training — full COCO, 36 epochs, A100 (single run)
- [ ] Record teacher mAP, then set `--teacher-min-map` to a realistic gate
      before launching the ablation
- [ ] Phase 2A — 9 runs, `bash scripts/run_ablation.sh`
- [ ] Attention visualization notebook (teacher vs. student)
- [ ] Results / Findings / Limitations in README
- [ ] TensorRT FP16/INT8 latency-vs-accuracy sweep on the best checkpoint
- [ ] Hugging Face Spaces demo; blog series
