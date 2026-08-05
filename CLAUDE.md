# RT-DETR Knowledge Distillation

## Project overview

**Title:** Knowledge Distillation for Efficient Real-Time Detection Transformers
**Author:** Umut Onur Yaşar
**Deliverable:** Tech report — an honest Results section in `README.md`, a deployed
demo, and a 3-part blog series. **No arXiv paper.**
**Objective:** A controlled KD study on RT-DETR — logit, feature, CWD, and two
novel RT-DETR-specific methods — plus edge deployment analysis on an RTX 3050.

**Status (2026-08-05): the experimental campaign is COMPLETE.** Phase 2A (nine
runs, seed 42) and Phase 2B (λ-swap 2×2 + three-seed repeats of four
configurations) have run; `README.md` reports the numbers and is the primary
deliverable. What remains is Phase 3: TensorRT export, FPS benchmarking, the
demo and the blog series. Do **not** re-plan or re-run the ablation.

**Scope decision (July 2026, still in force):** the earlier 18/23-run,
full-COCO, 3-seed plan (Phase 2D/2E, `run14`–`run17`, MGD in the matrix) is
**superseded**. Anything in the repo still describing that plan is stale, not a
to-do. `scripts/run_final.sh` implements it, was **never run**, produced no
reported number, and is kept only as a starting point if the project is ever
extended to full COCO.

**Architecture decision (2026-05-23):** the ablation is **own-architecture**
KD — an R50 teacher trained in this repo distilling into the simplified R18
student in `src/models/rtdetr.py` (100 queries, 3 decoder layers, vanilla MHA,
C4+C5 encoder memory), forced by the 4 GB RTX 3050 VRAM budget. Query-KD and the
cross-attention terms need post-norm decoder queries and dense `[Q, N]` attention
maps, which the canonical deformable teacher does not expose. The canonical
`lyuwenyu/RT-DETR` adapter (`src/models/rtdetr_teacher.py`, git submodule under
`third_party/RT-DETR`) is retained for an **optional** cross-architecture
comparison that was never exercised. Implementation differences are documented
in README § Architecture.

---

## Repository structure

```
rt-detr-kd/
├── README.md                    # THE deliverable: results, findings, limitations
├── CLAUDE.md                    # this file — current state, protocol, decisions
├── docs/                        # historical process record (see docs/README.md)
│   ├── README.md                # condensed index + the four result-invalidating bugs
│   ├── AUDIT.md                 # pre-campaign audit: findings, fixes, limits
│   ├── FIXES.md                 # methodology fix batch
│   ├── TRAINER_FIXES.md         # resume correctness
│   └── TECH_REPORT_PLAN.md      # the plan the campaign was run against
├── configs/
│   ├── rtdetr_r18vd_coco.yml    # student
│   ├── rtdetr_r50vd_coco.yml    # own teacher
│   ├── rtdetr_r34vd_coco.yml    # unused in the campaign
│   └── kd/
│       ├── cwd_kd.yml           # used by run 4
│       ├── query_kd.yml         # used by runs 5, 6
│       ├── stage_adaptive_kd.yml# used by runs 7, 8, 9, 10
│       └── archive/             # unused variants; see archive/README.md
├── src/                         # tested core — models, distillation, data, losses, trainer
├── tools/                       # train_kd, eval, benchmark_fps, export_trt, calibrate_lambda,
│                                # aggregate_results, make_select_split, verify_teacher_kd
├── serve/                       # FastAPI inference server
├── tests/                       # pytest, CPU-only, runs in CI
├── scripts/
│   ├── download_coco_subset.sh
│   ├── download_coco_full.sh
│   ├── run_ablation.sh          # THE campaign script — every reported number
│   └── run_final.sh             # NOT PART OF THIS STUDY — never run
├── notebooks/                   # ablation_analysis, visualize_attention, colab_training
│                                # (all three are stale — see "Known gaps")
└── runs/                        # gitignored; campaign artifacts live on Drive
```

Dependencies: **`pyproject.toml` + `uv.lock` are authoritative**; `uv sync` is
the supported install. `requirements*.txt` are non-authoritative mirrors that
exist only for the Dockerfile's layer caching. Python is pinned by
`.python-version`.

---

## Campaign — complete

| Stage | Data | Epochs | Runs | Purpose |
|-------|------|--------|------|---------|
| Teacher | Full COCO 118K | 36 | 1 | Own R50 teacher, **0.142 mAP**. Reused by every KD run. |
| Phase 2A | COCO 30K subset (27.5K train / 2.5K selection) | 36 | 9 (runs 0–8), seed 42 | The ablation |
| Phase 2B | same | 36 | runs 9–10 + seed repeats | λ-swap 2×2; seeds 43/44 on four configurations |

Identical splits, identical teacher weights and identical settings across runs;
the only thing that varies is the run's one intended variable.

**Model selection:** per-epoch evaluation and best-checkpoint selection use a
2,500-image selection split carved *from the training pool* by
`tools/make_select_split.py` (seed 42, generated once). `val2017` is evaluated
once at the end of each run and never influences checkpoint choice.

### The runs (`scripts/run_ablation.sh`)

| Run | Tag | KD type | Varies | λ |
|-----|-----|---------|--------|---|
| 0 | `run00_baseline` | none | — | — |
| 1 | `run01_logit_binary_t4` | logit | `--logit-mode binary` | 24.23 |
| 2 | `run02_logit_softmax_t4` | logit | `--logit-mode softmax` | 5.317 |
| 3 | `run03_feature` | feature | — | 6.249 |
| 4 | `run04_cwd` | cwd | — | 11.78 |
| 5 | `run05_query_hungarian` | query | `--query-matching hungarian` | 3.518 |
| 6 | `run06_query_index` | query | `--query-matching index` | 3.577 |
| 7 | `run07_stage_adaptive_cosine` | stage_adaptive | `--schedule cosine` | 6.324 |
| 8 | `run08_stage_adaptive_invcos` | stage_adaptive | `--schedule inverse_cosine` | 22.51 |
| 9 | `run09_stage_cosine_lam22` | stage_adaptive | cosine @ invcos's λ | 22.51 |
| 10 | `run10_stage_invcos_lam6` | stage_adaptive | invcos @ cosine's λ | 6.324 |

T=4 for every KD run. λ is **not** 1.0 — it is per-method calibrated (below) and
was passed to the script as environment overrides; the script's own defaults are
still 1.0 placeholders. MGD exists as a commented-out run 11 and is not part of
the reported matrix.

Phase 2B repeated runs 0, 5, 6 and 9 at seeds 43 and 44 (`SEED=` plus
`ONLY_RUNS=`), giving the README's n=3 headline table. Every other row is n=1.

A failed run records itself in `$OUTPUT_ROOT/failures.txt` and the ablation
continues; the script exits non-zero at the end if that file is non-empty.
Completed runs are skipped on re-invocation (checkpoint + `eval.log` present).

---

## Deliberate decisions — do not "fix" these

- **λ is per-method calibrated, not 1.0.** `tools/calibrate_lambda.py` measures
  median(L_det)/median(L_KD) over 20 batches at init so every KD term starts at
  detection-loss scale; raw KD magnitudes span ~1e5×, so λ=1.0 would have
  compared methods at wildly different effective strengths. Values (own R50
  teacher @ 0.142, seed 42): logit_binary 24.23, logit_softmax 5.317, feature
  6.249, cwd 11.78, query_hungarian 3.518, query_index 3.577, stage_cosine
  6.324, stage_invcos 22.51. **The calibration rule turned out to matter more
  than the method-design choices under test** — see README Findings 2 and 3.
- **LR is pinned to `--lr-head 1e-4 --lr-backbone 1e-5`.** `train_kd.py`'s
  default `lr_head=1e-3` collapses this architecture (teacher 0.027 vs 0.142)
  while the training loss keeps falling. Never run the campaign at the default.
- **Logit-KD defaults to per-class binary KL** (`logit_mode="binary"`), not
  Hinton softmax KL, because RT-DETR trains classification with sigmoid focal
  loss. Softmax mode is run 2.
- **Query-KD defaults to prediction-space Hungarian matching.** Index-wise
  truncation is run 6 — and it won; see README Finding 1.
- **`query_matching` is intentionally absent from `configs/kd/query_kd.yml`** so
  the CLI flag governs it. A `--kd-cfg` key contradicting an explicitly passed
  CLI flag raises rather than silently winning.
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

| Role | Backbone | Params (exact) | mAP | Source |
|------|----------|----------------|-----|--------|
| Student (simplified) | ResNet-18 | 15,948,564 | 0.0388–0.0676 | trained here |
| Teacher (own) | ResNet-50 | 28,869,908 | 0.142 | trained here |
| RT-DETR-L (cross-arch option, unused) | ResNet-50 | 32M | 53.1 | lyuwenyu/RT-DETR |

Both trained-here counts are ~0.5M below earlier drafts: the encoder's C3 fusion
branch produced a tensor nothing consumed, so 477,824 student / 576,128 teacher
parameters never received a gradient. Removed pre-campaign; encoder outputs are
bit-identical (`docs/AUDIT.md`, fix P-1).

---

## Training details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| LR backbone / head | **1e-5 / 1e-4** (pinned; the 1e-3 default collapses the model) |
| Weight decay | 1e-4 |
| LR schedule | Cosine + 500-iter warmup, stepped per optimizer step |
| Epochs | 36 |
| Batch size | 4 (RTX 3050) / 16 (A100) |
| Grad accumulation | 2 (RTX 3050) / 1 (A100) |
| Image size | 512 (640 OOMs on 4 GB with teacher + student) |
| AMP | fp16 |
| Teacher | Frozen, eval mode, `no_grad`, outputs detached |
| Seeds | 42 everywhere; 43/44 for the four Phase 2B repeats |

Data order and augmentation are a pure function of `(seed, epoch)`: the loaders
carry an explicit seeded `torch.Generator` and the trainer re-seeds it per epoch,
so a resumed run continues the stream an uninterrupted one would have followed.
Resumed runs are statistically, not bitwise, reproducible (`cudnn.benchmark` +
fp16).

---

## Environment

- **Local:** Ubuntu 24.04 · RTX 3050 4 GB · Ryzen 5800H · 16 GB RAM
- **Colab:** Pro+ · A100 40 GB — where the teacher and every campaign run executed
- **Python:** 3.12 (`.python-version`) · **PyTorch:** 2.5.1+cu121
- **Dependency management:** `uv` (`pyproject.toml` + `uv.lock`). Run everything
  as `uv run …`.
- **Tests:** `uv run pytest tests/ -q` — must stay green (86 passed, 2 skipped).

---

## Known gaps

- **The two Colab notebooks that drove the teacher run and the ablation are not
  in the repo** — they live only in the author's Drive. `notebooks/colab_training.ipynb`
  predates them and describes the superseded 23-run plan. Until they are
  committed, the README→script→config chain is complete but the *orchestration*
  that supplied λ, `SEED` and `ONLY_RUNS` per run is not.
- **`notebooks/ablation_analysis.ipynb` and `notebooks/visualize_attention.ipynb`
  reference the superseded run tags** (`run05_logit_l1.0_t4`,
  `run08_feature_l1.0`) and will not find the campaign's directories.
- **`runs/` is gitignored**, so no campaign `eval.log` is in version control.
  The reported numbers trace to the script and configs, not to committed logs.

---

## Status

### Done
- [x] Full pipeline — 5 KD methods in the matrix (+ MGD/combined/partial archived),
      unified loss wrapper, config-driven
- [x] Methodology fix batch (`docs/FIXES.md`) and trainer resume fixes
      (`docs/TRAINER_FIXES.md`)
- [x] Pre-campaign audit (`docs/AUDIT.md`) — see `docs/README.md` for the four
      findings that would have silently invalidated results
- [x] Cross-architecture teacher adapter with mAP sanity gate (built, unused)
- [x] TensorRT FP32/FP16/INT8 export with entropy calibration
- [x] FastAPI inference server; DETR-style top-k decoding; results aggregation
- [x] CI: pytest on every push (CPU-only)
- [x] Own R50 teacher — full COCO, 36 epochs, A100. **0.142 mAP**
- [x] λ calibration (`tools/calibrate_lambda.py`)
- [x] Phase 2A — runs 0–8, seed 42
- [x] Phase 2B — λ-swap runs 9–10; seeds 43/44 on runs 0, 5, 6, 9
- [x] Results / Findings / Limitations in README

### Next — Phase 3
- [ ] Commit the two real Colab notebooks (teacher run, ablation run)
- [ ] TensorRT FP16/INT8 latency-vs-accuracy sweep on the best checkpoint
- [ ] FPS benchmarking on the RTX 3050
- [ ] Attention visualization notebook refreshed against real campaign tags
- [ ] Hugging Face Spaces demo; blog series
