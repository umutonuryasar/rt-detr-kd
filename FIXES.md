# Fix Batch — Methodology & Correctness Revision (July 2026)

This batch addresses paper-facing methodology issues and training-correctness
bugs identified in a full-repo review, applied *before* the Phase 2A ablation
runs so results are produced under the corrected setup.

## What changed and why

### 1. Query-KD: Hungarian matching (`src/distillation/query_kd.py`)
Decoder queries have no canonical ordering — index-wise truncation to the
first min(Q_s, Q_t) queries matched semantically unrelated pairs. Default is
now per-image bipartite matching in **prediction space** (sigmoid class-prob
L1 + box L1 cost, scipy `linear_sum_assignment` under `no_grad`), with an
embedding-cosine fallback when predictions are unavailable. The old behavior
survives as `query_matching: index` strictly for the ablation defense. Also:
`teacher_queries=None` (canonical lyuwenyu teacher) now degrades to a logged
zero-loss instead of crashing.

### 2. Logit-KD: binary KL (`src/distillation/logit_kd.py`)
RT-DETR trains classification with per-class **sigmoid** focal loss; applying
categorical softmax KL imposed a distribution family the logits were never
trained under. Default `mode="binary"` computes per-class binary KL between
temperature-scaled sigmoid probabilities (numerically stable via
BCE-with-logits minus teacher entropy). `mode="softmax"` retained for
ablation.

### 3. Feature-KD / CWD: per-scale 2-D alignment
(`src/distillation/feature_kd.py`, `cwd.py`, plumbing in `kd_loss.py`,
`stage_adaptive_kd.py`, `src/models/encoder.py`, `rtdetr.py`, `rtdetr_kd.py`,
`rtdetr_teacher.py`)
Encoder sequences are concatenations of flattened scales (student C4+C5,
teacher C3+C4+C5); 1-D interpolation over the concatenated axis blended
tokens across scale boundaries. Models now expose per-scale `(H, W)` shapes;
losses split sequences per scale, pair from the coarsest end (C5↔C5, C4↔C4),
drop the teacher's extra C3, and interpolate in 2-D only within a scale when
needed. Legacy 1-D path remains as a warned fallback when shapes are absent.

### 4. CWD attribution corrected
CWD is Shu et al., "Channel-wise Knowledge Distillation for Dense
Prediction", ICCV 2021 — not Yang et al. (whose "Focal and Global KD" is a
different method, CVPR 2022). Fixed in docstrings, `kd_loss.py`, and README.

### 5. Attention/query terms with the canonical teacher — now explicit
The deformable-attention lyuwenyu teacher exposes neither dense cross-attention
maps nor post-norm decoder queries, so the attention-alignment terms (and all
of Query-KD) were silently inactive against it while the README advertised
them. `KDLoss` now logs a one-time warning when a method's advertised signal
is missing, and the README states the limitation per method.

### 6. MosaicWrapper double normalization (`src/data/transforms.py`, `tools/train_kd.py`)
The wrapper was fed an already-Normalized dataset; the mosaic path converted
normalized floats (~[-2.1, 2.6]) to uint8 (garbage) and normalized *again*.
New contract: the wrapped dataset must return raw PIL images
(`transforms=None`), and the full pipeline runs exactly once — on the single
image or on the assembled mosaic canvas. Tensor-returning datasets are
rejected with a clear error. `train_kd.py` wiring updated accordingly; as a
bonus the mosaic canvas now also receives flip/jitter augmentation.

### 7. `capture_attn` flag (`src/models/decoder.py`, `rtdetr.py`, `tools/train_kd.py`)
`need_weights=True` was hardcoded, forcing the slow attention path and
storing [L, B, H, Q, N] maps for every method including the baseline.
Attention capture is now enabled only for methods that consume it
(feature/combined/query/stage_adaptive) — VRAM/speed savings on the 4 GB
budget and a clean baseline runtime profile.

### 8. Optional model-selection split (`src/trainer_kd.py`, `tools/train_kd.py`)
Best-checkpoint selection previously used the same val set that final numbers
are reported on. With `--select-img/--select-ann` (a split carved from
training data), per-epoch eval + checkpoint selection use the selection split
and val is evaluated once at the end. Default behavior unchanged when the
flags are absent.

### 9. Hygiene
`requirements.txt`: removed `albumentations`, `timm`, `einops`, `tqdm`
(never imported). `configs/kd/query_kd.yml`: documents the new
`query_matching` key. New CLI flags: `--logit-mode`, `--query-matching`,
`--select-img`, `--select-ann`.

## Paper-facing implications for Phase 2A

- Report Logit-KD as binary-KL by default; the binary-vs-softmax comparison is
  a ready-made ablation row.
- Report Query-KD with Hungarian matching; the hungarian-vs-index comparison
  directly quantifies the matching contribution (reviewers will ask).
- State per method whether the attention term was active (own teacher only).
- Any pre-fix training runs used cross-scale-blended feature alignment and
  (if `--mosaic` was on) corrupted mosaic images — re-run, do not mix.

## Tests

`tests/test_fixes.py` adds 15 regression tests: binary-KL identity/positivity,
Hungarian permutation-invariance (and index-mode sensitivity as documented),
None-teacher degradation, per-scale alignment equivalence to direct MSE on
shared scales, single-normalization pinning for mosaic, tensor-input
rejection, and capture_attn on/off behavior. Existing tests pass unchanged.
