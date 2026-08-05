# Pre-campaign audit — RT-DETR KD

> **Historical document — kept verbatim as evidence.** Written before the
> campaign ran; statements about what *will* happen, open action items, and the
> test count below describe the repo as of 2026-07-27. The campaign has since
> completed. Current state: [`../README.md`](../README.md) (results) and
> [`../CLAUDE.md`](../CLAUDE.md) (protocol). Condensed index:
> [`README.md`](README.md).
>
> Two action items this document left open were closed by the campaign:
> `--teacher-min-map` is now `0.10` in `scripts/run_ablation.sh` (the own R50
> teacher scored 0.142), and P-3 was folded into the λ-calibration pass
> (`tools/calibrate_lambda.py`).

**Date:** 2026-07-27 · **revised** after proposal review (second batch: P-1, P-2, P-4, P-5 applied)
**Scope:** everything that runs during the ~11 h full-COCO teacher training and the nine 27.5K-subset ablation runs.
**Test suite:** `uv run pytest tests/ -q` → **85 passed, 2 skipped** (was 47 passed, 2 skipped before the audit; 38 regression tests in `tests/test_audit_regressions.py`).

Every P0/P1 fix below has a regression test that **fails against the pre-fix code and passes after**; the verification method for each is stated explicitly, and where the "fails before" could not be shown by simply reverting (the test imports a function that did not previously exist) the old behaviour was reproduced directly and the result is quoted.

**Review outcome.** P-1, P-2, P-4 and P-5 were accepted and are now applied — they have moved into *Fixes* below with their verification evidence. **P-3 remains open and deliberately deferred** (see the proposals section). The two documentation items previously marked accepted-as-is (`CLAUDE.md` / `scripts/run_final.sh` stale plan; README parameter counts) were also corrected.

---

## Findings

| Sev | Location | Finding | Status |
|-----|----------|---------|--------|
| P0 | `src/distillation/cwd.py:118` | KL sum divided by `B` instead of `B*D` — CWD loss inflated ~256×, making λ=1.0 mean something ~256× larger than for every other method | **fixed** |
| P0 | `tools/train_kd.py` (resume block) | Bare `except Exception` swallowed any checkpoint-load failure and silently restarted training from epoch 1, discarding completed epochs + optimizer/scheduler/scaler state. Reproduced live during this audit | **fixed** |
| P1 | `tools/train_kd.py` (DataLoader) | Training data order was seeded from the *global* RNG at iterator creation, so it depended on how much randomness each KD method's construction consumed. The nine runs saw **different data** | **fixed** |
| P1 | `tools/train_kd.py:build_cfg_from_args` | A `--kd-cfg` YAML key silently overrode an explicitly passed CLI flag (the `query_matching` trap, structurally still open) | **fixed** |
| P1 | `src/trainer_kd.py:build_lr_scheduler` | Cosine LR restarts upward past `total_iters` (which is an estimate) | **fixed** |
| P1 | `src/trainer_kd.py` save/load | Resume restored an inconsistent state: no RNG state persisted, and the loader generator restarted at epoch 1's ordering | **fixed** |
| P2 | `tools/benchmark_fps.py:92,104` | `--fp16` measured fp32 — input was cast back with `.float()` on every call | **fixed** |
| P2 | `tools/benchmark_fps.py:main` | Model built with `capture_attn=True`, so FPS was measured off `nn.MultiheadAttention`'s fused fast path — a number no deployed model sees, and inconsistent with the TensorRT table | **fixed** |
| P2 | `src/trainer_kd.py:evaluate` | Did not clamp `x0/y0 ≥ 0` while `tools/eval.py` and the teacher gate do — selection mAP and reported mAP used different post-processing | **fixed** |
| P2 | `README.md:154` | CWD equation omitted the `1/C` normalizer and the reference's τ² factor | **fixed** |
| P2 | `README.md:146` | Claimed "teacher C3+C4+C5"; the own-architecture R50 teacher uses C4+C5, same as the student | **fixed** |
| P2 | `src/models/encoder.py:220` | Docstring claimed "Return concatenation of all 3 encoded scales"; only 2 are returned | **fixed (doc)** — superseded by P-1, which removed the dead branch outright |
| P2 | `src/distillation/stage_adaptive_kd.py:111` | Docstring said epoch is 0-indexed; `KDTrainer` passes 1-indexed | **fixed (doc)** |
| P1 | `src/models/encoder.py` | `c3_fused` computed and discarded → `fusion_c3` + `input_proj[0]` never receive gradients. 477,824 dead params in the student, 576,128 in the teacher; ~15% of forward time | **fixed (P-1)** |
| P2 | `src/distillation/cwd.py` | Reference CWD scales by τ²; omitted here | **fixed (P-2)** |
| P2 | `tools/train_kd.py` | `capture_attn` was threaded into the student config only; the teacher always captured attention maps | **fixed (P-4)** |
| P2 | `scripts/run_ablation.sh` | `set -e` aborted the entire ablation if any single run failed | **fixed (P-5)** |
| P2 | `README.md` params columns | Student listed as 17M and the own teacher as 32M (the canonical RT-DETR-L count, not this repo's teacher) | **fixed (doc)** |
| P2 | `CLAUDE.md`, `scripts/run_final.sh` | Described the superseded 18/23-run plan (run14–run17, MGD, 3 seeds, full-COCO Phase 2D/2E) | **fixed (doc)** |
| P1 | `src/distillation/logit_kd.py:109,122` | `binary` averages over B·Q·C, `softmax` uses `batchmean` (sum over C) — run01 and run02 are not λ-matched | **deferred (P-3)** |
| P2 | `configs/*.yml` | `train:` / `data:` / `checkpoint:` blocks are unconditionally overwritten by CLI values — dead config | **accepted-as-is** |
| P2 | `src/models/rtdetr.py:145` | `build_rtdetr` ignores `pretrained_backbone`, `freeze_bn`, `freeze_stages`, `num_csp_blocks` | **accepted-as-is** |
| P2 | `src/distillation/mgd.py:86` | Uses legacy 1-D interpolation over the concatenated token axis, not per-scale 2-D alignment | **accepted-as-is** |
| P2 | `src/distillation/cwd.py:59` | CWD always builds a learnable 1×1 align conv even when `D_s == D_t`, where Feature-KD/Query-KD use `nn.Identity` | **accepted-as-is** |

---

## Fixes

### P0-1 — CWD loss inflated ~256× (`src/distillation/cwd.py`)

**What was wrong.** The KL was accumulated with `reduction="sum"` over the `[B*D, N]` view and then divided by `B` alone. Shu et al. normalize by the channel count (`L = τ²/C · Σ_c Σ_n ...`), so the implemented loss was `D` = `hidden_dim` = 256× too large. The comment above the line asserted the opposite — that dividing by `B*D` would make the loss "256 times too small".

**Why it mattered.** CWD is one of the nine runs and the only literature baseline in the matrix. At λ=1.0 the detection loss would have been swamped: the smoke logs show it directly.

```
runs/smoke/cwd.log:  Epoch 1 [0/50] loss=1247.9194  det=5.9632  kd=1241.9563
```

The KD term was 208× the detection term; the run would have optimized almost purely for channel-distribution matching and the CWD-vs-everything comparison would have measured λ, not the method. After the fix, on an equivalent tiny run: `loss=10.81 det=6.89 kd=3.93` — the same order as Feature-KD, Query-KD and Logit-KD.

**Regression test.** `test_cwd_kl_is_normalized_by_batch_and_channels` pins the alignment conv to an exact identity, computes the summed KL in closed form, and asserts the returned value equals `summed/(B*D)` *and* is **not** `summed/B`. `test_cwd_matches_batchmean_reduction` cross-checks against torch's own `batchmean` at τ=2.

**Fails-before verification.** The pre-fix `CWDLoss` was loaded from `git show HEAD:src/distillation/cwd.py` and run against the test's inputs:
```
OLD CWD: got=2.879000  test_expects(B*D)=0.719750  old(B)=2.879000  ratio=4.0x   (D=4 in the test)
  test FAILS on old code: True
```

The τ² deviation was flagged separately as **P-2** and has since been applied — see *Fixes from the accepted proposals* below.

---

### P0-2 — a failed resume silently restarted training from epoch 1 (`tools/train_kd.py`)

**What was wrong.** The resume path was:

```python
try:
    ckpt_probe = torch.load(...)
    if isinstance(ckpt_probe, dict) and "epoch" in ckpt_probe:
        start_epoch = trainer.load_checkpoint(args.student_weights)
    ...
except Exception as exc:
    logger.warning(f"Could not fully load checkpoint for resume: {exc}. "
                   "Model weights were loaded earlier; continuing from epoch 1.")
```

Any failure inside `load_checkpoint` — a malformed checkpoint, a dtype problem, a missing key, a disk error — was caught, logged at WARNING, and training continued **from epoch 1** with a fresh optimizer, a fresh scheduler and a fresh AMP scaler, while the model kept whatever weights the earlier `strict=False` load had put in. The run would look healthy in the log.

**Why it mattered.** This is the exact failure mode the Colab workflow depends on not happening. `run_ablation.sh` restarts dropped runs, and `checkpoint_latest.pth` exists specifically so a 36-epoch run resumes where it stopped. A silent restart burns hours of A100 time and, worse, produces a checkpoint whose LR schedule and optimizer moments do not correspond to its epoch count.

**Observed live, not hypothetically.** While validating the RNG-persistence fix (P1-4), `torch.load(map_location=cuda)` moved the saved RNG tensors onto the GPU, `torch.cuda.set_rng_state_all` rejected them, and the run reported:

```
[WARNING] train_kd: Could not fully load checkpoint for resume: RNG state must be a
torch.ByteTensor. Model weights were loaded earlier; continuing from epoch 1.
[INFO] train_kd: Starting from epoch 1
[INFO] src.trainer_kd: Epoch 1/4 ...
```

Two completed epochs were silently retrained. Both defects are fixed: the RNG tensors are normalised back to CPU `ByteTensor`s on load, and the swallow is gone.

**Fix.** The block is extracted into `resume_if_checkpoint(trainer, student_weights)`. A file *without* an `"epoch"` key keeps the graceful path (backbone-only weights are a legitimate input and must not abort). A file *with* an `"epoch"` key is a resume request, and any failure now propagates.

**Regression tests.** `test_failed_resume_raises_instead_of_restarting` (trainer whose `load_checkpoint` raises the exact live error → must raise, not return 0), `test_successful_resume_returns_checkpoint_epoch`, `test_bare_state_dict_still_starts_from_epoch_one` (the graceful path must survive), `test_missing_weights_path_is_not_a_resume`, `test_cuda_mapped_rng_state_restores`.

**Fails-before verification.** The pre-fix code returned `start_epoch = 0` on this input rather than raising — reproduced end-to-end above (log quoted). Post-fix, the same two-phase run gives:
```
[INFO] src.trainer_kd: Loaded checkpoint from epoch 2: .../checkpoint_latest.pth
[INFO] train_kd: Starting from epoch 3
Epoch 3/4 loss_total=9.9316   (uninterrupted reference: 9.9523)
Epoch 4/4 loss_total=9.5570   (uninterrupted reference: 9.5455)
```

---

### P1-1 — the nine runs trained on different data orders (`tools/train_kd.py`)

**What was wrong.** `train_loader` was constructed with `generator=None`. A shuffling `DataLoader` then seeds its `RandomSampler` — and every worker's augmentation seed — from the **global** torch RNG *at iterator-creation time*. That state depends on how much randomness the process consumed beforehand, which differs per KD method:

- baseline builds no teacher at all,
- `CWDLoss` / `MGDLoss` build a Xavier-initialised `Conv1d`,
- `FeatureKDLoss` / `QueryKDLoss` build `nn.Identity` (no draws),
- the optional teacher mAP gate iterates `val_loader`, consuming one more draw.

So `--seed 42` did not pin the data stream. It pinned model init only.

**Why it mattered.** The protocol requires that two runs differ only in the intended variable. They did not: they differed in the training data order and in every augmentation decision. This is confirmed by the pre-fix smoke logs — the *first batch of the first epoch*, before any training has occurred, already shows different detection losses, which is only possible if the images differ:

```
runs/smoke/baseline.log       Epoch 1 [0/50] det=7.2332
runs/smoke/logit_binary.log   Epoch 1 [0/50] det=7.4966
runs/smoke/feature.log        Epoch 1 [0/50] det=7.4956
runs/smoke/cwd.log            Epoch 1 [0/50] det=5.9632     ← different batch entirely
```

**Fix.** `make_loader_generator(seed)` builds an explicit `torch.Generator` seeded from `--seed`, passed to the train/val/select loaders together with `seed_worker` (which re-seeds python's `random` and numpy inside each worker from `torch.initial_seed()`, since the augmentation pipeline draws from python's `random`). `KDTrainer` additionally re-seeds the generator to `seed + epoch` at the start of every epoch, which makes epoch *N*'s order a pure function of `(seed, N)` — see P1-4.

**Verified live.** Same seven KD types, same tiny split, after the fix:

```
baseline      det=6.8869      query_hung    det=6.8873
logit         det=6.8869      query_index   det=6.8873
cwd           det=6.8871      stage_cos     det=6.8873
feature       det=6.8873
```

Spread collapsed from ~1.5 to 4e-4. The residual is fp16/cuDNN nondeterminism plus the deliberate `capture_attn` difference (attention-consuming methods take `nn.MultiheadAttention`'s math path rather than the fused path), not a data difference.

**Regression tests.** `test_data_order_independent_of_prior_rng_consumption` (burning 100 000 global draws must not change the epoch order), `test_data_order_still_depends_on_seed` (the fix pins order to the seed, it does not make it constant), `test_unseeded_loader_demonstrates_the_bug` (asserts the *old* construction **does** change order — this is the fails-before proof, kept in-suite so the bug cannot silently return), `test_seed_worker_makes_python_and_numpy_streams_deterministic`.

---

### P1-2 — a KD YAML key could silently override an explicit CLI flag (`tools/train_kd.py`)

**What was wrong.** The `--kd-cfg` merge runs *after* argument parsing and unconditionally assigned YAML values over the parsed ones. `configs/kd/query_kd.yml` carries a comment explaining that `query_matching` is deliberately omitted for exactly this reason — the workaround was in place, the trap was not.

**Why it mattered.** Runs 5/6 differ only in `--query-matching`, runs 7/8 only in `--schedule`, runs 1/2 only in `--logit-mode`. A single added YAML key would have collapsed a pair into two identical runs, and nothing in the log would have said so.

**Fix.** `_apply_kd_cfg_overrides()` compares each YAML key against the corresponding CLI flag. Overriding a flag left at its default is logged at INFO (`KD config override: tau: 1.0 -> 4.0`). Contradicting an **explicitly passed** flag raises before any model is built. Values are compared numerically-tolerantly, so the live configs — which restate values the script also passes (`kd_lambda: 1.0` with `--kd-lambda 1.0`) — stay legal. All nine invocations were checked: none conflicts.

**Verified live.**
```
$ uv run python tools/train_kd.py --kd-type cwd --tau 1.0 --kd-cfg conflict.yml   # tau: 8.0
ValueError: --kd-cfg contradicts explicitly passed CLI flags:
  --tau 1.0 (CLI) vs tau: 8.0 (--kd-cfg)
exit=1
```

**Regression tests.** `test_kd_cfg_conflicting_with_explicit_cli_flag_raises`, `test_kd_cfg_may_override_a_default_valued_flag`, `test_kd_cfg_agreeing_with_explicit_flag_is_not_a_conflict`.

**Fails-before note.** These tests import `_apply_kd_cfg_overrides`, which did not exist pre-fix, so reverting produces a collection error rather than an assertion failure. The pre-fix behaviour was a two-line unconditional `student_cfg["kd"][cfg_key] = kd_yaml[yaml_key]` loop with no comparison of any kind — there was no code path that could have raised.

---

### P1-3 — cosine LR schedule turns back upward past the horizon (`src/trainer_kd.py`)

**What was wrong.** `lr_lambda` computed `progress = (iter - warmup) / (total_iters - warmup)` with no clamp. `total_iters` is an *estimate* (`len(loader) * epochs // accumulate_steps`); once `progress > 1` the cosine rises again.

**Why it mattered.** With the current numbers the estimate slightly over-counts, so it does not trigger — but it is one config change away (a different `accumulate_steps`, a `drop_last` change, or resuming with a larger `--epochs`), and the failure is a silent LR ramp during the final epochs, which is exactly when a detector's mAP is decided.

**Regression test.** `test_lr_schedule_never_rises_after_total_iters` steps 140 iterations against a 100-iteration horizon and asserts the LR stays at the floor.

**Fails-before verification.** Pre-fix `build_lr_scheduler` loaded from `git show HEAD:src/trainer_kd.py`:
```
OLD LR: lr@100=0.000000 lr@139=0.396044; post-horizon iters above floor: 39
  test FAILS on old code: True
```

---

### P1-4 — resume restored an inconsistent state (`src/trainer_kd.py`, `tools/train_kd.py`)

**What was wrong.** Two gaps. (a) No RNG state was persisted, so a resumed run's augmentation stream diverged from an uninterrupted one. (b) Even with the P1-1 generator fix, the generator is rebuilt at process start, so resuming at epoch *N* would have replayed **epoch 1's** data order.

**Why it mattered.** Colab sessions drop; `run_ablation.sh` is built around restarting. Without this, "same seed" stops implying "same run" the moment a run is interrupted — and whether a run got interrupted is not recorded anywhere in the results.

**Fix.** `save_checkpoint` persists python / numpy / torch / CUDA RNG state; `load_checkpoint` restores it, normalising `map_location`-relocated tensors back to CPU `ByteTensor`s and warning (not crashing) if the GPU count differs. `KDTrainer` re-seeds the train loader generator to `seed + epoch` at the top of every epoch, so epoch *N*'s order is `(seed, N)` and nothing else.

**Regression tests.** `test_epoch_data_order_is_resume_identical` (drives the real `KDTrainer.train_epoch` for epochs 1→3 and compares against a fresh trainer jumping straight to epoch 3), `test_checkpoint_roundtrip_restores_rng_state` (perturbs all three streams, resumes, asserts the next draws match), `test_cuda_mapped_rng_state_restores`.

**Fails-before verification.** The underlying invariant was checked against both constructions directly:
```
PRE-FIX (no per-epoch reseed): resumed == uninterrupted @epoch3?  False
POST-FIX (per-epoch reseed) : resumed == uninterrupted @epoch3?  True
POST-FIX epochs still differ from each other?                    True
```
For the RNG persistence, pre-fix checkpoints contain no `rng_state` key at all, so the round-trip test cannot pass by construction.

**Limitation.** Resume is *statistically* identical, not bitwise: `cudnn.benchmark=True` plus fp16 makes GPU reductions non-reproducible. The observed post-resume loss deltas were ~0.1% (9.9316 vs 9.9523 at epoch 3).

---

### P2 fixes

- **`tools/benchmark_fps.py` — `--fp16` measured fp32.** Both the warmup and the measurement loop called `model(dummy.float())`, casting the half-precision input straight back. Fixed: the model is halved and the fp16 tensor is fed as-is. No live impact (`run_ablation.sh` does not pass `--fp16`), but the Phase 3 FP32/FP16/INT8 table would have reported two identical columns.
- **`tools/benchmark_fps.py` — benchmarked with attention capture on.** The config default is `capture_attn: True`, which forces `need_weights=True` and takes `nn.MultiheadAttention` off its fused path. The FPS column would have been uniformly pessimistic and inconsistent with the TensorRT latency table (ONNX export has no such branch). Now forced to `False`.
- **`src/trainer_kd.py:evaluate` — box clamping.** `tools/eval.py` and the teacher mAP gate clamp `x0/y0` to ≥ 0; the trainer's per-epoch evaluator did not. Selection mAP and reported mAP now use identical post-processing.
- **Documentation.** README CWD equation (`1/C` + τ²), README multi-scale claim (own teacher is C4+C5, not C3+C4+C5), `StageAdaptiveKDLoss.forward` epoch indexing (1-indexed as passed by the trainer).

---

## Fixes from the accepted proposals (second batch)

### P-1 — dead C3 fusion branch removed (`src/models/encoder.py`, `src/models/backbone.py`)

**What was wrong.** `HybridEncoder.forward` computed `c3_fused = self.fusion_c3(cat([c3, c4_up]))` and never used it. Nothing downstream read it, so `fusion_c3` and `input_proj[0]` received no gradient and stayed at initialisation for the entire run — while still costing a full `RepCSP` forward at stride 8, the most expensive spatial resolution in the network.

**Why it mattered.** ~15% of encoder forward time across a ~11 h teacher run and nine 36-epoch ablation runs, spent producing a tensor that is thrown away, plus a README parameter column that counted 0.48M parameters which never train. Removing it *later* would have invalidated every checkpoint already written, so this was the last free moment.

**What changed.**
- `fusion_c3` deleted; the C3 input projection deleted.
- `input_proj` re-indexed: `[0] → C4`, `[1] → C5` (was `[0] → C3`, `[1] → C4`, `[2] → C5`). Every read was audited — all three lived in `HybridEncoder.forward`; nothing outside the encoder indexes `input_proj`.
- `ResNetBackbone.forward` no longer returns `'0'` (C3). **C3 is still computed** — `layer3`/`layer4` are stacked on it — it is simply not returned. Keys `'1'`/`'2'` keep their original scale meaning rather than being renumbered.
- `self.num_scales` now reports the number of memory scales actually built (2), instead of silently describing a projection list that no longer matches.

**The gate: bit-identical outputs.** `test_p1_encoder_output_is_bit_identical_to_pre_change` reconstructs the pre-change encoder via `git show HEAD:src/models/encoder.py`, remaps the shared weights onto the post-change module (asserting the two state dicts line up exactly — a mismatch would mean the re-indexing is wrong), feeds both the same input, and requires `torch.equal`. Run in both `eval()` and `train()` mode. It passes, so the branch was in fact dead.

**Parameter counts, measured against the real pre-change encoder at production `hidden_dim=256`:**

```
resnet18: encoder params 1910272 -> 1432448  delta=477824 (expected 477824)  match=True
resnet50: encoder params 2598400 -> 2022272  delta=576128 (expected 576128)  match=True
```

`test_p1_removes_exactly_the_audited_parameter_count` asserts both deltas *and* that the delta equals exactly the parameters of `input_proj.0.*` + `fusion_c3.*` in the old model — a different number would mean the edit removed something other than the audit target. Full-model totals: student **16,426,388 → 15,948,564**, own teacher **29,446,036 → 28,869,908**.

**Fails-before verification.** Against the pre-change modules loaded from `HEAD`:
```
test_p1_encoder_has_no_untrained_parameters -> 18 params without grad
    (e.g. ['input_proj.0.conv.weight', 'input_proj.0.bn.weight'])   FAILS on old code: True
test_p1_backbone_no_longer_returns_c3       -> old keys ['0','1','2']  FAILS on old code: True
test_p1_removes_exactly_the_audited_parameter_count -> old-vs-old delta 0  FAILS on old code: True
```

**Live smoke after the change** (RTX 3050, AMP, 300-image split, one epoch, all KD types): `none`, `cwd`, `feature`, `query`, `stage_adaptive` all exit 0. Student params log as 15,948,564 and the teacher as 28,869,908. Data order is still identical across methods — batch-0 detection loss is 8.6751 (baseline) / 8.6755 (cwd) / 8.6756 (feature), i.e. the P1-1 guarantee survives the architecture change.

**What it changes about the results.** No measured output — encoder outputs are bit-identical. It does change (a) reported parameter counts, downward and more honestly, (b) FPS/latency, upward, (c) the RNG stream at initialisation, so student weights differ from any previously trained checkpoint. Since no campaign checkpoint exists, (c) costs nothing.

---

### P-2 — CWD τ² factor added (`src/distillation/cwd.py`)

**What was wrong.** Shu et al. define `L = τ²/C · Σ_c Σ_n KL(...)`; the τ² gradient-scaling factor was omitted.

**Why it mattered — and why it is safe.** At the ablation's fixed `tau: 1.0` the two forms are numerically identical, so **no planned run changes**. Without the factor the effective KD weight scales with `1/τ²`, so any future temperature sweep would have confounded a τ ablation with a λ ablation.

**Fix.** The loss returns `batchmean_KL * self.tau ** 2`. The module docstring, which previously documented the omission, now documents the factor.

**Regression tests.** `test_cwd_includes_tau_squared_factor` (parametrised over τ ∈ {0.5, 1.0, 2.0, 4.0}, compared against a closed-form reference implementing `τ²/C · ΣΣ`), and `test_cwd_tau_one_is_unchanged_by_the_tau_squared_fix`, which pins the "no planned run changes" claim by requiring the τ=1.0 value to equal the plain batchmean to `rel=1e-9`.

**Fails-before verification.** Pre-change `CWDLoss` from `HEAD` against the same inputs:
```
tau=0.5: old=10.011160  reference=0.417132   FAILS on old code: True
tau=2.0: old= 1.090265  reference=0.726844   FAILS on old code: True
tau=4.0: old= 0.283440  reference=0.755840   FAILS on old code: True
```
(The τ=1.0 row also differs there because `HEAD` still carries the pre-P0-1 `/B` divisor; the τ² factor *alone* is a no-op at τ=1, which is what `test_cwd_tau_one_is_unchanged_by_the_tau_squared_fix` asserts against the current code.)

---

### P-4 — `capture_attn` threaded into the teacher config (`tools/train_kd.py`)

**What was wrong.** `cfg["model"]["capture_attn"]` was set on the **student** config only. `teacher_cfg_dict` comes from a separate YAML that never sets the key, so `build_rtdetr` defaulted it to `True` and the teacher stored `[L, B, H, Q, N]` attention maps on every forward — including for `logit` and `cwd`, which never read them (~98 MB at batch 16 / 512 px, plus the non-fused attention path).

**Fix.** The flag is one setting, not two. `apply_capture_attn(kd_type, student_cfg, teacher_cfg)` sets the same value on both and returns it; `_ATTN_KD_TYPES` moved to module scope. `True` for `feature` / `combined` / `query` / `stage_adaptive`; `False` for `none` / `logit` / `cwd` / `mgd`. No separate flag was introduced.

**Timing.** Applied **before** the first ablation run, so all nine runs are uniform. This was the condition on accepting it: it perturbs the teacher's fp16 outputs for the logit and CWD runs (fused vs. math attention rounding), which would be a confound if introduced mid-campaign.

**Regression tests.** `test_capture_attn_is_threaded_into_teacher_config` (parametrised over all eight KD types, asserting student and teacher agree and that the value matches the group), `test_capture_attn_reaches_the_built_teacher_model` (end-to-end: builds a teacher for `cwd` and for `feature` and checks `decoder.capture_attn` and every decoder layer), `test_capture_attn_without_teacher_cfg_still_sets_student` (the baseline path passes no teacher config).

**Fails-before verification.** Pre-fix, the teacher config was never touched, so `teacher_cfg["model"]` had no `capture_attn` key at all and `build_rtdetr` fell back to `True` for every KD type — the parametrised test fails on all four non-attention types. Post-fix live smoke logs `Decoder attention capture: False (kd_type=cwd)` and `True (kd_type=feature)`, and both models are now built from that value.

---

### P-5 — one failed run no longer aborts the ablation (`scripts/run_ablation.sh`)

**What was wrong.** `set -euo pipefail` meant a single dead run (OOM, dropped mount, corrupt image) cancelled every run after it — defeating the skip-if-done logic that exists specifically for unattended overnight execution.

**Fix.** A `run_or_record` wrapper calls `run_experiment`, and on non-zero exit prints a distinct banner and appends to `$OUTPUT_ROOT/failures.txt` instead of aborting. The script exits non-zero at the end iff that file is non-empty, and clears it at the start so a clean re-run does not inherit stale entries.

**Skip and failure are visually distinct**, as required:
- completed: `✓ Already complete — skipping (…/checkpoint_best.pth + eval.log present)`
- crashed: `✗ FAILED — run 3 (run03_feature_l1.0) exited with status 1.`

and only crashes appear in `failures.txt`. The final banner reads `Ablation study complete!` or `Ablation finished WITH FAILURES` followed by the ledger.

**One non-obvious hazard, handled.** Invoking a function under `||` disables `errexit` inside its *entire* body. Without care, a failed `train_kd.py` would have fallen through to benchmarking and evaluating a checkpoint that does not exist — turning a clean failure into a fabricated `eval.log`. Every stage inside `run_experiment` therefore carries an explicit `|| return $?`.

**Verified with an injected failure** (stub `python` on `PATH`, run 3 forced to exit 1):
```
POST-FIX  runs 0,1,2 ok → run 3 ✗ FAILED → runs 4,5,6,7,8 all still executed
          exit=1;  failures.txt: "run 3   run03_feature_l1.0   exit=1"
          run03 dir contains ONLY train.log — no fps.log, no eval.log, no checkpoint
re-run    run 3 retried, runs 0-2/4-8 "✓ Already complete — skipping", exit=0, ledger cleared

PRE-FIX   runs 0,1,2 ok → run 3 dies → script aborts; exit=1
          runs 4-8 never started (only 4 run directories created)
```

---

## Proposals — open

P-1, P-2, P-4 and P-5 were reviewed, accepted and applied; they now live in
*Fixes from the accepted proposals* above, with their verification evidence.
One proposal remains open.

### P-3 · Logit-KD binary and softmax modes are not λ-matched — **deferred to λ calibration**

**Status: deferred deliberately. Both formulations stay canonical; neither reduction is to be changed.**
The rationale for deferring: effective KD magnitudes cannot be balanced against a teacher that does not exist yet, so this is folded into a single λ-calibration pass across *all* KD methods once the teacher is trained and real magnitudes can be measured — rather than by rescaling one of the two losses now on the basis of smoke-run numbers.

The finding stands as recorded. `_binary_kl` uses `reduction="mean"` over `[B·Q, C]` (averaging over classes); `_softmax_kl` uses `KLDivLoss(reduction="batchmean")` over the same view (**summing** over classes, averaging over `B·Q`). The two differ by a factor of roughly `C = 80` in normalization before the KL magnitudes are even considered. Smoke evidence at λ=1.0, T=4: `logit_binary kd=0.2511` vs `logit_softmax kd=1.3563`.

Both are the canonical form of their own formulation, so neither is "wrong" — but run01 vs run02 is presented as a *formulation* ablation, and as it stands it also varies the effective KD weight. Until the calibration pass lands, run 1 vs run 2 answers "binary at λ=1 vs softmax at λ=1", not "binary vs softmax at matched strength", and the write-up must say so.

Options that remain on the table for the calibration pass: (1) leave both canonical and state the caveat; (2) normalize `softmax` by `C` so both are per-(query, class) means; (3) add a run with softmax at a λ chosen to match the binary KD-term magnitude.


## Verified clean

Traced end-to-end and found correct. Listed so the next reader knows what was actually checked rather than assumed.

**Training batch — dataset → transforms → collate → forward → detection loss → KD loss → backward → optimizer → scheduler**
- Box format is normalized `cxcywh` at every hop. `COCODetection` converts pixel `xywh` → normalized `cxcywh` against the *original* image size; `Resize` is a plain square resize so normalized boxes need no rescaling; `RandomHorizontalFlip` correctly maps `cx → 1-cx`; the matcher and `RTDETRLoss` convert to `xyxy` only transiently for GIoU; the decoder's `bbox_head` emits `sigmoid` outputs in `[0,1]`. No format mixing found.
- Targets are moved to the device in `train_epoch` with a correct `isinstance(v, torch.Tensor)` guard (`image_id` int and `orig_size` tuple are correctly left alone); `RTDETRLoss` re-applies `.to(device)` defensively.
- `collate_fn` stacks images and keeps targets as a list — correct for variable object counts.
- Images with zero annotations: filtered out of the training set (`remove_no_annotations=True`), kept for val/select (`False`). The matcher returns empty index pairs for `size == 0`, and `RTDETRLoss` falls back to `pred_boxes.sum() * 0.0` — a graph-connected zero, not a bare constant, so backward does not break.
- KD projection parameters: `build_optimizer` deliberately walks `model.student`, and `train_kd.py` adds `loss_fn.parameters()` as a third param group at `lr_head`. Verified that CWD's `align` conv is in that group and receives gradients; Feature-KD/Query-KD use `nn.Identity` at `student_dim == teacher_dim == 256`, so their empty param list correctly skips `add_param_group`.
- Gradient accumulation: `zero_grad()` at epoch start discards any tail-of-epoch partial accumulation; loss is divided by `accumulate_steps` before backward; clipping covers both model and loss-module params.
- AMP interlock: `unscale_` before clipping, and the scheduler steps only when the scaler did not skip the optimizer step (detected via a reduced scale). This is correct and non-obvious.
- Teacher isolation: `RTDETRWithKD` freezes teacher params, overrides `train()` to keep the teacher in `eval()`, runs it under `no_grad`, **and** explicitly detaches every returned tensor. Each KD loss detaches its teacher input again. `build_optimizer` cannot see teacher params. Covered by existing tests.

**Evaluation batch — forward → top-k decode → denormalization → category mapping → pycocotools**
- Preprocessing matches training: both use `build_transforms(train=…)` with the same `Resize(img_size)` → `ToTensor` → `Normalize`, and `run_ablation.sh` passes the same `--img-size` to `train_kd.py` and `eval.py`.
- `orig_size` handling is correct for non-square images: the square resize squashes aspect ratio, and boxes are normalized against the *original* dimensions, so multiplying by `orig_w`/`orig_h` recovers pixel coordinates exactly. No letterboxing to undo.
- `_topk_decode` correctly recovers `labels = idx % C` and `query = idx // C` and gathers the matching boxes.
- COCO 80↔91 category mapping round-trips: `COCO91_TO_80` is built by enumerating `_COCO_CATEGORIES_80`, and both evaluators invert it with the same list.
- The `results_epoch{N}.json` temp file is written and unlinked per evaluation; the final val evaluation cannot collide with the last selection-split evaluation.

**Save/resume cycle** — model, loss-module, optimizer, scheduler, scaler, `global_step`, `best_map`, `epoch`, `cfg` and (now) RNG state are all persisted. Optimizer param-group ordering is reconstructed identically because `build_optimizer` + `add_param_group` are deterministic. Documented limitations below.

**Two runs differing in one flag** — checked all four ablation pairs (1/2 `--logit-mode`, 5/6 `--query-matching`, 7/8 `--schedule`, and 3/4 as the closest method pair). After the P1-1 fix, student initialisation, data order and augmentation are identical. The remaining intended difference is `capture_attn`, which is on for `feature`/`query`/`stage_adaptive` and off for `baseline`/`logit`/`cwd` — a deliberate decision, uniform within each method.

**Selection-split leakage** — `tools/make_select_split.py` asserts the splits are disjoint and cover the input; `KDTrainer` evaluates the selection split per epoch and only touches `val` once at the end. Best-checkpoint selection never sees val. Clean.

**Live smoke** — all seven KD configurations used by the nine-run matrix (`none`, `logit`, `cwd` + `cwd_kd.yml`, `feature`, `query` + `query_kd.yml` × both matchings, `stage_adaptive` + `stage_adaptive_kd.yml`) run end-to-end on GPU with AMP after every change in this batch; all exit 0 with no warnings beyond the expected "no teacher weights" notice.

---

## Not verifiable statically — limits of this audit

- **Teacher quality.** Whether the trained R50 teacher reaches a mAP worth distilling from is unknown until it trains. `--teacher-min-map` defaults to `0.0` (gate disabled) and `run_ablation.sh` leaves `TEACHER_MIN_MAP=0.0`. **Set it to a real threshold once the teacher's mAP is known** — it is the cheapest possible guard against nine runs distilling from a broken checkpoint.
- **Long-horizon fp16 stability.** No NaN/inf path was found by inspection, and `GradScaler` handles the standard cases, but 36 epochs of AMP behaviour cannot be established from 1–4-epoch smoke runs.
- **A100 memory headroom** at the full-COCO teacher settings (batch size, 512 vs 640 px). Only the 4 GB RTX 3050 path was exercised locally.
- **Bitwise reproducibility.** Not achievable with `cudnn.benchmark=True` + fp16. Reruns are statistically identical, not bit-identical; observed drift ~1e-4 relative on a single batch, ~0.1% on an epoch loss. If you need bitwise runs, that is a separate change (`cudnn.deterministic=True`, `use_deterministic_algorithms`) with a real speed cost.
- **pycocotools mAP at scale.** The evaluator was exercised only on 48–200-image splits. The mapping and decode logic are verified; absolute mAP values are not.
- **The lyuwenyu cross-architecture teacher path** (`src/models/rtdetr_teacher.py`, `third_party/RT-DETR`) was not exercised — the nine-run matrix uses the own teacher. Its import-context tests pass, but no forward pass was run.
- **`tools/export_trt.py`** (Phase 3) was not audited beyond a scan for swallowed exceptions.
- **Whether a missing/corrupt image file aborts a run.** It does — observed during smoke setup (`FileNotFoundError` in a worker kills the run). Correct behaviour, but worth knowing before an unattended 11 h job: verify the image directory is complete first.

---

**Verdict: the repo is ready to start the teacher run** — every P0/P1 finding is fixed and test-covered, the four accepted proposals (including the architecture change, which had to land before the first checkpoint) are applied and verified, and the only open item, **P-3**, is a deliberate deferral that affects how run 1 vs run 2 is *described*, not whether any run is valid. The one remaining action is operational, not blocking: set `--teacher-min-map` to a real threshold once the teacher's mAP is known, before launching the nine ablation runs.

### Post-architecture-change smoke test (run this by hand first)

The teacher configuration has not been trained since P-1 changed the encoder. Before committing ~11 h of A100 time, run the same configuration locally for 3 epochs on a ~2,000-image slice.

Carve the slice once (the 30K subset minus 2,000 images; only the `_select.json` side is used here):

```bash
uv run python tools/make_select_split.py \
    --ann  ~/data/coco/annotations/instances_train2017_30k.json \
    --num-select 2000 \
    --seed 42 \
    --out-prefix ~/data/coco/annotations/teacher_smoke
```

Then the dry run — R50, no KD, 3 epochs, 512 px, batch 2 (a single R50 at 512 px fits 4 GB; batch 4 does not leave headroom for the evaluator):

```bash
uv run python tools/train_kd.py \
    --student-cfg configs/rtdetr_r50vd_coco.yml \
    --kd-type none \
    --epochs 3 \
    --batch-size 2 \
    --accumulate-steps 8 \
    --img-size 512 \
    --num-workers 2 \
    --seed 42 \
    --use-amp \
    --output-dir runs/teacher_smoke \
    --coco-train ~/data/coco/train2017_30k \
    --coco-val   ~/data/coco/train2017_30k \
    --train-ann  ~/data/coco/annotations/teacher_smoke_select.json \
    --val-ann    ~/data/coco/annotations/teacher_smoke_select.json \
    --select-img ~/data/coco/train2017_30k \
    --select-ann ~/data/coco/annotations/teacher_smoke_select.json
```

What to check in the output:

- `Student params: 28,869,908` — the post-P-1 R50 count (this run trains the teacher *as* the model, hence the "Student" label).
- `Decoder attention capture: False (kd_type=none)`.
- `loss_total` decreasing across the three epochs, and a non-zero `select mAP` by epoch 3 (it will be small — 3 epochs on 2K images).
- Peak VRAM below ~3.5 GB (`nvidia-smi` alongside). If it OOMs, drop to `--batch-size 1 --accumulate-steps 16`; the A100 run uses a larger batch regardless, so this only gates the local check.
- A clean exit, and `runs/teacher_smoke/checkpoint_latest.pth` written.
