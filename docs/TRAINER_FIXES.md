# Trainer Fixes — resume correctness (apply before the teacher run)

> **Historical document — kept verbatim as evidence.** Applied before the
> teacher run, as the title says; the campaign has since completed. The
> resume path was audited again in [`AUDIT.md`](AUDIT.md) §§ P0-2, P1-4, which
> added RNG persistence and per-epoch loader re-seeding on top of these three
> fixes. Condensed index: [`README.md`](README.md).

Three fixes to `src/trainer_kd.py`, found while reviewing the smoke-run logs.

## 1. LR scheduler state was not saved/restored — **real bug**

`save_checkpoint` stored model, optimizer, scaler and `best_map`, but NOT the
LR scheduler. The scheduler steps **per iteration**, so a resumed run rebuilt
it from scratch at iteration 0: the linear warmup restarted and the cosine
curve was wrong for the entire remainder of training.

Measured (warmup=10, total=100, interrupted at iteration 30):

| | next 3 LRs after resume |
|---|---|
| uninterrupted reference | 8.72e-04, 8.60e-04, 8.47e-04 |
| resume **before** fix | 1.00e-04, 2.00e-04, 3.00e-04 ← warmup restarted |
| resume **after** fix | 8.72e-04, 8.60e-04, 8.47e-04 ✓ |

`global_step` is now persisted too, so TensorBoard curves stay continuous
across a resume. Old checkpoints without scheduler state still load, with a
warning.

## 2. Rolling `latest.pth` every epoch

`save_checkpoint` only wrote on `save_every` multiples and on mAP improvement.
An interrupted run therefore resumed from the last *improving* epoch, silently
discarding every epoch since. Each epoch now also writes
`checkpoint_latest.pth` (fixed name, overwritten) — the resume anchor.

Resume with:
```bash
--student-weights runs/<run>/checkpoint_latest.pth
```

## 3. LR scheduler advanced on AMP-skipped steps

`scaler.step()` SKIPS the optimizer when gradients contain inf/NaN — routine
while the scaler calibrates in the first iterations. The scheduler stepped
anyway, which (a) produced the `lr_scheduler.step() before optimizer.step()`
warning seen in every smoke log and (b) advanced the schedule by phantom
steps. The scheduler is now gated on the optimizer having actually stepped
(detected via the scaler reducing its scale — the idiom from the PyTorch AMP
docs).

Verified with a real `GradScaler`: 3 inf-gradient steps advance the schedule
by 0; 4 good steps advance it by 4.

## Verification

```bash
uv run pytest tests/ -q          # existing suite must stay green
```

Then a resume smoke test (2-epoch run, interrupt, resume):
```bash
uv run python tools/train_kd.py --kd-type logit --epochs 4 \
  --coco-train ~/data/coco/train2017_30k --coco-val ~/data/coco/train2017_30k \
  --train-ann ~/data/coco/annotations/instances_smoke_select.json \
  --val-ann ~/data/coco/annotations/instances_smoke_select.json \
  --teacher-source own --seed 42 --output-dir runs/smoke/resume_test
# Ctrl-C after epoch 2, then:
uv run python tools/train_kd.py --kd-type logit --epochs 4 \
  --coco-train ~/data/coco/train2017_30k --coco-val ~/data/coco/train2017_30k \
  --train-ann ~/data/coco/annotations/instances_smoke_select.json \
  --val-ann ~/data/coco/annotations/instances_smoke_select.json \
  --teacher-source own --seed 42 --output-dir runs/smoke/resume_test \
  --student-weights runs/smoke/resume_test/checkpoint_latest.pth

# Expect: "Loaded checkpoint from epoch 2", "Starting from epoch 3",
# NO lr_scheduler warning, and lr continuing the curve (not restarting).
```

