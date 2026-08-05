# Process record

How the pipeline was corrected before it produced any reported number, and the
plan it was executed against. These are **historical documents** — they are
kept verbatim as evidence, not maintained as current-state descriptions. The
current state of the project lives in [`../README.md`](../README.md) (results)
and [`../CLAUDE.md`](../CLAUDE.md) (protocol, decisions, status).

| Document | Date | What it records |
|---|---|---|
| [`FIXES.md`](FIXES.md) | July 2026 | Methodology fix batch — the formulation choices the study is built on |
| [`TRAINER_FIXES.md`](TRAINER_FIXES.md) | 2026-07-27 | Resume correctness in `src/trainer_kd.py` |
| [`AUDIT.md`](AUDIT.md) | 2026-07-27 | Full pre-campaign audit: every P0/P1/P2 finding, its fix, and its fails-before evidence |
| [`TECH_REPORT_PLAN.md`](TECH_REPORT_PLAN.md) | July 2026 | The plan the campaign was run against (Weeks 0–4) |

---

## The four bugs that would have silently invalidated results

Each was found *before* any campaign checkpoint existed, each has a regression
test that fails against the pre-fix code, and each is the kind of defect that
produces a plausible-looking training log and a wrong number.

**1. Data order was seeded from the global RNG, not from `--seed`.**
`DataLoader(shuffle=True, generator=None)` seeds its sampler from the global
torch RNG *at iterator-creation time*, so the data stream depended on how much
randomness each KD method's construction had already consumed — a baseline
builds no teacher, `CWDLoss` builds a Xavier-initialised conv, the mAP gate
consumes another draw. The nine runs would have trained on **different data**
while the protocol claimed they differed only in the KD term. Pre-fix, batch 0
of epoch 1 already showed detection losses of 7.23 / 7.50 / 5.96 across
methods; post-fix the spread is 4e-4.
→ [`AUDIT.md`](AUDIT.md) § P1-1

**2. Resume restarted the LR schedule from iteration 0.**
The scheduler steps per iteration and was not saved in the checkpoint, so a
resumed run replayed the linear warmup and then followed a cosine curve that
was wrong for the rest of training. Measured at iteration 30 of 100: an
uninterrupted run continues at 8.72e-04, a resumed one restarted at 1.00e-04.
Colab sessions drop; `run_ablation.sh` is built around restarting.
→ [`TRAINER_FIXES.md` § 1](TRAINER_FIXES.md), and the companion defect —
a bare `except Exception` that swallowed a failed resume and silently
restarted from epoch 1 — in [`AUDIT.md`](AUDIT.md) § P0-2.

**3. CWD loss was inflated ~256×.**
The KL was summed over the `[B*D, N]` view and divided by `B` alone instead of
`B*D`. CWD is the study's only literature baseline; at λ=1.0 the KD term
measured 208× the detection term in the smoke logs, so the CWD row would have
measured λ, not the method.
→ [`AUDIT.md`](AUDIT.md) § P0-1

**4. `lr_head=1e-3` collapses this architecture while the loss still falls.**
`train_kd.py`'s default LR produced a teacher at 0.027 mAP versus 0.142 at
1e-4 — with the training loss decreasing in both cases and every predicted
class probability stuck below 0.12. Every campaign run therefore pins
`--lr-head 1e-4 --lr-backbone 1e-5`. This one was found during the campaign,
not the audit; it is recorded in the README's *Methodology notes* and pinned in
`scripts/run_ablation.sh`.

Two further audit findings mattered for cross-run comparability rather than
correctness: a `--kd-cfg` YAML key could silently override an explicitly passed
CLI flag (which would have collapsed an ablation pair into two identical runs),
and `capture_attn` was threaded into the student config only, so the teacher
stored attention maps for methods that never read them.

---

## What is deliberately still open

- **P-3 — the two Logit-KD modes are not λ-matched.** `binary` averages over
  classes, `softmax` sums over them; both are canonical for their own
  formulation. This was deferred to the λ-calibration pass, which
  `tools/calibrate_lambda.py` then performed per method — the campaign's
  logit runs use λ=24.23 (binary) and λ=5.317 (softmax), which is that
  factor-of-C difference measured rather than assumed.
  → [`AUDIT.md`](AUDIT.md) § P-3
- **Resumed runs are not bit-exact** — `cudnn.benchmark=True` plus fp16 makes
  GPU reductions non-reproducible. Reruns are statistically identical
  (~0.1% on an epoch loss), not bitwise. Reported as a README limitation.
- Items marked **accepted-as-is** in `AUDIT.md`'s findings table (dead config
  blocks, `build_rtdetr` ignoring several backbone keys, MGD's legacy 1-D
  alignment, CWD's always-built align conv) were judged not worth the churn
  and remain as documented.
