# RT-DETR KD — Tech Report Plan

**Scope decision (July 2026):** No arXiv paper. Deliverables are (1) a complete
repo with an honest Results section, (2) a deployed demo, (3) a 3-part blog
series. Phase 2A's ablation runs happen; full-COCO Phase 2D/2E is cut.
Single seed (42) throughout — reported openly as a limitation, no
statistical-significance claims.

**Compute split:** smoke tests + TRT/FPS benchmarks on the RTX 3050;
teacher training + the 9 ablation runs on Colab A100 (3050 would need ~34h
per run — infeasible ×9; A100 ≈ 3.5–4.5h per run, ~1.5–2 GPU-days total).

---

## Week 0 — Push + pre-flight (RTX 3050)

- [ ] Apply the fix-batch zip, run the manual pycache steps from FIXES.md,
      `pytest -q` green, push.
- [ ] Apply this work package (plan + updated `run_ablation.sh` +
      `tools/make_select_split.py` + `configs/kd/query_kd.yml`), push.
- [ ] Generate the selection split ONCE (fixed seed — identical for all runs):
      `python tools/make_select_split.py --ann .../instances_train2017_30k.json --num-select 2500 --seed 42`
- [ ] Smoke runs: each KD type, ~200 images × 2 epochs on the 3050.
      Verifies post-fix wiring, loss curves decreasing, the one-time
      warnings firing where expected, and the capture_attn VRAM savings.
- [ ] **DECISION — teacher budget.** Recommendation: own R50 on FULL COCO,
      36 epochs, A100 (~1 GPU-day; teacher quality drives every downstream
      number). Fallback if Colab budget is tight: 30K_train split, 72 epochs
      (~4–5h) — weaker teacher, smaller KD gains, must be stated in Results.

## Week 1 — Teacher + first runs (Colab A100)

- [ ] Train own R50 teacher (single run; checkpoint to Drive; resume-safe).
- [ ] Record teacher mAP explicitly; set `--teacher-min-map` to a realistic
      gate for the chosen regime (0.40 is unrealistic for the subset regime).
- [ ] Launch runs 0–4 (`bash scripts/run_ablation.sh`; skip-if-done makes
      session drops cheap).

## Week 2 — Remaining runs + analysis

- [ ] Runs 5–8 (+ optional 9/MGD if budget allows).
- [ ] `tools/aggregate_results.py` table. Keep select-mAP (checkpoint
      selection) and final val-mAP (reported) clearly separated.
- [ ] 1–2 figures from the attention-viz notebook (teacher vs student maps).
- [ ] **Narrative gate** — all three outcomes produce a report:
      (i) both novels win → straightforward findings;
      (ii) hungarian ≤ index or inverse_cosine ≈ cosine → matching/curriculum
      claim is reframed as a negative/neutral finding (CIFAR-v2 lesson:
      honest nuance is the brand);
      (iii) nothing beats baseline → "what doesn't transfer to DETR-family
      KD under tight budgets" is itself the finding.

## Week 3 — Writing + deployment (3050)

- [ ] README Results section: main table, Findings, Limitations
      (30K subset, single seed, teacher strength, 512px).
- [ ] TRT FP16/INT8 sweep on the best checkpoint + latency/FPS table
      ("trained on 4GB, X FPS at INT8" is the efficiency-narrative line).
- [ ] Update the HF Spaces demo with the best checkpoint.
- [ ] Draft all three blog posts (drafts only; publishing is Week 4).

## Week 4 — Publish + visibility

- [ ] Repo: "Status: Complete", release tag `v1.0`, CHANGELOG entry.
- [ ] Blog series, one per week from here:
      1. *Why softmax KL is the wrong distillation loss for sigmoid-trained
         detectors* (anchored by Run 1 vs Run 2).
      2. *Your queries don't line up: why matching matters in DETR
         distillation* (Run 5 vs Run 6).
      3. *What I learned distilling RT-DETR on a 4GB GPU* (process, TRT/FPS,
         honest results — links the repo and the CIFAR paper).
- [ ] One LinkedIn post per blog post + cross-links from GitHub/HF profiles.

---

## Standing rules

- Selection split decides checkpoints; val2017 is touched once per run.
- Same seed, same split, same teacher weights across all 9 runs.
- Any pre-fix training artifacts are discarded; never mix.
- If a run's outcome kills a claim, the claim dies in the writeup — the
  report's credibility is the actual career asset.
