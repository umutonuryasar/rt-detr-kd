# RT-DETR Knowledge Distillation

**Systematic knowledge distillation study for real-time detection transformers — a 9-run controlled ablation across 5 KD methods with 2 novel contributions, TensorRT INT8 edge deployment on a 4 GB GPU.**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/umutonuryasar/rt-detr-kd/actions/workflows/ci.yml/badge.svg)

> **Status:** Phase 2A ablation in progress. Tech-report scope: results land in this README together with a 3-part blog series — findings are reported here, honestly, whichever way they go.

---

## Motivation

RT-DETR achieves state-of-the-art detection accuracy but its 32M-parameter ResNet-50 backbone is ill-suited to edge hardware: a direct swap to ResNet-18 (15.9M params in this student) costs several mAP points with no principled recovery strategy. Knowledge distillation transfers structural and semantic signal from a frozen teacher to a lightweight student, but most KD literature targets CNN detectors — it is unclear how logit-level versus feature-level versus query-level distillation interact with the transformer encoder-decoder architecture that RT-DETR uses. This work runs a controlled ablation across five KD methods on a fixed 4 GB RTX 3050 budget, introduces two transformer-specific techniques (Query-KD and Stage-Adaptive KD), and carries the best configuration through TensorRT INT8 quantization to a deployable FastAPI server.

---

## What I Built

- **9-run ablation:** baseline, Logit-KD (binary vs. softmax formulation), Feature-KD, CWD (ICCV'21), Query-KD (Hungarian vs. index matching), and Stage-Adaptive KD (cosine vs. inverse-cosine curriculum-direction control) — every claim ships with its own control run
- **Leakage-free model selection** — best checkpoints are chosen on a 2.5K selection split carved *from the training pool* (`tools/make_select_split.py`); the reported val set is evaluated once per run and never drives checkpoint selection
- **Feature-KD** with encoder MSE + decoder cross-attention cosine alignment, projecting student features to teacher channel width; multi-scale sequences are aligned **per scale in 2-D**, never blended across scale boundaries
- **CWD** (Shu et al., ICCV'21) — channel-wise softmax KL baseline for fair literature comparison
- **Query-KD** *(novel)* — distils RT-DETR's decoder object queries directly via per-image bipartite matching in prediction space (which object does each query describe?); robust to the 100 vs. 300 query-count mismatch. Legacy index-wise truncation is kept as an ablation baseline
- **Stage-Adaptive KD** *(novel)* — cosine curriculum that shifts weight from feature distillation (structural alignment, early training) to logit distillation (semantic refinement, late training), trained against an inverse-schedule control
- **Cross-architecture teacher adapter** (`src/models/rtdetr_teacher.py`) loading canonical [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) weights with a mAP sanity gate at training start
- **TensorRT INT8 export** with entropy calibration, FP32/FP16/INT8 latency sweep, and a latency-vs-accuracy table (`tools/export_trt.py`)
- **FastAPI inference server** for single-image and batch detection endpoints
- **Automated results aggregation** (`tools/aggregate_results.py`) producing CSV + Markdown tables

---

## Results

All numbers are mAP@[.5:.95] on COCO val2017. Checkpoints are selected on a
2,500-image split carved from the training pool; val2017 is evaluated once per
run and never influences selection.

### Headline: three seeds

These four configurations were repeated at seeds 42/43/44. The spread within a
configuration is what makes the differences between them interpretable.

| Configuration | mAP (mean ± std, n=3) | Δ baseline |
|---|---|---|
| Baseline (no KD) | 0.0388 ± 0.0028 | — |
| Query-KD, Hungarian matching *(novel)* | 0.0377 ± 0.0007 | −0.0011 |
| Query-KD, index truncation *(control)* | 0.0448 ± 0.0010 | +0.0060 |
| **Stage-Adaptive, cosine, λ=22.51** | **0.0676 ± 0.0005** | **+0.0288 (+74%)** |
| Teacher (own R50) | 0.142 | reference |

### Single-seed exploration (seed 42)

The remaining configurations were run once. Differences among them are of the
same order as the baseline's seed-to-seed spread (±0.0028), so they are
reported as observations, not as a ranking.

| Configuration | λ | mAP (n=1) |
|---|---|---|
| Baseline (no KD) | — | 0.0419 |
| Logit-KD, binary KL | 24.23 | 0.0348 |
| Logit-KD, softmax KL | 5.317 | 0.0374 |
| Feature-KD (encoder + attention) | 6.249 | 0.0464 |
| CWD — Shu et al. ICCV'21 | 11.78 | 0.0442 |
| Stage-Adaptive, cosine | 6.324 | 0.0553 |
| Stage-Adaptive, inverse-cosine | 22.51 | 0.0657 |

### The λ 2×2

Calibration set λ per method by matching the KD term to the detection loss at
initialization. For a schedule-based method that measurement is taken at the
epoch-1 mixture, which differs between the two schedule directions — so cosine
and inverse-cosine received very different λ. Crossing them isolates the
schedule direction from the weight:

| | λ=6.324 | λ=22.51 |
|---|---|---|
| **cosine** | 0.0553 | **0.0672** |
| **inverse-cosine** | 0.0430 | 0.0657 |

Raising λ helps in both directions (+0.012 and +0.023). At matched λ the
direction effect is ~0.001, i.e. nothing.

---

## Findings

**1. Query matching did not help; index truncation beat it.** Query-KD was
built on the argument that decoder queries have no canonical ordering, so
student and teacher queries must be matched (Hungarian, in prediction space)
rather than paired by index. The control says otherwise: index truncation wins
by +0.0071 at n=3, with within-configuration spread of ±0.001, and the same
ordering holds in every seed. Hungarian matching also lands *below* the
baseline — worse than not distilling at all.

A plausible mechanism is target instability: the assignment is recomputed every
step, so early in training a given student query chases a different teacher
query from batch to batch. Index pairing is arbitrary but stationary.

Scope: this holds for a **same-architecture teacher with an equal query count**.
The original argument for matching concerned cross-architecture distillation and
a 100-vs-300 query mismatch, which this setup does not test.

**2. λ dominates schedule design.** The first-pass result — inverse-cosine
beating cosine — did not survive its own control. It was a λ artifact: at
matched λ the two schedules are within 0.001 of each other, while the λ
difference is worth 0.012–0.023. The curriculum-direction claim is not
supported.

**3. Calibrating λ by a uniform rule is not the same as calibrating it
fairly.** The rule here (match the KD term to detection-loss scale at
initialization) is applied identically to every method, yet it systematically
under-weights schedule-based methods, because their epoch-1 mixture is not
representative of the loss they actually train under. The best result in this
study came from a *control* run that crossed the calibrated λ values, not from
the calibrated configuration itself.

**4. Logit-KD hurt in both formulations.** Binary KL (0.0348) and softmax KL
(0.0374) both landed below the seed-42 baseline (0.0419). The premise that
sigmoid-matched binary KL should outperform categorical softmax KL — because
RT-DETR trains classification with sigmoid focal loss — is not supported here;
if anything the ordering is reversed. Single seed, so the ordering between the
two is not itself a claim.

**5. KD reduced run-to-run variance.** Baseline: ±0.0028. Every KD
configuration measured at n=3: ±0.0005 to ±0.0010. The effect is independent of
whether the configuration improved mean mAP.

---

## Limitations

- **λ was not swept per method.** The 2×2 above varies λ only within
  Stage-Adaptive. Feature-KD, CWD and the others were each run at a single
  calibrated λ. Since λ turned out to matter more than the method-design
  choices under test, the correct reading of the headline number is *the best
  configuration found*, not *Stage-Adaptive is the best method* — other methods
  may well close the gap at a higher λ.
- **One teacher.** All runs distil from the same R50 checkpoint (0.142 mAP).
  Whether these orderings hold under a stronger teacher is untested.
- **Weak absolute regime.** A 15.9M student on a 30K subset at 512 px reaches
  ~0.04–0.07 mAP. Conclusions are about relative behaviour in this regime, not
  about production-grade detection.
- **Three seeds.** Enough to separate a 0.007 difference from a 0.001 spread;
  not enough for a distributional claim.
- **Five of eleven configurations are single-seed** and are reported as
  observations only.
- **Resumed runs are not bit-exact reproducible** — DataLoader shuffle RNG state
  is not restored across a resume. Uninterrupted runs at a fixed seed are
  reproducible.

---

## Methodology notes

**λ calibration.** `tools/calibrate_lambda.py` measures median(L_det) /
median(L_KD) over 20 batches at initialization and sets λ per method so every KD
term starts at detection-loss scale. Values used (teacher = own R50 @ 0.142,
seed 42): logit_binary 24.23, logit_softmax 5.317, feature 6.249, cwd 11.78,
query_hungarian 3.518, query_index 3.577, stage_cosine 6.324, stage_invcos
22.51. Re-running the script with the same arguments reproduces these.

**Learning rate.** `train_kd.py` defaults to `lr_head=1e-3`, which collapses
this architecture: the teacher reached 0.027 mAP at 1e-3 versus 0.142 at 1e-4,
with the training loss falling in both cases. All runs pin `--lr-head 1e-4
--lr-backbone 1e-5`. The failure mode is worth naming — a falling loss alongside
a collapsed mAP, with every predicted class probability stuck below 0.12.

---

## Architecture

### Student vs. teacher

The main ablation pairs the simplified student with an **own-architecture R50 teacher** trained in this repo: Query-KD and the cross-attention alignment terms require signals (post-norm decoder queries, dense $[Q, N]$ attention maps) that the canonical deformable-attention teacher does not expose. This teacher reaches 0.142 mAP on val2017 — well below the canonical RT-DETR (53.1), a direct consequence of the deliberate simplifications (100 queries, 3 decoder layers, vanilla MHA, C4+C5 memory) and the 512 px / 36-epoch / 30K-subset budget. What the ablation needs is a teacher meaningfully stronger than the student so KD has signal to transfer, not a state-of-the-art teacher; the gap is documented as a scope limitation, not hidden. The **canonical lyuwenyu teacher adapter** is retained for an optional cross-architecture comparison covering the methods that survive without those signals (Logit-KD, CWD, encoder-only Feature-KD); upstream teacher mAPs below are from the official release.

| Role | Backbone | Source | Params | mAP@\[.5:.95\] |
|------|----------|--------|--------|---------------|
| Student (simplified, this repo) | ResNet-18 | trained here | 15.9M | TBD |
| Teacher (own, main ablation) | ResNet-50 | trained here | 28.9M | 0.142 |
| Teacher RT-DETR-M (cross-arch option) | ResNet-34 | lyuwenyu/RT-DETR | 25M | 51.3 |
| Teacher RT-DETR-L (cross-arch option) | ResNet-50 | lyuwenyu/RT-DETR | 32M | 53.1 |

Rows trained here are exact counts for the architecture as it ships (student 15,948,564; own teacher 28,869,908). Both are ~0.5M lower than earlier drafts of this table: a pre-campaign audit found the encoder's C3 fusion branch produced a tensor nothing consumed, so 477,824 student / 576,128 teacher parameters never received a gradient. The branch was removed before any checkpoint existed; encoder outputs are bit-identical (`tests/test_audit_regressions.py::test_p1_encoder_output_is_bit_identical_to_pre_change`). Cross-architecture rows are the upstream published counts.

### Implementation differences (student only)

Every simplification is forced by the 4 GB RTX 3050 VRAM budget for dual-model (teacher + student) forward passes.

| Component | Canonical RT-DETR | This student | Reason |
|-----------|-------------------|--------------|--------|
| Object queries | 300 | 100 | OOMs at 300 with dual fp16 forward |
| Decoder layers | 6 | 3 | OOMs at 6 layers with dual forward pass |
| Cross-attention | Multi-scale deformable | Vanilla MHA | Deformable kernel doubles backward memory |
| Encoder memory | C3 + C4 + C5 | C4 + C5 only | C3 token count (6400 @ 640²) saturates VRAM |

Ablation runs execute on a Colab A100; the RTX 3050 hosts smoke tests and the TensorRT/FPS benchmarks. The student architecture is identical everywhere. The study measures *relative KD-method ranking* under a fixed budget, which is the claim these simplifications support.

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
├── tools/            # train_kd, eval, benchmark_fps, export_trt, serve,
│                     # aggregate_results, make_select_split
├── tests/            # pytest suite incl. methodology regression tests — runs on every push
├── scripts/          # run_ablation.sh (9-run Phase 2A); run_final.sh (full-COCO — out of current scope)
├── notebooks/        # ablation_analysis, visualize_attention, colab_training
├── third_party/      # lyuwenyu/RT-DETR submodule — canonical teacher weights + config
└── .github/          # CI: pytest on push
```

---

## Distillation Methods

Total loss for all methods: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \lambda \cdot \mathcal{L}_{\text{KD}}$

### Logit-KD

RT-DETR trains its classification head with a **per-class sigmoid** focal loss — logits are independent binary scores, not a categorical distribution. The default formulation therefore uses per-class *binary* KL between temperature-scaled sigmoid probabilities, which matches the distribution family the logits were trained under:

$$\mathcal{L}_{\text{logit}} = T^2 \cdot \frac{1}{|BQC|}\sum_{b,q,c} \mathrm{KL}_{\text{bin}}\!\left(\sigma\!\left(\tfrac{t_{bqc}}{T}\right) \,\Big\|\, \sigma\!\left(\tfrac{s_{bqc}}{T}\right)\right)$$

The classic Hinton et al. (2015) categorical softmax KL is available via `logit_mode: softmax` as an ablation — note it imposes a categorical structure the logits do not have. $T \in \{2, 4, 8\}$. Applied to the classification head only.

### Feature-KD

Encoder MSE with a learned projection layer + decoder cross-attention cosine alignment:

$$\mathcal{L}_{\text{feat}} = \mathrm{MSE}\!\left(\mathrm{proj}(s_{\text{enc}}),\, t_{\text{enc}}\right)$$

$$\mathcal{L}_{\text{attn}} = 1 - \cos\!\left(s_{\text{attn}},\, t_{\text{attn}}\right)$$

$$\mathcal{L}_{\text{KD}} = w_f \cdot \mathcal{L}_{\text{feat}} + \alpha \cdot \mathcal{L}_{\text{attn}}$$

Encoder sequences are concatenations of flattened scales; alignment is **per scale in 2-D** — sequences are split back into their scale chunks, paired from the coarsest end (C5↔C5, C4↔C4), and any extra fine teacher scale is dropped rather than blended in. The own-architecture R50 teacher shares the student's C4+C5 memory, so the pairing is exact there; the dropping rule matters for the cross-architecture (lyuwenyu, C3+C4+C5) teacher.

> **Note (canonical teacher):** the lyuwenyu deformable-attention teacher does not expose dense $[Q, N]$ cross-attention maps, so $\mathcal{L}_{\text{attn}}$ is **inactive** in the cross-architecture setup — Feature-KD reduces to the encoder term there. The main ablation uses the own-architecture teacher, where both terms are active; results tables state which configuration each run used.

### CWD — Channel-Wise Distillation (Shu et al., ICCV'21)

Spatially-normalized channel distributions aligned via KL divergence:

$$\mathcal{L}_{\text{CWD}} = \frac{1}{C}\sum_{c=1}^{C} \mathrm{KL}\!\left(\tilde{t}_c \,\Big\|\, \tilde{s}_c\right), \qquad \tilde{t}_c = \mathrm{softmax}\!\left(\tfrac{t_c}{\tau}\right)_{\text{spatial}}$$

Normalized by the channel count as in the reference, so $\lambda$ stays on the same scale as the other methods. The reference's additional $\tau^2$ gradient-scaling factor is omitted; it is a no-op at the ablation's fixed $\tau = 1.0$ and must be restored before any $\tau$ sweep.

### Query-KD *(novel)*

Distils RT-DETR's decoder object queries — a transformer-specific signal unavailable to CNN-detector KD methods. Decoder queries carry no canonical ordering (which query represents which object is image-dependent), so correspondence is built per image via **bipartite matching in prediction space**: the cost combines the L1 distance between sigmoid class-probability vectors and the L1 distance between predicted boxes, then Hungarian assignment pairs each student query with the teacher query describing the most similar object. This is robust to the 100 vs. 300 query-count mismatch and, unlike a shared student–teacher–GT Hungarian assignment, requires no joint matcher.

$$\mathcal{L}_{\text{query}} = \mathrm{MSE}\big(q_s^{(i)}, q_t^{(\pi(i))}\big) + \alpha \cdot \left(1 - \cos\!\left(A_s^{\text{dec}},\, A_t^{\text{dec}}\right)\right)$$

where $\pi$ is the per-image assignment. Index-wise truncation (`query_matching: index`) is retained strictly as an ablation baseline to quantify the value of matching.

**Distinction from prior work.** DETRDistill (ICLR'23) aligns query-prediction pairs after a *joint* Hungarian assignment against ground truth, which breaks when teacher and student query counts differ; the matching here is student↔teacher directly and needs no labels. MimicDet (ECCV'20) mimics RPN attention in two-stage detectors; the decoder cross-attention term here is specific to RT-DETR's encoder-memory interaction, which has no CNN analogue.

> **Note (canonical teacher):** the lyuwenyu teacher adapter does not expose post-norm decoder query embeddings (deformable decoder), so Query-KD requires the same-architecture (own) teacher; against the canonical teacher it degrades to a logged no-op.

### Stage-Adaptive KD *(novel)*

Cosine curriculum shifting from feature distillation (structural alignment) to logit distillation (semantic refinement) as training progresses:

$$w_f(e) = \cos\!\left(\frac{\pi e}{2E}\right), \qquad w_l(e) = 1 - w_f(e)$$

$$\mathcal{L}_{\text{KD}}^{\text{SA}}(e) = w_f(e)\cdot\mathcal{L}_{\text{feat}} + w_l(e)\cdot\mathcal{L}_{\text{logit}}$$

where $e$ is the current epoch and $E$ is total epochs. The schedule shape (cosine / linear / step / sigmoid / inverse-cosine) is configurable. The ablation trains the **inverse-cosine schedule as a curriculum-direction control**: if reversing the curriculum performs on par with the proposed direction, the scheduling claim is rejected rather than defended.

---

## Roadmap

**Done**
- [x] Full distillation pipeline — 5 KD methods, unified loss wrapper, config-driven
- [x] Methodology fix batch — prediction-space Hungarian query matching, sigmoid-matched binary logit KL, per-scale 2-D feature alignment, mosaic single-normalization, leakage-free checkpoint selection (see `FIXES.md`)
- [x] Cross-architecture teacher adapter with mAP sanity gate
- [x] TensorRT FP32 / FP16 / INT8 export with entropy calibration (`tools/export_trt.py`)
- [x] FastAPI inference server with single-image and batch endpoints
- [x] DETR-style top-k decoding (fixes ~2 mAP vs. per-query argmax)
- [x] Automated results aggregation — CSV + Markdown (`tools/aggregate_results.py`)
- [x] CI test suite: pytest incl. methodology regression tests on every push (CPU-only)

**In progress**
- [ ] Own R50 teacher training (single run; reused across all ablation runs)
- [ ] Phase 2A ablation — 9 runs on COCO 30K subset, 36 epochs, seed 42
- [ ] Attention visualization notebook (teacher vs. student cross-attention maps)

**Next**
- [ ] Results, Findings, and Limitations sections in this README
- [ ] TensorRT FP16/INT8 latency-vs-accuracy sweep on the best checkpoint
- [ ] Hugging Face Spaces demo with the best checkpoint
- [ ] 3-part blog series on the methodology findings

**Deliberately out of scope**
- Full-COCO 118K multi-seed runs and a standalone paper — the single-seed, subset-scale tech-report format keeps the claims proportionate to the compute behind them.

---

## Author

**Umut Onur Yasar** — Applied AI Research Engineer
[GitHub](https://github.com/umutonuryasar) · [LinkedIn](https://linkedin.com/in/umutonuryasar)