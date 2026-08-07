# Changelog

## [1.0.0] — 2026-08-06
- Tech report complete: 9-run KD ablation on 30K COCO subset (n=3 seeds on key configs).
- All three novel claims refuted by their controls; λ-magnitude confound identified.
- fp16 precision sweep: 160.8 FPS / 64 MB, mAP-neutral vs fp32.
- Scope: TRT/INT8 and live demo excluded (see Limitations).
