# RT-DETR Knowledge Distillation

## Project overview

**Title:** Knowledge Distillation for Efficient Real-Time Detection Transformers  
**Author:** Umut Onur Yaşar  
**Target:** arXiv cs.CV (1–2 ay)  
**Objective:** Systematic KD study on RT-DETR — logit, feature, combined, and partial ablations — with edge deployment analysis on RTX 3050.

---

## Repository structure

```
rt_detr_kd/
├── CLAUDE.md
├── configs/
│   ├── rtdetr_r18vd_coco.yml          # Student (RT-DETR-S, ResNet-18)
│   ├── rtdetr_r50vd_coco.yml          # Teacher (RT-DETR-L, ResNet-50)
│   ├── rtdetr_r34vd_coco.yml          # Teacher capacity: R34  [NEW]
│   └── kd/
│       ├── logit_kd.yml
│       ├── feature_kd.yml
│       ├── combined_kd.yml            # Logit + Feature simultaneously [NEW]
│       ├── encoder_only_kd.yml        # Partial: encoder MSE only [NEW]
│       └── attention_only_kd.yml      # Partial: attention cosine only [NEW]
├── src/
│   ├── distillation/
│   │   ├── logit_kd.py
│   │   ├── feature_kd.py
│   │   ├── kd_loss.py                 # Combined loss wrapper
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
│   └── export_trt.py                  # ONNX → TensorRT INT8 export [NEW]
├── notebooks/
│   ├── ablation_analysis.ipynb
│   └── visualize_attention.ipynb
├── scripts/
│   ├── download_coco_subset.sh
│   └── run_ablation.sh                # Update for 14 runs
└── runs/
    └── run00_baseline/
```

---

## Phase 1 — Code & infra tasks (week 1)

Bu fazda Colab açmaya gerek yok. Tüm değişiklikler local'de yapılıp commit edilecek.

### 1. lr_scheduler bug fix

**Dosya:** `src/trainer_kd.py`

**Sorun:** `lr_scheduler.step()` `optimizer.step()`'ten önce çağrılıyor. PyTorch 1.1.0+ bunu warning olarak gösteriyor ama ilk warmup adımı skip ediliyor.

**Fix:**
```python
# YANLIŞ (mevcut):
lr_scheduler.step()
optimizer.step()

# DOĞRU:
optimizer.step()
lr_scheduler.step()
```

Tüm ablation run'ları bu düzeltmeden sonra çalıştırılacak — tutarlılık için run00_baseline de bu fix ile tekrar çalıştırılmalı.

---

### 2. Combined KD config

**Yeni dosya:** `configs/kd/combined_kd.yml`

Logit-KD ve Feature-KD'yi aynı anda uygulayan config. `kd_loss.py` wrapper'ı zaten buna hazır — sadece config gerekiyor.

```yaml
kd_type: combined
kd_lambda: 1.0        # toplam KD loss ağırlığı
logit_weight: 0.5     # combined içinde logit payı
feature_weight: 0.5   # combined içinde feature payı
temperature: 4        # logit-KD sıcaklığı
alpha: 0.5            # feature-KD içinde attention payı
```

**run09** bu config ile çalışacak.

---

### 3. Partial KD configs

**Yeni dosya:** `configs/kd/encoder_only_kd.yml`

```yaml
kd_type: feature
kd_lambda: 1.0
alpha: 0.0            # attention terimi kapalı → sadece encoder MSE
```

**Yeni dosya:** `configs/kd/attention_only_kd.yml`

```yaml
kd_type: feature
kd_lambda: 1.0
alpha: 1.0            # sadece attention cosine terimi
# encoder MSE'yi sıfırlamak için feature_kd.py'de feat_weight: 0.0 ekle
```

`feature_kd.py`'ye `feat_weight` parametresi eklenmesi gerekiyor:

```python
class FeatureKDLoss(nn.Module):
    def __init__(self, student_dim=256, teacher_dim=256,
                 alpha=0.5, feat_weight=1.0):  # feat_weight ekle
        ...
        self.feat_weight = feat_weight

    def forward(self, ...):
        ...
        loss_kd = self.feat_weight * loss_feat + self.alpha * loss_attn
```

**run10** encoder-only, **run11** attention-only ile çalışacak.

---

### 4. Teacher capacity config

**Yeni dosya:** `configs/rtdetr_r34vd_coco.yml`

Mevcut `rtdetr_r50vd_coco.yml`'den kopyala, backbone'u ResNet-34'e değiştir.

**run12** teacher=R34, **run13** teacher=R50 (mevcut) olarak çalışacak.

---

### 5. TensorRT export script iskeleti

**Yeni dosya:** `tools/export_trt.py`

Phase 3'te doldurulacak. Şimdi sadece iskelet:

```python
"""
ONNX → TensorRT INT8 export.
Kullanim:
  python tools/export_trt.py \
    --cfg configs/rtdetr_r18vd_coco.yml \
    --weights runs/best.pth \
    --precision int8 \
    --output runs/best_int8.trt \
    --calib-data /data/coco/val2017
"""

# TODO Phase 3:
# 1. torch.onnx.export(model, ...)
# 2. tensorrt INT8 calibration (ImageBatcher on COCO val)
# 3. engine serialize → .trt dosyasına yaz
# 4. benchmark: trt vs torch fp32 vs torch fp16
```

---

### 6. run_ablation.sh güncelleme

Mevcut script 8 run için yazılı. 14 run'a genişlet:

```bash
# run00 baseline (done — fix sonrasi tekrar calistir)
# run01-06 logit-KD ablation
# run07-08 feature-KD ablation
# run09 combined-KD
# run10 encoder-only
# run11 attention-only
# run12 teacher=R34
# run13 teacher=R50 (upper bound)
```

---

### 7. Colab notebook iskeleti

**Yeni dosya:** `notebooks/colab_training.ipynb`

Colab Pro+'da A100 ile çalışacak session yönetimi:

```python
# Drive mount
from google.colab import drive
drive.mount('/content/drive')

# Repo clone / pull
!git clone https://github.com/umutonuryasar/RT-DETR /content/rt_detr
# veya mevcut klonsa:
# %cd /content/rt_detr && git pull

# Checkpoint resume
# Her run output-dir'i Drive'a yönlendir:
OUTPUT_BASE = "/content/drive/MyDrive/rt_detr_runs"

# Session kopma durumunda kaldığı yerden devam:
# --resume runs/runXX/checkpoint_last.pth
```

---

## Full run plan (14 runs)

| Run | KD type | λ | T | Notlar |
|-----|---------|---|---|--------|
| 00 | Baseline | — | — | fix sonrasi tekrar |
| 01 | Logit | 0.5 | 2 | |
| 02 | Logit | 0.5 | 4 | |
| 03 | Logit | 0.5 | 8 | |
| 04 | Logit | 1.0 | 2 | |
| 05 | Logit | 1.0 | 4 | |
| 06 | Logit | 1.0 | 8 | |
| 07 | Feature | 0.5 | — | |
| 08 | Feature | 1.0 | — | best projected ★ |
| 09 | Combined | 1.0 | 4 | logit+feature |
| 10 | Encoder-only | 1.0 | — | partial ablation |
| 11 | Attention-only | 1.0 | — | partial ablation |
| 12 | Feature (teacher=R34) | 1.0 | — | capacity analizi |
| 13 | Feature (teacher=R50) | 1.0 | — | capacity upper bound |

A100 tahmini: ~40-50 saat toplam.

---

## Paper tables (hedef çıktılar)

**Table 1 — Ana ablation:**
KD Type | λ | T | mAP@[.5:.95] | FPS | VRAM (GB)

**Table 2 — Partial KD:**
Method | L_feat | L_attn | mAP | Delta vs baseline

**Table 3 — Edge deployment:**
Model | Precision | Latency (ms) | Throughput (FPS) | mAP

**Table 4 — Teacher capacity:**
Teacher backbone | Teacher mAP | Student mAP (KD) | mAP gap

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

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| LR backbone | 1e-4 |
| LR transformer head | 1e-3 |
| Weight decay | 1e-4 |
| LR schedule | Cosine + 500-iter linear warmup |
| Batch size | 4 (RTX 3050) / 16 (A100 Colab) |
| Grad accumulation | 2 (RTX) / 1 (A100) |
| Epochs | 36 |
| Image size | 640×640 |
| AMP | fp16 |
| Teacher | Frozen, eval mode throughout |

---

## Distillation formulation

```
L_total = L_det + λ · L_KD

Logit-KD:    L_KD = T² · KL( softmax(t/T) ‖ softmax(s/T) )
Feature-KD:  L_KD = feat_weight · MSE(proj(s_enc), t_enc) + α · (1 - cos_sim(s_attn, t_attn))
Combined-KD: L_KD = logit_weight · L_logit + feature_weight · L_feature
```

---

## Environment

- **Local:** Ubuntu 24.04.3 · RTX 3050 4GB · Ryzen 5800H · 16GB RAM
- **Colab:** Pro+ · A100 40GB
- **Python:** 3.12
- **PyTorch:** 2.x + CUDA 12.x

---

## Phase checklist

### Phase 1 — Code & infra
- [x] lr_scheduler bug fix (`optimizer.step()` önce) — kod zaten doğruydu, değişiklik gerekmedi
- [x] `configs/kd/combined_kd.yml` oluştur
- [x] `configs/kd/encoder_only_kd.yml` oluştur
- [x] `configs/kd/attention_only_kd.yml` oluştur
- [x] `feature_kd.py`'ye `feat_weight` parametresi ekle
- [x] `configs/rtdetr_r34vd_coco.yml` oluştur
- [x] `tools/export_trt.py` iskeleti oluştur
- [x] `scripts/run_ablation.sh` 14 run'a güncelle
- [x] `notebooks/colab_training.ipynb` iskeleti oluştur

### Phase 2 — Training (A100, Colab)
- [ ] run00 baseline (lr fix sonrasi)
- [ ] run01–06 logit-KD
- [ ] run07–08 feature-KD
- [ ] run09 combined-KD
- [ ] run10 encoder-only
- [ ] run11 attention-only
- [ ] run12 teacher=R34
- [ ] run13 teacher=R50
- [ ] Her run için attention map'leri kaydet

### Phase 3 — Analysis & deployment
- [ ] `tools/export_trt.py` tamamla (ONNX → TRT INT8)
- [ ] FP32 / FP16 / INT8 latency tablosu (RTX 3050 + T4)
- [ ] COCO category-level AP analizi
- [ ] `visualize_attention.ipynb`: KD öncesi/sonrası en iyi 4-5 görsel
- [ ] Teacher capacity eğrisi grafik
- [ ] `ablation_analysis.ipynb`: tüm tablolar + Pareto plot (paper-ready)

### Phase 4 — Paper & submission
- [ ] Paper yaz (LaTeX / Overleaf, arxiv.sty)
- [ ] GitHub cleanup: description, topics, README görselleri, LICENSE
- [ ] arXiv submit
- [ ] Portfolio + LinkedIn güncelle
