"""
ONNX -> TensorRT INT8 export.
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
# 3. engine serialize -> .trt dosyasina yaz
# 4. benchmark: trt vs torch fp32 vs torch fp16
