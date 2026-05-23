#!/usr/bin/env python3
"""ONNX → TensorRT export with FP32 / FP16 / INT8 calibration + latency benchmark.

Pipeline:
  1. torch.onnx.export — dynamic batch axis, opset 17.
  2. Build a TensorRT engine in the requested precision. For INT8, an
     entropy calibrator iterates over a calibration directory of images.
  3. Benchmark the engine at batch=1 with CUDA events: 50 warm-up iterations
     followed by 500 measurement iterations; report mean ± std latency,
     median, p95, and engine size.

The Torch model is built from the same YAML config the training pipeline
uses (``configs/rtdetr_r18vd_coco.yml``), and weights are loaded from a
checkpoint produced by ``tools/train_kd.py``.

Usage:
    # FP32
    python tools/export_trt.py \\
        --cfg configs/rtdetr_r18vd_coco.yml \\
        --weights runs/feature_kd_l1.0/checkpoint_best.pth \\
        --precision fp32 \\
        --output runs/feature_kd_l1.0/model_fp32.trt

    # INT8 with COCO val2017 calibration (~500 images)
    python tools/export_trt.py \\
        --cfg configs/rtdetr_r18vd_coco.yml \\
        --weights runs/feature_kd_l1.0/checkpoint_best.pth \\
        --precision int8 \\
        --calib-dir /data/coco/val2017 \\
        --calib-num 500 \\
        --output runs/feature_kd_l1.0/model_int8.trt
"""

import sys
import os
import argparse
import logging
import statistics
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from src.models.rtdetr import build_rtdetr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("export_trt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ONNX → TensorRT INT8 export + benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cfg", required=True, help="Model config YAML.")
    p.add_argument("--weights", required=True, help="PyTorch checkpoint .pth.")
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"],
                   default="fp16", help="TensorRT engine precision.")
    p.add_argument("--output", required=True, help="Output .trt path.")
    p.add_argument("--input-size", type=int, default=640, help="Square input size.")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version.")

    # INT8 calibration
    p.add_argument("--calib-dir", default=None,
                   help="Directory of calibration images (required for int8).")
    p.add_argument("--calib-num", type=int, default=500,
                   help="Number of calibration images to use.")
    p.add_argument("--calib-batch", type=int, default=8,
                   help="Calibration batch size.")
    p.add_argument("--calib-cache", default=None,
                   help="Cache file for the INT8 calibration table.")

    # Benchmarking
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--skip-benchmark", action="store_true",
                   help="Build the engine but skip the latency benchmark.")
    p.add_argument("--keep-onnx", action="store_true",
                   help="Keep the intermediate ONNX file (default: delete).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: PyTorch → ONNX
# ---------------------------------------------------------------------------

def export_onnx(weights_path: str, cfg_path: str, onnx_path: str,
                input_size: int, opset: int) -> None:
    """Export the Torch model to ONNX with a dynamic batch axis."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = build_rtdetr(cfg)
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Wrap to return concrete tensors (ONNX dislikes dicts)
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            out = self.m(x)
            return out["pred_logits"], out["pred_boxes"]

    wrapped = OnnxWrapper(model)
    dummy = torch.zeros(1, 3, input_size, input_size)

    logger.info(f"Exporting ONNX to {onnx_path} (opset {opset})...")
    torch.onnx.export(
        wrapped, dummy, onnx_path,
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        dynamic_axes={"images": {0: "batch"},
                      "pred_logits": {0: "batch"},
                      "pred_boxes": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    logger.info(f"ONNX file size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Step 2: ONNX → TensorRT engine
# ---------------------------------------------------------------------------

def build_engine(
    onnx_path: str,
    output_path: str,
    precision: str,
    input_size: int,
    calib_dir: str | None,
    calib_num: int,
    calib_batch: int,
    calib_cache: str | None,
) -> None:
    """Build a TensorRT engine from an ONNX file in the chosen precision."""
    try:
        import tensorrt as trt
    except ImportError:
        raise RuntimeError(
            "TensorRT not installed. Install via NVIDIA's tar/wheel for your "
            "CUDA + Python versions: https://developer.nvidia.com/tensorrt"
        )

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            logger.warning("Platform reports no fast FP16 — building anyway.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            logger.warning("Platform reports no fast INT8 — building anyway.")
        if calib_dir is None:
            raise ValueError("--calib-dir is required for precision=int8.")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _build_int8_calibrator(
            calib_dir, calib_num, calib_batch, input_size, calib_cache, trt
        )

    # Dynamic-batch profile: opt at 1, max at 8 (typical for detection).
    profile = builder.create_optimization_profile()
    profile.set_shape("images",
                      (1, 3, input_size, input_size),
                      (1, 3, input_size, input_size),
                      (8, 3, input_size, input_size))
    config.add_optimization_profile(profile)

    logger.info(f"Building {precision.upper()} engine (this can take 1-10 min)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed — see TRT logger output.")

    with open(output_path, "wb") as f:
        f.write(serialized)
    logger.info(f"Engine saved: {output_path} "
                f"({os.path.getsize(output_path) / 1e6:.1f} MB)")


def _build_int8_calibrator(calib_dir, calib_num, calib_batch, input_size,
                            calib_cache, trt):
    """Construct an entropy calibrator that streams normalized COCO images."""
    from PIL import Image
    import torchvision.transforms.functional as TF

    image_paths = sorted(Path(calib_dir).glob("*.jpg"))[:calib_num]
    if not image_paths:
        raise RuntimeError(f"No .jpg files in {calib_dir}")
    logger.info(f"INT8 calibration: {len(image_paths)} images, batch {calib_batch}")

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.batches = [
                image_paths[i:i + calib_batch]
                for i in range(0, len(image_paths), calib_batch)
            ]
            self.cursor = 0
            self.cache_file = calib_cache
            # Pinned host buffer + a single CUDA device buffer (re-used).
            self.shape = (calib_batch, 3, input_size, input_size)
            self.host_buf = np.zeros(self.shape, dtype=np.float32)
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
            self.cuda = cuda
            self.dev_buf = cuda.mem_alloc(self.host_buf.nbytes)

        def get_batch_size(self):
            return calib_batch

        def get_batch(self, names):
            if self.cursor >= len(self.batches):
                return None
            paths = self.batches[self.cursor]
            self.cursor += 1
            self.host_buf.fill(0)
            for i, p in enumerate(paths):
                img = Image.open(p).convert("RGB").resize((input_size, input_size))
                t = TF.to_tensor(img)
                t = TF.normalize(t, mean, std)
                self.host_buf[i] = t.numpy()
            self.cuda.memcpy_htod(self.dev_buf, self.host_buf)
            return [int(self.dev_buf)]

        def read_calibration_cache(self):
            if self.cache_file and os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            if self.cache_file:
                with open(self.cache_file, "wb") as f:
                    f.write(cache)

    return Calibrator()


# ---------------------------------------------------------------------------
# Step 3: Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_engine(engine_path: str, input_size: int, warmup: int, iters: int) -> dict:
    """Measure batch-1 latency of a TRT engine with CUDA-event timing."""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    context.set_input_shape("images", (1, 3, input_size, input_size))

    # Allocate I/O buffers
    bindings = []
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = int(np.prod(shape))
        host = np.zeros(size, dtype=dtype)
        dev = cuda.mem_alloc(host.nbytes)
        bindings.append(int(dev))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append((name, host, dev))
        else:
            outputs.append((name, host, dev))
        context.set_tensor_address(name, int(dev))

    # Warmup
    stream = cuda.Stream()
    for _ in range(warmup):
        context.execute_async_v3(stream.handle)
    stream.synchronize()

    # Measurement using CUDA events.
    start_evt = cuda.Event()
    end_evt   = cuda.Event()
    latencies = []
    for _ in range(iters):
        start_evt.record(stream)
        context.execute_async_v3(stream.handle)
        end_evt.record(stream)
        end_evt.synchronize()
        latencies.append(start_evt.time_till(end_evt))

    arr = np.array(latencies)
    return {
        "engine_mb":    os.path.getsize(engine_path) / 1e6,
        "mean_ms":      float(arr.mean()),
        "std_ms":       float(arr.std()),
        "median_ms":    float(np.median(arr)),
        "p95_ms":       float(np.percentile(arr, 95)),
        "fps_mean":     1000.0 / float(arr.mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    onnx_path = str(Path(args.output).with_suffix(".onnx"))
    export_onnx(args.weights, args.cfg, onnx_path, args.input_size, args.opset)

    build_engine(
        onnx_path=onnx_path,
        output_path=args.output,
        precision=args.precision,
        input_size=args.input_size,
        calib_dir=args.calib_dir,
        calib_num=args.calib_num,
        calib_batch=args.calib_batch,
        calib_cache=args.calib_cache,
    )

    if not args.keep_onnx:
        os.remove(onnx_path)

    if args.skip_benchmark:
        return

    logger.info("Benchmarking engine latency...")
    stats = benchmark_engine(args.output, args.input_size, args.warmup, args.iters)
    print("\n" + "=" * 60)
    print(f"  TensorRT {args.precision.upper()} engine benchmark")
    print("=" * 60)
    print(f"  Engine size : {stats['engine_mb']:.1f} MB")
    print(f"  Mean lat.   : {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
    print(f"  Median lat. : {stats['median_ms']:.2f} ms")
    print(f"  P95 lat.    : {stats['p95_ms']:.2f} ms")
    print(f"  Mean FPS    : {stats['fps_mean']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
