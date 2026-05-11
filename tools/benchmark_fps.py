#!/usr/bin/env python3
"""FPS benchmarking for RT-DETR models.

Protocol (matches CS229 project spec):
  - Batch size: 1
  - Precision: fp32 (no AMP)
  - Warmup: 50 iterations (not measured)
  - Measurement: 500 iterations
  - Device: CUDA (with CUDA event timing for accuracy)
  - Report: mean ± std FPS over 500 iterations

Usage
-----
python tools/benchmark_fps.py \\
    --weights runs/feature_kd_l1.0/checkpoint_best.pth \\
    --cfg configs/rtdetr_r18vd_coco.yml \\
    --input-size 640 \\
    --warmup 50 \\
    --iters 500 \\
    --device cuda

python tools/benchmark_fps.py --cfg configs/rtdetr_r50vd_coco.yml --no-weights
"""

import sys
import time
import argparse
import logging
import statistics
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from src.models.rtdetr import build_rtdetr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("benchmark_fps")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RT-DETR FPS Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cfg", default="configs/rtdetr_r18vd_coco.yml",
                   help="Model config YAML.")
    p.add_argument("--weights", default=None,
                   help="Path to checkpoint .pth file.")
    p.add_argument("--no-weights", action="store_true",
                   help="Skip weight loading (benchmark with random init).")
    p.add_argument("--input-size", type=int, default=640,
                   help="Input image size (square).")
    p.add_argument("--warmup", type=int, default=50,
                   help="Number of warmup iterations.")
    p.add_argument("--iters", type=int, default=500,
                   help="Number of measurement iterations.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Compute device.")
    p.add_argument("--fp16", action="store_true", default=False,
                   help="Benchmark in fp16 (for reference — primary is fp32).")
    return p.parse_args()


def measure_fps_cuda(
    model: torch.nn.Module,
    device: torch.device,
    input_size: int,
    warmup: int,
    iters: int,
    fp16: bool = False,
) -> tuple[float, float, list[float]]:
    """Measure per-image FPS using CUDA events for high-precision timing.

    Returns:
        (mean_fps, std_fps, all_fps_values)
    """
    model.eval()
    dtype = torch.float16 if fp16 else torch.float32
    dummy = torch.zeros(1, 3, input_size, input_size, dtype=dtype, device=device)

    # Warmup
    logger.info(f"Warming up for {warmup} iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy.float())
    torch.cuda.synchronize()

    # Measurement with CUDA events
    logger.info(f"Measuring {iters} iterations...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    iter_times_ms = []

    with torch.no_grad():
        for _ in range(iters):
            start_event.record()
            _ = model(dummy.float())
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            iter_times_ms.append(elapsed_ms)

    fps_values = [1000.0 / t for t in iter_times_ms]
    mean_fps = statistics.mean(fps_values)
    std_fps = statistics.stdev(fps_values) if len(fps_values) > 1 else 0.0

    return mean_fps, std_fps, fps_values


def measure_fps_cpu(
    model: torch.nn.Module,
    device: torch.device,
    input_size: int,
    warmup: int,
    iters: int,
) -> tuple[float, float, list[float]]:
    """Measure FPS on CPU using Python time.perf_counter."""
    model.eval()
    dummy = torch.zeros(1, 3, input_size, input_size)

    # Warmup
    logger.info(f"Warming up for {warmup} iterations (CPU)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    # Measurement
    logger.info(f"Measuring {iters} iterations (CPU)...")
    iter_times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            iter_times.append(t1 - t0)

    fps_values = [1.0 / t for t in iter_times]
    mean_fps = statistics.mean(fps_values)
    std_fps = statistics.stdev(fps_values) if len(fps_values) > 1 else 0.0
    return mean_fps, std_fps, fps_values


def print_percentiles(fps_values: list[float]) -> None:
    arr = np.array(fps_values)
    print(f"  Min     FPS : {arr.min():.1f}")
    print(f"  P5      FPS : {np.percentile(arr, 5):.1f}")
    print(f"  Median  FPS : {np.median(arr):.1f}")
    print(f"  P95     FPS : {np.percentile(arr, 95):.1f}")
    print(f"  Max     FPS : {arr.max():.1f}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Build model
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Building model from config: {args.cfg}")
    model = build_rtdetr(cfg)
    logger.info(f"  Parameters: {model.num_parameters:,}")

    if args.weights and not args.no_weights:
        logger.info(f"Loading weights: {args.weights}")
        ckpt = torch.load(args.weights, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    else:
        logger.info("No weights loaded — using random initialization.")

    model = model.to(device)

    # Print model summary
    trainable_params = model.num_trainable_parameters
    total_params = model.num_parameters
    logger.info(f"  Trainable: {trainable_params:,} / Total: {total_params:,}")
    logger.info(f"  Input size: {args.input_size}×{args.input_size}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Precision: {'fp16' if args.fp16 else 'fp32'}")

    # Run benchmark
    if device.type == "cuda":
        torch.cuda.empty_cache()
        mean_fps, std_fps, fps_values = measure_fps_cuda(
            model, device,
            input_size=args.input_size,
            warmup=args.warmup,
            iters=args.iters,
            fp16=args.fp16,
        )
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        mean_fps, std_fps, fps_values = measure_fps_cpu(
            model, device,
            input_size=args.input_size,
            warmup=args.warmup,
            iters=args.iters,
        )
        gpu_name = "CPU"
        vram_mb = 0.0

    # Print results
    print("\n" + "=" * 60)
    print("RT-DETR FPS Benchmark Results")
    print("=" * 60)
    print(f"  Model       : {cfg.get('model', {}).get('backbone', 'unknown')}")
    print(f"  Device      : {gpu_name}")
    print(f"  Input size  : {args.input_size}×{args.input_size}")
    print(f"  Precision   : {'fp16' if args.fp16 else 'fp32'}")
    print(f"  Iterations  : {args.iters} (after {args.warmup} warmup)")
    print(f"  Params      : {total_params:,}")
    if device.type == "cuda":
        print(f"  Peak VRAM   : {vram_mb:.1f} MB")
    print()
    print(f"  Mean FPS    : {mean_fps:.1f} ± {std_fps:.1f}")
    print(f"  Mean lat.   : {1000.0 / mean_fps:.2f} ms/image")
    print()
    print_percentiles(fps_values)
    print("=" * 60)


if __name__ == "__main__":
    main()
