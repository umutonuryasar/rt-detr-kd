#!/usr/bin/env python3
"""Aggregate per-run results into a paper-ready CSV + Markdown table.

Walks a runs directory, parses each run's ``eval.log`` for COCO metrics,
``fps.log`` for throughput, and ``train.log`` for peak VRAM, then emits:

  - ``results.csv``   — one row per run, all metrics.
  - ``results.md``    — Markdown table grouped as it should appear in the paper.

Phase 2E aggregation: pass ``--seed-aggregate`` to compute mean ± std across
matching runs (e.g. ``run08_feature_l1.0_seed42``, ``_seed1337``, ``_seed2025``
collapse into one row labelled ``run08_feature_l1.0`` with mean ± std).

Usage:
    python tools/aggregate_results.py --runs-dir runs
    python tools/aggregate_results.py --runs-dir runs_final --seed-aggregate
"""

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Regex patterns — match the formats emitted by tools/eval.py and benchmark_fps.py.
_MAP_PATTERNS = {
    "mAP":     re.compile(r"AP@\[\.5:\.95\]\s*:\s*([\d.]+)"),
    "AP50":    re.compile(r"AP@\.50\s*:\s*([\d.]+)"),
    "AP75":    re.compile(r"AP@\.75\s*:\s*([\d.]+)"),
    "AP_S":    re.compile(r"AP-small\s*:\s*([\d.]+)"),
    "AP_M":    re.compile(r"AP-medium\s*:\s*([\d.]+)"),
    "AP_L":    re.compile(r"AP-large\s*:\s*([\d.]+)"),
}
_FPS_MEAN_PAT = re.compile(r"Mean FPS\s*:\s*([\d.]+)\s*[±+\-]?\s*([\d.]+)?")
_FPS_LAT_PAT  = re.compile(r"Mean lat\.\s*:\s*([\d.]+)\s*ms/image")
_VRAM_PAT     = re.compile(r"Peak VRAM\s*:\s*([\d.]+)\s*MB")
_PARAMS_PAT   = re.compile(r"Params\s*:\s*([\d,]+)")
_SEED_SUFFIX  = re.compile(r"_seed(\d+)$")


def _parse_log(path: Path, patterns: dict) -> dict:
    """Apply each regex to the file contents; return the last match per key."""
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    out = {}
    for key, pat in patterns.items():
        matches = pat.findall(text)
        if matches:
            val = matches[-1]
            if isinstance(val, tuple):
                val = val[0]
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                out[key] = None
    return out


def _parse_fps_log(path: Path) -> dict:
    """Extract FPS, std, latency, VRAM, params from a benchmark_fps log."""
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    out = {}
    m = _FPS_MEAN_PAT.search(text)
    if m:
        out["fps_mean"] = float(m.group(1))
        out["fps_std"]  = float(m.group(2)) if m.group(2) else 0.0
    m = _FPS_LAT_PAT.search(text)
    if m:
        out["latency_ms"] = float(m.group(1))
    m = _VRAM_PAT.search(text)
    if m:
        out["vram_mb"] = float(m.group(1))
    m = _PARAMS_PAT.search(text)
    if m:
        out["params"] = int(m.group(1).replace(",", ""))
    return out


def collect_run(run_dir: Path) -> dict:
    """Collect all metrics for a single run directory."""
    row = {"run": run_dir.name}
    row.update(_parse_log(run_dir / "eval.log", _MAP_PATTERNS))
    row.update(_parse_fps_log(run_dir / "fps.log"))
    return row


def collect_all(runs_dir: Path) -> list[dict]:
    """Collect rows for every direct sub-directory of ``runs_dir``."""
    rows = []
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        rows.append(collect_run(sub))
    return rows


def aggregate_by_seed(rows: list[dict]) -> list[dict]:
    """Collapse ``<tag>_seed<S>`` rows into one row per tag with mean ± std.

    Run name format expected: ``<tag>_seed<digits>``. Rows without that
    suffix are passed through unchanged.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    pass_through: list[dict] = []
    for r in rows:
        m = _SEED_SUFFIX.search(r["run"])
        if not m:
            pass_through.append(r)
            continue
        tag = r["run"][: m.start()]
        grouped[tag].append(r)

    out = list(pass_through)
    for tag, group in grouped.items():
        agg: dict = {"run": tag, "n_seeds": len(group)}
        for key in ("mAP", "AP50", "AP75", "AP_S", "AP_M", "AP_L",
                    "fps_mean", "latency_ms", "vram_mb", "params"):
            vals = [r[key] for r in group if key in r and r[key] is not None]
            if not vals:
                continue
            agg[key] = statistics.mean(vals)
            if len(vals) > 1:
                agg[key + "_std"] = statistics.stdev(vals)
            else:
                agg[key + "_std"] = 0.0
        out.append(agg)
    return sorted(out, key=lambda r: r["run"])


def write_csv(rows: list[dict], out_path: Path) -> None:
    """Write rows to CSV with a stable column order."""
    if not rows:
        out_path.write_text("")
        return
    base_keys = ["run", "n_seeds", "mAP", "mAP_std", "AP50", "AP75",
                 "AP_S", "AP_M", "AP_L",
                 "fps_mean", "fps_std", "latency_ms",
                 "vram_mb", "params"]
    extra_keys = sorted({k for r in rows for k in r if k not in base_keys})
    fieldnames = base_keys + extra_keys
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _fmt(value: Optional[float], std: Optional[float] = None, digits: int = 4) -> str:
    """Format a value (optionally with ± std) for the Markdown table."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if std is not None and std > 0:
        return f"{value:.{digits}f} ± {std:.{digits}f}"
    return f"{value:.{digits}f}"


def write_markdown(rows: list[dict], out_path: Path) -> None:
    """Write a Markdown table summarizing the runs."""
    headers = ["Run", "mAP", "AP50", "AP75",
               "AP_S", "AP_M", "AP_L",
               "FPS", "Lat (ms)", "VRAM (MB)"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        row = [
            r["run"],
            _fmt(r.get("mAP"), r.get("mAP_std")),
            _fmt(r.get("AP50"), r.get("AP50_std"), 3),
            _fmt(r.get("AP75"), r.get("AP75_std"), 3),
            _fmt(r.get("AP_S"), r.get("AP_S_std"), 3),
            _fmt(r.get("AP_M"), r.get("AP_M_std"), 3),
            _fmt(r.get("AP_L"), r.get("AP_L_std"), 3),
            _fmt(r.get("fps_mean"), r.get("fps_mean_std"), 1),
            _fmt(r.get("latency_ms"), r.get("latency_ms_std"), 2),
            _fmt(r.get("vram_mb"), r.get("vram_mb_std"), 0),
        ]
        lines.append("| " + " | ".join(row) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate RT-DETR KD ablation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runs-dir", required=True,
                   help="Directory containing per-run sub-directories.")
    p.add_argument("--csv", default=None,
                   help="Output CSV path (default: <runs-dir>/results.csv).")
    p.add_argument("--md", default=None,
                   help="Output Markdown path (default: <runs-dir>/results.md).")
    p.add_argument("--seed-aggregate", action="store_true",
                   help="Collapse <tag>_seed<S> rows into mean ± std (Phase 2E).")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.is_dir():
        raise SystemExit(f"Not a directory: {runs_dir}")

    rows = collect_all(runs_dir)
    if args.seed_aggregate:
        rows = aggregate_by_seed(rows)

    csv_path = Path(args.csv) if args.csv else runs_dir / "results.csv"
    md_path  = Path(args.md)  if args.md  else runs_dir / "results.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"Aggregated {len(rows)} runs.")
    print(f"  CSV: {csv_path}")
    print(f"  MD : {md_path}")


if __name__ == "__main__":
    main()
