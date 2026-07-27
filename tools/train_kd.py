#!/usr/bin/env python3
"""Main entry point for RT-DETR Knowledge Distillation training.

Usage
-----
# Feature-KD with lambda=1.0 (best config)
python tools/train_kd.py \\
    --student-cfg configs/rtdetr_r18vd_coco.yml \\
    --teacher-cfg configs/rtdetr_r50vd_coco.yml \\
    --kd-type feature \\
    --kd-lambda 1.0 \\
    --temperature 4 \\
    --epochs 36 \\
    --batch-size 4 \\
    --output-dir runs/feature_kd_l1.0 \\
    --coco-train /data/coco/train2017 \\
    --coco-val /data/coco/val2017 \\
    --train-ann /data/coco/annotations/instances_train2017.json \\
    --val-ann /data/coco/annotations/instances_val2017.json \\
    --teacher-weights /path/to/teacher.pth

# Baseline (no KD)
python tools/train_kd.py \\
    --student-cfg configs/rtdetr_r18vd_coco.yml \\
    --kd-type none \\
    --epochs 36 \\
    --output-dir runs/baseline
"""

import sys
import os
import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.models.rtdetr import build_rtdetr
from src.models.rtdetr_kd import RTDETRWithKD
from src.distillation.kd_loss import KDLoss
from src.losses.detection_loss import RTDETRLoss
from src.data.coco_dataset import COCODetection, collate_fn
from src.data.transforms import build_transforms, MosaicWrapper
from src.trainer_kd import KDTrainer, build_optimizer, build_lr_scheduler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_kd")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Re-seed python/numpy RNGs inside each DataLoader worker.

    PyTorch derives every worker's torch seed from the loader's generator, but
    the augmentation pipeline in src/data/transforms.py draws from python's
    ``random`` module. Deriving those seeds from ``torch.initial_seed()`` ties
    augmentation to the loader generator too, so augmentation is a pure
    function of (--seed, epoch, sample index).
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader_generator(seed: int) -> torch.Generator:
    """Build the DataLoader RNG for shuffling and worker seeding.

    WHY THIS EXISTS (cross-run comparability, not cosmetics):
    with ``generator=None`` a shuffling DataLoader seeds its sampler from the
    *global* torch RNG at iterator-creation time. That state depends on how
    much randomness the process consumed beforehand — and every KD method
    consumes a different amount (CWDLoss/MGDLoss build a Xavier-initialised
    Conv1d, FeatureKD/QueryKD do not, the baseline builds no teacher at all).
    Two runs that should differ only in ``--kd-type`` therefore saw *different
    training data orders*, confounding the ablation. Binding the loader to an
    explicit, seed-derived generator makes the data order a function of --seed
    alone.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RT-DETR Knowledge Distillation Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config files
    p.add_argument("--student-cfg", default="configs/rtdetr_r18vd_coco.yml",
                   help="Student model YAML config.")
    p.add_argument("--teacher-cfg", default="configs/rtdetr_r50vd_coco.yml",
                   help="Teacher model YAML config.")
    p.add_argument("--kd-cfg", default=None,
                   help="Optional KD-specific YAML config. Its keys override "
                        "the CLI defaults, but a key that contradicts an "
                        "EXPLICITLY passed CLI flag is a hard error — see "
                        "_apply_kd_cfg_overrides().")

    # KD settings
    p.add_argument(
        "--kd-type",
        default="feature",
        choices=[
            "logit", "feature", "combined",
            "cwd", "mgd", "query", "stage_adaptive",
            "none",
        ],
        help="Type of knowledge distillation. 'none' = baseline training.",
    )
    p.add_argument("--kd-lambda", type=float, default=1.0,
                   help="Weight for the KD loss term (λ).")
    p.add_argument("--temperature", type=float, default=4.0,
                   help="Temperature for logit/combined/stage_adaptive KD.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Attention weight in feature/combined/query/stage_adaptive KD.")
    p.add_argument("--feat-weight", type=float, default=1.0,
                   help="Encoder MSE weight inside feature/combined/stage_adaptive KD.")
    p.add_argument("--logit-weight", type=float, default=0.5,
                   help="Logit-component weight inside combined KD.")
    p.add_argument("--feature-weight", type=float, default=0.5,
                   help="Feature-component weight inside combined KD.")
    p.add_argument("--tau", type=float, default=1.0,
                   help="Spatial softmax temperature for CWD.")
    p.add_argument("--mask-ratio", type=float, default=0.75,
                   help="Token mask ratio for MGD.")
    p.add_argument("--schedule", default="cosine",
                   choices=["cosine", "linear", "step", "sigmoid", "inverse_cosine"],
                   help="Weight schedule for stage_adaptive KD.")
    p.add_argument("--logit-mode", default="binary",
                   choices=["binary", "softmax"],
                   help="Logit-KD formulation: 'binary' (per-class sigmoid KL, "
                        "matches RT-DETR's sigmoid-focal training — default) or "
                        "'softmax' (legacy Hinton categorical KL, for ablation).")
    p.add_argument("--query-matching", default="hungarian",
                   choices=["hungarian", "index"],
                   help="Query-KD correspondence: 'hungarian' (prediction-space "
                        "bipartite matching — default) or 'index' (legacy first-K "
                        "truncation, for ablation).")

    # Training
    p.add_argument("--epochs", type=int, default=36)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulate-steps", type=int, default=2,
                   help="Gradient accumulation steps (effective BS = batch_size * steps).")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--clip-max-norm", type=float, default=0.1)
    p.add_argument("--use-amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="use_amp", action="store_false")
    p.add_argument("--mosaic", action="store_true", default=False,
                   help="Enable Mosaic augmentation (p=0.5).")

    # Data paths
    p.add_argument("--coco-train", default="/data/coco/train2017")
    p.add_argument("--coco-val", default="/data/coco/val2017")
    p.add_argument("--train-ann",
                   default="/data/coco/annotations/instances_train2017.json")
    p.add_argument("--val-ann",
                   default="/data/coco/annotations/instances_val2017.json")
    p.add_argument("--select-img", default=None,
                   help="Optional image dir for a model-selection split carved "
                        "from training data. When set (with --select-ann), "
                        "per-epoch eval + best-checkpoint selection use this "
                        "split and val is only evaluated once at the end.")
    p.add_argument("--select-ann", default=None,
                   help="COCO annotation JSON for the model-selection split.")
    p.add_argument("--num-workers", type=int, default=4)

    # Weights
    p.add_argument("--teacher-weights", default=None,
                   help="Path to teacher pretrained weights (.pth).")
    p.add_argument("--student-weights", default=None,
                   help="Path to student pretrained/resume weights (.pth).")
    p.add_argument("--teacher-source", default="own",
                   choices=["own", "lyuwenyu"],
                   help="Teacher implementation: 'own' (simplified, defined in "
                        "src.models.rtdetr) or 'lyuwenyu' (canonical RT-DETR "
                        "from the original authors, via third_party submodule).")
    p.add_argument("--lyuwenyu-cfg", default=None,
                   help="If --teacher-source=lyuwenyu: path to one of their "
                        "configs/rtdetr/*.yml files.")

    # Teacher sanity gates (guard against silently-broken KD signal)
    p.add_argument("--teacher-max-missing-ratio", type=float, default=0.05,
                   help="Abort if more than this fraction of teacher state dict "
                        "keys are missing after load_state_dict.")
    p.add_argument("--teacher-min-map", type=float, default=0.0,
                   help="Run a 200-image eval pass on the teacher before training. "
                        "Abort if mAP < this threshold. 0.0 disables the gate. "
                        "Recommended: 0.40 (own teacher) / 0.45 (real RT-DETR).")
    p.add_argument("--teacher-gate-num-images", type=int, default=200,
                   help="Number of val images for the teacher mAP gate.")
    p.add_argument("--skip-teacher-gate", action="store_true", default=False,
                   help="Skip both teacher sanity gates (NOT recommended).")

    # Output
    p.add_argument("--output-dir", default="runs/kd_experiment")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# Maps the effective cfg["kd"] key to the CLI flag that also controls it.
# Used to detect the "YAML silently overrode my CLI flag" trap.
_KD_KEY_TO_CLI_FLAG = {
    "type": "--kd-type",
    "lambda": "--kd-lambda",
    "temperature": "--temperature",
    "alpha": "--alpha",
    "feat_weight": "--feat-weight",
    "logit_weight": "--logit-weight",
    "feature_weight": "--feature-weight",
    "tau": "--tau",
    "mask_ratio": "--mask-ratio",
    "schedule": "--schedule",
    "logit_mode": "--logit-mode",
    "query_matching": "--query-matching",
}


def _explicit_cli_flags(argv: Optional[list[str]] = None) -> set[str]:
    """Return the set of long option strings actually present on the command line."""
    argv = sys.argv[1:] if argv is None else argv
    seen = set()
    for token in argv:
        if token.startswith("--"):
            seen.add(token.split("=", 1)[0])
    return seen


def _values_differ(cli_value, yaml_value) -> bool:
    """Compare a CLI value against a YAML value tolerantly (1.0 == 1 == '1.0')."""
    if cli_value == yaml_value:
        return False
    try:
        return float(cli_value) != float(yaml_value)
    except (TypeError, ValueError):
        return str(cli_value) != str(yaml_value)


def _apply_kd_cfg_overrides(
    kd_cfg: dict,
    yaml_overrides: dict,
    explicit_flags: set[str],
) -> None:
    """Merge KD YAML keys into ``kd_cfg`` in place, refusing silent conflicts.

    The YAML merge runs AFTER argument parsing, so a key present in the KD
    config wins over the corresponding CLI flag. That is fine when the flag
    was left at its default, and a methodology-corrupting trap when it was
    not: two ablation runs that differ only in a CLI flag would silently
    train the same configuration. (This already happened once with
    ``query_matching``, which is why configs/kd/query_kd.yml omits it.)

    Overrides of default-valued flags are logged; overrides that contradict an
    explicitly passed flag raise before any training starts.
    """
    conflicts = []
    for key, value in yaml_overrides.items():
        old = kd_cfg.get(key)
        flag = _KD_KEY_TO_CLI_FLAG.get(key)
        if flag in explicit_flags and _values_differ(old, value):
            conflicts.append(f"  {flag} {old!r} (CLI) vs {key}: {value!r} (--kd-cfg)")
        elif _values_differ(old, value):
            logger.info(f"KD config override: {key}: {old!r} -> {value!r}")
        kd_cfg[key] = value

    if conflicts:
        raise ValueError(
            "--kd-cfg contradicts explicitly passed CLI flags:\n"
            + "\n".join(conflicts)
            + "\nThe YAML would silently win, which invalidates any ablation "
              "that varies the flag. Remove the key from the KD config or "
              "drop the CLI flag."
        )


# KD methods that actually read decoder cross-attention maps. For every other
# type (none, logit, cwd, mgd) capture is disabled so PyTorch can take
# nn.MultiheadAttention's fused fast path and skip storing [L, B, H, Q, N]
# maps — meaningful VRAM/speed savings on the 4 GB budget, and it keeps the
# baseline's runtime profile clean.
_ATTN_KD_TYPES = ("feature", "combined", "query", "stage_adaptive")


def apply_capture_attn(
    kd_type: str, student_cfg: dict, teacher_cfg: Optional[dict] = None
) -> bool:
    """Set ``model.capture_attn`` on the student AND teacher configs.

    The teacher config is a separate YAML that never sets the key, so before
    this it defaulted to True and the teacher stored dense attention maps on
    every forward — including for logit-KD and CWD, which never read them.
    Student and teacher must agree: the flag is one setting, not two.
    """
    capture = kd_type in _ATTN_KD_TYPES
    student_cfg.setdefault("model", {})["capture_attn"] = capture
    if teacher_cfg is not None:
        teacher_cfg.setdefault("model", {})["capture_attn"] = capture
    return capture


def build_cfg_from_args(args: argparse.Namespace) -> tuple[dict, dict]:
    """Build (student_cfg, teacher_cfg) from parsed arguments and YAML files."""
    # Load base config files
    student_cfg = load_yaml(args.student_cfg)
    teacher_cfg = load_yaml(args.teacher_cfg)

    # Override with CLI flags
    student_cfg.setdefault("train", {})
    student_cfg["train"].update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulate_steps": args.accumulate_steps,
        "img_size": args.img_size,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "weight_decay": args.weight_decay,
        "warmup_iters": args.warmup_iters,
        "clip_max_norm": args.clip_max_norm,
        "use_amp": args.use_amp,
    })
    student_cfg["data"] = {
        "train_ann": args.train_ann,
        "val_ann": args.val_ann,
        "train_img": args.coco_train,
        "val_img": args.coco_val,
        "select_ann": args.select_ann,
        "select_img": args.select_img,
        "num_workers": args.num_workers,
    }
    student_cfg["checkpoint"] = {
        "output_dir": args.output_dir,
        "save_every": args.save_every,
    }
    student_cfg["kd"] = {
        "type": args.kd_type,
        "lambda": args.kd_lambda,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "feat_weight": args.feat_weight,
        "logit_weight": args.logit_weight,
        "feature_weight": args.feature_weight,
        "tau": args.tau,
        "mask_ratio": args.mask_ratio,
        "schedule": args.schedule,
        "logit_mode": args.logit_mode,
        "query_matching": args.query_matching,
    }

    # Optionally load KD-specific YAML override.
    # Two YAML schemas are supported:
    #   (a) nested:  kd: { type: ..., lambda: ..., ... }
    #   (b) flat:    kd_type: ...,  kd_lambda: ..., tau: ..., mask_ratio: ...
    if args.kd_cfg and os.path.exists(args.kd_cfg):
        kd_yaml = load_yaml(args.kd_cfg)
        explicit_flags = _explicit_cli_flags()
        # Schema (a): nested
        if "kd" in kd_yaml and isinstance(kd_yaml["kd"], dict):
            _apply_kd_cfg_overrides(
                student_cfg["kd"], kd_yaml["kd"], explicit_flags
            )
        # Schema (b): flat top-level keys with kd_ prefix or known names
        flat_map = {
            "kd_type": "type",
            "kd_lambda": "lambda",
            "temperature": "temperature",
            "alpha": "alpha",
            "feat_weight": "feat_weight",
            "logit_weight": "logit_weight",
            "feature_weight": "feature_weight",
            "tau": "tau",
            "mask_ratio": "mask_ratio",
            "schedule": "schedule",
            "logit_mode": "logit_mode",
            "query_matching": "query_matching",
        }
        flat_overrides = {
            cfg_key: kd_yaml[yaml_key]
            for yaml_key, cfg_key in flat_map.items()
            if yaml_key in kd_yaml
        }
        _apply_kd_cfg_overrides(
            student_cfg["kd"], flat_overrides, explicit_flags
        )

    return student_cfg, teacher_cfg


def _check_teacher_state_dict(
    teacher,
    missing: list,
    unexpected: list,
    max_missing_ratio: float,
) -> None:
    """Abort training if too many teacher weights failed to load.

    A silently-mismatched teacher state dict is the highest-impact bug in a KD
    pipeline — every downstream experiment becomes meaningless. We require that
    at least ``1 - max_missing_ratio`` of the teacher's parameters loaded from
    the checkpoint.
    """
    total_keys = len(teacher.state_dict())
    n_missing = len(missing)
    n_unexpected = len(unexpected)
    missing_ratio = n_missing / max(total_keys, 1)

    logger.info(
        f"Teacher state-dict load: {total_keys - n_missing}/{total_keys} keys "
        f"loaded ({n_missing} missing, {n_unexpected} unexpected)."
    )
    if missing:
        logger.info(f"  First few missing keys: {missing[:5]}")
    if unexpected:
        logger.info(f"  First few unexpected keys: {unexpected[:5]}")

    if missing_ratio > max_missing_ratio:
        raise RuntimeError(
            f"Teacher state-dict mismatch too large: {missing_ratio:.1%} of keys "
            f"missing (threshold {max_missing_ratio:.1%}). Refusing to train — "
            f"a silently broken teacher produces meaningless KD signal. "
            f"Fix the checkpoint or pass --skip-teacher-gate."
        )


def _teacher_map_gate(
    teacher,
    val_loader,
    val_ann_file: str,
    device,
    num_images: int,
    min_map: float,
    use_amp: bool = True,
) -> float:
    """Run a quick mAP eval on the teacher and abort if below threshold.

    Returns the measured mAP. Bounded to ``num_images`` to keep startup time
    reasonable (~30 s on A100, ~2 min on RTX 3050).
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        logger.warning("pycocotools not installed; skipping teacher mAP gate.")
        return 0.0

    if not os.path.exists(val_ann_file):
        logger.warning(f"val_ann_file not found ({val_ann_file}); skipping teacher mAP gate.")
        return 0.0

    from src.data.coco_dataset import _COCO_CATEGORIES_80
    from src.trainer_kd import _topk_decode
    from torch.amp import autocast

    idx_to_coco_id = {i: cid for i, cid in enumerate(_COCO_CATEGORIES_80)}
    teacher.eval()
    results = []
    images_seen = 0

    with torch.no_grad():
        for images, targets in val_loader:
            if images_seen >= num_images:
                break
            images = images.to(device)
            with autocast('cuda', enabled=use_amp and device.type == "cuda"):
                outputs = teacher(images)
            # Use the same top-k decode as the training evaluator so the mAP
            # gate threshold is comparable to reported numbers.  Argmax-per-query
            # underestimates multi-label predictions and can abort valid setups.
            scores, labels, decoded_boxes = _topk_decode(
                outputs["pred_logits"], outputs["pred_boxes"], top_k=100
            )
            for i in range(images.size(0)):
                if images_seen >= num_images:
                    break
                images_seen += 1
                img_id = targets[i]["image_id"]
                if isinstance(img_id, torch.Tensor):
                    img_id = img_id.item()
                orig_h, orig_w = targets[i]["orig_size"]
                if isinstance(orig_h, torch.Tensor):
                    orig_h, orig_w = orig_h.item(), orig_w.item()
                cx = decoded_boxes[i, :, 0] * orig_w
                cy = decoded_boxes[i, :, 1] * orig_h
                bw = decoded_boxes[i, :, 2] * orig_w
                bh = decoded_boxes[i, :, 3] * orig_h
                x0 = (cx - bw / 2).clamp(min=0)
                y0 = (cy - bh / 2).clamp(min=0)
                for j in range(scores.size(1)):
                    s = scores[i, j].item()
                    if s < 0.05:
                        continue
                    cid = idx_to_coco_id.get(labels[i, j].item(), -1)
                    if cid < 0:
                        continue
                    results.append({
                        "image_id": img_id, "category_id": cid,
                        "bbox": [round(x0[j].item(), 2), round(y0[j].item(), 2),
                                 round(bw[j].item(), 2), round(bh[j].item(), 2)],
                        "score": round(s, 4),
                    })

    if not results:
        logger.error("Teacher produced no predictions on val sample — definitely broken.")
        raise RuntimeError("Teacher mAP gate: zero predictions above 0.05 score.")

    import tempfile, json as _json
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        _json.dump(results, f)
        results_path = f.name

    coco_gt = COCO(val_ann_file)
    coco_dt = coco_gt.loadRes(results_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = list({r["image_id"] for r in results})
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    teacher_map = float(coco_eval.stats[0])
    os.unlink(results_path)

    logger.info(f"Teacher mAP on {images_seen} val images: {teacher_map:.4f}")
    if teacher_map < min_map:
        raise RuntimeError(
            f"Teacher mAP {teacher_map:.4f} < required {min_map:.4f}. "
            f"Refusing to train — KD with a weak teacher is worse than no KD. "
            f"Verify weights/config or pass --skip-teacher-gate."
        )
    return teacher_map


def resume_if_checkpoint(trainer, student_weights: Optional[str]) -> int:
    """Resume from a full trainer checkpoint, returning the completed epoch.

    Backbone-only / bare state-dict weight files (no 'epoch' key) fall back
    gracefully: the model was already loaded with strict=False earlier and
    training starts at epoch 1.

    A file that DOES carry an 'epoch' key is a resume request, and a failure
    to restore it is fatal. It used to be caught by a bare ``except
    Exception`` that logged a warning and continued from epoch 1 — so an
    unattended Colab run that dropped and relaunched would silently throw
    away every completed epoch (and its optimizer, scheduler and scaler
    state) instead of stopping. This was observed live: a CUDA-mapped RNG
    tensor raised inside load_checkpoint and the run quietly restarted.
    """
    if not student_weights or not os.path.exists(student_weights):
        return 0

    try:
        ckpt_probe = torch.load(student_weights, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read --student-weights {student_weights}: {exc}"
        ) from exc

    is_trainer_checkpoint = isinstance(ckpt_probe, dict) and "epoch" in ckpt_probe
    del ckpt_probe

    if not is_trainer_checkpoint:
        logger.info(
            "Student weights appear to be a bare state-dict (no 'epoch' key); "
            "optimizer/scaler state not restored. Training from epoch 1."
        )
        return 0

    return trainer.load_checkpoint(student_weights)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    logger.info(f"Device: {args.device}")
    device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    cfg, teacher_cfg_dict = build_cfg_from_args(args)

    # After YAML merge the effective KD config may differ from CLI args.
    # Authoritative source from here on is cfg["kd"].
    kd_cfg = cfg["kd"]
    effective_kd_type = kd_cfg["type"]
    logger.info(
        f"Effective KD config: type={effective_kd_type}, "
        f"lambda={kd_cfg['lambda']}, temperature={kd_cfg['temperature']}, "
        f"alpha={kd_cfg['alpha']}, tau={kd_cfg['tau']}, "
        f"mask_ratio={kd_cfg['mask_ratio']}, schedule={kd_cfg['schedule']}, "
        f"logit_mode={kd_cfg.get('logit_mode', 'binary')}, "
        f"query_matching={kd_cfg.get('query_matching', 'hungarian')}"
    )

    # ---- Build models ----
    capture_attn = apply_capture_attn(effective_kd_type, cfg, teacher_cfg_dict)
    logger.info(f"Decoder attention capture: {capture_attn} (kd_type={effective_kd_type})")

    logger.info("Building student model...")
    student = build_rtdetr(cfg)
    logger.info(f"  Student params: {student.num_parameters:,}")

    if effective_kd_type != "none":
        if args.teacher_source == "lyuwenyu":
            logger.info("Building teacher from lyuwenyu/RT-DETR (canonical)...")
            if args.lyuwenyu_cfg is None:
                raise ValueError(
                    "--teacher-source=lyuwenyu requires --lyuwenyu-cfg pointing "
                    "to one of third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr/*.yml."
                )
            from src.models.rtdetr_teacher import build_lyuwenyu_teacher
            teacher = build_lyuwenyu_teacher(
                config=args.lyuwenyu_cfg,
                checkpoint=args.teacher_weights,
            )
            logger.info(f"  Teacher params: {teacher.num_parameters:,}")
        else:
            logger.info("Building teacher model (own simplified RT-DETR)...")
            teacher = build_rtdetr(teacher_cfg_dict)
            logger.info(f"  Teacher params: {teacher.num_parameters:,}")
            if args.teacher_weights:
                logger.info(f"Loading teacher weights from: {args.teacher_weights}")
                ckpt = torch.load(args.teacher_weights, map_location="cpu")
                state = ckpt.get("model_state_dict", ckpt)
                missing, unexpected = teacher.load_state_dict(state, strict=False)
                _check_teacher_state_dict(teacher, missing, unexpected,
                                           max_missing_ratio=args.teacher_max_missing_ratio)
            else:
                logger.warning(
                    "No --teacher-weights given; teacher will be randomly initialized. "
                    "KD signal is meaningless without a competent teacher. "
                    "Pass --teacher-weights or use --skip-teacher-gate to override."
                )

        model = RTDETRWithKD(student=student, teacher=teacher)
    else:
        # Baseline: no KD, train student directly
        model = student

    if args.student_weights:
        logger.info(f"Loading student weights from: {args.student_weights}")
        ckpt = torch.load(args.student_weights, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = student.load_state_dict(state, strict=False)
        if missing:
            logger.info(f"  student weight load: {len(missing)} missing keys (backbone-only weights?)")
        if unexpected:
            logger.info(f"  student weight load: {len(unexpected)} unexpected keys")

    # ---- Build loss ----
    num_classes = cfg["model"].get("num_classes", 80)
    hidden_dim = cfg["model"].get("hidden_dim", 256)
    # For the lyuwenyu teacher the hidden_dim lives in their YAML, not in our
    # teacher_cfg_dict. Read it from the model object if available.
    if effective_kd_type != "none":
        teacher_hidden_dim = getattr(
            teacher, "hidden_dim",
            teacher_cfg_dict.get("model", teacher_cfg_dict).get("hidden_dim", hidden_dim),
        )
    else:
        teacher_hidden_dim = hidden_dim

    if effective_kd_type != "none":
        loss_fn = KDLoss(
            kd_type=effective_kd_type,
            kd_lambda=kd_cfg["lambda"],
            temperature=kd_cfg["temperature"],
            alpha=kd_cfg["alpha"],
            feat_weight=kd_cfg["feat_weight"],
            logit_weight=kd_cfg["logit_weight"],
            feature_weight=kd_cfg["feature_weight"],
            tau=kd_cfg["tau"],
            mask_ratio=kd_cfg["mask_ratio"],
            num_classes=num_classes,
            student_dim=hidden_dim,
            teacher_dim=teacher_hidden_dim,
            total_epochs=args.epochs,
            schedule=kd_cfg["schedule"],
            logit_mode=kd_cfg.get("logit_mode", "binary"),
            query_matching=kd_cfg.get("query_matching", "hungarian"),
        )
    else:
        # Wrap RTDETRLoss to match KDLoss forward signature
        class BaselineLossWrapper(torch.nn.Module):
            def __init__(self, nc):
                super().__init__()
                self.det_loss = RTDETRLoss(num_classes=nc)
            def forward(self, model_outputs, targets, epoch: int = 0):
                # For baseline, model_outputs is the raw dict from RTDETR
                losses = self.det_loss(model_outputs, targets)
                losses["loss_total"] = losses["loss_det"]
                losses["loss_kd"] = torch.tensor(0.0, device=losses["loss_det"].device)
                return losses
        loss_fn = BaselineLossWrapper(num_classes)

    # ---- Build datasets ----
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    img_size = train_cfg.get("img_size", 640)

    logger.info("Building datasets...")
    train_transforms = build_transforms(train=True, img_size=img_size)
    val_transforms = build_transforms(train=False, img_size=img_size)

    if args.mosaic:
        # MosaicWrapper contract: the wrapped dataset must return RAW PIL
        # images; the full transform pipeline is applied exactly once (on
        # the single image, or on the assembled mosaic canvas). Wrapping an
        # already-transformed dataset would normalize twice and corrupt the
        # mosaic images.
        logger.info("Mosaic augmentation ENABLED (p=0.5)")
        raw_train_dataset = COCODetection(
            img_folder=data_cfg["train_img"],
            ann_file=data_cfg["train_ann"],
            transforms=None,
        )
        train_dataset = MosaicWrapper(
            raw_train_dataset,
            base_transform=train_transforms,
            img_size=img_size,
            p=0.5,
        )
    else:
        train_dataset = COCODetection(
            img_folder=data_cfg["train_img"],
            ann_file=data_cfg["train_ann"],
            transforms=train_transforms,
        )

    val_dataset = COCODetection(
        img_folder=data_cfg["val_img"],
        ann_file=data_cfg["val_ann"],
        transforms=val_transforms,
        remove_no_annotations=False,
    )

    logger.info(f"Train set: {len(train_dataset)} images")
    logger.info(f"Val set:   {len(val_dataset)} images")

    # Data order and augmentation must depend on --seed ONLY, never on how
    # much RNG the model/loss construction happened to consume. See
    # make_loader_generator().
    loader_generator = make_loader_generator(args.seed)
    val_loader_generator = make_loader_generator(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=True,
        generator=loader_generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        generator=val_loader_generator,
        worker_init_fn=seed_worker,
    )

    # Optional model-selection split (keeps the reported val set untouched
    # by best-checkpoint selection — see KDTrainer docstring).
    select_loader = None
    if data_cfg.get("select_ann") and data_cfg.get("select_img"):
        logger.info("Model-selection split ENABLED "
                    f"({data_cfg['select_ann']}) — best checkpoint will be "
                    "chosen on this split; val is evaluated once at the end.")
        select_dataset = COCODetection(
            img_folder=data_cfg["select_img"],
            ann_file=data_cfg["select_ann"],
            transforms=val_transforms,
            remove_no_annotations=False,
        )
        logger.info(f"Select set: {len(select_dataset)} images")
        select_loader = DataLoader(
            select_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
            generator=make_loader_generator(args.seed),
            worker_init_fn=seed_worker,
        )

    # ---- Teacher mAP sanity gate ----
    # Refuses to train if the teacher is broken (random init, bad weights, etc).
    # Skipped for baseline (no KD) and when --skip-teacher-gate is set.
    if (
        effective_kd_type != "none"
        and not args.skip_teacher_gate
        and args.teacher_min_map > 0.0
    ):
        logger.info(
            f"Running teacher mAP sanity gate "
            f"({args.teacher_gate_num_images} images, threshold {args.teacher_min_map})..."
        )
        teacher.to(device)
        _teacher_map_gate(
            teacher=teacher,
            val_loader=val_loader,
            val_ann_file=data_cfg["val_ann"],
            device=device,
            num_images=args.teacher_gate_num_images,
            min_map=args.teacher_min_map,
            use_amp=train_cfg.get("use_amp", True),
        )

    # ---- Optimizer and scheduler ----
    optimizer = build_optimizer(model, train_cfg)

    # Also include KDLoss learnable parameters (e.g., FeatureKD projection)
    kd_params = [p for p in loss_fn.parameters() if p.requires_grad]
    if kd_params:
        optimizer.add_param_group({"params": kd_params, "lr": train_cfg["lr_head"]})

    total_iters = len(train_loader) * args.epochs // train_cfg["accumulate_steps"]
    scheduler = build_lr_scheduler(optimizer, train_cfg["warmup_iters"], total_iters)

    # ---- Train ----
    trainer = KDTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        select_loader=select_loader,
        select_ann_file=data_cfg.get("select_ann"),
        train_loader_generator=loader_generator,
        seed=args.seed,
    )

    # Resume from a full trainer checkpoint (has epoch / optimizer / scaler).
    # Backbone-only weight files (no 'epoch' key) fall back gracefully: model
    # was already loaded above with strict=False; start_epoch stays 0.
    start_epoch = resume_if_checkpoint(trainer, args.student_weights)

    logger.info(f"Starting from epoch {start_epoch + 1}")
    trainer.train(args.epochs, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
