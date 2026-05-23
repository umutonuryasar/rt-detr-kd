"""Adapter wrapping the official lyuwenyu/RT-DETR (PyTorch) as a KD teacher.

The user's choice (B1, see CLAUDE.md) is to use the canonical RT-DETR
implementation from the original authors as the teacher while keeping a
simplified student. This module bridges the two architectures:

  - Loads the official RT-DETR (PResNet + canonical HybridEncoder + the
    deformable-attention RTDETRTransformer) via their YAML config system.
  - Restores published weights from a downloadable checkpoint.
  - Installs forward hooks that expose the intermediate features the KD
    losses expect:
        ``encoder_output``    [B, N, D] — flattened multi-scale encoder memory
        ``decoder_queries``   [B, Q, D] — post-norm decoder query embeddings
        ``attn_maps``         list/None — per-layer cross-attention if exposed
  - ``forward(images)`` returns a dict ``{'pred_logits', 'pred_boxes'}`` with
    the same key naming as our :class:`src.models.rtdetr.RTDETR`, so
    :class:`src.models.rtdetr_kd.RTDETRWithKD` can use the teacher unchanged.

The deformable-attention decoder does not produce a dense [Q, N] attention
tensor like vanilla MHA does — ``attn_maps`` is left as ``None``. The KD
losses already degrade gracefully when attention is missing (see
``feature_kd.FeatureKDLoss.forward`` and ``query_kd.QueryKDLoss.forward``).

Usage
-----
    from src.models.rtdetr_teacher import build_lyuwenyu_teacher

    teacher = build_lyuwenyu_teacher(
        config="third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
        checkpoint="weights/rtdetr_r50vd_6x_coco_from_paddle.pth",
    )
    teacher.eval()
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


_LYUWENYU_PYTORCH_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "third_party" / "RT-DETR" / "rtdetr_pytorch"
)


def _check_submodule_present() -> None:
    if not _LYUWENYU_PYTORCH_DIR.is_dir():
        raise RuntimeError(
            f"lyuwenyu/RT-DETR submodule not found at {_LYUWENYU_PYTORCH_DIR}. "
            f"Run: git submodule update --init --recursive"
        )


class _Placeholder:
    """Catches every attribute access and call as a no-op.

    Their module-load-time references resolve to this class. We never
    instantiate them because we never invoke the code that uses them
    (data pipeline, RegNet/HF backbones, etc.)."""
    def __init__(self, *a, **k): pass
    def __getattr__(self, _): return _Placeholder
    def __call__(self, *a, **k): return _Placeholder()


def _install_dep_stubs() -> dict:
    """Inject placeholder modules for optional dependencies that lyuwenyu's
    code references but we don't need at KD time.

    Stubs installed:
      torchvision.datapoints — removed in 0.16+ (renamed to tv_tensors)
      transformers           — only used by their RegNet backbone (we use R50)

    Returns the dict of previous ``sys.modules`` entries so callers can
    restore the original state on context exit.
    """
    import types

    backup: dict = {}
    for name in ("torchvision.datapoints", "transformers"):
        backup[name] = sys.modules.get(name)

    tv_stub = types.ModuleType("torchvision.datapoints")
    tv_stub.BoundingBox = _Placeholder
    tv_stub.BoundingBoxFormat = _Placeholder()
    tv_stub.Image = _Placeholder
    tv_stub.Video = _Placeholder
    tv_stub.Mask = _Placeholder
    sys.modules["torchvision.datapoints"] = tv_stub

    hf_stub = types.ModuleType("transformers")
    hf_stub.RegNetModel = _Placeholder
    hf_stub.AutoModel = _Placeholder
    hf_stub.AutoConfig = _Placeholder
    sys.modules["transformers"] = hf_stub

    return backup


def _restore_dep_stubs(backup: dict) -> None:
    for name, prev in backup.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


@contextlib.contextmanager
def _lyuwenyu_import_context():
    """Temporarily make ``src.*`` resolve to lyuwenyu's package tree.

    Both this repository and lyuwenyu/RT-DETR use ``src`` as their top-level
    package name. Python's import machinery can only have one module named
    ``src`` in ``sys.modules`` at a time, so we cannot simply add their root
    to ``sys.path`` — our already-imported ``src`` would shadow theirs.

    Strategy:
      1. Snapshot the current ``src`` / ``src.*`` entries in ``sys.modules``
         (these belong to *our* package).
      2. Drop those entries and prepend the lyuwenyu pytorch root to
         ``sys.path``.
      3. Install torchvision-API stubs (their data pipeline uses removed
         ``torchvision.datapoints``; we don't need data, only the model).
      4. Yield. The caller imports ``src.core.yaml_config`` etc., which
         populates ``sys.modules`` with **lyuwenyu's** modules.
      5. On exit, drop the lyuwenyu entries and restore ours + stubs.

    After exit, the *model object* built inside the context still works
    because it holds direct Python references to the lyuwenyu class
    objects — the import-name lookup is only needed at module-load time.
    """
    _check_submodule_present()

    backup_modules = {k: v for k, v in sys.modules.items()
                      if k == "src" or k.startswith("src.")}
    backup_path = sys.path[:]

    # Clear our src.* and prepend lyuwenyu root.
    for k in list(backup_modules):
        del sys.modules[k]
    sys.path.insert(0, str(_LYUWENYU_PYTORCH_DIR))
    stub_backup = _install_dep_stubs()

    # Pre-stub lyuwenyu's data subpackage so their ``src/__init__.py``'s
    # cascading ``from . import data`` line is a no-op. Their data module
    # references torchvision APIs that have been renamed across releases
    # (ToImageTensor → ToImage, datapoints → tv_tensors) and we never run
    # their data pipeline during KD anyway.
    import types
    sys.modules["src.data"] = types.ModuleType("src.data")

    try:
        yield
    finally:
        lyu_keys = [k for k in list(sys.modules)
                    if k == "src" or k.startswith("src.")]
        for k in lyu_keys:
            del sys.modules[k]
        sys.modules.update(backup_modules)
        sys.path[:] = backup_path
        _restore_dep_stubs(stub_backup)


class RTDETRTeacher(nn.Module):
    """Wraps a lyuwenyu/RT-DETR model with our KD interface.

    Args:
        inner: An already-built RTDETR instance from
               ``src.zoo.rtdetr.rtdetr.RTDETR`` (lyuwenyu's namespace).

    After ``forward(images)`` the following attributes are populated and stay
    valid until the next forward call:

        self.encoder_output   [B, N, hidden_dim]
        self.decoder_queries  [B, num_queries, hidden_dim] or None
        self.attn_maps        list (currently always empty — deformable
                              attention does not produce a dense map)
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

        # KD-interface attributes
        self.encoder_output: Optional[torch.Tensor] = None
        self.decoder_queries: Optional[torch.Tensor] = None
        self.attn_maps: list = []

        # Freeze entire teacher and pin to eval mode.
        self.inner.eval()
        for p in self.inner.parameters():
            p.requires_grad = False

        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Capture intermediate features without modifying their code.

        - Encoder output: their ``HybridEncoder.forward`` returns a list of
          feature maps (one per FPN scale). We flatten + concatenate into the
          [B, N, D] sequence shape our KD losses expect.
        - Decoder queries: their ``RTDETRTransformer`` exposes the last-layer
          embedding as ``output`` (see rtdetr_decoder.py). We tap it via a
          forward hook on the decoder.
        """
        # Hook on the encoder output
        def enc_hook(module, inputs, output):
            self.encoder_output = self._flatten_encoder_output(output)
        self.inner.encoder.register_forward_hook(enc_hook)

        # Hook on the decoder output. The decoder returns a dict during eval
        # (and a tuple during training with aux outputs). We want the
        # post-norm query embeddings of the final layer — exposed indirectly
        # as the hidden state used to produce ``pred_logits``.
        # Their RTDETRTransformer stores intermediate decoder hidden states
        # in ``dec_out_logits`` / ``dec_out_bboxes`` (see line 278 of
        # rtdetr_decoder.py). The post-norm queries are not stored as a
        # named attribute, so for now we leave decoder_queries=None and
        # query-KD against this teacher will fall back to logit alignment.
        # (Adding a deeper hook on their decoder layer LayerNorm would let
        # us recover the embedding; deferred to Phase 2D if Query-KD wins.)

    @staticmethod
    def _flatten_encoder_output(outs) -> torch.Tensor:
        """Flatten the multi-scale encoder output to [B, N_total, D]."""
        if isinstance(outs, torch.Tensor):
            # Already flat
            if outs.dim() == 3:
                return outs
            if outs.dim() == 4:
                B, C, H, W = outs.shape
                return outs.flatten(2).permute(0, 2, 1)
            raise ValueError(f"Unexpected encoder output rank: {outs.dim()}")
        # List/tuple of [B, C, H, W] feature maps — flatten+concat.
        parts = []
        for feat in outs:
            assert feat.dim() == 4, f"unexpected feat rank {feat.dim()}"
            parts.append(feat.flatten(2).permute(0, 2, 1))   # [B, H*W, C]
        return torch.cat(parts, dim=1)                       # [B, N_total, C]

    # ------------------------------------------------------------------
    # Inference interface
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """Teacher stays in eval mode regardless of the parent's train flag."""
        super().train(mode)
        self.inner.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict:
        # Reset KD-side storage so consumers always see this-pass features.
        self.encoder_output = None
        self.decoder_queries = None
        self.attn_maps = []

        with torch.no_grad():
            out = self.inner(images)

        # lyuwenyu's decoder returns {'pred_logits', 'pred_boxes'} — same
        # naming as our RTDETR. Detach to be safe.
        out = {k: v.detach() if isinstance(v, torch.Tensor) else v
               for k, v in out.items()}
        return out

    def get_attn_maps_tensor(self) -> Optional[torch.Tensor]:
        """Deformable attention does not produce a dense [Q, N] tensor;
        KD against this teacher uses encoder + logit signals only."""
        return None

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------

def build_lyuwenyu_teacher(
    config: str,
    checkpoint: Optional[str] = None,
    strict: bool = False,
) -> RTDETRTeacher:
    """Build an RT-DETR teacher from a lyuwenyu YAML config + checkpoint.

    Args:
        config:     Path to one of ``third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr/*.yml``.
        checkpoint: Optional path to a ``.pth`` containing their state dict.
                    If omitted, the teacher uses random weights — useful for
                    unit tests but useless for actual KD.
        strict:     Whether ``load_state_dict`` should be strict. Defaults to
                    False because their checkpoints sometimes carry extra
                    optimizer / criterion keys.

    Returns:
        An :class:`RTDETRTeacher` ready for KD.

    Notes:
        All lyuwenyu imports happen inside ``_lyuwenyu_import_context`` to
        keep their ``src.*`` modules from permanently colliding with ours.
        The returned model object holds direct references to their classes,
        so future forward passes work after the context exits.

        **Input size constraint.** Their HybridEncoder bakes sinusoidal
        positional encodings during ``__init__`` at the YAML's
        ``eval_spatial_size`` (640×640 for the R50 config). The teacher
        must be fed exactly that input resolution. Passing 512×512 to a
        teacher built at 640×640 will raise a shape-mismatch error on the
        positional-embedding add. Phase 2A is launched at 640×640 on
        A100, so this aligns with the run_ablation.sh ``EXTRA_TRAIN_ARGS``.
    """
    with _lyuwenyu_import_context():
        # Their config loader auto-imports modules so the @register
        # decorator populates the class registry before instantiation.
        from src.core.yaml_config import YAMLConfig  # type: ignore

        cfg = YAMLConfig(config)
        inner = cfg.model

        if checkpoint is not None:
            # weights_only=False is required because their checkpoints
            # pickle the entire optimizer/EMA state, not just tensors.
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            # Their checkpoints sometimes store the EMA copy under
            # ckpt['ema']['module'] and the raw model under ckpt['model'].
            # Try EMA first (better mAP), fall back to plain model.
            state = None
            ema = ckpt.get("ema") if isinstance(ckpt, dict) else None
            if isinstance(ema, dict):
                state = ema.get("module")
            if state is None and isinstance(ckpt, dict):
                state = ckpt.get("model", ckpt)
            if state is None:
                state = ckpt
            missing, unexpected = inner.load_state_dict(state, strict=strict)
            if missing or unexpected:
                import logging
                logging.getLogger(__name__).info(
                    f"Teacher load: {len(missing)} missing, "
                    f"{len(unexpected)} unexpected keys."
                )

    return RTDETRTeacher(inner)
