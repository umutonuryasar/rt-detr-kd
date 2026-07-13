"""Regression tests for the methodology/correctness fix batch.

Covers:
  1. Logit-KD binary mode (sigmoid-matched) — zero at identity, positive
     otherwise, gradients flow; softmax mode still available.
  2. Query-KD Hungarian matching — permutation-invariant where index
     matching is not; graceful zero-loss degradation when teacher queries
     are None (canonical-teacher case).
  3. Feature-KD / CWD per-scale alignment — matching scales are compared
     directly (no cross-scale blending); shape-sum validation.
  4. MosaicWrapper — output is normalized exactly once and tensor-input
     datasets are rejected (double-normalization guard).
  5. capture_attn=False — decoder stores no attention maps.
"""

import pytest
import torch

from src.distillation.logit_kd import LogitKDLoss
from src.distillation.query_kd import QueryKDLoss
from src.distillation.feature_kd import FeatureKDLoss, split_by_shapes
from src.distillation.cwd import CWDLoss


# ---------------------------------------------------------------------------
# 1. Logit-KD binary mode
# ---------------------------------------------------------------------------

def test_logit_kd_binary_zero_at_identity():
    """Binary KL must be ~0 when student equals teacher."""
    loss_fn = LogitKDLoss(temperature=4.0, mode="binary")
    logits = torch.randn(2, 8, 5)
    loss = loss_fn(logits, logits.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_logit_kd_binary_positive_and_grads():
    loss_fn = LogitKDLoss(temperature=4.0, mode="binary")
    s = torch.randn(2, 8, 5, requires_grad=True)
    t = torch.randn(2, 8, 5)
    loss = loss_fn(s, t)
    assert loss.item() > 0
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0


def test_logit_kd_softmax_mode_still_available():
    loss_fn = LogitKDLoss(temperature=4.0, mode="softmax")
    s = torch.randn(2, 8, 5, requires_grad=True)
    t = torch.randn(2, 8, 5)
    loss = loss_fn(s, t)
    assert torch.isfinite(loss)


def test_logit_kd_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        LogitKDLoss(mode="categorical")


# ---------------------------------------------------------------------------
# 2. Query-KD Hungarian matching
# ---------------------------------------------------------------------------

def test_query_kd_hungarian_permutation_invariant():
    """Permuting teacher queries must not change the Hungarian loss.

    This is the property index-wise truncation lacks — and the reason the
    matching fix exists.
    """
    torch.manual_seed(0)
    s_q = torch.randn(2, 6, 16)
    t_q = torch.randn(2, 8, 16)

    hungarian = QueryKDLoss(16, 16, matching="hungarian")
    base = hungarian(s_q, t_q)["loss_query"]

    perm = torch.randperm(8)
    permuted = hungarian(s_q, t_q[:, perm, :])["loss_query"]
    assert permuted.item() == pytest.approx(base.item(), rel=1e-5)


def test_query_kd_index_mode_is_permutation_sensitive():
    """Sanity check that the legacy ablation baseline behaves as documented."""
    torch.manual_seed(0)
    s_q = torch.randn(2, 6, 16)
    t_q = torch.randn(2, 8, 16)

    index = QueryKDLoss(16, 16, matching="index")
    base = index(s_q, t_q)["loss_query"]
    perm = torch.randperm(8)
    permuted = index(s_q, t_q[:, perm, :])["loss_query"]
    # With random tensors an order-changing permutation virtually always
    # changes the index-matched loss.
    assert permuted.item() != pytest.approx(base.item(), rel=1e-6)


def test_query_kd_prediction_space_matching_runs_and_backprops():
    torch.manual_seed(0)
    s_q = torch.randn(2, 6, 16, requires_grad=True)
    t_q = torch.randn(2, 8, 16)
    s_preds = {"pred_logits": torch.randn(2, 6, 4), "pred_boxes": torch.rand(2, 6, 4)}
    t_preds = {"pred_logits": torch.randn(2, 8, 4), "pred_boxes": torch.rand(2, 8, 4)}

    loss_fn = QueryKDLoss(16, 16, matching="hungarian")
    out = loss_fn(s_q, t_q, student_preds=s_preds, teacher_preds=t_preds)
    assert torch.isfinite(out["loss_kd"])
    out["loss_kd"].backward()
    assert s_q.grad is not None and s_q.grad.abs().sum() > 0


def test_query_kd_none_teacher_degrades_to_zero():
    """Canonical-teacher case: no crash, zero loss, warning logged."""
    s_q = torch.randn(2, 6, 16)
    loss_fn = QueryKDLoss(16, 16)
    out = loss_fn(s_q, None)
    assert out["loss_kd"].item() == 0.0
    assert out["loss_query"].item() == 0.0


def test_query_kd_invalid_matching_rejected():
    with pytest.raises(ValueError, match="matching"):
        QueryKDLoss(matching="random")


# ---------------------------------------------------------------------------
# 3. Per-scale feature alignment
# ---------------------------------------------------------------------------

def test_feature_kd_per_scale_no_cross_scale_blending():
    """When shared scales are spatially identical, per-scale alignment must
    equal a direct MSE over those scales — the teacher's extra fine scale is
    dropped, and no interpolation happens."""
    torch.manual_seed(0)
    B, D = 2, 16
    # Student: C4 (4x4) + C5 (2x2) = 20 tokens. Teacher: C3 (8x8) + same = 84.
    s_shapes = [(4, 4), (2, 2)]
    t_shapes = [(8, 8), (4, 4), (2, 2)]
    s_enc = torch.randn(B, 20, D, requires_grad=True)
    t_enc = torch.randn(B, 84, D)

    loss_fn = FeatureKDLoss(student_dim=D, teacher_dim=D)
    out = loss_fn(s_enc, t_enc, student_shapes=s_shapes, teacher_shapes=t_shapes)

    # Manual reference: last 20 teacher tokens (C4+C5) vs student directly.
    ref = torch.nn.functional.mse_loss(
        s_enc.permute(0, 2, 1), t_enc.permute(0, 2, 1)[:, :, -20:]
    )
    assert out["loss_feat"].item() == pytest.approx(ref.item(), rel=1e-5)
    out["loss_kd"].backward()
    assert s_enc.grad is not None


def test_feature_kd_legacy_fallback_without_shapes():
    """Without shapes the old 1-D path must keep working (compat)."""
    s_enc = torch.randn(2, 20, 16, requires_grad=True)
    t_enc = torch.randn(2, 84, 16)
    loss_fn = FeatureKDLoss(student_dim=16, teacher_dim=16)
    out = loss_fn(s_enc, t_enc)
    assert torch.isfinite(out["loss_feat"])


def test_split_by_shapes_validates_token_count():
    with pytest.raises(ValueError, match="tokens"):
        split_by_shapes(torch.randn(2, 16, 19), [(4, 4), (2, 2)])


def test_cwd_per_scale_alignment_runs():
    s_enc = torch.randn(2, 20, 16, requires_grad=True)
    t_enc = torch.randn(2, 84, 16)
    loss_fn = CWDLoss(student_channels=16, teacher_channels=16)
    loss = loss_fn(s_enc, t_enc,
                   student_shapes=[(4, 4), (2, 2)],
                   teacher_shapes=[(8, 8), (4, 4), (2, 2)])
    assert torch.isfinite(loss)
    loss.backward()
    assert s_enc.grad is not None


# ---------------------------------------------------------------------------
# 4. Mosaic single normalization
# ---------------------------------------------------------------------------

class _TinyPILDataset:
    """Four solid-gray PIL images with one centered box each."""

    def __init__(self, value: int = 128, size: int = 64):
        from PIL import Image
        self.img = Image.new("RGB", (size, size), (value, value, value))

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        target = {
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            "labels": torch.tensor([1]),
            "image_id": idx,
            "orig_size": (64, 64),
        }
        return self.img.copy(), target


def test_mosaic_normalizes_exactly_once():
    """A solid 128-gray mosaic normalized ONCE lands on known values.

    Double normalization (the old bug) would push channel means far outside
    the expected band; this pins the correct single-normalization output.
    """
    from src.data.transforms import (
        MosaicWrapper, Compose, Resize, ToTensor, Normalize,
        _IMAGENET_MEAN, _IMAGENET_STD,
    )

    base = Compose([Resize(64), ToTensor(), Normalize()])
    wrapper = MosaicWrapper(_TinyPILDataset(), base_transform=base,
                            img_size=64, p=1.0)  # always mosaic
    img, target = wrapper[0]

    expected = torch.tensor(
        [(128 / 255 - m) / s for m, s in zip(_IMAGENET_MEAN, _IMAGENET_STD)]
    )
    got = img.mean(dim=(1, 2))
    assert torch.allclose(got, expected, atol=1e-2), (
        f"channel means {got.tolist()} != expected {expected.tolist()} — "
        f"normalization applied more than once?"
    )
    assert target["boxes"].shape[-1] == 4


def test_mosaic_rejects_pretransformed_dataset():
    """Tensor-returning datasets must be rejected (double-norm guard)."""
    from src.data.transforms import MosaicWrapper

    class _TensorDataset:
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return torch.rand(3, 64, 64), {"boxes": torch.zeros(0, 4),
                                           "labels": torch.zeros(0).long()}

    wrapper = MosaicWrapper(_TensorDataset(), base_transform=None,
                            img_size=64, p=1.0)
    with pytest.raises(TypeError, match="double normalization"):
        _ = wrapper[0]


# ---------------------------------------------------------------------------
# 5. capture_attn flag
# ---------------------------------------------------------------------------

def test_decoder_capture_attn_off_stores_nothing():
    from src.models.decoder import RTDETRDecoder

    dec = RTDETRDecoder(num_classes=4, hidden_dim=16, num_queries=6,
                        num_decoder_layers=2, nhead=2, dim_feedforward=32,
                        capture_attn=False)
    memory = torch.randn(2, 10, 16)
    out = dec(memory)
    assert out["pred_logits"].shape == (2, 6, 4)
    assert dec.get_attn_maps_tensor() is None


def test_decoder_capture_attn_on_stores_maps():
    from src.models.decoder import RTDETRDecoder

    dec = RTDETRDecoder(num_classes=4, hidden_dim=16, num_queries=6,
                        num_decoder_layers=2, nhead=2, dim_feedforward=32,
                        capture_attn=True)
    memory = torch.randn(2, 10, 16)
    _ = dec(memory)
    maps = dec.get_attn_maps_tensor()
    assert maps is not None and maps.shape == (2, 2, 2, 6, 10)
