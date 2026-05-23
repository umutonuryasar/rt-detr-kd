"""Smoke tests for every supported KD loss.

Each test builds the loss on CPU with tiny tensors, runs a forward pass,
verifies the output is finite, and confirms backward produces non-zero
gradients on at least one student-side parameter. These are the tests that
would have caught the CLI/YAML plumbing bug.
"""

import pytest
import torch

from src.distillation.kd_loss import KDLoss, SUPPORTED_KD_TYPES


def _make_model_outputs(num_classes=4, num_queries=8, hidden_dim=16,
                        num_layers=2, num_heads=2, n_tokens=10):
    """Synthesize the dict shape that RTDETRWithKD.forward() emits."""
    s_logits = torch.randn(2, num_queries, num_classes, requires_grad=True)
    s_boxes  = torch.rand(2, num_queries, 4, requires_grad=True)
    s_enc    = torch.randn(2, n_tokens, hidden_dim, requires_grad=True)
    s_attn   = torch.rand(num_layers, 2, num_heads, num_queries, n_tokens)
    s_queries = torch.randn(2, num_queries, hidden_dim, requires_grad=True)

    t_logits = torch.randn(2, num_queries, num_classes)
    t_boxes  = torch.rand(2, num_queries, 4)
    t_enc    = torch.randn(2, n_tokens, hidden_dim)
    t_attn   = torch.rand(num_layers, 2, num_heads, num_queries, n_tokens)
    t_queries = torch.randn(2, num_queries, hidden_dim)

    outputs = {
        "student": {"pred_logits": s_logits, "pred_boxes": s_boxes},
        "teacher": {"pred_logits": t_logits, "pred_boxes": t_boxes},
        "student_enc_out": s_enc, "teacher_enc_out": t_enc,
        "student_attn": s_attn,  "teacher_attn": t_attn,
        "student_queries": s_queries, "teacher_queries": t_queries,
    }
    student_params = [s_logits, s_boxes, s_enc, s_queries]
    return outputs, student_params


def _make_targets(num_classes=4):
    return [
        {"labels": torch.tensor([0, 1], dtype=torch.long),
         "boxes":  torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.2, 0.2, 0.1, 0.1]])},
        {"labels": torch.tensor([2], dtype=torch.long),
         "boxes":  torch.tensor([[0.7, 0.7, 0.2, 0.2]])},
    ]


@pytest.mark.parametrize("kd_type", SUPPORTED_KD_TYPES)
def test_kd_loss_forward_backward(kd_type):
    """Each KD type must: (a) instantiate with no errors, (b) run forward
    producing a finite total loss, (c) backward producing non-zero gradient
    on at least one student-side input."""
    outputs, student_params = _make_model_outputs(num_classes=4, hidden_dim=16)
    targets = _make_targets(num_classes=4)

    loss_fn = KDLoss(
        kd_type=kd_type, kd_lambda=1.0,
        num_classes=4, student_dim=16, teacher_dim=16,
        total_epochs=10,
    )
    losses = loss_fn(outputs, targets, epoch=0)

    assert "loss_total" in losses, f"{kd_type}: loss_total missing"
    assert torch.isfinite(losses["loss_total"]), f"{kd_type}: non-finite total loss"
    assert "loss_kd" in losses, f"{kd_type}: loss_kd missing"

    losses["loss_total"].backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in student_params)
    assert has_grad, f"{kd_type}: no student gradient flowed"


def test_invalid_kd_type_rejected():
    """KDLoss must reject unsupported kd_type strings."""
    with pytest.raises(ValueError, match="kd_type"):
        KDLoss(kd_type="not_a_real_kd_type")


@pytest.mark.parametrize("temperature", [-1.0, 0.0])
def test_logit_kd_bad_temperature(temperature):
    """Temperature must be > 0."""
    from src.distillation.logit_kd import LogitKDLoss
    with pytest.raises(ValueError):
        LogitKDLoss(temperature=temperature)


@pytest.mark.parametrize("mask_ratio", [-0.1, 0.0, 1.0, 1.1])
def test_mgd_bad_mask_ratio(mask_ratio):
    """mask_ratio must be in the open interval (0, 1)."""
    from src.distillation.mgd import MGDLoss
    with pytest.raises(ValueError):
        MGDLoss(mask_ratio=mask_ratio)


def test_cwd_zero_tau_rejected():
    from src.distillation.cwd import CWDLoss
    with pytest.raises(ValueError):
        CWDLoss(tau=0.0)


def test_stage_adaptive_weights_anneal_correctly():
    """w_feat should start at 1.0 and decay to 0.0 across training (cosine)."""
    from src.distillation.stage_adaptive_kd import StageAdaptiveKDLoss
    from src.distillation.feature_kd import FeatureKDLoss
    from src.distillation.logit_kd import LogitKDLoss

    sa = StageAdaptiveKDLoss(
        feature_loss=FeatureKDLoss(16, 16),
        logit_loss=LogitKDLoss(temperature=4.0),
        total_epochs=10,
    )
    w_feat_start, w_logit_start = sa._weights(0)
    w_feat_end,   w_logit_end   = sa._weights(10)
    assert w_feat_start == pytest.approx(1.0)
    assert w_logit_start == pytest.approx(0.0)
    assert w_feat_end == pytest.approx(0.0, abs=1e-6)
    assert w_logit_end == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "schedule,start_w_feat,end_w_feat",
    [
        ("cosine",          1.0, 0.0),
        ("linear",          1.0, 0.0),
        ("step",            1.0, 0.0),
        ("sigmoid",         pytest.approx(1.0, abs=0.01), pytest.approx(0.0, abs=0.01)),
        ("inverse_cosine",  0.0, 1.0),  # control: 0 → 1 instead of 1 → 0
    ],
)
def test_stage_adaptive_schedule_shapes(schedule, start_w_feat, end_w_feat):
    """Each schedule shape must produce the expected boundary weights."""
    from src.distillation.stage_adaptive_kd import StageAdaptiveKDLoss
    from src.distillation.feature_kd import FeatureKDLoss
    from src.distillation.logit_kd import LogitKDLoss

    sa = StageAdaptiveKDLoss(
        feature_loss=FeatureKDLoss(16, 16),
        logit_loss=LogitKDLoss(temperature=4.0),
        total_epochs=20,
        schedule=schedule,
    )
    w_feat_start, _ = sa._weights(0)
    w_feat_end,   _ = sa._weights(20)
    if isinstance(start_w_feat, float):
        assert w_feat_start == pytest.approx(start_w_feat, abs=1e-6)
    else:
        assert w_feat_start == start_w_feat
    if isinstance(end_w_feat, float):
        assert w_feat_end == pytest.approx(end_w_feat, abs=1e-6)
    else:
        assert w_feat_end == end_w_feat


def test_stage_adaptive_rejects_bad_schedule():
    from src.distillation.stage_adaptive_kd import StageAdaptiveKDLoss
    from src.distillation.feature_kd import FeatureKDLoss
    from src.distillation.logit_kd import LogitKDLoss
    with pytest.raises(ValueError, match="schedule"):
        StageAdaptiveKDLoss(
            feature_loss=FeatureKDLoss(16, 16),
            logit_loss=LogitKDLoss(),
            total_epochs=10,
            schedule="quadratic",
        )


def test_kd_loss_handles_missing_attention():
    """Feature/Query KD must tolerate None attention maps (e.g., during eval)."""
    outputs, _ = _make_model_outputs(hidden_dim=16)
    outputs["student_attn"] = None
    outputs["teacher_attn"] = None
    targets = _make_targets()

    for kd_type in ("feature", "query"):
        loss_fn = KDLoss(kd_type=kd_type, num_classes=4,
                         student_dim=16, teacher_dim=16, total_epochs=10)
        losses = loss_fn(outputs, targets)
        assert torch.isfinite(losses["loss_total"])
