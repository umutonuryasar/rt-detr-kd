"""Smoke tests for RTDETR and RTDETRWithKD wrapper.

These cover the load-bearing invariants that the KD pipeline relies on:
output shape, intermediate-feature storage, and frozen-teacher gradient
isolation.
"""

import pytest
import torch

from src.models.rtdetr import RTDETR
from src.models.rtdetr_kd import RTDETRWithKD


# Tiny model — order-of-magnitude smaller than the production student to
# keep CI runtime under a second.
TINY_KW = dict(
    backbone_name="resnet18",
    num_classes=4,
    hidden_dim=32,
    num_queries=8,
    num_decoder_layers=1,
    nhead=2,
    dim_feedforward=64,
    dropout=0.0,
    num_encoder_layers=1,
    pretrained_backbone=False,  # skip torchvision download in CI
    freeze_stages=0,
)


def test_rtdetr_forward_shape(tiny_images):
    model = RTDETR(**TINY_KW)
    model.eval()
    out = model(tiny_images)
    assert "pred_logits" in out and "pred_boxes" in out
    B = tiny_images.size(0)
    assert out["pred_logits"].shape == (B, TINY_KW["num_queries"], TINY_KW["num_classes"])
    assert out["pred_boxes"].shape  == (B, TINY_KW["num_queries"], 4)
    # Boxes are sigmoid'ed → must lie in [0, 1]
    assert ((out["pred_boxes"] >= 0) & (out["pred_boxes"] <= 1)).all()


def test_rtdetr_exposes_kd_features(tiny_images):
    """encoder_output, decoder_queries, and attn_maps must be populated."""
    model = RTDETR(**TINY_KW)
    model.eval()
    _ = model(tiny_images)
    assert model.encoder_output is not None
    assert model.encoder_output.dim() == 3  # [B, N, D]
    assert model.decoder_queries is not None
    assert model.decoder_queries.shape == (
        tiny_images.size(0), TINY_KW["num_queries"], TINY_KW["hidden_dim"]
    )
    attn = model.get_attn_maps_tensor()
    assert attn is not None
    # [L, B, H, Q, N]
    assert attn.shape[0] == TINY_KW["num_decoder_layers"]
    assert attn.shape[1] == tiny_images.size(0)


def test_kd_wrapper_freezes_teacher(tiny_images):
    student = RTDETR(**TINY_KW)
    teacher = RTDETR(**TINY_KW)
    kd = RTDETRWithKD(student, teacher)

    # Teacher must report no trainable parameters
    teacher_trainable = sum(p.requires_grad for p in teacher.parameters())
    assert teacher_trainable == 0

    # Teacher must remain in eval mode after kd.train()
    kd.train()
    assert not teacher.training
    assert student.training

    # Forward pass populates both
    out = kd(tiny_images)
    assert "student" in out and "teacher" in out
    assert out["student_enc_out"].requires_grad
    assert not out["teacher_enc_out"].requires_grad


def test_kd_wrapper_no_teacher_grad_leakage(tiny_images):
    """Gradient flow into teacher params after backward must be exactly zero."""
    student = RTDETR(**TINY_KW)
    teacher = RTDETR(**TINY_KW)
    kd = RTDETRWithKD(student, teacher)

    out = kd(tiny_images)
    loss = out["student"]["pred_logits"].sum() + out["teacher_enc_out"].sum()
    loss.backward()

    for p in teacher.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, (
            "Teacher parameter received gradient — KD wrapper is leaking."
        )


def test_topk_decode_shape():
    """The eval-time top-k helper must return correctly-shaped tensors."""
    from src.trainer_kd import _topk_decode
    B, Q, C = 3, 50, 80
    logits = torch.randn(B, Q, C)
    boxes  = torch.rand(B, Q, 4)
    scores, labels, decoded = _topk_decode(logits, boxes, top_k=20)
    assert scores.shape == (B, 20)
    assert labels.shape == (B, 20)
    assert decoded.shape == (B, 20, 4)
    # Labels must be valid class indices
    assert (labels >= 0).all() and (labels < C).all()
    # Scores must be in (0, 1) (sigmoid output)
    assert (scores >= 0).all() and (scores <= 1).all()
