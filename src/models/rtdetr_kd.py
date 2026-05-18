"""RT-DETR wrapper for Knowledge Distillation.

Runs the student in training mode and the teacher frozen in eval mode.
Exposes all intermediate features required by both logit-KD and feature-KD.

Returned dictionary structure from forward():
    {
        'student': {
            'pred_logits': [B, Q_s, num_classes],
            'pred_boxes':  [B, Q_s, 4],
        },
        'teacher': {
            'pred_logits': [B, Q_t, num_classes],
            'pred_boxes':  [B, Q_t, 4],
        },
        'student_enc_out':  [B, N_s, D],   # encoder output
        'teacher_enc_out':  [B, N_t, D],
        'student_attn':     Tensor [L, B, H, Q_s, N_s] or None,
        'teacher_attn':     Tensor [L, B, H, Q_t, N_t] or None,
        'student_queries':  Tensor [B, Q_s, D] or None,   # decoder queries
        'teacher_queries':  Tensor [B, Q_t, D] or None,
    }
"""

import torch
import torch.nn as nn
from typing import Optional

from .rtdetr import RTDETR


class RTDETRWithKD(nn.Module):
    """Wrapper combining student and teacher RTDETR models for KD training.

    The teacher's parameters are frozen and the model is kept in eval mode
    throughout training. All teacher computations run under torch.no_grad().

    Args:
        student: Student RTDETR model (will be trained).
        teacher: Teacher RTDETR model (frozen, eval mode).
    """

    def __init__(self, student: RTDETR, teacher: RTDETR):
        super().__init__()
        self.student = student
        self.teacher = teacher

        # Freeze teacher completely
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        """Freeze all teacher parameters and set to eval mode."""
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "RTDETRWithKD":
        """Override train() to always keep teacher in eval mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict:
        """Run both student and teacher forward passes.

        Args:
            images: Input image batch [B, 3, H, W].

        Returns:
            Dictionary with student outputs, teacher outputs, and all
            intermediate features needed for knowledge distillation.
        """
        # ---- Student forward (training mode, gradients flow) ----
        student_out = self.student(images)
        student_enc = self.student.encoder_output           # [B, N_s, D]
        student_attn = self.student.get_attn_maps_tensor()  # [L, B, H, Q_s, N_s] or None
        student_queries = self.student.decoder_queries      # [B, Q_s, D] or None

        # ---- Teacher forward (eval mode, no gradients) ----
        with torch.no_grad():
            teacher_out = self.teacher(images)
            teacher_enc = self.teacher.encoder_output           # [B, N_t, D]
            teacher_attn = self.teacher.get_attn_maps_tensor()  # [L, B, H, Q_t, N_t] or None
            teacher_queries = self.teacher.decoder_queries      # [B, Q_t, D] or None

            # Detach to be safe (no_grad already prevents grad flow, but
            # explicit detach ensures nothing leaks through clone/cat ops)
            teacher_out = {k: v.detach() for k, v in teacher_out.items()}
            teacher_enc = teacher_enc.detach() if teacher_enc is not None else None
            teacher_attn = teacher_attn.detach() if teacher_attn is not None else None
            teacher_queries = teacher_queries.detach() if teacher_queries is not None else None

        return {
            "student": student_out,
            "teacher": teacher_out,
            "student_enc_out": student_enc,
            "teacher_enc_out": teacher_enc,
            "student_attn": student_attn,
            "teacher_attn": teacher_attn,
            "student_queries": student_queries,
            "teacher_queries": teacher_queries,
        }

    @property
    def num_student_parameters(self) -> int:
        return self.student.num_parameters

    @property
    def num_teacher_parameters(self) -> int:
        return self.teacher.num_parameters


def build_kd_model(student_cfg: dict, teacher_cfg: dict) -> RTDETRWithKD:
    """Build an RTDETRWithKD model from config dictionaries.

    Args:
        student_cfg: Config dict for the student model.
        teacher_cfg: Config dict for the teacher model.

    Returns:
        RTDETRWithKD instance with frozen teacher.
    """
    from .rtdetr import build_rtdetr

    student = build_rtdetr(student_cfg)
    teacher = build_rtdetr(teacher_cfg)
    return RTDETRWithKD(student, teacher)
