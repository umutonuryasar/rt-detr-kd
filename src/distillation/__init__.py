"""Knowledge Distillation losses for RT-DETR."""

from .logit_kd import LogitKDLoss
from .feature_kd import FeatureKDLoss
from .cwd import CWDLoss
from .mgd import MGDLoss
from .query_kd import QueryKDLoss
from .stage_adaptive_kd import StageAdaptiveKDLoss
from .kd_loss import KDLoss, SUPPORTED_KD_TYPES

__all__ = [
    "LogitKDLoss",
    "FeatureKDLoss",
    "CWDLoss",
    "MGDLoss",
    "QueryKDLoss",
    "StageAdaptiveKDLoss",
    "KDLoss",
    "SUPPORTED_KD_TYPES",
]
