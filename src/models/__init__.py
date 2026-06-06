"""
Origin and attribution:
  Project: DP-SCL.
  Purpose: Public model exports.

Reference sources:
  DP-SCL model exports include project-specific temporal encoders and a
  supervised contrastive objective based on Khosla et al.:
  https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html
"""

from .common import (
    ActionWeightedInput,
    AugmentationModule,
    EarlyPredictionMask,
    LearnableQueryPool,
    MySelfAttention,
    SupConLoss,
)
from .supcon import DPSCLModel, SupConClassifier, SupConEncoder, SupConProjectionHead

__all__ = [
    "ActionWeightedInput",
    "AugmentationModule",
    "EarlyPredictionMask",
    "LearnableQueryPool",
    "MySelfAttention",
    "SupConLoss",
    "SupConClassifier",
    "SupConEncoder",
    "DPSCLModel",
    "SupConProjectionHead",
]
