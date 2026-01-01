"""
LSE-DINOv2: Local Scale Equivariant DINOv2

A DINOv2 Vision Transformer enhanced with Deep Equilibrium Model (DEM)
based local scale adaptation for improved scale equivariance.
"""

from .configuration_lse_dinov2 import LSEDinoV2Config
from .modeling_lse_dinov2 import (
    LSEDinoV2ForImageClassification,
    LSEDinoV2PreTrainedModel,
    DEMAdapter,
    PerlayerAdapterParams,
)

__all__ = [
    "LSEDinoV2Config",
    "LSEDinoV2ForImageClassification",
    "LSEDinoV2PreTrainedModel",
    "DEMAdapter",
    "PerlayerAdapterParams",
]

