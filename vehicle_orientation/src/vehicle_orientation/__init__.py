"""Vehicle orientation dataset, training, and SAM3 inference helpers."""

from .model import CLASS_NAMES, OrientationDecision, decide_orientation
from .preprocessing import (
    DEFAULT_FISHEYE_CONFIG,
    CropResult,
    FisheyeConfig,
    build_split_circular_mask,
    deduplicate_mask_indices,
    extract_mask_crop,
)

__all__ = [
    "CLASS_NAMES",
    "DEFAULT_FISHEYE_CONFIG",
    "CropResult",
    "FisheyeConfig",
    "OrientationDecision",
    "build_split_circular_mask",
    "decide_orientation",
    "deduplicate_mask_indices",
    "extract_mask_crop",
]
