"""SAM3-assisted dataset tools for HL320 sign-type classification."""

from .clustering import SignCandidate, SignDetection, cluster_sign_detections
from .dataset import CLASS_NAMES
from .detection_filters import filter_detections, reject_instance_by_geometry, reject_instance_by_size
from .features import build_numeric_features, extract_sign_crops, numeric_feature_names

__all__ = [
    "CLASS_NAMES",
    "SignCandidate",
    "SignDetection",
    "build_numeric_features",
    "cluster_sign_detections",
    "extract_sign_crops",
    "filter_detections",
    "numeric_feature_names",
    "reject_instance_by_geometry",
    "reject_instance_by_size",
]
