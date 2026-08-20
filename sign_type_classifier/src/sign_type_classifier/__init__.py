"""SAM3-assisted dataset tools for HL320 sign-type classification."""

from .clustering import SignCandidate, SignDetection, cluster_sign_detections
from .dataset import CLASSIFIER_CLASS_NAMES, CLASS_NAMES, NOT_A_SIGN_CLASS
from .detection_filters import filter_detections, reject_instance_by_geometry, reject_instance_by_size
from .features import build_numeric_features, extract_sign_crops, numeric_feature_names
from .model import SignTypeDecision, SignTypeEnsembleClassifier, blend_probabilities

__all__ = [
    "CLASS_NAMES",
    "CLASSIFIER_CLASS_NAMES",
    "NOT_A_SIGN_CLASS",
    "SignCandidate",
    "SignDetection",
    "SignTypeDecision",
    "SignTypeEnsembleClassifier",
    "blend_probabilities",
    "build_numeric_features",
    "cluster_sign_detections",
    "extract_sign_crops",
    "filter_detections",
    "numeric_feature_names",
    "reject_instance_by_geometry",
    "reject_instance_by_size",
]
