"""EfficientNet-B0 model construction and mask-based orientation inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .preprocessing import classifier_tensor_array, extract_mask_crop


CLASS_NAMES = ("front", "rear", "other")


@dataclass(frozen=True)
class OrientationDecision:
    predicted_class: str
    semantic_label: str
    confidence: float
    margin: float
    probabilities: tuple[float, float, float]
    fallback: bool
    fallback_reason: str | None


def decide_orientation(
    probabilities: Sequence[float],
    *,
    min_confidence: float = 0.70,
    min_margin: float = 0.10,
) -> OrientationDecision:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.shape != (len(CLASS_NAMES),):
        raise ValueError(f"Expected {len(CLASS_NAMES)} probabilities, got {probs.shape}")
    if not np.isfinite(probs).all() or np.any(probs < 0.0):
        raise ValueError(f"Probabilities must be finite and non-negative, got {probs.tolist()}")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("Probability sum must be positive")
    probs = probs / total
    order = np.argsort(-probs, kind="stable")
    top_index, second_index = int(order[0]), int(order[1])
    predicted = CLASS_NAMES[top_index]
    confidence = float(probs[top_index])
    margin = confidence - float(probs[second_index])
    reason = None
    if predicted == "other":
        reason = "other"
    elif confidence < float(min_confidence):
        reason = "low_confidence"
    elif margin < float(min_margin):
        reason = "low_margin"
    fallback = reason is not None
    semantic_label = "front" if fallback else predicted
    return OrientationDecision(
        predicted_class=predicted,
        semantic_label=semantic_label,
        confidence=confidence,
        margin=margin,
        probabilities=tuple(float(value) for value in probs),
        fallback=fallback,
        fallback_reason=reason,
    )


def build_efficientnet_b0(*, pretrained: bool, num_classes: int = len(CLASS_NAMES)) -> Any:
    try:
        import torch.nn as nn  # noqa: WPS433
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0  # noqa: WPS433
    except ImportError as exc:
        raise ImportError("EfficientNet-B0 requires torch and torchvision") from exc
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    input_features = int(model.classifier[-1].in_features)
    model.classifier[-1] = nn.Linear(input_features, int(num_classes))
    return model


def set_backbone_trainable(model: Any, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = bool(trainable)
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def resolve_device(value: str) -> str:
    import torch  # noqa: WPS433

    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


class VehicleOrientationClassifier:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        import torch  # noqa: WPS433

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Vehicle orientation checkpoint does not exist: {self.checkpoint_path}")
        self.device = resolve_device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"Invalid vehicle orientation checkpoint: {self.checkpoint_path}")
        class_names = tuple(checkpoint.get("class_names", ()))
        if class_names != CLASS_NAMES:
            raise ValueError(f"Checkpoint classes must be {CLASS_NAMES}, got {class_names}")
        self.input_size = int(checkpoint.get("input_size", 224))
        self.crop_padding = float(checkpoint.get("crop_padding", 0.12))
        self.model = build_efficientnet_b0(pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

    def classify(
        self,
        image_rgb: np.ndarray,
        masks: Sequence[np.ndarray],
        *,
        min_confidence: float,
        min_margin: float,
    ) -> list[OrientationDecision]:
        import torch  # noqa: WPS433

        if not masks:
            return []
        tensors = []
        for mask in masks:
            crop = extract_mask_crop(image_rgb, mask, padding=self.crop_padding)
            tensors.append(classifier_tensor_array(crop.masked_rgb, size=self.input_size))
        batch = torch.from_numpy(np.stack(tensors, axis=0)).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(batch), dim=1).detach().cpu().numpy()
        return [
            decide_orientation(row, min_confidence=min_confidence, min_margin=min_margin)
            for row in probabilities
        ]

