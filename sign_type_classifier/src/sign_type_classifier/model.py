"""Numeric and EfficientNet branches for reviewed sign-type classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dataset import CLASSIFIER_CLASS_NAMES


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SignTypeDecision:
    label: str
    confidence: float
    margin: float
    probabilities: tuple[float, ...]
    image_probabilities: tuple[float, ...]
    numeric_probabilities: tuple[float, ...]


def build_efficientnet_b0(*, pretrained: bool, num_classes: int = len(CLASSIFIER_CLASS_NAMES)) -> Any:
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


def build_numeric_mlp(
    *,
    num_features: int,
    num_classes: int = len(CLASSIFIER_CLASS_NAMES),
    hidden_size: int = 128,
    dropout: float = 0.20,
) -> Any:
    try:
        import torch.nn as nn  # noqa: WPS433
    except ImportError as exc:
        raise ImportError("The numeric sign classifier requires PyTorch") from exc
    if num_features <= 0 or hidden_size <= 0:
        raise ValueError("num_features and hidden_size must be positive")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError(f"dropout must be in [0,1), got {dropout}")
    return nn.Sequential(
        nn.Linear(int(num_features), int(hidden_size)),
        nn.ReLU(inplace=True),
        nn.Dropout(float(dropout)),
        nn.Linear(int(hidden_size), max(32, int(hidden_size) // 2)),
        nn.ReLU(inplace=True),
        nn.Dropout(float(dropout)),
        nn.Linear(max(32, int(hidden_size) // 2), int(num_classes)),
    )


def set_image_backbone_trainable(model: Any, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = bool(trainable)
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def compute_numeric_normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = _numeric_matrix(values)
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def standardize_numeric_features(values: np.ndarray, *, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    matrix = _numeric_matrix(values)
    mean_array = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_array = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean_array.shape != (matrix.shape[1],) or std_array.shape != mean_array.shape:
        raise ValueError(
            f"Normalization shape mismatch: values={matrix.shape}, mean={mean_array.shape}, std={std_array.shape}"
        )
    if not np.isfinite(mean_array).all() or not np.isfinite(std_array).all() or np.any(std_array <= 0.0):
        raise ValueError("Numeric normalization statistics must be finite and std must be positive")
    return ((matrix - mean_array[None, :]) / std_array[None, :]).astype(np.float32, copy=False)


def blend_probabilities(
    image_probabilities: np.ndarray,
    numeric_probabilities: np.ndarray,
    *,
    image_weight: float,
) -> np.ndarray:
    image_values = np.asarray(image_probabilities, dtype=np.float64)
    numeric_values = np.asarray(numeric_probabilities, dtype=np.float64)
    if image_values.shape != numeric_values.shape or image_values.ndim != 2:
        raise ValueError(
            f"Probability arrays must have equal [N,C] shapes, got {image_values.shape} and {numeric_values.shape}"
        )
    weight = float(image_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"image_weight must be in [0,1], got {weight}")
    blended = weight * image_values + (1.0 - weight) * numeric_values
    totals = blended.sum(axis=1, keepdims=True)
    if not np.isfinite(blended).all() or np.any(blended < 0.0) or np.any(totals <= 0.0):
        raise ValueError("Probabilities must be finite, non-negative, and have positive row sums")
    return (blended / totals).astype(np.float32)


def decisions_from_probabilities(
    probabilities: np.ndarray,
    *,
    image_probabilities: np.ndarray,
    numeric_probabilities: np.ndarray,
    class_names: Sequence[str] = CLASSIFIER_CLASS_NAMES,
) -> list[SignTypeDecision]:
    values = np.asarray(probabilities, dtype=np.float64)
    image_values = np.asarray(image_probabilities, dtype=np.float64)
    numeric_values = np.asarray(numeric_probabilities, dtype=np.float64)
    names = tuple(str(value) for value in class_names)
    if values.shape != image_values.shape or values.shape != numeric_values.shape:
        raise ValueError("Ensemble and branch probability shapes must match")
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(f"Expected probabilities with shape [N,{len(names)}], got {values.shape}")
    output = []
    for row, image_row, numeric_row in zip(values, image_values, numeric_values):
        order = np.argsort(-row, kind="stable")
        top = int(order[0])
        second = int(order[1]) if len(order) > 1 else top
        output.append(
            SignTypeDecision(
                label=names[top],
                confidence=float(row[top]),
                margin=float(row[top] - row[second]) if second != top else float(row[top]),
                probabilities=tuple(float(value) for value in row),
                image_probabilities=tuple(float(value) for value in image_row),
                numeric_probabilities=tuple(float(value) for value in numeric_row),
            )
        )
    return output


def resolve_device(value: str) -> str:
    import torch  # noqa: WPS433

    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


class SignTypeEnsembleClassifier:
    """Load one training checkpoint and classify sign context crops in batches."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        import torch  # noqa: WPS433

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Sign-type checkpoint does not exist: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid sign-type checkpoint: {self.checkpoint_path}")
        self.class_names = tuple(checkpoint.get("class_names", ()))
        if self.class_names != CLASSIFIER_CLASS_NAMES:
            raise ValueError(f"Checkpoint classes must be {CLASSIFIER_CLASS_NAMES}, got {self.class_names}")
        self.feature_names = tuple(str(value) for value in checkpoint.get("numeric_feature_names", ()))
        if not self.feature_names:
            raise ValueError("Checkpoint has no numeric_feature_names")
        self.input_size = int(checkpoint.get("input_size", 224))
        self.image_weight = float(checkpoint["fusion_image_weight"])
        self.numeric_mean = np.asarray(checkpoint["numeric_mean"], dtype=np.float32)
        self.numeric_std = np.asarray(checkpoint["numeric_std"], dtype=np.float32)
        self.device = resolve_device(device)

        self.image_model = build_efficientnet_b0(pretrained=False, num_classes=len(self.class_names))
        self.numeric_model = build_numeric_mlp(
            num_features=len(self.feature_names),
            num_classes=len(self.class_names),
            hidden_size=int(checkpoint.get("numeric_hidden_size", 128)),
            dropout=float(checkpoint.get("numeric_dropout", 0.20)),
        )
        self.image_model.load_state_dict(checkpoint["image_model_state_dict"], strict=True)
        self.numeric_model.load_state_dict(checkpoint["numeric_model_state_dict"], strict=True)
        self.image_model.to(self.device).eval()
        self.numeric_model.to(self.device).eval()

    def classify(
        self,
        context_images_rgb: Sequence[np.ndarray],
        numeric_features: np.ndarray,
    ) -> list[SignTypeDecision]:
        import torch  # noqa: WPS433

        if not context_images_rgb:
            return []
        matrix = standardize_numeric_features(
            numeric_features,
            mean=self.numeric_mean,
            std=self.numeric_std,
        )
        if matrix.shape[0] != len(context_images_rgb):
            raise ValueError(
                f"Image/numeric batch mismatch: {len(context_images_rgb)} images and {matrix.shape[0]} feature rows"
            )
        image_batch = np.stack(
            [classifier_tensor_array(image, size=self.input_size) for image in context_images_rgb],
            axis=0,
        )
        with torch.inference_mode():
            image_logits = self.image_model(torch.from_numpy(image_batch).to(self.device))
            numeric_logits = self.numeric_model(torch.from_numpy(matrix).to(self.device))
            image_probabilities = torch.softmax(image_logits, dim=1).cpu().numpy()
            numeric_probabilities = torch.softmax(numeric_logits, dim=1).cpu().numpy()
        probabilities = blend_probabilities(
            image_probabilities,
            numeric_probabilities,
            image_weight=self.image_weight,
        )
        return decisions_from_probabilities(
            probabilities,
            image_probabilities=image_probabilities,
            numeric_probabilities=numeric_probabilities,
            class_names=self.class_names,
        )


def classifier_tensor_array(image_rgb: np.ndarray, *, size: int) -> np.ndarray:
    from PIL import Image, ImageOps  # noqa: WPS433

    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image.shape}")
    pil_image = Image.fromarray(image, mode="RGB")
    contained = ImageOps.contain(pil_image, (size, size), method=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), color=(127, 127, 127))
    canvas.paste(contained, ((size - contained.width) // 2, (size - contained.height) // 2))
    values = np.asarray(canvas, dtype=np.float32) / 255.0
    values = (values - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.transpose(values, (2, 0, 1)).astype(np.float32, copy=False)


def _numeric_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"Numeric features must have non-empty shape [N,F], got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Numeric features contain NaN or infinity")
    return matrix
