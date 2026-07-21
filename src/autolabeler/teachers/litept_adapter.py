from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import traceback
import types
from typing import Any

import numpy as np


class LitePTUnavailableError(RuntimeError):
    pass


@dataclass
class LitePTModelBundle:
    model: Any
    cfg: Any
    device: Any
    device_name: str
    device_capability: tuple[int, int] | None
    num_classes: int
    class_names: list[str]
    pointrope_backend: str
    dataset_name: str
    config_file: str
    checkpoint_file: str
    training_id_to_source_id: list[int] | None = None
    output_ignore_index: int = 255


def _add_litept_to_path(litept_root: str) -> None:
    root = str(Path(litept_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_litept_model(
    *,
    litept_root: str,
    checkpoint: str,
    config_path: str,
    litept_dataset: str,
    litept_config: str | None,
    device: str | None = None,
    force_torch_pointrope: bool = False,
) -> LitePTModelBundle:
    root = Path(litept_root).resolve()
    dataset = _validate_litept_dataset(litept_dataset)
    config_file = _resolve_litept_config(root, dataset, litept_config, validate_exists=True)
    checkpoint_file = _resolve_checkpoint_path(Path(checkpoint).resolve())
    pointrope_backend = "cuda"
    if force_torch_pointrope:
        _install_torch_pointrope_module(root)
        pointrope_backend = "torch"

    try:
        import torch
        from models import build_model
        from utils.config import Config
    except Exception as exc:  # pragma: no cover - depends on external LitePT env
        raise LitePTUnavailableError(
            "Failed to import LitePT runtime. Run this script inside the LitePT conda env "
            "and make sure LitePT dependencies/custom ops are installed. "
            f"litept_root={root} error={type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc(limit=12)}"
        ) from exc

    try:
        cfg = Config.fromfile(str(config_file))
        model = build_model(cfg.model)
    except Exception as exc:  # pragma: no cover - depends on external LitePT env
        raise LitePTUnavailableError(
            f"Failed to build LitePT model from config: {config_file} "
            f"error={type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}"
        ) from exc

    checkpoint_obj = _torch_load_checkpoint(torch, checkpoint_file)
    state_dict = _extract_state_dict(checkpoint_obj)
    state_dict = _normalize_state_dict_keys(state_dict, model.state_dict().keys())
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise LitePTUnavailableError(
            f"Checkpoint does not match LitePT config. checkpoint={checkpoint_file} config={config_file} "
            f"error={type(exc).__name__}: {exc}"
        ) from exc

    torch_device = _select_litept_device(torch, device)
    device_name, device_capability = _describe_torch_device(torch, torch_device)
    _validate_litept_device(torch_device, device_name, device_capability)
    print(
        f"LitePT device: {torch_device} name={device_name} "
        f"capability={device_capability} pointrope={pointrope_backend}",
        flush=True,
    )
    model.to(torch_device)
    model.eval()

    class_names = list(getattr(getattr(cfg, "data", {}), "names", []))
    num_classes = int(getattr(getattr(cfg, "data", {}), "num_classes", getattr(cfg.model, "num_classes", 0)))
    if not class_names:
        class_names = [f"class_{index}" for index in range(num_classes)]
    training_id_to_source_id = list(getattr(cfg, "training_id_to_source_id", range(num_classes)))
    if len(training_id_to_source_id) != num_classes:
        raise LitePTUnavailableError(
            f"training_id_to_source_id must contain {num_classes} ids, got {len(training_id_to_source_id)} "
            f"in config: {config_file}"
        )
    training_id_to_source_id = [int(value) for value in training_id_to_source_id]
    output_ignore_index = int(getattr(cfg, "output_ignore_index", 255))
    if any(value < 0 or value > np.iinfo(np.uint16).max for value in training_id_to_source_id):
        raise LitePTUnavailableError("training_id_to_source_id values must fit uint16")
    if output_ignore_index < 0 or output_ignore_index > np.iinfo(np.uint16).max:
        raise LitePTUnavailableError("output_ignore_index must fit uint16")

    return LitePTModelBundle(
        model=model,
        cfg=cfg,
        device=torch_device,
        device_name=device_name,
        device_capability=device_capability,
        num_classes=num_classes,
        class_names=class_names,
        pointrope_backend=pointrope_backend,
        dataset_name=dataset,
        config_file=str(config_file),
        checkpoint_file=str(checkpoint_file),
        training_id_to_source_id=training_id_to_source_id,
        output_ignore_index=output_ignore_index,
    )


def normalize_litept_strength(intensity: np.ndarray) -> np.ndarray:
    strength = np.asarray(intensity, dtype=np.float32)
    finite = np.isfinite(strength)
    if finite.any() and float(np.nanmax(strength[finite])) > 1.5:
        strength = strength / 255.0
    return strength.astype(np.float32, copy=False)


def remap_training_predictions(labels: np.ndarray, training_id_to_source_id: list[int]) -> np.ndarray:
    training_labels = np.asarray(labels)
    if not np.issubdtype(training_labels.dtype, np.integer):
        raise ValueError(f"Training predictions must have integer dtype, got {training_labels.dtype}")
    source_ids = np.asarray(training_id_to_source_id, dtype=np.uint16)
    if training_labels.size and (
        int(training_labels.min()) < 0 or int(training_labels.max()) >= source_ids.size
    ):
        raise ValueError(f"Training predictions contain ids outside [0, {source_ids.size - 1}]")
    return source_ids[training_labels]


_normalize_strength = normalize_litept_strength


def _validate_litept_dataset(litept_dataset: str) -> str:
    if litept_dataset not in {"custom", "waymo", "nuscenes"}:
        raise ValueError(f"Unsupported LitePT dataset: {litept_dataset}. Expected one of: custom, waymo, nuscenes")
    return litept_dataset


def _resolve_litept_config(
    root: Path,
    litept_dataset: str,
    litept_config: str | None,
    *,
    validate_exists: bool = True,
) -> Path:
    if litept_config is None:
        if litept_dataset == "custom":
            raise ValueError("--litept-config is required when --litept-dataset custom")
        path = root / "configs" / litept_dataset / "semseg-litept-small-v1m1.py"
    else:
        candidate = Path(litept_config).expanduser()
        path = candidate if candidate.is_absolute() else root / candidate
    if validate_exists and (not path.exists() or not path.is_file()):
        raise FileNotFoundError(f"LitePT config does not exist: {path}")
    return path.resolve()


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"LitePT checkpoint does not exist: {path}")
    candidates = []
    for pattern in ("*.pth", "*.pt", "*.ckpt"):
        candidates.extend(sorted(path.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(f"No .pth/.pt/.ckpt checkpoint files found in: {path}")
    if len(candidates) > 1:
        names = ", ".join(candidate.name for candidate in candidates[:8])
        raise LitePTUnavailableError(
            f"Checkpoint path is a directory with multiple checkpoint files. "
            f"Pass one file explicitly. directory={path} candidates={names}"
        )
    return candidates[0].resolve()


def _torch_load_checkpoint(torch_module, checkpoint_file: Path):
    try:
        return torch_module.load(str(checkpoint_file), map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(str(checkpoint_file), map_location="cpu")


def _extract_state_dict(checkpoint_obj) -> OrderedDict:
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return OrderedDict(checkpoint_obj[key])
        return OrderedDict(checkpoint_obj)
    raise LitePTUnavailableError("Unsupported LitePT checkpoint format: expected a dict/state_dict")


def _normalize_state_dict_keys(state_dict: OrderedDict, model_keys) -> OrderedDict:
    model_keys = set(model_keys)
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(key.startswith("module.") for key in keys) and not any(key.startswith("module.") for key in model_keys):
        return OrderedDict((key[7:], value) for key, value in state_dict.items())
    if not any(key.startswith("module.") for key in keys) and any(key.startswith("module.") for key in model_keys):
        return OrderedDict((f"module.{key}", value) for key, value in state_dict.items())
    return state_dict


def _select_litept_device(torch_module, requested_device: str | None):
    if requested_device:
        return torch_module.device(requested_device)
    if not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    for index in range(torch_module.cuda.device_count()):
        major, _minor = torch_module.cuda.get_device_capability(index)
        if major >= 8:
            return torch_module.device(f"cuda:{index}")
    return torch_module.device("cuda:0")


def _describe_torch_device(torch_module, device) -> tuple[str, tuple[int, int] | None]:
    if device.type != "cuda":
        return str(device), None
    index = device.index if device.index is not None else torch_module.cuda.current_device()
    return torch_module.cuda.get_device_name(index), tuple(torch_module.cuda.get_device_capability(index))


def _validate_litept_device(device, device_name: str, device_capability: tuple[int, int] | None) -> None:
    if device.type != "cuda":
        raise LitePTUnavailableError(f"LitePT semantic config requires a CUDA GPU. Selected device={device}")
    if device_capability is None or device_capability[0] < 8:
        raise LitePTUnavailableError(
            "LitePT semantic config uses FlashAttention, which requires Ampere or newer GPU "
            f"(compute capability >= 8.0). Selected device={device} name={device_name} "
            f"capability={device_capability}. Use --force-torch-pointrope when the LitePT environment supports "
            "the PyTorch fallback, or select a compatible CUDA device."
        )


def _install_torch_pointrope_module(litept_root: Path) -> None:
    module_path = litept_root / "libs" / "pointrope" / "pointrope_torch.py"
    if not module_path.exists():
        raise FileNotFoundError(f"LitePT torch PointROPE fallback does not exist: {module_path}")
    spec = importlib.util.spec_from_file_location("_autolabeler_litept_pointrope_torch", module_path)
    if spec is None or spec.loader is None:
        raise LitePTUnavailableError(f"Failed to load torch PointROPE fallback from: {module_path}")
    torch_pointrope = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(torch_pointrope)

    pointrope_package = types.ModuleType("libs.pointrope")
    pointrope_package.PointROPE = torch_pointrope.PointROPE
    pointrope_package.__all__ = ["PointROPE"]
    pointrope_package.__file__ = str(module_path)
    sys.modules["libs.pointrope"] = pointrope_package
