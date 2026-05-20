from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import importlib.util
from pathlib import Path
import sys
import traceback
import types
from typing import Any

import numpy as np

from ..data.bin_loader import H, W
from ..data.dataset_indexer import build_dataset_index
from ..data.frame_loader import load_frame
from ..data.schemas import SemanticSegmentationResult


@dataclass(frozen=True)
class LitePTInferencePlan:
    litept_root: str
    checkpoint: str
    litept_config: str
    litept_dataset: str
    input_dir: str
    output_dir: str
    frame_count: int
    frame_ids: list[str]
    max_frames: int | None = None


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


def build_litept_inference_plan(
    *,
    litept_root: str,
    checkpoint: str | None,
    input_dir: str,
    output_dir: str,
    input_format: str = "auto",
    litept_dataset: str = "nuscenes",
    litept_config: str | None = None,
    validate_checkpoint: bool = True,
    max_frames: int | None = None,
) -> LitePTInferencePlan:
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    root = Path(litept_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"LitePT root does not exist or is not a directory: {root}")

    dataset = _validate_litept_dataset(litept_dataset)
    ckpt = _resolve_litept_checkpoint(root, dataset, checkpoint)
    if validate_checkpoint and not ckpt.exists():
        raise FileNotFoundError(f"LitePT checkpoint does not exist: {ckpt}")
    config_file = _resolve_litept_config(root, dataset, litept_config, validate_exists=False)

    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input dir does not exist or is not a directory: {input_path}")

    records = build_dataset_index(str(input_path), input_format=input_format)
    if max_frames is not None:
        records = records[:max_frames]
    return LitePTInferencePlan(
        litept_root=str(root),
        checkpoint=str(ckpt),
        litept_config=str(config_file),
        litept_dataset=dataset,
        input_dir=str(input_path),
        output_dir=str(Path(output_dir).expanduser().resolve()),
        frame_count=len(records),
        frame_ids=[r.frame_id for r in records],
        max_frames=max_frames,
    )


def run_litept_inference(
    *,
    litept_root: str,
    checkpoint: str | None,
    input_dir: str,
    output_dir: str,
    input_format: str = "auto",
    config_path: str = "configs/classes.yaml",
    litept_dataset: str = "nuscenes",
    litept_config: str | None = None,
    max_frames: int | None = None,
    device: str | None = None,
    force_torch_pointrope: bool = False,
) -> list[SemanticSegmentationResult]:
    plan = build_litept_inference_plan(
        litept_root=litept_root,
        checkpoint=checkpoint,
        input_dir=input_dir,
        output_dir=output_dir,
        input_format=input_format,
        litept_dataset=litept_dataset,
        litept_config=litept_config,
        validate_checkpoint=True,
        max_frames=max_frames,
    )
    _add_litept_to_path(plan.litept_root)

    records = build_dataset_index(plan.input_dir, input_format=input_format)
    if max_frames is not None:
        records = records[:max_frames]
    model = _load_litept_model(
        litept_root=plan.litept_root,
        checkpoint=plan.checkpoint,
        config_path=config_path,
        litept_dataset=plan.litept_dataset,
        litept_config=plan.litept_config,
        device=device,
        force_torch_pointrope=force_torch_pointrope,
    )

    results: list[SemanticSegmentationResult] = []
    for record in records:
        frame = load_frame(record.lidar_path, record.frame_id, input_format=input_format)
        semantic_mask, confidence_mask = _predict_frame(model, frame.points_range)
        results.append(
            SemanticSegmentationResult(
                frame_id=record.frame_id,
                semantic_mask=semantic_mask,
                confidence_mask=confidence_mask,
                pseudo_label_version="litept_pretrained_v0",
                provenance="student_predicted",
                metadata={
                    "label_space": f"litept_{model.dataset_name}_semseg",
                    "litept_dataset": model.dataset_name,
                    "litept_config": model.config_file,
                    "checkpoint": model.checkpoint_file,
                    "class_names": model.class_names,
                    "num_classes": model.num_classes,
                    "ignore_index": 255,
                    "device": str(model.device),
                    "device_name": model.device_name,
                    "device_capability": model.device_capability,
                    "pointrope_backend": model.pointrope_backend,
                },
            )
        )
    return results


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
):
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
        from utils.config import Config
        from models import build_model  # imports model registries as side effects
    except Exception as exc:  # pragma: no cover - depends on external LitePT env
        raise LitePTUnavailableError(
            "Failed to import LitePT runtime. Run this script inside the LItePT conda env "
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
        class_names = [f"class_{i}" for i in range(num_classes)]

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
    )


def _select_litept_device(torch_module, requested_device: str | None):
    if requested_device:
        return torch_module.device(requested_device)
    if not torch_module.cuda.is_available():
        return torch_module.device("cpu")

    for idx in range(torch_module.cuda.device_count()):
        major, _minor = torch_module.cuda.get_device_capability(idx)
        if major >= 8:
            return torch_module.device(f"cuda:{idx}")
    return torch_module.device("cuda:0")


def _describe_torch_device(torch_module, device) -> tuple[str, tuple[int, int] | None]:
    if device.type != "cuda":
        return str(device), None
    index = device.index if device.index is not None else torch_module.cuda.current_device()
    return torch_module.cuda.get_device_name(index), tuple(torch_module.cuda.get_device_capability(index))


def _validate_litept_device(device, device_name: str, device_capability: tuple[int, int] | None) -> None:
    if device.type != "cuda":
        raise LitePTUnavailableError(
            "LitePT semantic config uses FlashAttention and requires a CUDA GPU. "
            f"Selected device={device}"
        )
    if device_capability is None or device_capability[0] < 8:
        raise LitePTUnavailableError(
            "LitePT semantic config uses FlashAttention, which requires Ampere or newer GPU "
            f"(compute capability >= 8.0). Selected device={device} name={device_name} "
            f"capability={device_capability}. On your machine this likely means the process still sees "
            "Tesla T4 instead of RTX A4000. Check CUDA_VISIBLE_DEVICES with: "
            "CUDA_VISIBLE_DEVICES=1 python -c \"import torch; print(torch.cuda.get_device_name(0), "
            "torch.cuda.get_device_capability(0))\""
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


def _predict_frame(model: LitePTModelBundle, points_range: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch
        import torch.nn.functional as F
        from datasets.transform import Compose, TRANSFORMS
        from datasets.utils import collate_fn
    except Exception as exc:  # pragma: no cover - depends on external LitePT env
        raise LitePTUnavailableError("Failed to import LitePT inference transforms") from exc

    arr = np.asarray(points_range, dtype=np.float32)
    if arr.shape != (H, W, 4):
        raise ValueError(f"LitePT expects organized frame shape {(H, W, 4)}, got {arr.shape}")

    flat = arr.reshape(H * W, 4)
    valid = np.isfinite(flat[:, :3]).all(axis=1)
    valid &= ~(
        (np.abs(flat[:, 0]) < 1e-4)
        & (np.abs(flat[:, 1]) < 1e-4)
        & (np.abs(flat[:, 2]) < 1e-4)
    )
    valid_indices = np.flatnonzero(valid)

    semantic_flat = np.full((H * W,), 255, dtype=np.uint16)
    confidence_flat = np.zeros((H * W,), dtype=np.float32)
    if valid_indices.size == 0:
        return semantic_flat.reshape(H, W), confidence_flat.reshape(H, W)

    data_dict = {
        "coord": flat[valid_indices, :3].astype(np.float32, copy=False),
        "strength": _normalize_strength(flat[valid_indices, 3]).reshape(-1, 1),
        "segment": np.full((valid_indices.size,), -1, dtype=np.int64),
        "index_valid_keys": ["coord", "strength", "segment"],
    }

    test_cfg = model.cfg.data.test.test_cfg
    voxelize = TRANSFORMS.build(test_cfg.voxelize)
    post_transform = Compose(test_cfg.post_transform)

    fragments = []
    for part in voxelize(data_dict):
        fragments.append(post_transform(part))

    pred = torch.zeros((valid_indices.size, model.num_classes), device=model.device)
    counts = torch.zeros((valid_indices.size, 1), device=model.device)
    with torch.no_grad():
        for fragment in fragments:
            input_dict = collate_fn([fragment])
            for key, value in list(input_dict.items()):
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.to(model.device, non_blocking=True)
            idx_part = input_dict["index"].long()
            pred_part = model.model(input_dict)["seg_logits"]
            pred_part = F.softmax(pred_part, dim=-1)
            pred.index_add_(0, idx_part, pred_part)
            counts.index_add_(0, idx_part, torch.ones((idx_part.numel(), 1), device=model.device))

    pred = pred / counts.clamp_min(1.0)
    confidence, labels = pred.max(dim=1)
    semantic_flat[valid_indices] = labels.detach().cpu().numpy().astype(np.uint16)
    confidence_flat[valid_indices] = confidence.detach().cpu().numpy().astype(np.float32)
    return semantic_flat.reshape(H, W), confidence_flat.reshape(H, W)


def _validate_litept_dataset(litept_dataset: str) -> str:
    if litept_dataset not in {"nuscenes", "waymo"}:
        raise ValueError(f"Unsupported LitePT dataset: {litept_dataset}. Expected one of: nuscenes, waymo")
    return litept_dataset


def _resolve_litept_checkpoint(root: Path, litept_dataset: str, checkpoint: str | None) -> Path:
    if checkpoint is None:
        return (root / "pth" / litept_dataset / "model_best.pth").resolve()
    return Path(checkpoint).expanduser().resolve()


def _resolve_litept_config(
    root: Path,
    litept_dataset: str,
    litept_config: str | None,
    *,
    validate_exists: bool = True,
) -> Path:
    if litept_config is None:
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
        names = ", ".join(p.name for p in candidates[:8])
        raise LitePTUnavailableError(
            f"Checkpoint path is a directory with multiple checkpoint files. "
            f"Pass one file explicitly. directory={path} candidates={names}"
        )
    return candidates[0].resolve()


def _torch_load_checkpoint(torch_module, checkpoint_file: Path):
    try:
        return torch_module.load(str(checkpoint_file), map_location="cpu", weights_only=False)
    except TypeError:  # older torch versions do not support weights_only
        return torch_module.load(str(checkpoint_file), map_location="cpu")


def _extract_state_dict(checkpoint_obj) -> OrderedDict:
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return OrderedDict(checkpoint_obj[key])
    if isinstance(checkpoint_obj, dict):
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


def _normalize_strength(intensity: np.ndarray) -> np.ndarray:
    strength = np.asarray(intensity, dtype=np.float32)
    finite = np.isfinite(strength)
    if finite.any() and float(np.nanmax(strength[finite])) > 1.5:
        strength = strength / 255.0
    return strength.astype(np.float32, copy=False)


def _candidate_entrypoints(root: Path) -> list[str]:
    names = []
    for pattern in ("*infer*.py", "*test*.py", "*demo*.py", "*eval*.py"):
        names.extend(str(p.relative_to(root)) for p in root.rglob(pattern) if p.is_file())
    return sorted(names)
