from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .litept_adapter import normalize_litept_strength


IGNORE_TRAINING_ID = -1
DEFAULT_OUTPUT_IGNORE_ID = 255
GENERATED_FILES = (
    "litept_custom_config.py",
    "train_litept_custom.py",
    "pretrained_backbone.pth",
    "run_manifest.json",
    "taxonomy.json",
    "class_statistics.json",
)


@dataclass(frozen=True)
class TaxonomyClass:
    name: str
    source_id: int
    training_id: int | None
    color: list[int] | None = None
    ignore: bool = False


@dataclass(frozen=True)
class FinetuneFrame:
    frame_id: str
    points_path: str
    mask_path: str
    split: str
    point_count: int
    mask_shape: list[int]
    source_label_counts: dict[str, int]
    unknown_source_ids: list[int]


@dataclass(frozen=True)
class LitePTFinetunePlan:
    litept_root: str
    export_dir: str
    labeler_dir: str
    output_dir: str
    checkpoint: str
    checkpoint_exists: bool
    config_path: str
    dataset_dir: str
    experiment_dir: str
    taxonomy: dict[str, Any]
    frames: list[FinetuneFrame]
    train_frame_ids: list[str]
    val_frame_ids: list[str]
    epochs: int
    batch_size: int
    num_workers: int
    num_gpus: int
    val_ratio: float
    seed: int
    grid_size: float
    head_lr: float
    backbone_lr: float
    force_torch_pointrope: bool

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frame_count"] = len(self.frames)
        payload["train_frame_count"] = len(self.train_frame_ids)
        payload["val_frame_count"] = len(self.val_frame_ids)
        return payload


def build_finetune_plan(
    *,
    litept_root: str,
    export_dir: str,
    labeler_dir: str,
    output_dir: str,
    checkpoint: str | None = None,
    epochs: int = 30,
    batch_size: int = 4,
    num_workers: int = 4,
    num_gpus: int = 1,
    val_ratio: float = 0.2,
    seed: int = 42,
    grid_size: float = 0.05,
    head_lr: float = 2e-4,
    backbone_lr: float = 2e-5,
    force_torch_pointrope: bool = False,
) -> LitePTFinetunePlan:
    _validate_training_options(
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        num_gpus=num_gpus,
        val_ratio=val_ratio,
        grid_size=grid_size,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
    )

    root = Path(litept_root).expanduser().resolve()
    export_path = Path(export_dir).expanduser().resolve()
    labeler_path = Path(labeler_dir).expanduser().resolve()
    out_path = Path(output_dir).expanduser().resolve()
    _require_dir(root, "LitePT root")
    _require_dir(export_path, "Point-labeler export")
    _require_dir(labeler_path, "Point-labeler dataset")

    labels_xml = labeler_path / "labels.xml"
    taxonomy = build_training_taxonomy(read_labels_xml(labels_xml))
    frame_ids = discover_exported_frame_ids(export_path)
    train_ids, val_ids = split_frame_ids(frame_ids, val_ratio=val_ratio, seed=seed)
    split_by_id = {frame_id: "train" for frame_id in train_ids}
    split_by_id.update({frame_id: "val" for frame_id in val_ids})

    frames = [
        inspect_finetune_frame(
            frame_id=frame_id,
            points_path=labeler_path / "velodyne" / f"{frame_id}.bin",
            mask_path=export_path / frame_id / "semantic_mask.npy",
            split=split_by_id[frame_id],
            known_source_ids=set(taxonomy["source_id_to_training_id"]) | set(taxonomy["ignore_source_ids"]),
        )
        for frame_id in frame_ids
    ]

    checkpoint_path = (
        Path(checkpoint).expanduser().resolve()
        if checkpoint is not None
        else (root / "pth" / "waymo" / "model_best.pth").resolve()
    )
    return LitePTFinetunePlan(
        litept_root=str(root),
        export_dir=str(export_path),
        labeler_dir=str(labeler_path),
        output_dir=str(out_path),
        checkpoint=str(checkpoint_path),
        checkpoint_exists=checkpoint_path.is_file(),
        config_path=str(out_path / "litept_custom_config.py"),
        dataset_dir=str(out_path / "dataset"),
        experiment_dir=str(out_path / "experiment"),
        taxonomy=taxonomy,
        frames=frames,
        train_frame_ids=train_ids,
        val_frame_ids=val_ids,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        num_gpus=num_gpus,
        val_ratio=val_ratio,
        seed=seed,
        grid_size=grid_size,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        force_torch_pointrope=force_torch_pointrope,
    )


def read_labels_xml(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Point-labeler taxonomy does not exist: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Cannot parse point-labeler taxonomy: {path}: {exc}") from exc

    classes: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for label in root.findall("label"):
        name = (label.findtext("name") or "").strip()
        raw_id = (label.findtext("id") or "").strip()
        if not name or not raw_id:
            raise ValueError(f"Each <label> in {path} must contain non-empty <name> and <id>")
        try:
            source_id = int(raw_id)
        except ValueError as exc:
            raise ValueError(f"Invalid label id {raw_id!r} for class {name!r} in {path}") from exc
        if source_id < 0 or source_id > np.iinfo(np.uint16).max:
            raise ValueError(f"Class {name!r} id must fit uint16, got {source_id}")

        folded = name.casefold()
        if source_id in seen_ids:
            raise ValueError(f"Duplicate class id {source_id} in {path}")
        if folded in seen_names:
            raise ValueError(f"Duplicate class name {name!r} in {path}")
        seen_ids.add(source_id)
        seen_names.add(folded)

        color = _parse_label_color(label.findtext("color"), class_name=name)
        classes.append({"name": name, "source_id": source_id, "color": color})

    if not classes:
        raise ValueError(f"No <label> entries found in {path}")
    return sorted(classes, key=lambda item: item["source_id"])


def build_training_taxonomy(classes: list[dict[str, Any]]) -> dict[str, Any]:
    taxonomy_classes: list[TaxonomyClass] = []
    source_to_training: dict[int, int] = {}
    training_to_source: list[int] = []
    class_names: list[str] = []
    ignore_source_ids: list[int] = []

    for item in sorted(classes, key=lambda value: int(value["source_id"])):
        name = str(item["name"])
        source_id = int(item["source_id"])
        is_ignore = source_id == DEFAULT_OUTPUT_IGNORE_ID or name.casefold() in {"ignore", "ignored"}
        if is_ignore:
            ignore_source_ids.append(source_id)
            taxonomy_classes.append(
                TaxonomyClass(
                    name=name,
                    source_id=source_id,
                    training_id=None,
                    color=item.get("color"),
                    ignore=True,
                )
            )
            continue

        training_id = len(training_to_source)
        source_to_training[source_id] = training_id
        training_to_source.append(source_id)
        class_names.append(name)
        taxonomy_classes.append(
            TaxonomyClass(
                name=name,
                source_id=source_id,
                training_id=training_id,
                color=item.get("color"),
            )
        )

    if not class_names:
        raise ValueError("Taxonomy contains no trainable classes")
    return {
        "class_names": class_names,
        "num_classes": len(class_names),
        "classes": [asdict(item) for item in taxonomy_classes],
        "source_id_to_training_id": source_to_training,
        "training_id_to_source_id": training_to_source,
        "ignore_source_ids": sorted(ignore_source_ids),
        "training_ignore_index": IGNORE_TRAINING_ID,
        "output_ignore_index": DEFAULT_OUTPUT_IGNORE_ID,
    }


def discover_exported_frame_ids(export_dir: Path) -> list[str]:
    frame_ids = sorted(
        path.name
        for path in export_dir.iterdir()
        if path.is_dir() and (path / "semantic_mask.npy").is_file()
    )
    if len(frame_ids) < 2:
        raise ValueError(
            f"At least 2 exported frames are required for train/validation split, found {len(frame_ids)} in {export_dir}"
        )
    return frame_ids


def split_frame_ids(
    frame_ids: list[str],
    *,
    val_ratio: float,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    if len(frame_ids) < 2:
        raise ValueError("At least 2 frames are required for train/validation split")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    ordered = sorted(frame_ids)
    val_count = min(len(ordered) - 1, max(1, int(math.ceil(len(ordered) * val_ratio))))
    shuffled_indices = np.random.default_rng(seed).permutation(len(ordered))
    val_indices = set(int(index) for index in shuffled_indices[:val_count])
    train = [frame_id for index, frame_id in enumerate(ordered) if index not in val_indices]
    val = [frame_id for index, frame_id in enumerate(ordered) if index in val_indices]
    return train, val


def inspect_finetune_frame(
    *,
    frame_id: str,
    points_path: Path,
    mask_path: Path,
    split: str,
    known_source_ids: set[int],
) -> FinetuneFrame:
    if not points_path.is_file():
        raise FileNotFoundError(f"Missing point cloud for frame {frame_id}: {points_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing semantic mask for frame {frame_id}: {mask_path}")

    float_count = points_path.stat().st_size // np.dtype(np.float32).itemsize
    if points_path.stat().st_size % np.dtype(np.float32).itemsize != 0 or float_count % 4 != 0:
        raise ValueError(f"Point cloud {points_path} is not float32 XYZI data")
    point_count = float_count // 4
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"Semantic mask {mask_path} must have integer dtype, got {mask.dtype}")
    if mask.size != point_count:
        raise ValueError(
            f"Frame {frame_id} point/mask size mismatch: points={point_count}, mask={mask.size}, shape={mask.shape}"
        )

    unique_ids, counts = np.unique(mask, return_counts=True)
    label_counts = {str(int(label_id)): int(count) for label_id, count in zip(unique_ids, counts)}
    unknown_ids = sorted(int(label_id) for label_id in unique_ids if int(label_id) not in known_source_ids)
    return FinetuneFrame(
        frame_id=frame_id,
        points_path=str(points_path.resolve()),
        mask_path=str(mask_path.resolve()),
        split=split,
        point_count=point_count,
        mask_shape=[int(value) for value in mask.shape],
        source_label_counts=label_counts,
        unknown_source_ids=unknown_ids,
    )


def prepare_finetune_run(
    plan: LitePTFinetunePlan,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = Path(plan.output_dir)
    _prepare_output_directory(output_dir, overwrite=overwrite)
    checkpoint = Path(plan.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LitePT pretrained checkpoint does not exist: {checkpoint}")

    dataset_dir = Path(plan.dataset_dir)
    source_to_training = {
        int(source_id): int(training_id)
        for source_id, training_id in plan.taxonomy["source_id_to_training_id"].items()
    }
    class_counts = {
        "all": Counter(),
        "train": Counter(),
        "val": Counter(),
        "ignored": {"all": 0, "train": 0, "val": 0},
        "unknown_source_ids": Counter(),
    }
    frame_summaries: list[dict[str, Any]] = []

    for frame in plan.frames:
        points = np.fromfile(frame.points_path, dtype=np.float32).reshape(-1, 4)
        mask = np.load(frame.mask_path, allow_pickle=False).reshape(-1)
        valid = valid_litept_points(points)
        segments = remap_source_labels(mask, source_to_training)
        if not np.any(valid):
            raise ValueError(f"Frame {frame.frame_id} contains no valid non-zero XYZ points")
        if not np.any(segments[valid] >= 0):
            raise ValueError(
                f"Frame {frame.frame_id} contains no valid points with trainable taxonomy labels"
            )
        coord = points[valid, :3].astype(np.float32, copy=False)
        strength = normalize_litept_strength(points[valid, 3]).reshape(-1, 1)
        strength = np.nan_to_num(strength, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        segment = segments[valid].astype(np.int32, copy=False)

        frame_dir = dataset_dir / frame.split / frame.frame_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / "coord.npy", coord)
        np.save(frame_dir / "strength.npy", strength)
        np.save(frame_dir / "segment.npy", segment)

        training_ids, counts = np.unique(segment, return_counts=True)
        for training_id, count in zip(training_ids, counts):
            if int(training_id) == IGNORE_TRAINING_ID:
                class_counts["ignored"]["all"] += int(count)
                class_counts["ignored"][frame.split] += int(count)
            else:
                class_counts["all"][int(training_id)] += int(count)
                class_counts[frame.split][int(training_id)] += int(count)
        for source_id in frame.unknown_source_ids:
            class_counts["unknown_source_ids"][source_id] += int(frame.source_label_counts[str(source_id)])

        frame_summaries.append(
            {
                "frame_id": frame.frame_id,
                "split": frame.split,
                "input_points": frame.point_count,
                "valid_points": int(np.count_nonzero(valid)),
                "ignored_points": int(np.count_nonzero(segment == IGNORE_TRAINING_ID)),
                "unknown_source_ids": frame.unknown_source_ids,
                "dataset_path": str(frame_dir),
            }
        )

    taxonomy_path = output_dir / "taxonomy.json"
    taxonomy_path.write_text(json.dumps(plan.taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
    statistics = _class_statistics_payload(plan.taxonomy, class_counts)
    (output_dir / "class_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config_path = Path(plan.config_path)
    config_path.write_text(render_litept_config(plan), encoding="utf-8")
    launcher_path = output_dir / "train_litept_custom.py"
    launcher_path.write_text(render_training_launcher(plan), encoding="utf-8")
    backbone_path = output_dir / "pretrained_backbone.pth"
    removed_keys = prepare_backbone_checkpoint(checkpoint, backbone_path)

    manifest = plan.to_jsonable()
    manifest.update(
        {
            "prepared": True,
            "pretrained_backbone": str(backbone_path),
            "training_launcher": str(launcher_path),
            "removed_checkpoint_keys": removed_keys,
            "taxonomy_path": str(taxonomy_path),
            "class_statistics": str(output_dir / "class_statistics.json"),
            "frame_summaries": frame_summaries,
        }
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def valid_litept_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"points must have shape [N,>=3], got {arr.shape}")
    valid = np.isfinite(arr[:, :3]).all(axis=1)
    valid &= ~(
        (np.abs(arr[:, 0]) < 1e-4)
        & (np.abs(arr[:, 1]) < 1e-4)
        & (np.abs(arr[:, 2]) < 1e-4)
    )
    return valid


def remap_source_labels(labels: np.ndarray, source_to_training: dict[int, int]) -> np.ndarray:
    source = np.asarray(labels)
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError(f"labels must have integer dtype, got {source.dtype}")
    result = np.full(source.shape, IGNORE_TRAINING_ID, dtype=np.int32)
    for source_id, training_id in source_to_training.items():
        result[source == source_id] = training_id
    return result


def render_litept_config(plan: LitePTFinetunePlan) -> str:
    base_config = Path(plan.litept_root) / "configs" / "waymo" / "semseg-litept-small-v1m1.py"
    class_names = plan.taxonomy["class_names"]
    source_ids = plan.taxonomy["training_id_to_source_id"]
    num_classes = int(plan.taxonomy["num_classes"])
    dataset_root = str(Path(plan.dataset_dir))
    experiment_dir = str(Path(plan.experiment_dir))
    save_freq = max(1, min(5, plan.epochs))
    return f'''_base_ = [{str(base_config)!r}]

batch_size = {plan.batch_size}
num_worker = {plan.num_workers}
mix_prob = 0.0
enable_amp = True
enable_wandb = False
seed = {plan.seed}
save_path = {experiment_dir!r}

epoch = {plan.epochs}
eval_epoch = {plan.epochs}
optimizer = dict(type="AdamW", lr={plan.head_lr!r}, weight_decay=0.005)
param_dicts = [dict(keyword="backbone", lr={plan.backbone_lr!r})]
scheduler = dict(
    type="OneCycleLR",
    max_lr=[{plan.head_lr!r}, {plan.backbone_lr!r}],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

model = dict(
    num_classes={num_classes},
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

dataset_type = "DefaultDataset"
data_root = {dataset_root!r}
ignore_index = -1
names = {class_names!r}
training_id_to_source_id = {source_ids!r}
output_ignore_index = {int(plan.taxonomy["output_ignore_index"])}

train_transform = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="GridSample", grid_size={plan.grid_size!r}, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={{"grid_size": {plan.grid_size!r}}}),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "grid_size"),
        feat_keys=("coord", "strength"),
    ),
]

val_transform = [
    dict(type="Copy", keys_dict={{"segment": "origin_segment"}}),
    dict(
        type="GridSample",
        grid_size={plan.grid_size!r},
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=True,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
        feat_keys=("coord", "strength"),
    ),
]

data = dict(
    _delete_=True,
    num_classes={num_classes},
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=train_transform,
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=val_transform,
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={{"segment": "origin_segment"}}),
            dict(
                type="GridSample",
                grid_size={plan.grid_size!r},
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size={plan.grid_size!r},
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
        ignore_index=ignore_index,
    ),
)

hooks = [
    dict(type="CheckpointLoader", strict=False),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq={save_freq}),
    dict(type="PreciseEvaluator", test_last=False),
]
'''


def render_training_launcher(plan: LitePTFinetunePlan) -> str:
    litept_root = str(Path(plan.litept_root))
    train_script = str(Path(plan.litept_root) / "tools" / "train.py")
    pointrope_module = str(Path(plan.litept_root) / "libs" / "pointrope" / "pointrope_torch.py")
    return f'''#!/usr/bin/env python3
import importlib.util
import runpy
import sys
import types

LITEPT_ROOT = {litept_root!r}
TRAIN_SCRIPT = {train_script!r}
POINTROPE_MODULE = {pointrope_module!r}

if LITEPT_ROOT not in sys.path:
    sys.path.insert(0, LITEPT_ROOT)

if {plan.force_torch_pointrope!r}:
    spec = importlib.util.spec_from_file_location("_finetune_pointrope_torch", POINTROPE_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load PointROPE fallback from: {{POINTROPE_MODULE}}")
    pointrope_torch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pointrope_torch)

    pointrope_package = types.ModuleType("libs.pointrope")
    pointrope_package.PointROPE = pointrope_torch.PointROPE
    pointrope_package.__all__ = ["PointROPE"]
    pointrope_package.__file__ = POINTROPE_MODULE
    sys.modules["libs.pointrope"] = pointrope_package
    print(f"LitePT training PointROPE backend: torch ({{POINTROPE_MODULE}})", flush=True)

sys.argv[0] = TRAIN_SCRIPT
runpy.run_path(TRAIN_SCRIPT, run_name="__main__")
'''


def filter_seg_head_state_dict(state_dict: dict[str, Any]) -> tuple[OrderedDict, list[str]]:
    filtered = OrderedDict()
    removed: list[str] = []
    for key, value in state_dict.items():
        normalized = key[7:] if key.startswith("module.") else key
        if normalized.startswith("seg_head."):
            removed.append(key)
        else:
            filtered[key] = value
    return filtered, removed


def prepare_backbone_checkpoint(source: Path, destination: Path) -> list[str]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on LitePT environment
        raise RuntimeError(
            "PyTorch is required to prepare the pretrained backbone. "
            "Run finetune_litept.py with the same Python interpreter that successfully runs LitePT inference. "
            f"python={sys.executable} import_error={type(exc).__name__}: {exc}. "
            "Check both interpreters with: "
            "'python3 -c \"import sys, torch; print(sys.executable, torch.__version__)\"' and "
            f"'{sys.executable} -c \"import sys, torch; print(sys.executable, torch.__version__)\"'."
        ) from exc

    try:
        checkpoint = torch.load(str(source), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        checkpoint = torch.load(str(source), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported LitePT checkpoint format: {source}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"LitePT checkpoint has no state_dict mapping: {source}")
    filtered, removed = filter_seg_head_state_dict(state_dict)
    if not removed:
        raise ValueError(f"No seg_head parameters found in pretrained checkpoint: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": filtered}, str(destination))
    return removed


def build_training_command(plan: LitePTFinetunePlan, *, resume: bool = False) -> list[str]:
    config_path = Path(plan.config_path)
    experiment_dir = Path(plan.experiment_dir)
    if resume:
        weight = experiment_dir / "model" / "model_last.pth"
        if not config_path.is_file():
            raise FileNotFoundError(f"Cannot resume: generated config does not exist: {config_path}")
        if not weight.is_file():
            raise FileNotFoundError(f"Cannot resume: model_last.pth does not exist: {weight}")
    else:
        weight = Path(plan.output_dir) / "pretrained_backbone.pth"
        if not config_path.is_file():
            raise FileNotFoundError(f"Generated LitePT config does not exist: {config_path}")
        if not weight.is_file():
            raise FileNotFoundError(f"Prepared backbone checkpoint does not exist: {weight}")

    train_entrypoint = (
        Path(plan.output_dir) / "train_litept_custom.py"
        if plan.force_torch_pointrope
        else Path(plan.litept_root) / "tools" / "train.py"
    )
    if plan.force_torch_pointrope and not train_entrypoint.is_file():
        raise FileNotFoundError(f"Generated LitePT training launcher does not exist: {train_entrypoint}")

    return [
        sys.executable,
        str(train_entrypoint),
        "--config-file",
        str(config_path),
        "--num-gpus",
        str(plan.num_gpus),
        "--num-machines",
        "1",
        "--dist-url",
        "auto",
        "--options",
        f"save_path={experiment_dir}",
        f"resume={str(resume).lower()}",
        f"weight={weight}",
    ]


def run_training(plan: LitePTFinetunePlan, *, resume: bool = False) -> int:
    command = build_training_command(plan, resume=resume)
    env = dict(os.environ)
    pythonpath = [str(Path(plan.litept_root)), str(Path(__file__).resolve().parents[3] / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    completed = subprocess.run(command, cwd=plan.litept_root, env=env, check=False)
    return int(completed.returncode)


def _prepare_output_directory(output_dir: Path, *, overwrite: bool) -> None:
    generated_paths = [output_dir / "dataset", output_dir / "experiment"]
    generated_paths.extend(output_dir / name for name in GENERATED_FILES)
    existing = [path for path in generated_paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing[:6])
        raise FileExistsError(f"Finetune output already contains generated files: {names}. Use --overwrite.")
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _class_statistics_payload(taxonomy: dict[str, Any], counts: dict[str, Any]) -> dict[str, Any]:
    classes = []
    for training_id, (name, source_id) in enumerate(
        zip(taxonomy["class_names"], taxonomy["training_id_to_source_id"])
    ):
        classes.append(
            {
                "name": name,
                "source_id": int(source_id),
                "training_id": training_id,
                "all_points": int(counts["all"][training_id]),
                "train_points": int(counts["train"][training_id]),
                "val_points": int(counts["val"][training_id]),
            }
        )
    return {
        "classes": classes,
        "ignored_points": counts["ignored"],
        "unknown_source_ids": {
            str(source_id): int(count)
            for source_id, count in sorted(counts["unknown_source_ids"].items())
        },
    }


def _parse_label_color(value: str | None, *, class_name: str) -> list[int] | None:
    if value is None or not value.strip():
        return None
    tokens = value.split()
    if len(tokens) != 3:
        raise ValueError(f"Class {class_name!r} color must contain 3 integers, got {value!r}")
    color = [int(token) for token in tokens]
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError(f"Class {class_name!r} color channels must be in [0,255], got {color}")
    return color


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")


def _validate_training_options(
    *,
    epochs: int,
    batch_size: int,
    num_workers: int,
    num_gpus: int,
    val_ratio: float,
    grid_size: float,
    head_lr: float,
    backbone_lr: float,
) -> None:
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive because LitePT enables persistent workers")
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if grid_size <= 0.0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if head_lr <= 0.0 or backbone_lr <= 0.0:
        raise ValueError("head_lr and backbone_lr must be positive")
