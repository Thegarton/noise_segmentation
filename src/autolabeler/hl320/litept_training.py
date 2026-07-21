from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


GENERATED_FILES = (
    "hl320_litept_config.py",
    "train_hl320_litept.py",
    "run_manifest.json",
    "class_statistics.json",
)


@dataclass(frozen=True)
class HL320LitePTTrainingPlan:
    litept_root: str
    dataset_root: str
    output_dir: str
    config_path: str
    experiment_dir: str
    taxonomy: dict[str, Any]
    feature_names: list[str]
    litept_in_channels: int
    train_frame_ids: list[str]
    val_frame_ids: list[str]
    train_counts: list[int]
    val_counts: list[int]
    epochs: int
    batch_size: int
    num_workers: int
    num_gpus: int
    grid_size: float
    lr: float
    weight_decay: float
    class_weighting: str
    max_class_weight: float
    noise_frame_repeat: int
    seed: int
    force_torch_pointrope: bool

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def build_training_plan(
    *,
    litept_root: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    epochs: int = 100,
    batch_size: int = 2,
    num_workers: int = 4,
    num_gpus: int = 1,
    grid_size: float = 0.05,
    lr: float = 0.002,
    weight_decay: float = 0.005,
    class_weighting: str = "sqrt_inverse",
    max_class_weight: float = 10.0,
    noise_frame_repeat: int = 4,
    seed: int = 42,
    force_torch_pointrope: bool = False,
) -> HL320LitePTTrainingPlan:
    _validate_options(
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        num_gpus=num_gpus,
        grid_size=grid_size,
        lr=lr,
        weight_decay=weight_decay,
        class_weighting=class_weighting,
        max_class_weight=max_class_weight,
        noise_frame_repeat=noise_frame_repeat,
    )
    litept_path = Path(litept_root).expanduser().resolve()
    dataset_path = Path(dataset_root).expanduser().resolve()
    out_path = Path(output_dir).expanduser().resolve()
    if not litept_path.is_dir():
        raise FileNotFoundError(f"LitePT root does not exist: {litept_path}")
    manifest = read_dataset_manifest(dataset_path)
    taxonomy = manifest["taxonomy"]
    num_classes = int(taxonomy["num_classes"])
    train_counts = split_counts(manifest, split="train", num_classes=num_classes)
    val_counts = split_counts(manifest, split="val", num_classes=num_classes)
    if not any(count > 0 for count in train_counts):
        raise ValueError("HL320 dataset has no train points with trainable labels")
    if any(count <= 0 for count in train_counts):
        missing = [
            taxonomy["class_names"][index]
            for index, count in enumerate(train_counts)
            if count <= 0
        ]
        raise ValueError(f"Every active class must have train points, missing: {missing}")
    train_frame_ids = [frame["frame_id"] for frame in manifest["frames"] if frame["split"] == "train"]
    val_frame_ids = [frame["frame_id"] for frame in manifest["frames"] if frame["split"] == "val"]
    if not train_frame_ids:
        raise ValueError("HL320 dataset contains no training frames")
    if not val_frame_ids:
        raise ValueError("HL320 dataset contains no validation frames")
    return HL320LitePTTrainingPlan(
        litept_root=str(litept_path),
        dataset_root=str(dataset_path),
        output_dir=str(out_path),
        config_path=str(out_path / "hl320_litept_config.py"),
        experiment_dir=str(out_path / "experiment"),
        taxonomy=taxonomy,
        feature_names=[str(name) for name in manifest["feature_names"]],
        litept_in_channels=int(manifest["litept_in_channels"]),
        train_frame_ids=train_frame_ids,
        val_frame_ids=val_frame_ids,
        train_counts=train_counts,
        val_counts=val_counts,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        num_gpus=num_gpus,
        grid_size=grid_size,
        lr=lr,
        weight_decay=weight_decay,
        class_weighting=class_weighting,
        max_class_weight=max_class_weight,
        noise_frame_repeat=noise_frame_repeat,
        seed=seed,
        force_torch_pointrope=force_torch_pointrope,
    )


def prepare_training_run(plan: HL320LitePTTrainingPlan, *, overwrite: bool = False) -> dict[str, Any]:
    out_path = Path(plan.output_dir)
    _prepare_output_directory(out_path, overwrite=overwrite)
    dataset_dir = Path(plan.dataset_root) / "dataset"
    train_split_path, oversampling = write_oversampled_train_split(plan, dataset_dir=dataset_dir)
    class_weights = compute_class_weights(
        plan.train_counts,
        method=plan.class_weighting,
        max_weight=plan.max_class_weight,
    )
    config_path = Path(plan.config_path)
    config_path.write_text(
        render_litept_config(plan, class_weights=class_weights, train_split=train_split_path.name),
        encoding="utf-8",
    )
    launcher_path = out_path / "train_hl320_litept.py"
    launcher_path.write_text(render_training_launcher(plan), encoding="utf-8")
    statistics = class_statistics(plan, class_weights=class_weights, oversampling=oversampling)
    (out_path / "class_statistics.json").write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        **plan.to_jsonable(),
        "prepared": True,
        "training_launcher": str(launcher_path),
        "class_weights": class_weights,
        "oversampling": oversampling,
        "class_statistics": str(out_path / "class_statistics.json"),
    }
    (out_path / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_training_command(plan: HL320LitePTTrainingPlan, *, resume: bool = False) -> list[str]:
    config_path = Path(plan.config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Generated LitePT config does not exist: {config_path}")
    train_entrypoint = (
        Path(plan.output_dir) / "train_hl320_litept.py"
        if plan.force_torch_pointrope
        else Path(plan.litept_root) / "tools" / "train.py"
    )
    if not train_entrypoint.is_file():
        raise FileNotFoundError(f"LitePT training entrypoint does not exist: {train_entrypoint}")
    command = [
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
        f"save_path={plan.experiment_dir}",
        f"resume={str(resume).lower()}",
    ]
    if resume:
        weight = Path(plan.experiment_dir) / "model" / "model_last.pth"
        if not weight.is_file():
            raise FileNotFoundError(f"Cannot resume: model_last.pth does not exist: {weight}")
        command.append(f"weight={weight}")
    return command


def run_training(plan: HL320LitePTTrainingPlan, *, resume: bool = False) -> int:
    command = build_training_command(plan, resume=resume)
    env = dict(os.environ)
    pythonpath = [str(Path(plan.litept_root)), str(Path(__file__).resolve().parents[3] / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    completed = subprocess.run(command, cwd=plan.litept_root, env=env, check=False)
    return int(completed.returncode)


def read_dataset_manifest(dataset_root: Path) -> dict[str, Any]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"HL320 dataset root does not exist: {dataset_root}")
    manifest_path = dataset_root / "hl320_dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"HL320 dataset manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = ("taxonomy", "feature_names", "litept_in_channels", "frames")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"HL320 dataset manifest is missing keys: {missing}")
    if not (dataset_root / "dataset" / "train").is_dir():
        raise FileNotFoundError(f"HL320 dataset has no train split: {dataset_root / 'dataset' / 'train'}")
    if not (dataset_root / "dataset" / "val").is_dir():
        raise FileNotFoundError(f"HL320 dataset has no val split: {dataset_root / 'dataset' / 'val'}")
    return manifest


def split_counts(manifest: dict[str, Any], *, split: str, num_classes: int) -> list[int]:
    counts = [0 for _ in range(num_classes)]
    for frame in manifest["frames"]:
        if frame["split"] != split:
            continue
        for training_id, count in frame.get("training_class_counts", {}).items():
            index = int(training_id)
            if 0 <= index < num_classes:
                counts[index] += int(count)
    return counts


def compute_class_weights(train_counts: list[int], *, method: str, max_weight: float) -> list[float] | None:
    if method == "none":
        return None
    if method != "sqrt_inverse":
        raise ValueError(f"Unsupported class weighting method: {method}")
    import numpy as np

    counts = np.asarray(train_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("train_counts must be a non-empty one-dimensional list")
    if np.any(counts <= 0):
        raise ValueError(f"All active classes must have train points, got {train_counts}")
    raw = np.sqrt(counts.sum() / (counts.size * counts))
    raw = np.clip(raw, 1.0 / max_weight, max_weight)
    normalized = raw / raw.mean()
    return [round(float(value), 6) for value in normalized]


def write_oversampled_train_split(
    plan: HL320LitePTTrainingPlan,
    *,
    dataset_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    noise_training_ids = {
        training_id
        for training_id, name in enumerate(plan.taxonomy["class_names"])
        if "noise" in str(name).casefold()
    }
    manifest = read_dataset_manifest(Path(plan.dataset_root))
    frame_training_ids = {
        frame["frame_id"]: {int(value) for value in frame.get("training_class_counts", {})}
        for frame in manifest["frames"]
        if frame["split"] == "train"
    }
    entries: list[str] = []
    repeated_frames: dict[str, int] = {}
    for frame_id in plan.train_frame_ids:
        has_noise = bool(frame_training_ids.get(frame_id, set()) & noise_training_ids)
        repeat = plan.noise_frame_repeat if has_noise else 1
        entries.extend([f"train/{frame_id}"] * repeat)
        if repeat > 1:
            repeated_frames[frame_id] = repeat
    split_path = dataset_dir / "train_oversampled.json"
    split_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return split_path, {
        "strategy": "repeat_frames_with_noise_classes",
        "noise_training_ids": sorted(noise_training_ids),
        "noise_class_names": [
            plan.taxonomy["class_names"][training_id]
            for training_id in sorted(noise_training_ids)
        ],
        "noise_frame_repeat": plan.noise_frame_repeat,
        "original_train_frames": len(plan.train_frame_ids),
        "effective_train_samples": len(entries),
        "repeated_frames": repeated_frames,
    }


def class_statistics(
    plan: HL320LitePTTrainingPlan,
    *,
    class_weights: list[float] | None,
    oversampling: dict[str, Any],
) -> dict[str, Any]:
    classes = []
    for training_id, (name, source_id) in enumerate(
        zip(plan.taxonomy["class_names"], plan.taxonomy["training_id_to_source_id"])
    ):
        classes.append(
            {
                "name": name,
                "source_id": int(source_id),
                "training_id": training_id,
                "train_points": int(plan.train_counts[training_id]),
                "val_points": int(plan.val_counts[training_id]),
                "loss_weight": float(class_weights[training_id]) if class_weights is not None else 1.0,
            }
        )
    return {
        "classes": classes,
        "class_weighting": {
            "method": plan.class_weighting,
            "max_weight": plan.max_class_weight,
        },
        "oversampling": oversampling,
    }


def render_litept_config(
    plan: HL320LitePTTrainingPlan,
    *,
    class_weights: list[float] | None,
    train_split: str,
) -> str:
    default_runtime = Path(plan.litept_root) / "configs" / "_base_" / "default_runtime.py"
    dataset_root = str(Path(plan.dataset_root) / "dataset")
    experiment_dir = str(Path(plan.experiment_dir))
    class_names = plan.taxonomy["class_names"]
    source_ids = plan.taxonomy["training_id_to_source_id"]
    num_classes = int(plan.taxonomy["num_classes"])
    save_freq = max(1, min(5, plan.epochs))
    return f'''_base_ = [{str(default_runtime)!r}]

batch_size = {plan.batch_size}
num_worker = {plan.num_workers}
mix_prob = 0.0
empty_cache = False
enable_amp = True
enable_wandb = False
seed = {plan.seed}
save_path = {experiment_dir!r}

epoch = {plan.epochs}
eval_epoch = {plan.epochs}
optimizer = dict(type="AdamW", lr={plan.lr!r}, weight_decay={plan.weight_decay!r})
param_dicts = None
scheduler = dict(
    type="OneCycleLR",
    max_lr={plan.lr!r},
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

model = dict(
    type="DefaultSegmentorV2",
    num_classes={num_classes},
    backbone_out_channels=72,
    backbone=dict(
        type="LitePT",
        in_channels={plan.litept_in_channels},
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", weight={class_weights!r}, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

dataset_type = "DefaultDataset"
data_root = {dataset_root!r}
ignore_index = -1
names = {class_names!r}
training_id_to_source_id = {source_ids!r}
output_ignore_index = 255
feature_names = {plan.feature_names!r}
class_weights = {class_weights!r}

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
    num_classes={num_classes},
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split={train_split!r},
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


def render_training_launcher(plan: HL320LitePTTrainingPlan) -> str:
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
    spec = importlib.util.spec_from_file_location("_hl320_pointrope_torch", POINTROPE_MODULE)
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


def _prepare_output_directory(output_dir: Path, *, overwrite: bool) -> None:
    generated_paths = [output_dir / "experiment"]
    generated_paths.extend(output_dir / name for name in GENERATED_FILES)
    existing = [path for path in generated_paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing[:6])
        raise FileExistsError(f"HL320 LitePT output already contains generated files: {names}. Use --overwrite.")
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_options(
    *,
    epochs: int,
    batch_size: int,
    num_workers: int,
    num_gpus: int,
    grid_size: float,
    lr: float,
    weight_decay: float,
    class_weighting: str,
    max_class_weight: float,
    noise_frame_repeat: int,
) -> None:
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive because LitePT enables persistent workers")
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    if grid_size <= 0.0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if lr <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if class_weighting not in {"none", "sqrt_inverse"}:
        raise ValueError(f"class_weighting must be one of none/sqrt_inverse, got {class_weighting}")
    if max_class_weight < 1.0:
        raise ValueError(f"max_class_weight must be at least 1, got {max_class_weight}")
    if noise_frame_repeat < 1:
        raise ValueError(f"noise_frame_repeat must be at least 1, got {noise_frame_repeat}")
