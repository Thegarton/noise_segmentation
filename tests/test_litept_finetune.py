from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import autolabeler.teachers.litept_finetune as litept_finetune
from autolabeler.teachers.litept_finetune import (
    build_finetune_plan,
    build_training_command,
    build_training_taxonomy,
    compute_class_weights,
    filter_seg_head_state_dict,
    prepare_finetune_run,
    remap_source_labels,
    split_frame_ids,
    valid_litept_points,
)


def write_labels_xml(path: Path) -> None:
    path.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<config>\n"
        "  <label><id>2</id><name>Car</name><color>10 20 30</color></label>\n"
        "  <label><id>10</id><name>Ground</name><color>40 50 60</color></label>\n"
        "  <label><id>20</id><name>Dust noise</name><color>70 80 90</color></label>\n"
        "  <label><id>255</id><name>ignore</name><color>120 120 120</color></label>\n"
        "</config>\n",
        encoding="utf-8",
    )


def make_finetune_inputs(tmp_path: Path, frame_count: int = 5) -> tuple[Path, Path, Path]:
    litept_root = tmp_path / "LitePT"
    (litept_root / "configs" / "waymo").mkdir(parents=True)
    (litept_root / "configs" / "waymo" / "semseg-litept-small-v1m1.py").write_text(
        "# base config\n",
        encoding="utf-8",
    )
    checkpoint = litept_root / "pth" / "waymo" / "model_best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake-checkpoint")

    labeler_dir = tmp_path / "labeler"
    (labeler_dir / "velodyne").mkdir(parents=True)
    write_labels_xml(labeler_dir / "labels.xml")

    export_dir = tmp_path / "export"
    for index in range(frame_count):
        frame_id = f"frame_{index:03d}"
        points = np.asarray(
            [
                [0.5, 0.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, 255.0],
                [np.nan, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 0.5],
            ],
            dtype=np.float32,
        )
        points.tofile(labeler_dir / "velodyne" / f"{frame_id}.bin")
        frame_dir = export_dir / frame_id
        frame_dir.mkdir(parents=True)
        np.save(frame_dir / "semantic_mask.npy", np.asarray([[2, 10], [255, 77]], dtype=np.uint16))
    return litept_root, labeler_dir, export_dir


def test_sparse_taxonomy_and_unknown_labels_map_to_dense_training_ids():
    taxonomy = build_training_taxonomy(
        [
            {"name": "Car", "source_id": 2, "color": [1, 2, 3]},
            {"name": "Ground", "source_id": 10, "color": [4, 5, 6]},
            {"name": "ignore", "source_id": 255, "color": [120, 120, 120]},
        ]
    )

    assert taxonomy["class_names"] == ["Car", "Ground"]
    assert taxonomy["source_id_to_training_id"] == {2: 0, 10: 1}
    assert taxonomy["training_id_to_source_id"] == [2, 10]
    assert taxonomy["ignore_source_ids"] == [255]
    labels = np.asarray([2, 10, 255, 77], dtype=np.uint16)
    assert remap_source_labels(labels, {2: 0, 10: 1}).tolist() == [0, 1, -1, -1]


def test_split_randomly_selects_validation_frames_deterministically():
    frame_ids = [f"frame_{index:03d}" for index in range(10)]
    train, val = split_frame_ids(frame_ids, val_ratio=0.2, seed=42)
    repeated_train, repeated_val = split_frame_ids(
        list(reversed(frame_ids)),
        val_ratio=0.2,
        seed=42,
    )
    other_train, other_val = split_frame_ids(frame_ids, val_ratio=0.2, seed=7)

    assert train == repeated_train
    assert val == repeated_val == ["frame_005", "frame_006"]
    assert other_val == ["frame_000", "frame_008"]
    assert other_train != train
    assert sorted(train + val) == frame_ids
    with pytest.raises(ValueError, match="At least 2 frames"):
        split_frame_ids(["only"], val_ratio=0.2)


def test_class_aware_split_keeps_singleton_and_every_class_in_train():
    frame_ids = [f"frame_{index:03d}" for index in range(6)]
    frame_class_ids = {
        "frame_000": {0, 9},
        "frame_001": {0},
        "frame_002": {0, 1},
        "frame_003": {1},
        "frame_004": {2},
        "frame_005": {2},
    }
    train, val = split_frame_ids(
        frame_ids,
        val_ratio=0.5,
        seed=42,
        frame_class_ids=frame_class_ids,
    )

    assert "frame_000" in train
    train_classes = set().union(*(frame_class_ids[frame_id] for frame_id in train))
    assert train_classes == {0, 1, 2, 9}
    assert val
    assert set(train).isdisjoint(val)


def test_sqrt_inverse_class_weights_favor_rare_classes():
    weights = compute_class_weights([1_000_000, 10_000, 100], max_weight=10.0)

    assert weights is not None
    assert weights[0] < weights[1] < weights[2]
    assert sum(weights) == pytest.approx(3.0, abs=1e-5)
    assert compute_class_weights([1, 2], method="none") is None


def test_valid_points_filters_zero_and_nonfinite_xyz():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [np.nan, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert valid_litept_points(points).tolist() == [False, True, False]


def test_prepare_builds_default_dataset_config_and_statistics(tmp_path: Path, monkeypatch):
    litept_root, labeler_dir, export_dir = make_finetune_inputs(tmp_path)
    output_dir = tmp_path / "finetune"
    plan = build_finetune_plan(
        litept_root=str(litept_root),
        export_dir=str(export_dir),
        labeler_dir=str(labeler_dir),
        output_dir=str(output_dir),
    )

    def fake_prepare_checkpoint(source: Path, destination: Path) -> list[str]:
        destination.write_bytes(b"backbone")
        return ["seg_head.weight", "seg_head.bias"]

    monkeypatch.setattr(litept_finetune, "prepare_backbone_checkpoint", fake_prepare_checkpoint)
    manifest = prepare_finetune_run(plan)

    assert plan.train_frame_ids == ["frame_000", "frame_001", "frame_003", "frame_004"]
    assert plan.val_frame_ids == ["frame_002"]
    coord = np.load(output_dir / "dataset" / "train" / "frame_000" / "coord.npy")
    strength = np.load(output_dir / "dataset" / "train" / "frame_000" / "strength.npy")
    segment = np.load(output_dir / "dataset" / "train" / "frame_000" / "segment.npy")
    assert coord.tolist() == [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert strength.reshape(-1).tolist() == [
        pytest.approx(10.0 / 255.0),
        1.0,
        pytest.approx(0.5 / 255.0),
    ]
    assert segment.tolist() == [0, 1, -1]
    assert manifest["removed_checkpoint_keys"] == ["seg_head.weight", "seg_head.bias"]

    taxonomy = json.loads((output_dir / "taxonomy.json").read_text(encoding="utf-8"))
    statistics = json.loads((output_dir / "class_statistics.json").read_text(encoding="utf-8"))
    config = (output_dir / "litept_custom_config.py").read_text(encoding="utf-8")
    compile(config, str(output_dir / "litept_custom_config.py"), "exec")
    assert taxonomy["training_id_to_source_id"] == [2, 10]
    assert taxonomy["excluded_classes"] == [
        {"name": "Dust noise", "source_id": 20, "reason": "no_train_points"}
    ]
    assert statistics["unknown_source_ids"] == {"77": 5}
    assert statistics["excluded_classes"][0]["name"] == "Dust noise"
    assert statistics["classes"][0]["loss_weight"] == pytest.approx(1.0)
    assert "dataset_type = \"DefaultDataset\"" in config
    assert "num_classes=2" in config
    assert "training_id_to_source_id = [2, 10]" in config
    assert "split='train_oversampled.json'" in config
    assert 'weight=[1.0, 1.0]' in config
    assert "enable_wandb = False" in config
    train_entries = json.loads(
        (output_dir / "dataset" / "train_oversampled.json").read_text(encoding="utf-8")
    )
    assert train_entries == [f"train/{frame_id}" for frame_id in plan.train_frame_ids]

    command = build_training_command(plan)
    assert command[0] == sys.executable
    assert f"weight={output_dir / 'pretrained_backbone.pth'}" in command
    assert "resume=false" in command


def test_force_torch_pointrope_uses_generated_training_launcher(tmp_path: Path, monkeypatch):
    litept_root, labeler_dir, export_dir = make_finetune_inputs(tmp_path)
    output_dir = tmp_path / "finetune"
    plan = build_finetune_plan(
        litept_root=str(litept_root),
        export_dir=str(export_dir),
        labeler_dir=str(labeler_dir),
        output_dir=str(output_dir),
        force_torch_pointrope=True,
    )

    def fake_prepare_checkpoint(source: Path, destination: Path) -> list[str]:
        destination.write_bytes(b"backbone")
        return ["seg_head.weight", "seg_head.bias"]

    monkeypatch.setattr(litept_finetune, "prepare_backbone_checkpoint", fake_prepare_checkpoint)
    prepare_finetune_run(plan)

    launcher_path = output_dir / "train_litept_custom.py"
    launcher = launcher_path.read_text(encoding="utf-8")
    compile(launcher, str(launcher_path), "exec")
    assert 'sys.modules["libs.pointrope"] = pointrope_package' in launcher
    assert "LitePT training PointROPE backend: torch" in launcher

    command = build_training_command(plan)
    assert command[:2] == [sys.executable, str(launcher_path)]


def test_noise_frames_are_kept_in_train_and_oversampled(tmp_path: Path, monkeypatch):
    litept_root, labeler_dir, export_dir = make_finetune_inputs(tmp_path)
    mask_path = export_dir / "frame_000" / "semantic_mask.npy"
    mask = np.load(mask_path)
    mask.reshape(-1)[1] = 20
    np.save(mask_path, mask)
    output_dir = tmp_path / "finetune"
    plan = build_finetune_plan(
        litept_root=str(litept_root),
        export_dir=str(export_dir),
        labeler_dir=str(labeler_dir),
        output_dir=str(output_dir),
        noise_frame_repeat=4,
    )

    def fake_prepare_checkpoint(source: Path, destination: Path) -> list[str]:
        destination.write_bytes(b"backbone")
        return ["seg_head.weight", "seg_head.bias"]

    monkeypatch.setattr(litept_finetune, "prepare_backbone_checkpoint", fake_prepare_checkpoint)
    manifest = prepare_finetune_run(plan)

    assert "frame_000" in plan.train_frame_ids
    assert plan.taxonomy["class_names"] == ["Car", "Ground", "Dust noise"]
    entries = json.loads(
        (output_dir / "dataset" / "train_oversampled.json").read_text(encoding="utf-8")
    )
    assert entries.count("train/frame_000") == 4
    assert manifest["oversampling"]["repeated_frames"]["frame_000"] == 4
    assert manifest["oversampling"]["noise_class_names"] == ["Dust noise"]


def test_filter_checkpoint_removes_only_segmentation_head():
    state_dict = OrderedDict(
        [
            ("module.backbone.embedding.weight", object()),
            ("module.seg_head.weight", object()),
            ("seg_head.bias", object()),
            ("backbone.block.weight", object()),
        ]
    )
    filtered, removed = filter_seg_head_state_dict(state_dict)
    assert list(filtered) == ["module.backbone.embedding.weight", "backbone.block.weight"]
    assert removed == ["module.seg_head.weight", "seg_head.bias"]


def test_finetune_dry_run_cli_needs_no_torch(tmp_path: Path):
    litept_root, labeler_dir, export_dir = make_finetune_inputs(tmp_path, frame_count=2)
    script = Path(__file__).resolve().parents[1] / "scripts" / "finetune_litept.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--litept-root",
            str(litept_root),
            "--export-dir",
            str(export_dir),
            "--labeler-dir",
            str(labeler_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    payload = json.loads(result.stdout)
    assert payload["frame_count"] == 2
    assert payload["train_frame_ids"] == ["frame_001"]
    assert payload["val_frame_ids"] == ["frame_000"]
    assert payload["taxonomy"]["source_id_to_training_id"] == {"2": 0, "10": 1}
    assert payload["taxonomy"]["excluded_classes"][0]["source_id"] == 20
    assert not (tmp_path / "out").exists()
