from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from autolabeler.hl320.csv_points import HL320_FEATURE_NAMES
from autolabeler.hl320.dataset import build_hl320_dataset
from autolabeler.hl320.litept_training import build_training_command, build_training_plan, prepare_training_run


def test_hl320_litept_prepare_writes_from_scratch_config(tmp_path: Path):
    dataset_root = make_dataset(tmp_path)
    litept_root = make_litept_root(tmp_path)

    plan = build_training_plan(
        litept_root=litept_root,
        dataset_root=dataset_root,
        output_dir=tmp_path / "train_out",
        epochs=12,
        batch_size=2,
        num_workers=1,
        force_torch_pointrope=True,
    )
    manifest = prepare_training_run(plan, overwrite=True)
    command = build_training_command(plan, resume=False)
    config_text = Path(plan.config_path).read_text(encoding="utf-8")

    assert manifest["prepared"] is True
    assert f"in_channels={3 + len(HL320_FEATURE_NAMES)}" in config_text
    assert "weight=None" not in config_text
    assert "CheckpointLoader" in config_text
    assert "feat_keys=(\"coord\", \"strength\")" in config_text
    assert command[0] == sys.executable
    assert command[1].endswith("train_hl320_litept.py")
    assert "weight=" not in " ".join(command)


def test_hl320_litept_dry_run_cli_does_not_import_torch(tmp_path: Path):
    dataset_root = make_dataset(tmp_path)
    litept_root = make_litept_root(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_hl320_litept_from_scratch.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--litept-root",
            str(litept_root),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(tmp_path / "train_out"),
            "--dry-run",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["litept_in_channels"] == 3 + len(HL320_FEATURE_NAMES)
    assert payload["training_command"] is None


def make_dataset(tmp_path: Path) -> Path:
    csv_dir = tmp_path / "csv"
    labels_dir = tmp_path / "labels"
    classes_yaml = tmp_path / "classes.yaml"
    csv_dir.mkdir()
    labels_dir.mkdir()
    classes_yaml.write_text(
        "semantic_classes:\n"
        "  background: 0\n"
        "  CAR: 2\n"
        "  PEDESTRIAN: 5\n"
        "  adhesive_noise: 18\n"
        "  ignore: 255\n",
        encoding="utf-8",
    )
    frame_specs = {
        "000000": [2, 18],
        "000001": [5, 18],
        "000002": [2, 5],
        "000003": [2, 5],
    }
    for frame_id, labels in frame_specs.items():
        (csv_dir / f"{frame_id}.csv").write_text(
            "x y z intensity slot pixel blockID Cxd Cyd\n"
            "1 0 0 10 2 0 0 10 20\n"
            "2 0 0 20 2 1 0 20 30\n"
            "3 0 0 30 2 1 1 20 30\n",
            encoding="utf-8",
        )
        frame_label_dir = labels_dir / frame_id
        frame_label_dir.mkdir()
        np.save(frame_label_dir / "semantic_mask.npy", np.asarray(labels, dtype=np.uint16))
    out = tmp_path / "hl320_dataset"
    build_hl320_dataset(
        csv_dir=csv_dir,
        labels_dir=labels_dir,
        output_dir=out,
        classes_yaml=classes_yaml,
        val_ratio=0.25,
        seed=4,
        overwrite=True,
    )
    return out


def make_litept_root(tmp_path: Path) -> Path:
    root = tmp_path / "LitePT"
    (root / "configs" / "_base_").mkdir(parents=True)
    (root / "configs" / "_base_" / "default_runtime.py").write_text("weight = None\nresume = False\n", encoding="utf-8")
    (root / "tools").mkdir()
    (root / "tools" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (root / "libs" / "pointrope").mkdir(parents=True)
    (root / "libs" / "pointrope" / "pointrope_torch.py").write_text("class PointROPE: pass\n", encoding="utf-8")
    return root
