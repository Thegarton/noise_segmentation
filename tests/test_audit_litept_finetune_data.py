from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def test_audit_frame_explains_dropped_and_ignored_points(tmp_path: Path):
    script = load_script()
    labeler_dir = tmp_path / "labeler"
    export_dir = labeler_dir / "export"
    (labeler_dir / "velodyne").mkdir(parents=True)
    (export_dir / "000003").mkdir(parents=True)
    points = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [2.0, 0.0, 0.0, 3.0],
            [3.0, 0.0, 0.0, 4.0],
        ],
        dtype=np.float32,
    )
    points.tofile(labeler_dir / "velodyne" / "000003.bin")
    np.save(export_dir / "000003" / "semantic_mask.npy", np.asarray([2, 2, 255, 99], dtype=np.uint16))

    summary = script.audit_frame(
        frame_id="000003",
        labeler_dir=labeler_dir,
        export_dir=export_dir,
        output_dir=None,
        source_to_training={2: 0},
        ignore_source_ids={255},
        id_to_name={2: "CAR", 255: "ignore"},
        excluded_by_id={},
        inspect_index=2,
    )

    assert summary["points"] == 4
    assert summary["valid_points"] == 3
    assert summary["dropped_near_zero_xyz"] == 1
    assert summary["training_segment_counts"] == {"-1": 2, "0": 1}
    assert summary["ignored_reason_counts"] == {
        "ignore_source_id": 1,
        "unknown_or_not_trainable_source_id": 1,
    }
    assert summary["inspect_index"]["raw_point_index"] == 3
    assert summary["inspect_index"]["source_id"] == 99
    assert summary["inspect_index"]["prepared_segment"] == -1


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_litept_finetune_data.py"
    spec = importlib.util.spec_from_file_location("audit_litept_finetune_data", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
