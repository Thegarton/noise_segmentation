from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from autolabeler.hl320.csv_points import load_hl320_csv
from autolabeler.hl320.teacher import TeacherPolicy, evaluate_hl320_teacher_frame


def test_teacher_rules_keep_shape_handle_nan_and_protect_noise_from_sam3(tmp_path: Path):
    paths = write_sequence(tmp_path)
    class_to_id = class_mapping()
    noise_groups = class_noise_groups()
    sam3_dir = write_sam3_context(tmp_path, frame_ids=["000001"], point_count=7)
    current = load_hl320_csv(paths["current"])
    prev_frame = load_hl320_csv(paths["prev"])
    next_frame = load_hl320_csv(paths["next"])

    result = evaluate_hl320_teacher_frame(
        frame=current,
        prev_frame=prev_frame,
        next_frame=next_frame,
        class_to_id=class_to_id,
        noise_groups=noise_groups,
        sam3_dir=sam3_dir,
        litept_dir=None,
        image_dir=None,
        policy=TeacherPolicy(),
    )

    assert result.labels.shape == (current.point_count,)
    assert result.confidence.shape == (current.point_count,)
    assert result.labels[0] == 20
    assert result.labels[1] == 2
    assert result.labels[2] == 27
    assert result.labels[3] in {26, 27}
    assert result.labels[5] == 24
    assert result.labels[6] == 255
    assert result.confidence[6] == 0.0
    assert result.features_debug["valid_xyz"].tolist()[-1] is False
    assert result.features_debug["temporal_transient"][0]
    assert result.features_debug["temporal_stable"][1]
    assert any(entry["rule"] == "dust_noise" for entry in result.reasons["points"]["0"])
    assert any("sam3_context_penalty" in entry["reasons"] for entry in result.reasons["points"]["0"])


def test_teacher_cli_writes_candidate_layout_manifest_and_debug_features(tmp_path: Path):
    paths = write_sequence(tmp_path)
    csv_dir = paths["csv_dir"]
    classes_yaml = write_classes_yaml(tmp_path)
    sam3_dir = write_sam3_context(tmp_path, frame_ids=["000000", "000001", "000002"], point_count=7)
    litept_dir = write_litept_context(tmp_path, frame_ids=["000000", "000001", "000002"], point_count=7)
    out_dir = tmp_path / "teacher_out"

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_hl320_teacher_candidates.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--csv-dir",
            str(csv_dir),
            "--classes-yaml",
            str(classes_yaml),
            "--sam3-dir",
            str(sam3_dir),
            "--litept-dir",
            str(litept_dir),
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["frame_count"] == 3
    assert (out_dir / "hl320_teacher_candidates_manifest.json").is_file()
    assert (out_dir / "teacher_candidates" / "000001.npy").is_file()
    assert np.load(out_dir / "000001" / "candidate_mask.npy").shape == (7,)
    assert np.load(out_dir / "000001" / "candidate_confidence.npy").shape == (7,)
    debug = np.load(out_dir / "000001" / "features_debug.npz")
    assert "return_count" in debug.files
    assert "temporal_transient" in debug.files
    reasons = json.loads((out_dir / "000001" / "candidate_reasons.json").read_text(encoding="utf-8"))
    assert reasons["rule_counts"]["multipath_noise"] >= 1


def write_sequence(tmp_path: Path) -> dict[str, Path]:
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    prev_path = csv_dir / "000000.csv"
    current_path = csv_dir / "000001.csv"
    next_path = csv_dir / "000002.csv"
    write_hl320_csv(
        prev_path,
        [
            [20, 0, 0, 20, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [10, 0, 0, 10, 10, 0, 200, 200, 0, 1, 0, 1, 1],
            [10, 0, 0, 10, 20, 0, 200, 200, 0, 2, 0, 2, 2],
            [12, 0, 0, 12, 20.1, 0, 20, 20, 0, 2, 1, 2, 2],
            [14, 0, 0, 14, 20.2, 0, 10, 10, 0, 2, 2, 2, 2],
            [8, 0, -1, 8, 30, -10, 100, 1, 0, 3, 0, "nan", "nan"],
            ["nan", 0, 0, "nan", "nan", "nan", 0, 0, 0, 4, 0, "nan", "nan"],
        ],
    )
    write_hl320_csv(
        current_path,
        [
            [5, 0, 0, 5, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [10, 0, 0, 10, 10, 0, 200, 200, 0, 1, 0, 1, 1],
            [10, 0, 0, 10, 20, 0, 200, 200, 0, 2, 0, 2, 2],
            [12, 0, 0, 12, 20.1, 0, 20, 20, 0, 2, 1, 2, 2],
            [14, 0, 0, 14, 20.2, 0, 10, 10, 0, 2, 2, 2, 2],
            [8, 0, -1, 8, 30, -10, 100, 1, 0, 3, 0, "nan", "nan"],
            ["nan", 0, 0, "nan", "nan", "nan", 0, 0, 0, 4, 0, "nan", "nan"],
        ],
    )
    write_hl320_csv(
        next_path,
        [
            [20, 0, 0, 20, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [10, 0, 0, 10, 10, 0, 200, 200, 0, 1, 0, 1, 1],
            [10, 0, 0, 10, 20, 0, 200, 200, 0, 2, 0, 2, 2],
            [12, 0, 0, 12, 20.1, 0, 20, 20, 0, 2, 1, 2, 2],
            [14, 0, 0, 14, 20.2, 0, 10, 10, 0, 2, 2, 2, 2],
            [8, 0, -1, 8, 30, -10, 100, 1, 0, 3, 0, "nan", "nan"],
            ["nan", 0, 0, "nan", "nan", "nan", 0, 0, 0, 4, 0, "nan", "nan"],
        ],
    )
    return {"csv_dir": csv_dir, "prev": prev_path, "current": current_path, "next": next_path}


def write_hl320_csv(path: Path, rows: list[list[object]]) -> None:
    path.write_text(
        "x y z distance azimuth vertical intensity reflectivity slot pixel blockID Cxd Cyd\n"
        + "\n".join(" ".join(str(value) for value in row) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def write_sam3_context(tmp_path: Path, *, frame_ids: list[str], point_count: int) -> Path:
    del point_count
    sam3_dir = tmp_path / "sam3"
    for frame_id in frame_ids:
        frame_dir = sam3_dir / frame_id
        frame_dir.mkdir(parents=True)
        np.save(frame_dir / "semantic_mask.npy", np.full((4, 4), 2, dtype=np.int32))
        np.save(frame_dir / "confidence.npy", np.full((4, 4), 0.9, dtype=np.float32))
    return sam3_dir


def write_litept_context(tmp_path: Path, *, frame_ids: list[str], point_count: int) -> Path:
    litept_dir = tmp_path / "litept"
    for frame_id in frame_ids:
        frame_dir = litept_dir / frame_id
        frame_dir.mkdir(parents=True)
        labels = np.full(point_count, 255, dtype=np.int32)
        labels[0] = 20
        confidence = np.zeros(point_count, dtype=np.float32)
        confidence[0] = 0.8
        np.save(frame_dir / "semantic_mask.npy", labels)
        np.save(frame_dir / "confidence.npy", confidence)
    return litept_dir


def write_classes_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "classes.yaml"
    path.write_text(
        "semantic_classes:\n"
        "  background: 0\n"
        "  CAR: 2\n"
        "  dust_noise: 20\n"
        "  long_distance_noise: 22\n"
        "  underground_mirror_noise: 24\n"
        "  multipath_noise: 26\n"
        "  multiple_range_noise: 27\n"
        "  crosstalk_noise_1: 15\n"
        "  adhesive_noise: 23\n"
        "  ignore: 255\n"
        "noise_groups:\n"
        "  dust_noise:\n"
        "    - dust_noise\n"
        "  multipath_noise:\n"
        "    - multipath_noise\n"
        "  multiple_range_noise:\n"
        "    - multiple_range_noise\n"
        "  underground_mirror_noise:\n"
        "    - underground_mirror_noise\n"
        "  crosstalk_noise:\n"
        "    - crosstalk_noise_1\n"
        "  adhesive_noise:\n"
        "    - adhesive_noise\n",
        encoding="utf-8",
    )
    return path


def class_mapping() -> dict[str, int]:
    return {
        "background": 0,
        "CAR": 2,
        "crosstalk_noise_1": 15,
        "dust_noise": 20,
        "adhesive_noise": 23,
        "underground_mirror_noise": 24,
        "multipath_noise": 26,
        "multiple_range_noise": 27,
        "ignore": 255,
    }


def class_noise_groups() -> dict[str, list[str]]:
    return {
        "dust_noise": ["dust_noise"],
        "underground_mirror_noise": ["underground_mirror_noise"],
        "multipath_noise": ["multipath_noise"],
        "multiple_range_noise": ["multiple_range_noise"],
        "crosstalk_noise": ["crosstalk_noise_1"],
        "adhesive_noise": ["adhesive_noise"],
    }
