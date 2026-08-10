from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_sam3_classes_logs.py"


def load_script():
    spec = importlib.util.spec_from_file_location("summarize_sam3_classes_logs", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_log(root: Path, frame_id: str, payload: dict) -> None:
    frame_dir = root / frame_id
    frame_dir.mkdir(parents=True)
    (frame_dir / "classes_log.json").write_text(json.dumps(payload), encoding="utf-8")


def test_summarizes_confidence_per_class_and_processing_time(tmp_path: Path):
    script = load_script()
    write_log(
        tmp_path,
        "000000",
        {
            "processing_time_seconds": 8.0,
            "instances": [
                {"label": "CAR", "class_id": 2, "score": 0.8},
                {"label": "CAR", "class_id": 2, "score": 0.6},
            ],
        },
    )
    write_log(
        tmp_path,
        "000001",
        {
            "processing_time_seconds": 12.0,
            "instances": [
                {"label": "CAR", "class_id": 2, "score": 1.0},
                {"label": "traffic_sign", "class_id": 10, "score": 0.7},
            ],
        },
    )

    result = script.summarize_classes_logs(tmp_path)

    assert result["logs_read"] == 2
    assert result["total_instances"] == 4
    assert result["processing_time"]["mean_seconds"] == pytest.approx(10.0)
    classes = {item["label"]: item for item in result["classes"]}
    assert classes["CAR"]["mean_confidence"] == pytest.approx(0.8)
    assert classes["CAR"]["instance_count"] == 3
    assert classes["CAR"]["frame_count"] == 2
    assert classes["traffic_sign"]["mean_confidence"] == pytest.approx(0.7)


def test_skips_bad_log_and_missing_timing_in_non_strict_mode(tmp_path: Path):
    script = load_script()
    write_log(
        tmp_path,
        "000000",
        {"instances": [{"label": "CAR", "class_id": 2, "score": 0.8}]},
    )
    bad_dir = tmp_path / "000001"
    bad_dir.mkdir()
    (bad_dir / "classes_log.json").write_text("not json", encoding="utf-8")

    result = script.summarize_classes_logs(tmp_path)

    assert result["logs_read"] == 1
    assert result["logs_skipped"] == 1
    assert result["processing_time"]["frame_count"] == 0
    assert result["processing_time"]["mean_seconds"] is None


def test_strict_mode_rejects_bad_instance_score(tmp_path: Path):
    script = load_script()
    write_log(tmp_path, "000000", {"instances": [{"label": "CAR", "score": "bad"}]})

    with pytest.raises(ValueError, match="score must be a finite number"):
        script.summarize_classes_logs(tmp_path, strict=True)
