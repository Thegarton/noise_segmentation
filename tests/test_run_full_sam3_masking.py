from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path


def test_configure_gpu_selects_one_physical_device(monkeypatch) -> None:
    script = load_script()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SAM3_PHYSICAL_GPU_ID", raising=False)

    script.configure_gpu(3)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ["SAM3_PHYSICAL_GPU_ID"] == "3"


def test_load_csv_class_tags_reads_mapping(tmp_path: Path) -> None:
    script = load_script()
    config = tmp_path / "tags.yaml"
    config.write_text(
        "class_tags:\n"
        '  traffic_cone: "锥桶(traffic_cone)"\n'
        '  ground_markings: "地面标识(ground_markings)"\n',
        encoding="utf-8",
    )

    tags = script.load_csv_class_tags(config)

    assert tags == {
        "traffic_cone": "锥桶(traffic_cone)",
        "ground_markings": "地面标识(ground_markings)",
    }


def test_load_matched_frames_resolves_zero_based_time_map_index(tmp_path: Path) -> None:
    script = load_script()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    time_rows = []
    for index in range(81):
        image_name = f"{16926 + index:06d}"
        time_rows.append(f"{image_name}>>{1786519580000000 + index}")
        if index >= 79:
            (image_dir / f"{image_name}.jpg").write_bytes(b"image")

    time_map = tmp_path / "imgTimeMap.txt"
    time_map.write_text("\n".join(time_rows), encoding="utf-8")
    match_map = tmp_path / "imgMatch.txt"
    match_map.write_text(
        "000125>>000079>>-2\n"
        "000126>>000080>>7\n",
        encoding="utf-8",
    )

    matches = script.load_matched_frames(
        image_dir=image_dir,
        img_time_map=time_map,
        img_match=match_map,
        start_frame=126,
        end_frame=126,
        recursive=False,
    )

    assert len(matches) == 1
    assert matches[0].frame_id == "000126"
    assert matches[0].image_reference == "000080"
    assert matches[0].image_name == "017006"
    assert matches[0].image_path == (image_dir / "017006.jpg").resolve()
    assert matches[0].diff_ms == 7.0


def test_load_matched_frames_accepts_image_name_from_time_map(tmp_path: Path) -> None:
    script = load_script()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "017006.png").write_bytes(b"image")
    time_map = tmp_path / "imgTimeMap.txt"
    time_map.write_text("017006>>1786519583605000\n", encoding="utf-8")
    match_map = tmp_path / "imgMatch.txt"
    match_map.write_text("000126>>017006>>7\n", encoding="utf-8")

    matches = script.load_matched_frames(
        image_dir=image_dir,
        img_time_map=time_map,
        img_match=match_map,
        start_frame=None,
        end_frame=None,
        recursive=False,
    )

    assert [(item.frame_id, item.image_name) for item in matches] == [("000126", "017006")]


def test_write_run_summary_csv_collects_classes_and_timestamps(tmp_path: Path) -> None:
    script = load_script()
    matches = []
    for index, (frame_id, timestamp) in enumerate(
        [
            ("000080", 1000),
            ("000081", 2000),
        ]
    ):
        matches.append(
            script.MatchedFrame(
                frame_id=frame_id,
                image_path=Path(f"/{index:06d}.jpg"),
                image_reference=f"{index:06d}",
                image_name=f"{index:06d}",
                image_timestamp=timestamp,
                diff_ms=0.0,
            )
        )
    args = argparse.Namespace(
        out_dir=str(tmp_path),
        image_dir=str(tmp_path),
        recursive=False,
        max_images=None,
        data_name=None,
    )

    summary_path = script.write_run_summary_csv(
        args,
        detected_classes={"ground_markings", "front_of_vehicle", "epoxy_floor"},
        frame_class_presence=[
            {"ground_markings", "front_of_vehicle"},
            {"front_of_vehicle", "epoxy_floor"},
        ],
        matches=matches,
    )

    with summary_path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["tags"] == (
        "环氧地坪(epoxy_floor); "
        "front_of_vehicle; "
        "地面标识(ground_markings)"
    )
    assert row["data_name"] == "data_name"
    assert row["start_frame"] == "000080"
    assert row["end_frame"] == "000081"
    assert row["start_timestamp"] == "1000"
    assert row["end_timestamp"] == "2000"
    assert row["frame_num"] == "2"
    assert json.loads(row["class_frame_counts"]) == {
        "epoxy_floor": 1,
        "front_of_vehicle": 2,
        "ground_markings": 1,
    }


def test_write_run_summary_csv_filters_by_count_and_consecutive_frames(tmp_path: Path) -> None:
    script = load_script()
    args = argparse.Namespace(
        out_dir=str(tmp_path),
        image_dir=str(tmp_path),
        recursive=False,
        max_images=None,
        data_name="sequence",
        class_min_frames=3,
        class_min_consecutive_frames=2,
    )
    for frame_id in range(5):
        (tmp_path / f"{frame_id:06d}.jpg").write_bytes(b"image")

    frame_class_presence = [
        {"consecutive", "scattered"},
        {"consecutive"},
        {"consecutive", "scattered", "too_rare"},
        set(),
        {"scattered"},
    ]
    summary_path = script.write_run_summary_csv(
        args,
        detected_classes={"consecutive", "scattered", "too_rare"},
        frame_class_presence=frame_class_presence,
        matches=None,
    )

    with summary_path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))

    assert row["tags"] == "consecutive"
    assert json.loads(row["class_frame_counts"]) == {
        "consecutive": 3,
        "scattered": 3,
        "too_rare": 1,
    }
    assert json.loads(row["class_max_consecutive_frames"]) == {
        "consecutive": 3,
        "scattered": 1,
        "too_rare": 1,
    }


def load_script():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = scripts_dir / "run_full_sam3_masking.py"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("run_full_sam3_masking_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
