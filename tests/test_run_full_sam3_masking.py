from __future__ import annotations

import argparse
import csv
import importlib.util
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


def test_short_cli_uses_production_defaults() -> None:
    script = load_script()
    args = script.build_parser().parse_args(
        [
            "--image-path",
            "/data/images",
            "--sensor-version",
            "HL320",
            "--img-match",
            "/data/imgMatch.txt",
            "--img-time-map",
            "/data/imgTimeMap.txt",
            "--output-path",
            "/output/run",
            "--gpu-num",
            "0",
            "--data-name",
            "dataset",
            "--start-frame",
            "101",
            "--end-frame",
            "408",
        ]
    )
    script.resolve_sensor_config_paths(args)

    assert args.image_dir == "/data/images"
    assert args.out_dir == "/output/run"
    assert args.gpu_id == 0
    assert Path(args.prompt_config).name == "sam3_text_prompts_v2.yaml"
    assert Path(args.classes_yaml).name == "classes_v2.yaml"
    assert Path(args.label_min_scores).name == "sam3_label_min_scores.yaml"
    assert Path(args.csv_tags_yaml).name == "sam3_csv_class_tags_zh_en.yaml"
    assert Path(args.sensor_config_dir).name.strip() == "HL320"
    assert Path(args.vehicle_orientation_checkpoint).name == "model_best.pth"
    assert args.min_score == 0.60
    assert args.class_min_frames == 2
    assert args.class_min_consecutive_frames == 2
    assert args.min_mask_size == 500
    assert args.overwrite is True
    assert args.log_json is True
    assert args.cache_visual_features is True
    assert args.save_intermediate_outputs is False
    assert args.vehicle_orientation_device == "cuda"
    assert args.vehicle_orientation_min_confidence == 0.60
    assert args.vehicle_orientation_min_margin == 0.10
    assert args.vehicle_orientation_nms_iou == 0.80
    assert args.vehicle_prompt_label == "vehicle"


def test_sensor_version_is_required() -> None:
    script = load_script()
    sensor_action = next(
        action for action in script.build_parser()._actions if action.dest == "sensor_version"
    )

    assert sensor_action.required is True


def test_sensor_config_paths_are_resolved_and_explicit_override_is_kept(tmp_path: Path) -> None:
    script = load_script()
    configs_root = tmp_path / "configs"
    sensor_dir = configs_root / "SENSOR_A"
    sensor_dir.mkdir(parents=True)
    for filename in script.SENSOR_CONFIG_FILENAMES.values():
        (sensor_dir / filename).write_text("config\n", encoding="utf-8")
    explicit_prompts = tmp_path / "custom_prompts.yaml"
    explicit_prompts.write_text("prompts\n", encoding="utf-8")
    args = argparse.Namespace(
        sensor_version="SENSOR_A",
        prompt_config=str(explicit_prompts),
        classes_yaml=None,
        label_min_scores=None,
        csv_tags_yaml=None,
    )

    script.resolve_sensor_config_paths(args, configs_root=configs_root)

    assert Path(args.prompt_config) == explicit_prompts.resolve()
    assert Path(args.classes_yaml) == (sensor_dir / "classes_v2.yaml").resolve()
    assert Path(args.label_min_scores) == (sensor_dir / "sam3_label_min_scores.yaml").resolve()
    assert Path(args.csv_tags_yaml) == (
        sensor_dir / "sam3_csv_class_tags_zh_en.yaml"
    ).resolve()


def test_main_short_cli_writes_only_csv(tmp_path: Path, monkeypatch) -> None:
    script = load_script()
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    image_dir.mkdir()
    output_dir.mkdir()
    (output_dir / "old-frame").mkdir()
    (output_dir / "old-frame" / "overlay.jpg").write_bytes(b"old")
    for image_name in ("000010", "000011"):
        (image_dir / f"{image_name}.jpg").write_bytes(b"image")

    time_map = tmp_path / "imgTimeMap.txt"
    time_map.write_text(
        "000010>>1000\n"
        "000011>>2000\n",
        encoding="utf-8",
    )
    match_map = tmp_path / "imgMatch.txt"
    match_map.write_text(
        "000101>>000010>>0\n"
        "000102>>000011>>0\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_sam3_single_image_folder(**kwargs):
        captured.update(kwargs)
        return script.FolderDetectionSummary(
            detected_classes=frozenset({"traffic_cone"}),
            frame_ids=("000101", "000102"),
            frame_class_presence=(
                frozenset({"traffic_cone"}),
                frozenset({"traffic_cone"}),
            ),
        )

    monkeypatch.setattr(script, "sam3_single_image_folder", fake_sam3_single_image_folder)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    script.main(
        [
            "--image-path",
            str(image_dir),
            "--sensor-version",
            "HL320",
            "--img-match",
            str(match_map),
            "--img-time-map",
            str(time_map),
            "--output-path",
            str(output_dir),
            "--gpu-num",
            "0",
            "--data-name",
            "dataset",
            "--start-frame",
            "101",
            "--end-frame",
            "102",
        ]
    )

    assert captured["save_outputs"] is False
    assert captured["cache_visual_features"] is True
    assert captured["log_json"] is True
    assert [path.name for path in output_dir.iterdir()] == ["sam3_run_summary.csv"]
    with (output_dir / "sam3_run_summary.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["tags"] == "锥桶(traffic_cone)"
    assert row["start_frame"] == "000101"
    assert row["end_frame"] == "000102"


def test_intermediate_output_flag_is_forwarded_to_sam3(tmp_path: Path, monkeypatch) -> None:
    script = load_script()
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    image_dir.mkdir()
    (image_dir / "000010.jpg").write_bytes(b"image")
    time_map = tmp_path / "imgTimeMap.txt"
    time_map.write_text("000010>>1000\n", encoding="utf-8")
    match_map = tmp_path / "imgMatch.txt"
    match_map.write_text("000101>>000010>>0\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_sam3_single_image_folder(**kwargs):
        captured.update(kwargs)
        frame_dir = Path(kwargs["out_dir"]) / "000101"
        frame_dir.mkdir(parents=True)
        (frame_dir / "metadata.json").write_text("{}", encoding="utf-8")
        return script.FolderDetectionSummary(
            detected_classes=frozenset(),
            frame_ids=("000101",),
            frame_class_presence=(frozenset(),),
        )

    monkeypatch.setattr(script, "sam3_single_image_folder", fake_sam3_single_image_folder)
    script.main(
        [
            "--image-path",
            str(image_dir),
            "--sensor-version",
            "HL320",
            "--img-match",
            str(match_map),
            "--img-time-map",
            str(time_map),
            "--output-path",
            str(output_dir),
            "--gpu-num",
            "0",
            "--data-name",
            "dataset",
            "--start-frame",
            "101",
            "--end-frame",
            "101",
            "--save-intermediate-outputs",
        ]
    )

    assert captured["save_outputs"] is True
    assert (output_dir / "000101" / "metadata.json").is_file()
    assert (output_dir / "sam3_run_summary.csv").is_file()


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


def test_load_matched_frames_uses_one_index_mode_for_numeric_image_names(tmp_path: Path) -> None:
    script = load_script()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    time_map = tmp_path / "imgTimeMap.txt"
    time_map.write_text(
        "\n".join(
            f"{index:06d}>>{1785742610000000 + index * 1000}"
            for index in range(1, 57)
        ),
        encoding="utf-8",
    )
    for index in range(1, 51):
        (image_dir / f"{index:06d}.jpg").write_bytes(b"image")
    match_map = tmp_path / "imgMatch.txt"
    match_map.write_text(
        "\n".join(
            f"{frame:06d}>>{frame - 1:06d}>>0"
            for frame in range(1, 51)
        ),
        encoding="utf-8",
    )

    matches = script.load_matched_frames(
        image_dir=image_dir,
        img_time_map=time_map,
        img_match=match_map,
        start_frame=1,
        end_frame=50,
        recursive=False,
    )

    assert matches[0].frame_id == "000001"
    assert matches[0].image_reference == "000000"
    assert matches[0].image_name == "000001"
    assert matches[0].image_timestamp == 1785742610001000
    assert matches[-1].frame_id == "000050"
    assert matches[-1].image_reference == "000049"
    assert matches[-1].image_name == "000050"
    assert matches[-1].image_timestamp == 1785742610050000


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
    assert list(row) == [
        "tags",
        "data_name",
        "start_frame",
        "end_frame",
        "start_timestamp",
        "end_timestamp",
        "frame_num",
    ]


def test_write_run_summary_csv_filters_by_count_and_consecutive_frames(tmp_path: Path) -> None:
    script = load_script()
    args = argparse.Namespace(
        out_dir=str(tmp_path),
        image_dir=str(tmp_path),
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
