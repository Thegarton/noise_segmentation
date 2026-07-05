from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from autolabeler.teachers.sam3_video_adapter import (
    Sam3VideoResult,
    build_sam3_video_command,
    load_camera_frame_manifest,
    load_sam3_video_result,
    sam3_video_to_semantic_mask,
)


def test_sam3_video_command_uses_conda_env():
    cmd = build_sam3_video_command(
        video="camera.mp4",
        prompt_config="prompts.yaml",
        frames_json="frames.json",
        output_dir="out/raw",
        sam3_video_script="/tools/sam3_video.py",
        conda_env="sam3",
    )

    assert cmd[:5] == ["conda", "run", "-n", "sam3", "python"]
    assert "--video" in cmd
    assert "camera.mp4" in cmd


def test_sam3_video_command_uses_conda_prefix():
    cmd = build_sam3_video_command(
        video="camera.mp4",
        prompt_config="prompts.yaml",
        frames_json="frames.json",
        output_dir="out/raw",
        sam3_video_script="/tools/sam3_video.py",
        conda_env="sam3",
        conda_prefix="/home/a60116606/miniconda3/envs/sam3",
        extra_args=["--sam3-root", "/home/a60116606/git_repo/sam3", "--sam3-model-path", "/home/a60116606/git_repo/sam3/sam3.1"],
    )

    assert cmd[:5] == ["conda", "run", "-p", "/home/a60116606/miniconda3/envs/sam3", "python"]
    assert cmd[-4:] == ["--sam3-root", "/home/a60116606/git_repo/sam3", "--sam3-model-path", "/home/a60116606/git_repo/sam3/sam3.1"]


def test_camera_manifest_filters_synced_frames(tmp_path: Path):
    manifest = tmp_path / "camera_frame_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "frames": [
                    {
                        "frame_id": "a",
                        "synced": True,
                        "video_frame_index": 7,
                        "image_path": "/tmp/a.jpg",
                        "lidar_timestamp_us": 1000,
                        "video_timestamp_us": 1010,
                        "delta_ms": 0.01,
                    },
                    {"frame_id": "b", "synced": False, "video_frame_index": 8},
                ],
            }
        ),
        encoding="utf-8",
    )

    frames = load_camera_frame_manifest(manifest)

    assert len(frames) == 1
    assert frames[0].frame_id == "a"
    assert frames[0].video_frame_index == 7


def test_sam3_video_npz_validation_rejects_wrong_mask_dtype(tmp_path: Path):
    path = tmp_path / "bad.npz"
    np.savez(path, masks=np.zeros((1, 2, 2), dtype=np.uint8), scores=np.ones(1), labels=np.asarray(["CAR"]), prompts=np.asarray(["car"]))

    with pytest.raises(ValueError, match="SAM3 video masks must be bool"):
        load_sam3_video_result(path)


def test_sam3_video_to_semantic_mask_resolves_overlap_and_low_score():
    masks = np.zeros((3, 3, 3), dtype=bool)
    masks[0, 1, 1] = True
    masks[1, 1, 1] = True
    masks[1, 2, 2] = True
    masks[2, 0, 0] = True
    result = Sam3VideoResult(
        masks=masks,
        scores=np.asarray([0.8, 0.95, 0.2], dtype=np.float32),
        labels=np.asarray(["CAR", "traffic_sign", "CAR"]),
        prompts=np.asarray(["car", "traffic sign", "car"]),
    )

    semantic = sam3_video_to_semantic_mask(result, label_to_id={"CAR": 2, "traffic_sign": 10}, min_score=0.7)

    assert semantic.semantic_mask[1, 1] == 10
    assert semantic.semantic_mask[2, 2] == 10
    assert semantic.semantic_mask[0, 0] == 0
    assert semantic.confidence[1, 1] == pytest.approx(0.95)
    assert semantic.accepted_instances == 2
    assert semantic.ignored_instances == 1


def test_run_sam3_video_teacher_smoke_without_real_conda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = load_run_sam3_video_teacher_script()
    manifest = tmp_path / "camera_frame_manifest.json"
    image = tmp_path / "frame_001.jpg"
    image.write_bytes(b"fake image bytes")
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "frames": [
                    {
                        "frame_id": "frame_001",
                        "synced": True,
                        "video_frame_index": 5,
                        "image_path": str(image),
                        "lidar_timestamp_us": 1000,
                        "video_timestamp_us": 1002,
                        "delta_ms": 0.002,
                    },
                    {
                        "frame_id": "frame_003",
                        "synced": True,
                        "video_frame_index": 7,
                        "image_path": str(image),
                        "lidar_timestamp_us": 2000,
                        "video_timestamp_us": 2002,
                        "delta_ms": 0.002,
                    },
                    {"frame_id": "frame_002", "synced": False, "video_frame_index": 6},
                ],
            }
        ),
        encoding="utf-8",
    )
    prompts = tmp_path / "prompts.yaml"
    prompts.write_text('CAR:\n  - "car"\n', encoding="utf-8")
    classes = tmp_path / "classes.yaml"
    classes.write_text("semantic_classes:\n  background: 0\n  CAR: 2\n  ignore: 255\n", encoding="utf-8")
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"fake video")
    out_dir = tmp_path / "out"

    def fake_run_sam3_video_teacher(**kwargs):
        assert kwargs["extra_args"][:2] == ["--max-video-frame-index", "5"]
        frames_payload = json.loads(Path(kwargs["frames_json"]).read_text(encoding="utf-8"))
        raw_out = Path(kwargs["output_dir"])
        for frame in frames_payload["frames"]:
            mask = np.zeros((1, 2, 3), dtype=bool)
            mask[0, 0, 1] = True
            np.savez(
                raw_out / f"{frame['frame_id']}.npz",
                masks=mask,
                scores=np.asarray([0.9], dtype=np.float32),
                labels=np.asarray(["CAR"]),
                prompts=np.asarray(["car"]),
            )

    monkeypatch.setattr(script, "run_sam3_video_teacher", fake_run_sam3_video_teacher)
    monkeypatch.setattr(script, "save_semantic_overlay", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sam3_video_teacher.py",
            "--video",
            str(video),
            "--camera-frame-manifest",
            str(manifest),
            "--prompt-config",
            str(prompts),
            "--classes-yaml",
            str(classes),
            "--out-dir",
            str(out_dir),
            "--sam3-video-script",
            str(tmp_path / "fake_sam3.py"),
            "--max-frames",
            "1",
            "--validate",
        ],
    )

    script.main()

    frame_out = out_dir / "frame_001"
    assert np.load(frame_out / "semantic_mask.npy").tolist() == [[0, 2, 0], [0, 0, 0]]
    assert np.load(frame_out / "confidence.npy")[0, 1] == pytest.approx(0.9)
    assert (frame_out / "sam3_video.npz").is_file()
    assert (frame_out / "metadata.json").is_file()
    assert (out_dir / "sam3_video_teacher_manifest.json").is_file()
    assert not (out_dir / "frame_002").exists()
    assert not (out_dir / "frame_003").exists()
    frames_payload = json.loads((out_dir / "sam3_video_frames.json").read_text(encoding="utf-8"))
    assert frames_payload["frames"] == [
        {"frame_id": "frame_001", "video_frame_index": 5, "image_path": str(image)}
    ]


def load_run_sam3_video_teacher_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_sam3_video_teacher.py"
    spec = importlib.util.spec_from_file_location("run_sam3_video_teacher", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
