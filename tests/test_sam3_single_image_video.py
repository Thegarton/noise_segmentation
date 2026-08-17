from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def test_build_single_image_command_uses_conda_prefix(tmp_path: Path):
    script = load_script()
    args = make_args(
        sam3_conda_prefix="/envs/sam3",
        sam3_root="/repo/sam3",
        sam3_model_path="/repo/sam3/sam3.1",
        overwrite=True,
        validate=True,
    )

    command = script.build_single_image_command(
        args=args,
        image_dir=tmp_path / "frames",
        out_dir=tmp_path / "out",
        single_image_script=tmp_path / "run_sam3_single_image_folder.py",
        max_images=None,
    )

    assert command[:5] == ["conda", "run", "-p", "/envs/sam3", "python"]
    assert "--image-dir" in command
    assert str(tmp_path / "frames") in command
    assert "--sam3-root" in command
    assert "/repo/sam3" in command
    assert "--sam3-model-path" in command
    assert "/repo/sam3/sam3.1" in command
    assert "--overwrite" in command
    assert "--validate" in command


def test_build_single_image_command_uses_conda_env(tmp_path: Path):
    script = load_script()
    args = make_args(sam3_conda_env="sam3")

    command = script.build_single_image_command(
        args=args,
        image_dir=tmp_path / "frames",
        out_dir=tmp_path / "out",
        single_image_script=tmp_path / "run_sam3_single_image_folder.py",
        max_images=1,
    )

    assert command[:5] == ["conda", "run", "-n", "sam3", "python"]
    assert command[-2:] == ["--max-images", "1"]


def test_build_single_image_command_passes_vehicle_orientation_options(tmp_path: Path):
    script = load_script()
    args = make_args(
        vehicle_orientation_checkpoint="/models/vehicle_orientation/model_best.pth",
        vehicle_prompt_label="vehicle",
        vehicle_orientation_device="cuda:0",
        vehicle_orientation_min_confidence=0.72,
        vehicle_orientation_min_margin=0.12,
        vehicle_orientation_nms_iou=0.81,
    )

    command = script.build_single_image_command(
        args=args,
        image_dir=tmp_path / "frames",
        out_dir=tmp_path / "out",
        single_image_script=tmp_path / "run_sam3_single_image_folder.py",
        max_images=None,
    )

    assert command[command.index("--vehicle-orientation-checkpoint") + 1] == "/models/vehicle_orientation/model_best.pth"
    assert command[command.index("--vehicle-prompt-label") + 1] == "vehicle"
    assert command[command.index("--vehicle-orientation-device") + 1] == "cuda:0"
    assert command[command.index("--vehicle-orientation-min-confidence") + 1] == "0.72"
    assert command[command.index("--vehicle-orientation-min-margin") + 1] == "0.12"
    assert command[command.index("--vehicle-orientation-nms-iou") + 1] == "0.81"


def test_build_single_image_command_passes_sam3_only(tmp_path: Path):
    script = load_script()
    args = make_args(sam3_only=True)

    command = script.build_single_image_command(
        args=args,
        image_dir=tmp_path / "frames",
        out_dir=tmp_path / "out",
        single_image_script=tmp_path / "run_sam3_single_image_folder.py",
        max_images=None,
    )

    assert "--sam3-only" in command


def test_extract_video_frames_preserves_source_frame_indices(tmp_path: Path, monkeypatch):
    script = load_script()
    video_path = tmp_path / "input.avi"
    video_path.write_bytes(b"video")
    frames_dir = tmp_path / "frames"
    fake_cv2 = build_fake_cv2(total_frames=6)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    result = script.extract_video_frames(
        video_path=video_path,
        frames_dir=frames_dir,
        start_frame=1,
        max_frames=2,
        frame_step=2,
        image_ext="jpg",
        jpeg_quality=90,
        overwrite=False,
    )

    assert [item.frame_id for item in result.frames] == ["000001", "000003"]
    assert [item.video_frame_index for item in result.frames] == [1, 3]
    assert [path.name for path in frames_dir.glob("*.jpg")] == ["000001.jpg", "000003.jpg"]


def test_extract_video_frames_requires_overwrite_for_existing_images(tmp_path: Path):
    script = load_script()
    video_path = tmp_path / "input.avi"
    video_path.write_bytes(b"video")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "000000.jpg").write_bytes(b"old")

    try:
        script.extract_video_frames(
            video_path=video_path,
            frames_dir=frames_dir,
            start_frame=0,
            max_frames=1,
            frame_step=1,
            image_ext="jpg",
            jpeg_quality=95,
            overwrite=False,
        )
    except FileExistsError as exc:
        assert "--overwrite" in str(exc)
    else:
        raise AssertionError("expected FileExistsError")


def make_args(**overrides):
    values = {
        "sam3_conda_prefix": None,
        "sam3_conda_env": None,
        "prompt_config": "prompts.yaml",
        "classes_yaml": "classes.yaml",
        "min_score": 0.7,
        "min_mask_size": 30,
        "label_min_scores": None,
        "sam3_root": None,
        "sam3_model_path": None,
        "vehicle_orientation_checkpoint": None,
        "vehicle_prompt_label": "vehicle",
        "vehicle_orientation_device": "auto",
        "vehicle_orientation_min_confidence": 0.70,
        "vehicle_orientation_min_margin": 0.10,
        "vehicle_orientation_nms_iou": 0.80,
        "use_fa3": False,
        "prompt_log": False,
        "overwrite": False,
        "validate": False,
        "max_prompts": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def build_fake_cv2(*, total_frames: int):
    class FakeCapture:
        def __init__(self, path: str):
            self.path = path
            self.current = 0
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == 7:
                return float(total_frames)
            if prop == 5:
                return 25.0
            return 0.0

        def set(self, prop, value):
            if prop == 1:
                self.current = int(value)
            return True

        def read(self):
            if self.current >= total_frames:
                return False, None
            frame = {"index": self.current}
            self.current += 1
            return True, frame

        def release(self):
            self.released = True

    def imwrite(path: str, frame, params):
        Path(path).write_text(str(frame["index"]), encoding="utf-8")
        return True

    return SimpleNamespace(
        CAP_PROP_POS_FRAMES=1,
        CAP_PROP_FPS=5,
        CAP_PROP_FRAME_COUNT=7,
        IMWRITE_JPEG_QUALITY=1,
        VideoCapture=FakeCapture,
        imwrite=imwrite,
    )


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_sam3_single_image_video.py"
    spec = importlib.util.spec_from_file_location("run_sam3_single_image_video", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
