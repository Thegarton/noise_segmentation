#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    args = parse_args()
    if args.sam3_root:
        sys.path.insert(0, str(Path(args.sam3_root).resolve()))

    import sam3.model_builder as sam3_model_builder  # noqa: WPS433

    prompts = load_prompt_config(args.prompts)
    requested_frames = load_requested_frames(args.frames_json)
    if not requested_frames:
        raise ValueError(f"No frames requested in {args.frames_json}")
    empty_mask_shape = read_resource_shape(args.video)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_local_or_default_predictor(
        sam3_model_builder,
        sam3_model_path=args.sam3_model_path,
        sam3_root=args.sam3_root,
    )
    response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(args.video),
        }
    )
    session_id = response["session_id"]

    frame_results: dict[str, list[dict[str, Any]]] = {frame["frame_id"]: [] for frame in requested_frames}
    frame_index_to_ids: dict[int, list[str]] = {}
    for frame in requested_frames:
        frame_index_to_ids.setdefault(int(frame["video_frame_index"]), []).append(str(frame["frame_id"]))

    try:
        for label, label_prompts in prompts.items():
            for prompt in label_prompts:
                reset_session(predictor, session_id)
                add_text_prompt(
                    predictor,
                    session_id=session_id,
                    frame_index=args.prompt_frame_index,
                    prompt=prompt,
                )
                for frame_index, outputs in propagate_in_video(predictor, session_id):
                    if frame_index not in frame_index_to_ids:
                        continue
                    masks, scores, track_ids, boxes = extract_output_arrays(outputs)
                    for frame_id in frame_index_to_ids[frame_index]:
                        for i in range(masks.shape[0]):
                            frame_results[frame_id].append(
                                {
                                    "mask": masks[i],
                                    "score": float(scores[i]),
                                    "label": label,
                                    "prompt": prompt,
                                    "track_id": int(track_ids[i]) if track_ids is not None else -1,
                                    "box": boxes[i] if boxes is not None else None,
                                }
                            )
    finally:
        close_session(predictor, session_id)

    written = []
    for frame in requested_frames:
        frame_id = str(frame["frame_id"])
        out_path = output_dir / f"{frame_id}.npz"
        write_frame_npz(out_path, frame_results[frame_id], empty_mask_shape=empty_mask_shape)
        written.append({"frame_id": frame_id, "path": str(out_path), "instances": len(frame_results[frame_id])})

    manifest = {"version": 1, "video": str(args.video), "frames": written}
    (output_dir / "sam3_31_video_wrapper_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def reset_session(predictor: Any, session_id: str) -> None:
    predictor.handle_request(request={"type": "reset_session", "session_id": session_id})


def add_text_prompt(predictor: Any, *, session_id: str, frame_index: int, prompt: str) -> None:
    predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": int(frame_index),
            "text": prompt,
        }
    )


def propagate_in_video(predictor: Any, session_id: str):
    for response in predictor.handle_stream_request(
        request={
            "type": "propagate_in_video",
            "session_id": session_id,
        }
    ):
        yield int(response["frame_index"]), response["outputs"]


def close_session(predictor: Any, session_id: str) -> None:
    predictor.handle_request(request={"type": "close_session", "session_id": session_id})


def build_local_or_default_predictor(model_builder_module: Any, *, sam3_model_path: str | None, sam3_root: str | None) -> Any:
    builder = model_builder_module.build_sam3_multiplex_video_predictor
    sam3_model_path = sam3_model_path or infer_local_model_path(sam3_root)
    if sam3_model_path is None:
        return builder()

    model_dir = Path(sam3_model_path).expanduser().resolve()
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "sam3.1_multiplex.pt"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing SAM3.1 config.json in local model directory: {model_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing SAM3.1 checkpoint sam3.1_multiplex.pt in local model directory: {model_dir}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    patch_hf_checkpoint_download(model_builder_module, checkpoint_path=checkpoint_path, model_dir=model_dir)
    signature = inspect.signature(builder)
    params = signature.parameters
    kwargs: dict[str, Any] = {}

    for name in ("checkpoint_path", "ckpt_path", "checkpoint", "ckpt", "weights_path"):
        if name in params:
            kwargs[name] = str(checkpoint_path)
            break

    for name in ("config_path", "cfg_path", "model_config_path", "config_file", "hf_config_path"):
        if name in params:
            kwargs[name] = str(config_path)
            break

    for name in ("model_path", "model_dir", "model_id", "repo_id"):
        if name in params:
            kwargs[name] = str(model_dir)
            break

    if not kwargs:
        return builder()
    return builder(**kwargs)


def infer_local_model_path(sam3_root: str | None) -> str | None:
    if sam3_root is None:
        return None
    candidate = Path(sam3_root).expanduser().resolve() / "sam3.1"
    if (candidate / "config.json").is_file() and (candidate / "sam3.1_multiplex.pt").is_file():
        return str(candidate)
    return None


def patch_hf_checkpoint_download(model_builder_module: Any, *, checkpoint_path: Path, model_dir: Path) -> None:
    def _local_download_ckpt_from_hf(*args: Any, **kwargs: Any) -> str:
        return str(checkpoint_path)

    model_builder_module.download_ckpt_from_hf = _local_download_ckpt_from_hf
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(model_dir))


def extract_output_arrays(outputs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if isinstance(outputs, dict):
        masks = first_present(outputs, ("masks", "pred_masks", "mask_logits", "out_mask_logits"))
        scores = first_present(outputs, ("scores", "pred_scores", "object_scores", "ious"))
        track_ids = first_present(outputs, ("track_ids", "obj_ids", "object_ids", "ids"))
        boxes = first_present(outputs, ("boxes_xyxy", "boxes", "pred_boxes"))
    elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        track_ids = outputs[0]
        masks = outputs[1]
        scores = outputs[2] if len(outputs) >= 3 else None
        boxes = outputs[3] if len(outputs) >= 4 else None
    else:
        raise ValueError(f"Unsupported SAM3 frame output type: {type(outputs)!r}")

    masks_np = normalize_masks(to_numpy(masks))
    count = masks_np.shape[0]
    scores_np = np.ones((count,), dtype=np.float32) if scores is None else normalize_vector(to_numpy(scores), count, dtype=np.float32)
    track_ids_np = None if track_ids is None else normalize_vector(to_numpy(track_ids), count, dtype=np.int64)
    boxes_np = None if boxes is None else normalize_boxes(to_numpy(boxes), count)
    return masks_np, scores_np, track_ids_np, boxes_np


def normalize_masks(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected masks as [N,H,W], got {arr.shape}")
    if arr.dtype == np.bool_:
        return arr
    return arr > 0


def normalize_vector(value: np.ndarray, count: int, *, dtype) -> np.ndarray:
    arr = np.asarray(value).reshape(-1)
    if arr.shape != (count,):
        raise ValueError(f"Expected vector with {count} values, got {arr.shape}")
    return arr.astype(dtype, copy=False)


def normalize_boxes(value: np.ndarray, count: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (count, 4):
        raise ValueError(f"Expected boxes with shape {(count, 4)}, got {arr.shape}")
    return arr


def write_frame_npz(path: Path, instances: list[dict[str, Any]], *, empty_mask_shape: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not instances:
        height, width = empty_mask_shape
        np.savez(
            path,
            masks=np.zeros((0, height, width), dtype=bool),
            scores=np.zeros((0,), dtype=np.float32),
            labels=np.asarray([], dtype=str),
            prompts=np.asarray([], dtype=str),
            track_ids=np.zeros((0,), dtype=np.int64),
        )
        return

    masks = np.stack([np.asarray(item["mask"], dtype=bool) for item in instances], axis=0)
    scores = np.asarray([item["score"] for item in instances], dtype=np.float32)
    labels = np.asarray([item["label"] for item in instances])
    prompts = np.asarray([item["prompt"] for item in instances])
    track_ids = np.asarray([item["track_id"] for item in instances], dtype=np.int64)
    boxes = [item["box"] for item in instances]
    payload = {
        "masks": masks,
        "scores": scores,
        "labels": labels,
        "prompts": prompts,
        "track_ids": track_ids,
    }
    if all(box is not None for box in boxes):
        payload["boxes_xyxy"] = np.stack([np.asarray(box, dtype=np.float32) for box in boxes], axis=0)
    np.savez(path, **payload)


def load_requested_frames(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"frames-json has no frames list: {path}")
    out = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        if "frame_id" not in frame or "video_frame_index" not in frame:
            continue
        out.append({"frame_id": str(frame["frame_id"]), "video_frame_index": int(frame["video_frame_index"])})
    return out


def read_resource_shape(resource_path: str | Path) -> tuple[int, int]:
    path = Path(resource_path)
    if path.is_dir():
        frames = sorted(list(path.glob("*.jpg")) + list(path.glob("*.jpeg")) + list(path.glob("*.png")))
        if not frames:
            raise ValueError(f"No image frames found in video resource directory: {path}")
        from PIL import Image  # noqa: WPS433

        with Image.open(frames[0]) as image:
            width, height = image.size
        return int(height), int(width)

    import cv2  # noqa: WPS433

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video resource: {path}")
    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read first frame from video resource: {path}")
        return int(frame.shape[0]), int(frame.shape[1])
    finally:
        capture.release()


def load_prompt_config(path: str | Path) -> dict[str, list[str]]:
    prompts: dict[str, list[str]] = {}
    current_key: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0 and ":" in stripped:
            key, value = [x.strip() for x in stripped.split(":", 1)]
            current_key = key
            if value:
                prompts[key] = [strip_quotes(x.strip()) for x in value.strip("[]").split(",") if x.strip()]
            else:
                prompts[key] = []
            continue
        if indent >= 2 and stripped.startswith("- ") and current_key is not None:
            prompts[current_key].append(strip_quotes(stripped[2:].strip()))
    return prompts


def first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Cannot convert None to numpy array")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3.1 video wrapper used by noise_segmentation teacher pipeline.")
    parser.add_argument("--video", required=True, help="MP4 video path or directory with numbered JPEG frames.")
    parser.add_argument("--prompts", required=True, help="YAML prompt config keyed by target class name.")
    parser.add_argument("--frames-json", required=True, help="JSON with requested frame_id/video_frame_index pairs.")
    parser.add_argument("--output-dir", required=True, help="Directory where <frame_id>.npz outputs are written.")
    parser.add_argument("--sam3-root", default=None, help="Optional path to the SAM3 repository.")
    parser.add_argument("--sam3-model-path", default=None, help="Optional local facebook/sam3.1 HuggingFace model directory.")
    parser.add_argument("--prompt-frame-index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
