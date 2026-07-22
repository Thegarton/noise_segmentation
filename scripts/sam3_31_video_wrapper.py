#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


MASK_KEYS = ("masks", "pred_masks", "out_binary_masks", "video_res_masks", "mask_logits", "out_mask_logits")
SCORE_KEYS = ("scores", "pred_scores", "object_scores", "out_probs", "ious")
TRACK_ID_KEYS = ("track_ids", "obj_ids", "out_obj_ids", "object_ids", "ids")
BOX_KEYS = ("boxes_xyxy", "boxes", "pred_boxes", "out_boxes_xywh")
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    args = parse_args()
    if args.sam3_root:
        sys.path.insert(0, str(Path(args.sam3_root).resolve()))

    sdpa_summary = configure_torch_sdpa_backend(args.sdpa_backend)
    log(f"torch SDPA backend: {json.dumps(sdpa_summary, ensure_ascii=False, sort_keys=True)}")

    import sam3.model_builder as sam3_model_builder  # noqa: WPS433

    decoder_sdpa_patch = patch_sam3_decoder_sdpa_kernel(str(sdpa_summary["selected"]))
    if decoder_sdpa_patch:
        log("patched sam3.model.decoder sdpa_kernel to force MATH backend")

    prompts = load_prompt_config(args.prompts)
    requested_frames = load_requested_frames(args.frames_json)
    if not requested_frames:
        raise ValueError(f"No frames requested in {args.frames_json}")
    log(f"requested frames: {len(requested_frames)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_resource_path = str(args.frames_resource)
    session_frames = requested_frames
    limited_resource_summary: dict[str, Any] | None = None
    if not args.use_original_video_resource:
        session_resource_path, session_frames, limited_resource_summary = prepare_limited_video_resource(
            frames_resource=args.frames_resource,
            requested_frames=requested_frames,
            output_dir=output_dir,
        )
        log(
            "limited SAM3 resource: "
            f"{limited_resource_summary['frames']} frames from {limited_resource_summary['source']} "
            f"at {limited_resource_summary['resource_path']}"
        )
    else:
        log("using original video resource; SAM3 may decode/cache the full video")
    empty_mask_shape = read_resource_shape(session_resource_path)

    log(f"building SAM3 predictor, use_fa3={args.use_fa3}")
    predictor = build_local_or_default_predictor(
        sam3_model_builder,
        sam3_model_path=args.sam3_model_path,
        sam3_root=args.sam3_root,
        use_fa3=args.use_fa3,
    )
    log("starting SAM3 session")
    response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(session_resource_path),
        }
    )
    session_id = response["session_id"]
    log(f"SAM3 session started: {session_id}")

    frame_results: dict[str, list[dict[str, Any]]] = {frame["frame_id"]: [] for frame in requested_frames}
    frame_index_to_ids: dict[int, list[str]] = {}
    for frame in session_frames:
        frame_index_to_ids.setdefault(int(frame["video_frame_index"]), []).append(str(frame["frame_id"]))

    try:
        for label, label_prompts in prompts.items():
            for prompt in label_prompts:
                log(f"prompt label={label!r} text={prompt!r}")
                reset_session(predictor, session_id)
                add_text_prompt(
                    predictor,
                    session_id=session_id,
                    frame_index=args.prompt_frame_index,
                    prompt=prompt,
                )
                remaining_frame_indices = set(frame_index_to_ids)
                for frame_index, outputs in propagate_in_video(predictor, session_id):
                    if args.max_video_frame_index is not None and frame_index > args.max_video_frame_index:
                        break
                    if frame_index not in frame_index_to_ids:
                        continue
                    if outputs is None or not has_mask_payload(outputs):
                        remaining_frame_indices.discard(frame_index)
                        if not remaining_frame_indices:
                            break
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
                    remaining_frame_indices.discard(frame_index)
                    if not remaining_frame_indices:
                        break
    finally:
        log("closing SAM3 session")
        close_session(predictor, session_id)

    written = []
    for frame in requested_frames:
        frame_id = str(frame["frame_id"])
        out_path = output_dir / f"{frame_id}.npz"
        write_frame_npz(out_path, frame_results[frame_id], empty_mask_shape=empty_mask_shape)
        written.append({"frame_id": frame_id, "path": str(out_path), "instances": len(frame_results[frame_id])})

    manifest = {
        "version": 1,
        "frames_resource": str(args.frames_resource),
        "session_resource_path": str(session_resource_path),
        "limited_resource": limited_resource_summary,
        "use_fa3": bool(args.use_fa3),
        "sdpa_backend": sdpa_summary,
        "decoder_sdpa_patch": bool(decoder_sdpa_patch),
        "frames": written,
    }
    (output_dir / "sam3_31_video_wrapper_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"wrote {len(written)} frame outputs")


def log(message: str) -> None:
    print(f"[sam3_31_video_wrapper] {message}", file=sys.stderr, flush=True)


def select_sdpa_backend(requested: str, capability: tuple[int, int] | None) -> str:
    normalized = requested.replace("_", "-")
    if normalized != "auto":
        return normalized
    if capability is not None and capability[0] < 8:
        return "math"
    return "default"


def configure_torch_sdpa_backend(requested: str) -> dict[str, Any]:
    try:
        import torch  # noqa: WPS433
    except ImportError:
        return {"requested": requested, "selected": "unavailable", "torch_available": False}

    capability = None
    device_name = None
    cuda_available = bool(torch.cuda.is_available()) if hasattr(torch, "cuda") else False
    if cuda_available:
        try:
            capability = tuple(int(value) for value in torch.cuda.get_device_capability())
            device_name = torch.cuda.get_device_name()
        except Exception as exc:  # pragma: no cover - defensive around CUDA driver state
            device_name = f"cuda query failed: {exc}"

    selected = select_sdpa_backend(requested, capability)
    if selected == "math":
        set_torch_sdp_flags(torch, math=True, flash=False, mem_efficient=False, cudnn=False)
    elif selected == "flash":
        set_torch_sdp_flags(torch, math=False, flash=True, mem_efficient=False, cudnn=False)
    elif selected in {"mem-efficient", "mem_efficient"}:
        set_torch_sdp_flags(torch, math=False, flash=False, mem_efficient=True, cudnn=False)

    return {
        "requested": requested,
        "selected": selected,
        "torch_available": True,
        "cuda_available": cuda_available,
        "cuda_device": device_name,
        "cuda_capability": list(capability) if capability is not None else None,
    }


def set_torch_sdp_flags(
    torch_module: Any,
    *,
    math: bool,
    flash: bool,
    mem_efficient: bool,
    cudnn: bool,
) -> None:
    cuda_backend = getattr(getattr(torch_module, "backends", None), "cuda", None)
    if cuda_backend is None:
        return
    setters = {
        "enable_math_sdp": math,
        "enable_flash_sdp": flash,
        "enable_mem_efficient_sdp": mem_efficient,
        "enable_cudnn_sdp": cudnn,
    }
    for name, enabled in setters.items():
        setter = getattr(cuda_backend, name, None)
        if setter is not None:
            setter(bool(enabled))


def patch_sam3_decoder_sdpa_kernel(selected: str) -> bool:
    if selected != "math":
        return False
    try:
        import sam3.model.decoder as decoder_module  # noqa: WPS433
    except ImportError:
        return False
    return patch_decoder_sdpa_kernel(decoder_module)


def patch_decoder_sdpa_kernel(decoder_module: Any) -> bool:
    if bool(getattr(decoder_module, "_noise_segmentation_math_sdpa_patch", False)):
        return True
    original_sdpa_kernel = getattr(decoder_module, "sdpa_kernel", None)
    backend = getattr(decoder_module, "SDPBackend", None)
    if original_sdpa_kernel is None or backend is None or not hasattr(backend, "MATH"):
        return False

    def math_sdpa_kernel(*args: Any, **kwargs: Any):
        del args, kwargs
        return original_sdpa_kernel(backend.MATH)

    decoder_module.sdpa_kernel = math_sdpa_kernel
    decoder_module._noise_segmentation_math_sdpa_patch = True
    return True


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


def build_local_or_default_predictor(
    model_builder_module: Any,
    *,
    sam3_model_path: str | None,
    sam3_root: str | None,
    use_fa3: bool,
) -> Any:
    builder = model_builder_module.build_sam3_multiplex_video_predictor
    sam3_model_path = sam3_model_path or infer_local_model_path(sam3_root)
    if sam3_model_path is None:
        return call_builder(builder, {"use_fa3": use_fa3})

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

    if "use_fa3" in params:
        kwargs["use_fa3"] = bool(use_fa3)
    return builder(**kwargs)


def call_builder(builder: Any, kwargs: dict[str, Any]) -> Any:
    params = inspect.signature(builder).parameters
    filtered = {key: value for key, value in kwargs.items() if key in params}
    return builder(**filtered)


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
        masks = first_present(outputs, MASK_KEYS)
        scores = first_present(outputs, SCORE_KEYS)
        track_ids = first_present(outputs, TRACK_ID_KEYS)
        boxes = first_present(outputs, BOX_KEYS)
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


def has_mask_payload(outputs: Any) -> bool:
    if isinstance(outputs, dict):
        return first_present(outputs, MASK_KEYS) is not None
    if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        return outputs[1] is not None
    return False


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


def prepare_limited_video_resource(
    *,
    frames_resource: str | Path,
    requested_frames: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    resource_dir = output_dir / "_sam3_limited_frames"
    if resource_dir.exists():
        shutil.rmtree(resource_dir)
    resource_dir.mkdir(parents=True, exist_ok=True)

    resource_path = Path(frames_resource)
    use_image_paths = all(frame.get("image_path") and Path(str(frame["image_path"])).is_file() for frame in requested_frames)
    if use_image_paths:
        source = "image_path"
        write_limited_frames_from_image_paths(resource_dir, requested_frames)
    elif resource_path.is_dir():
        source = "frames_dir"
        write_limited_frames_from_dir(resource_dir, frames_dir=resource_path, requested_frames=requested_frames)
    else:
        source = "video"
        write_limited_frames_from_video(resource_dir, video=resource_path, requested_frames=requested_frames)

    session_frames = []
    for local_index, frame in enumerate(requested_frames):
        session_frame = dict(frame)
        session_frame["original_video_frame_index"] = int(frame["video_frame_index"])
        session_frame["video_frame_index"] = local_index
        session_frames.append(session_frame)

    summary = {
        "resource_path": str(resource_dir),
        "source": source,
        "frames": len(session_frames),
        "original_video_frame_indices": [int(frame["video_frame_index"]) for frame in requested_frames],
    }
    return str(resource_dir), session_frames, summary


def write_limited_frames_from_image_paths(resource_dir: Path, requested_frames: list[dict[str, Any]]) -> None:
    for local_index, frame in enumerate(requested_frames):
        src = Path(str(frame["image_path"]))
        dst = resource_dir / f"{local_index:05d}.jpg"
        if src.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copy2(src, dst)
            continue

        from PIL import Image  # noqa: WPS433

        with Image.open(src) as image:
            image.convert("RGB").save(dst, quality=95)


def natural_sort_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def write_limited_frames_from_dir(
    resource_dir: Path,
    *,
    frames_dir: Path,
    requested_frames: list[dict[str, Any]],
) -> None:
    from PIL import Image  # noqa: WPS433

    image_files = sorted(
        [path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS],
        key=natural_sort_key,
    )
    if not image_files:
        raise ValueError(f"No valid image files found in frames directory: {frames_dir}")

    for local_index, frame in enumerate(requested_frames):
        source_index = int(frame["video_frame_index"])
        if source_index < 0 or source_index >= len(image_files):
            raise IndexError(
                f"Requested frame index {source_index} is out of range for {frames_dir}; "
                f"found {len(image_files)} images"
            )
        src = image_files[source_index]
        dst = resource_dir / f"{local_index:05d}.jpg"
        if src.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copy2(src, dst)
            continue
        with Image.open(src) as image:
            image.convert("RGB").save(dst, quality=95)


def write_limited_frames_from_video(
    resource_dir: Path,
    *,
    video: str | Path,
    requested_frames: list[dict[str, Any]],
) -> None:
    import cv2  # noqa: WPS433

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video resource: {video}")
    try:
        for local_index, frame in enumerate(requested_frames):
            source_index = int(frame["video_frame_index"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Could not read video frame {source_index} from {video}")
            dst = resource_dir / f"{local_index:05d}.jpg"
            if not cv2.imwrite(str(dst), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                raise ValueError(f"Could not write limited SAM3 frame: {dst}")
    finally:
        capture.release()


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
        row = {"frame_id": str(frame["frame_id"]), "video_frame_index": int(frame["video_frame_index"])}
        if frame.get("image_path") is not None:
            row["image_path"] = str(frame["image_path"])
        out.append(row)
    return out


def read_resource_shape(resource_path: str | Path) -> tuple[int, int]:
    path = Path(resource_path)
    if path.is_dir():
        frames = sorted(
            [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in VALID_IMAGE_EXTS],
            key=natural_sort_key,
        )
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
        if key in mapping and mapping[key] is not None:
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
    parser.add_argument(
        "--video",
        "--frames-dir",
        "--images-dir",
        "--img-dir",
        dest="frames_resource",
        required=True,
        help="MP4 video path or directory with numbered image frames.",
    )
    parser.add_argument("--prompts", required=True, help="YAML prompt config keyed by target class name.")
    parser.add_argument("--frames-json", required=True, help="JSON with requested frame_id/video_frame_index pairs.")
    parser.add_argument("--output-dir", required=True, help="Directory where <frame_id>.npz outputs are written.")
    parser.add_argument("--sam3-root", default=None, help="Optional path to the SAM3 repository.")
    parser.add_argument("--sam3-model-path", default=None, help="Optional local facebook/sam3.1 HuggingFace model directory.")
    parser.add_argument("--prompt-frame-index", type=int, default=0)
    parser.add_argument(
        "--use-fa3",
        action="store_true",
        help="Enable FlashAttention 3 in SAM3.1. Disabled by default for wider GPU/dtype compatibility.",
    )
    parser.add_argument(
        "--sdpa-backend",
        choices=("auto", "default", "math", "flash", "mem-efficient"),
        default="auto",
        help=(
            "PyTorch scaled-dot-product attention backend. "
            "auto selects math on pre-Ampere GPUs such as sm75 to avoid 'No available kernel'."
        ),
    )
    parser.add_argument(
        "--use-original-video-resource",
        action="store_true",
        help="Pass the original video directly to SAM3 instead of building a small requested-frame JPEG folder.",
    )
    parser.add_argument(
        "--max-video-frame-index",
        type=int,
        default=None,
        help="Stop SAM3 propagation after this video frame index.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
