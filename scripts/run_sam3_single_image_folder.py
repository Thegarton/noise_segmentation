#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolabeler.data.class_config import load_semantic_classes  # noqa: E402
from autolabeler.teachers.sam3_text_adapter import load_prompt_config  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_KEYS = ("masks", "pred_masks", "out_binary_masks", "video_res_masks", "mask_logits", "out_mask_logits")
SCORE_KEYS = ("scores", "pred_scores", "object_scores", "ious", "obj_scores", "out_probs")
BOX_KEYS = ("boxes_xyxy", "boxes", "pred_boxes", "out_boxes_xywh")


@dataclass(frozen=True)
class Sam3Instance:
    label: str
    class_id: int
    prompt: str
    score: float
    mask: np.ndarray
    box: np.ndarray | None = None


@dataclass(frozen=True)
class ImageResult:
    image_path: Path
    output_dir: Path
    image_size: tuple[int, int]
    instances: int
    class_pixel_counts: dict[str, int]


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(image_dir, recursive=args.recursive)
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError(f"--max-images must be positive, got {args.max_images}")
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    prompts_by_label = load_prompt_config(args.prompt_config)
    class_to_id = load_semantic_classes(args.classes_yaml)
    label_to_id = {label: int(class_to_id[label]) for label in prompts_by_label if label in class_to_id}
    flat_prompts = [(label, prompt) for label, prompts in prompts_by_label.items() for prompt in prompts if label in label_to_id]
    skipped_prompt_labels = sorted(label for label in prompts_by_label if label not in label_to_id)
    if args.max_prompts is not None:
        if args.max_prompts <= 0:
            raise ValueError(f"--max-prompts must be positive, got {args.max_prompts}")
        flat_prompts = flat_prompts[: args.max_prompts]
    if not flat_prompts:
        raise ValueError("Prompt config produced no prompts whose labels exist in classes yaml")

    sam3_root = Path(args.sam3_root).expanduser().resolve() if args.sam3_root else None
    model_dir = resolve_model_dir(sam3_root=sam3_root, sam3_model_path=args.sam3_model_path)
    configure_sam3_imports(sam3_root=sam3_root, model_dir=model_dir)
    predictor = build_predictor(model_dir=model_dir, use_fa3=args.use_fa3)

    results = []
    for index, image_path in enumerate(image_paths, start=1):
        frame_out = output_dir_for_image(out_dir, image_dir, image_path, recursive=args.recursive)
        if outputs_exist(frame_out) and not args.overwrite:
            metadata_path = frame_out / "metadata.json"
            results.append(
                {
                    "image": str(image_path),
                    "frame_id": frame_out.name,
                    "status": "exists",
                    "output_dir": str(frame_out),
                    "metadata": str(metadata_path),
                }
            )
            continue

        print(f"[{index:04d}/{len(image_paths):04d}] {image_path}", file=sys.stderr, flush=True)
        result = process_image(
            predictor=predictor,
            image_path=image_path,
            output_dir=frame_out,
            flat_prompts=flat_prompts,
            label_to_id=label_to_id,
            min_score=args.min_score,
            prompt_log=args.prompt_log,
        )
        metadata = {
            "version": 1,
            "image_path": str(result.image_path),
            "image_size": [result.image_size[0], result.image_size[1]],
            "sam3_root": str(sam3_root) if sam3_root is not None else None,
            "sam3_model_path": str(model_dir) if model_dir is not None else None,
            "sam3_checkpoint": str(model_dir / "sam3.1_multiplex.pt") if model_dir is not None else None,
            "prompt_config": str(Path(args.prompt_config)),
            "classes_yaml": str(Path(args.classes_yaml)),
            "min_score": float(args.min_score),
            "instances": result.instances,
            "class_pixel_counts": result.class_pixel_counts,
        }
        (frame_out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.validate:
            validate_outputs(frame_out)
        results.append(
            {
                "image": str(image_path),
                "frame_id": frame_out.name,
                "status": "created",
                "output_dir": str(frame_out),
                "metadata": str(frame_out / "metadata.json"),
                "instances": result.instances,
            }
        )

    manifest = {
        "version": 1,
        "image_dir": str(image_dir),
        "out_dir": str(out_dir),
        "prompt_config": str(Path(args.prompt_config)),
        "classes_yaml": str(Path(args.classes_yaml)),
        "sam3_root": str(sam3_root) if sam3_root is not None else None,
        "sam3_model_path": str(model_dir) if model_dir is not None else None,
        "min_score": float(args.min_score),
        "images": len(image_paths),
        "prompts": len(flat_prompts),
        "labels": label_to_id,
        "skipped_prompt_labels": skipped_prompt_labels,
        "frames": results,
    }
    manifest_path = out_dir / "sam3_single_image_folder_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(image_paths), "manifest": str(manifest_path)}, indent=2))


def process_image(
    *,
    predictor: Any,
    image_path: Path,
    output_dir: Path,
    flat_prompts: list[tuple[str, str]],
    label_to_id: dict[str, int],
    min_score: float,
    prompt_log: bool,
) -> ImageResult:
    from PIL import Image  # noqa: WPS433

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    session_id = start_session(predictor, image_path)
    instances: list[Sam3Instance] = []
    try:
        for prompt_idx, (label, prompt) in enumerate(flat_prompts, start=1):
            if prompt_log:
                print(f"  [{prompt_idx:03d}/{len(flat_prompts):03d}] {label}: {prompt}", file=sys.stderr, flush=True)
            reset_session(predictor, session_id)
            response = add_text_prompt(
                predictor,
                session_id=session_id,
                prompt=prompt,
                min_score=min_score,
            )
            masks, scores, boxes = extract_arrays(response.get("outputs"), height=height, width=width)
            for mask_index in range(masks.shape[0]):
                score = float(scores[mask_index])
                if score < min_score:
                    continue
                instances.append(
                    Sam3Instance(
                        label=label,
                        class_id=int(label_to_id[label]),
                        prompt=prompt,
                        score=score,
                        mask=masks[mask_index].astype(bool, copy=False),
                        box=None if boxes is None else boxes[mask_index],
                    )
                )
    finally:
        close_session(predictor, session_id)

    semantic_mask, confidence, class_pixel_counts = build_semantic_outputs(
        instances=instances,
        label_to_id=label_to_id,
        shape=(height, width),
    )
    save_image_outputs(
        image=image,
        image_path=image_path,
        output_dir=output_dir,
        instances=instances,
        semantic_mask=semantic_mask,
        confidence=confidence,
    )
    return ImageResult(
        image_path=image_path,
        output_dir=output_dir,
        image_size=(width, height),
        instances=len(instances),
        class_pixel_counts=class_pixel_counts,
    )


def start_session(predictor: Any, image_path: Path) -> str:
    response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(image_path),
        }
    )
    return str(response["session_id"])


def reset_session(predictor: Any, session_id: str) -> None:
    predictor.handle_request(request={"type": "reset_session", "session_id": session_id})


def add_text_prompt(predictor: Any, *, session_id: str, prompt: str, min_score: float) -> dict[str, Any]:
    return predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
            "output_prob_thresh": float(min_score),
        }
    )


def close_session(predictor: Any, session_id: str) -> None:
    predictor.handle_request(request={"type": "close_session", "session_id": session_id})


def build_semantic_outputs(
    *,
    instances: list[Sam3Instance],
    label_to_id: dict[str, int],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    height, width = shape
    semantic_mask = np.zeros((height, width), dtype=np.uint16)
    confidence = np.zeros((height, width), dtype=np.float32)
    for item in sorted(instances, key=lambda x: x.score):
        accept = item.mask & (item.score >= confidence)
        semantic_mask[accept] = np.uint16(item.class_id)
        confidence[accept] = np.float32(item.score)

    class_pixel_counts = {
        label: int(np.count_nonzero(semantic_mask == class_id))
        for label, class_id in label_to_id.items()
        if int(np.count_nonzero(semantic_mask == class_id)) > 0
    }
    return semantic_mask, confidence, class_pixel_counts


def save_image_outputs(
    *,
    image: Any,
    image_path: Path,
    output_dir: Path,
    instances: list[Sam3Instance],
    semantic_mask: np.ndarray,
    confidence: np.ndarray,
) -> None:
    from PIL import Image  # noqa: WPS433

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "semantic_mask.npy", semantic_mask)
    np.save(output_dir / "confidence.npy", confidence)
    save_instances_npz(output_dir / "instances.npz", instances, shape=semantic_mask.shape)

    image_np = np.asarray(image, dtype=np.uint8)
    overlay = make_overlay(image_np, semantic_mask)
    semantic_color = make_semantic_color(semantic_mask)
    Image.fromarray(overlay).save(output_dir / "overlay.jpg", quality=95)
    Image.fromarray(semantic_color).save(output_dir / "semantic_color.png")
    Image.fromarray(make_preview(image_np, semantic_color, overlay)).save(output_dir / "preview.jpg", quality=95)
    image.save(output_dir / "image.jpg", quality=95)

    instances_json = [
        {
            "label": item.label,
            "class_id": item.class_id,
            "prompt": item.prompt,
            "score": item.score,
            "box": None if item.box is None else np.asarray(item.box, dtype=np.float32).tolist(),
        }
        for item in instances
    ]
    (output_dir / "instances.json").write_text(json.dumps(instances_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "source_image.txt").write_text(str(image_path), encoding="utf-8")


def save_instances_npz(path: Path, instances: list[Sam3Instance], *, shape: tuple[int, int]) -> None:
    height, width = shape
    if instances:
        payload: dict[str, np.ndarray] = {
            "masks": np.stack([item.mask for item in instances], axis=0).astype(bool, copy=False),
            "scores": np.asarray([item.score for item in instances], dtype=np.float32),
            "labels": np.asarray([item.label for item in instances]),
            "class_ids": np.asarray([item.class_id for item in instances], dtype=np.uint16),
            "prompts": np.asarray([item.prompt for item in instances]),
        }
        if all(item.box is not None for item in instances):
            payload["boxes"] = np.stack([np.asarray(item.box, dtype=np.float32) for item in instances], axis=0)
    else:
        payload = {
            "masks": np.zeros((0, height, width), dtype=bool),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.asarray([], dtype=str),
            "class_ids": np.zeros((0,), dtype=np.uint16),
            "prompts": np.asarray([], dtype=str),
        }
    np.savez_compressed(path, **payload)


def make_overlay(image_np: np.ndarray, semantic_mask: np.ndarray, *, alpha: float = 0.45) -> np.ndarray:
    overlay = image_np.copy()
    for class_id in sorted(int(x) for x in np.unique(semantic_mask) if int(x) != 0):
        color = color_for_class(class_id)
        mask = semantic_mask == class_id
        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha
        ).astype(np.uint8)
    return overlay


def make_semantic_color(semantic_mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*semantic_mask.shape, 3), dtype=np.uint8)
    for class_id in sorted(int(x) for x in np.unique(semantic_mask) if int(x) != 0):
        color[semantic_mask == class_id] = color_for_class(class_id)
    return color


def make_preview(image_np: np.ndarray, semantic_color: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    separator = np.full((image_np.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([image_np, separator, semantic_color, separator, overlay], axis=1)


def color_for_class(class_id: int) -> np.ndarray:
    rng = np.random.default_rng(class_id * 1009 + 17)
    return rng.integers(40, 240, size=3, dtype=np.uint8)


def extract_arrays(outputs: Any, *, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if outputs is None:
        return empty_arrays(height, width)
    if isinstance(outputs, dict):
        masks_value = first_present(outputs, MASK_KEYS)
        scores_value = first_present(outputs, SCORE_KEYS)
        boxes_value = first_present(outputs, BOX_KEYS)
    elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        masks_value = outputs[1]
        scores_value = outputs[2] if len(outputs) >= 3 else None
        boxes_value = outputs[3] if len(outputs) >= 4 else None
    else:
        return empty_arrays(height, width)

    masks = normalize_masks(masks_value, height=height, width=width)
    scores = normalize_scores(scores_value, masks.shape[0])
    boxes = normalize_boxes(boxes_value, masks.shape[0])
    return masks, scores, boxes


def empty_arrays(height: int, width: int) -> tuple[np.ndarray, np.ndarray, None]:
    return np.zeros((0, height, width), dtype=bool), np.zeros((0,), dtype=np.float32), None


def normalize_masks(value: Any, *, height: int, width: int) -> np.ndarray:
    arr = to_numpy(value)
    if arr is None:
        return np.zeros((0, height, width), dtype=bool)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected masks [N,H,W], got {arr.shape}")
    return arr if arr.dtype == np.bool_ else arr > 0


def normalize_scores(value: Any, count: int) -> np.ndarray:
    arr = to_numpy(value)
    if arr is None:
        return np.ones((count,), dtype=np.float32)
    arr = arr.reshape(-1).astype(np.float32, copy=False)
    if arr.shape != (count,):
        return np.ones((count,), dtype=np.float32)
    return arr


def normalize_boxes(value: Any, count: int) -> np.ndarray | None:
    arr = to_numpy(value)
    if arr is None:
        return None
    arr = arr.astype(np.float32, copy=False)
    if arr.shape == (count, 4):
        return arr
    return None


def first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def collect_images(image_dir: Path, *, recursive: bool) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def output_dir_for_image(out_dir: Path, image_dir: Path, image_path: Path, *, recursive: bool) -> Path:
    if not recursive:
        return out_dir / image_path.stem
    rel = image_path.relative_to(image_dir)
    return out_dir / rel.with_suffix("")


def outputs_exist(frame_out: Path) -> bool:
    return (
        (frame_out / "semantic_mask.npy").is_file()
        and (frame_out / "confidence.npy").is_file()
        and (frame_out / "instances.npz").is_file()
        and (frame_out / "overlay.jpg").is_file()
        and (frame_out / "metadata.json").is_file()
    )


def validate_outputs(frame_out: Path) -> None:
    semantic = np.load(frame_out / "semantic_mask.npy")
    confidence = np.load(frame_out / "confidence.npy")
    if semantic.shape != confidence.shape:
        raise ValueError(f"semantic/confidence shape mismatch in {frame_out}: {semantic.shape} vs {confidence.shape}")
    with np.load(frame_out / "instances.npz", allow_pickle=False) as data:
        masks = np.asarray(data["masks"])
        scores = np.asarray(data["scores"])
        labels = np.asarray(data["labels"])
        class_ids = np.asarray(data["class_ids"])
        prompts = np.asarray(data["prompts"])
    if masks.ndim != 3 or masks.dtype != np.bool_:
        raise ValueError(f"instances masks must be bool[N,H,W], got {masks.shape} {masks.dtype}")
    if scores.shape != (masks.shape[0],):
        raise ValueError(f"instances scores mismatch in {frame_out}")
    if labels.shape != (masks.shape[0],) or class_ids.shape != (masks.shape[0],) or prompts.shape != (masks.shape[0],):
        raise ValueError(f"instances metadata mismatch in {frame_out}")


def configure_sam3_imports(*, sam3_root: Path | None, model_dir: Path | None) -> None:
    if sam3_root is not None:
        sys.path.insert(0, str(sam3_root))
    if model_dir is not None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", str(model_dir))


def resolve_model_dir(*, sam3_root: Path | None, sam3_model_path: str | None) -> Path | None:
    if sam3_model_path:
        model_dir = Path(sam3_model_path).expanduser().resolve()
    elif sam3_root is not None:
        model_dir = sam3_root / "sam3.1"
    else:
        return None
    if not (model_dir / "sam3.1_multiplex.pt").is_file():
        raise FileNotFoundError(f"Missing SAM3.1 checkpoint: {model_dir / 'sam3.1_multiplex.pt'}")
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing SAM3.1 config: {model_dir / 'config.json'}")
    return model_dir


def build_predictor(*, model_dir: Path | None, use_fa3: bool) -> Any:
    import sam3.model_builder as sam3_model_builder  # noqa: WPS433

    builder = sam3_model_builder.build_sam3_multiplex_video_predictor
    kwargs: dict[str, Any] = {
        "use_fa3": bool(use_fa3),
        "async_loading_frames": False,
    }
    if model_dir is not None:
        checkpoint_path = model_dir / "sam3.1_multiplex.pt"
        patch_hf_checkpoint_download(sam3_model_builder, checkpoint_path=checkpoint_path, model_dir=model_dir)
        kwargs["checkpoint_path"] = str(checkpoint_path)
    params = inspect.signature(builder).parameters
    return builder(**{key: value for key, value in kwargs.items() if key in params})


def patch_hf_checkpoint_download(model_builder_module: Any, *, checkpoint_path: Path, model_dir: Path) -> None:
    def _local_download_ckpt_from_hf(*args: Any, **kwargs: Any) -> str:
        return str(checkpoint_path)

    model_builder_module.download_ckpt_from_hf = _local_download_ckpt_from_hf
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(model_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SAM3.1 single-image prompt notebook flow over an image folder.")
    parser.add_argument("--image-dir", required=True, help="Directory with camera images.")
    parser.add_argument("--out-dir", required=True, help="Output directory. Each image gets its own subdirectory.")
    parser.add_argument("--prompt-config", default=str(REPO_ROOT / "configs" / "sam3_text_prompts_pointwise_v1.yaml"))
    parser.add_argument("--classes-yaml", default=str(REPO_ROOT / "configs" / "classes_pointwise_v1.yaml"))
    parser.add_argument("--sam3-root", default=None, help="Path to cloned SAM3 repo, e.g. /home/.../git_repo/sam3.")
    parser.add_argument("--sam3-model-path", default=None, help="Path to local facebook/sam3.1 model directory.")
    parser.add_argument("--min-score", type=float, default=0.70)
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--max-prompts", type=int, default=None, help="Optional smoke-test prompt limit per image.")
    parser.add_argument("--recursive", action="store_true", help="Read images recursively and mirror the relative output tree.")
    parser.add_argument("--use-fa3", action="store_true", help="Enable FlashAttention 3. Disabled by default.")
    parser.add_argument("--prompt-log", action="store_true", help="Print every prompt for every image.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
