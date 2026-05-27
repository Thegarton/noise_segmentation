from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Sam3TextResult:
    masks: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    prompts: np.ndarray
    boxes_xyxy: np.ndarray | None = None


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
                prompts[key] = [_strip_quotes(x.strip()) for x in value.strip("[]").split(",") if x.strip()]
            else:
                prompts[key] = []
            continue
        if indent >= 2 and stripped.startswith("- ") and current_key is not None:
            prompts[current_key].append(_strip_quotes(stripped[2:].strip()))
    return prompts


def write_prompt_json(path: str | Path, prompts: dict[str, list[str]]) -> None:
    Path(path).write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")


def run_sam3_text_teacher(
    *,
    conda_env: str,
    sam3_script: str | Path,
    image_path: str | Path,
    prompt_config: str | Path,
    output_npz: str | Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(sam3_script),
        "--image",
        str(image_path),
        "--prompts",
        str(prompt_config),
        "--output",
        str(output_npz),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, check=True)


def load_sam3_text_result(path: str | Path) -> Sam3TextResult:
    with np.load(path, allow_pickle=False) as data:
        masks = np.asarray(data["masks"])
        scores = np.asarray(data["scores"], dtype=np.float32)
        labels = np.asarray(data["labels"]).astype(str)
        prompts = np.asarray(data["prompts"]).astype(str)
        boxes = np.asarray(data["boxes_xyxy"], dtype=np.float32) if "boxes_xyxy" in data.files else None

    if masks.ndim != 3 or masks.dtype != np.bool_:
        raise ValueError(f"SAM3 masks must be bool[N,H,W], got {masks.shape} {masks.dtype}")
    if scores.shape != (masks.shape[0],):
        raise ValueError(f"SAM3 scores shape must be {(masks.shape[0],)}, got {scores.shape}")
    if labels.shape != (masks.shape[0],) or prompts.shape != (masks.shape[0],):
        raise ValueError("SAM3 labels/prompts must have one value per mask")
    if boxes is not None and boxes.shape != (masks.shape[0], 4):
        raise ValueError(f"SAM3 boxes_xyxy must have shape {(masks.shape[0], 4)}, got {boxes.shape}")
    return Sam3TextResult(masks=masks, scores=scores, labels=labels, prompts=prompts, boxes_xyxy=boxes)


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]
    return value

