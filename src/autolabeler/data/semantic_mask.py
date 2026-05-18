from __future__ import annotations

from pathlib import Path

import numpy as np

from .bin_loader import H, W


SEMANTIC_MASK_DTYPE = np.uint16


def load_semantic_mask(path: str) -> np.ndarray:
    mask = np.load(path)
    return validate_semantic_mask(mask)


def save_semantic_mask(path: str, mask: np.ndarray) -> None:
    valid = validate_semantic_mask(mask)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, valid.astype(SEMANTIC_MASK_DTYPE, copy=False))


def flatten_mask(mask: np.ndarray) -> np.ndarray:
    valid = validate_semantic_mask(mask)
    return valid.reshape(H * W)


def unflatten_mask(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.shape != (H * W,):
        raise ValueError(f"Semantic labels must have shape {(H * W,)}, got {arr.shape}")
    return arr.reshape(H, W).astype(SEMANTIC_MASK_DTYPE, copy=False)


def validate_semantic_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.shape != (H, W):
        raise ValueError(f"Semantic mask must have shape {(H, W)}, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Semantic mask dtype must be integer, got {arr.dtype}")
    return arr.astype(SEMANTIC_MASK_DTYPE, copy=False)
