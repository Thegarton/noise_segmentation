from __future__ import annotations

from pathlib import Path
import numpy as np

from .bin_loader import load_bin
from .csv_loader import load_csv
from .schemas import OrganizedLiDARFrame


def load_frame(path: str, frame_id: str, *, input_format: str = "auto", cache_bin_dir: str | None = None) -> OrganizedLiDARFrame:
    p = Path(path)
    fmt = input_format
    if fmt == "auto":
        fmt = p.suffix.lstrip(".").lower()

    if fmt == "bin":
        return load_bin(path, frame_id)
    if fmt == "csv":
        frame = load_csv(path, frame_id)
        if cache_bin_dir:
            out = Path(cache_bin_dir) / f"{frame_id}.bin"
            out.parent.mkdir(parents=True, exist_ok=True)
            frame.points_range.astype(np.float32).tofile(out)
            frame.meta["cached_bin_path"] = str(out)
        return frame

    raise ValueError(f"Unsupported input format: {input_format} for path {path}")
