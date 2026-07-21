from __future__ import annotations

from pathlib import Path

import numpy as np

from .csv_points import HL320Frame


DEFAULT_RAW_FLOAT_FIELDS = [
    "x",
    "y",
    "z",
    "distance",
    "azimuth",
    "vertical",
    "reflectivity",
    "intensity",
    "slot",
    "pixel",
    "block_id",
    "cxd",
    "cyd",
]


def load_hl320_raw_float_records(path: str | Path, *, fields: list[str] | None = None) -> HL320Frame:
    """Read a simple flat float32 HL320 dump.

    This intentionally handles only the explicit record layout passed by the caller. Vendor packet
    binaries should be converted with a packet-specific decoder before training data is built.
    """
    source = Path(path)
    names = fields or DEFAULT_RAW_FLOAT_FIELDS
    if len({"x", "y", "z", "intensity"} - set(name.lower() for name in names)) > 0:
        raise ValueError("Raw HL320 field schema must include x, y, z, and intensity")
    raw = np.fromfile(source, dtype=np.float32)
    if raw.size % len(names) != 0:
        raise ValueError(f"Raw HL320 file {source} has {raw.size} floats, not divisible by {len(names)} fields")
    table = raw.reshape(-1, len(names))
    index = {name.lower(): offset for offset, name in enumerate(names)}
    points = np.stack(
        [
            table[:, index["x"]],
            table[:, index["y"]],
            table[:, index["z"]],
            table[:, index["intensity"]],
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    fields_out = {
        _canonical(name): table[:, offset].astype(np.float32, copy=True)
        for offset, name in enumerate(names)
        if _canonical(name) not in {"x", "y", "z", "intensity"}
    }
    return HL320Frame(frame_id=source.stem, path=source, points=points, columns=list(names), fields=fields_out)


def _canonical(name: str) -> str:
    normalized = name.strip().lower()
    if normalized in {"blockid", "block_id"}:
        return "block_id"
    return normalized
