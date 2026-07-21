from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


IGNORE_FLOAT = np.nan

HL320_FEATURE_NAMES = [
    "intensity_norm",
    "reflectivity_norm",
    "distance",
    "azimuth_deg",
    "vertical_deg",
    "slot",
    "pixel",
    "block_id",
    "return_count",
    "echo_rank_by_distance",
    "nearest_echo_distance_delta",
    "strongest_echo_intensity_delta",
    "has_camera_projection",
]


@dataclass(frozen=True)
class HL320Frame:
    frame_id: str
    path: Path
    points: np.ndarray
    columns: list[str]
    fields: dict[str, np.ndarray]

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class EchoGroups:
    row_indices: np.ndarray
    echo_ids: np.ndarray
    group_keys: np.ndarray
    return_count_by_row: np.ndarray
    echo_rank_by_distance: np.ndarray
    nearest_echo_distance_delta: np.ndarray
    strongest_echo_intensity_delta: np.ndarray


def load_hl320_csv(path: str | Path) -> HL320Frame:
    source = Path(path)
    rows, columns = _read_table(source)
    if not rows:
        raise ValueError(f"HL320 CSV is empty: {source}")
    column_map = {name.strip().casefold(): index for index, name in enumerate(columns)}
    required = {name: _require_column(column_map, name, source) for name in ("x", "y", "z", "intensity")}
    optional_names = ("reflectivity", "distance", "azimuth", "vertical", "slot", "pixel", "blockid", "block_id", "cxd", "cyd")
    optional = {name: column_map[name] for name in optional_names if name in column_map}
    max_column = max([*required.values(), *optional.values()])

    points = np.zeros((len(rows), 4), dtype=np.float32)
    fields: dict[str, np.ndarray] = {}
    for name in optional:
        canonical = "block_id" if name in {"blockid", "block_id"} else name
        fields[canonical] = np.full((len(rows),), IGNORE_FLOAT, dtype=np.float32)

    for row_index, row in enumerate(rows):
        if len(row) <= max_column:
            raise ValueError(f"{source}:{row_index + 2}: expected at least {max_column + 1} columns, got {len(row)}")
        points[row_index, 0] = _parse_float(row[required["x"]], source, row_index + 2, "x")
        points[row_index, 1] = _parse_float(row[required["y"]], source, row_index + 2, "y")
        points[row_index, 2] = _parse_float(row[required["z"]], source, row_index + 2, "z")
        points[row_index, 3] = _parse_float(row[required["intensity"]], source, row_index + 2, "intensity")
        for name, column in optional.items():
            canonical = "block_id" if name in {"blockid", "block_id"} else name
            fields[canonical][row_index] = _parse_float(row[column], source, row_index + 2, canonical)

    if "distance" not in fields:
        fields["distance"] = np.linalg.norm(points[:, :3], axis=1).astype(np.float32)
    if "reflectivity" not in fields:
        fields["reflectivity"] = points[:, 3].copy()
    return HL320Frame(frame_id=source.stem, path=source, points=points, columns=columns, fields=fields)


def build_hl320_features(frame: HL320Frame) -> np.ndarray:
    groups = group_echo_returns(frame)
    point_count = frame.point_count
    block_id = _field(frame, "block_id", fill=-1.0)
    cxd = _field(frame, "cxd")
    cyd = _field(frame, "cyd")
    has_projection = np.isfinite(cxd) & np.isfinite(cyd)
    features = np.stack(
        [
            _normalize_u8(frame.points[:, 3]),
            _normalize_u8(_field(frame, "reflectivity")),
            _field(frame, "distance"),
            _field(frame, "azimuth"),
            _field(frame, "vertical"),
            _field(frame, "slot"),
            _field(frame, "pixel"),
            block_id,
            groups.return_count_by_row.astype(np.float32, copy=False),
            groups.echo_rank_by_distance.astype(np.float32, copy=False),
            groups.nearest_echo_distance_delta.astype(np.float32, copy=False),
            groups.strongest_echo_intensity_delta.astype(np.float32, copy=False),
            has_projection.astype(np.float32, copy=False),
        ],
        axis=1,
    )
    if features.shape != (point_count, len(HL320_FEATURE_NAMES)):
        raise AssertionError(f"Unexpected HL320 feature shape: {features.shape}")
    features[~np.isfinite(features)] = 0.0
    return features.astype(np.float32, copy=False)


def group_echo_returns(frame: HL320Frame) -> EchoGroups:
    point_count = frame.point_count
    slot = _field(frame, "slot")
    pixel = _field(frame, "pixel")
    block_id = _field(frame, "block_id", fill=-1.0)
    distance = _field(frame, "distance")
    intensity = frame.points[:, 3]

    if not (np.all(np.isfinite(slot)) and np.all(np.isfinite(pixel))):
        row_indices = np.arange(point_count, dtype=np.int64).reshape(point_count, 1)
        return EchoGroups(
            row_indices=row_indices,
            echo_ids=np.asarray([-1], dtype=np.int32),
            group_keys=np.arange(point_count, dtype=np.int64).reshape(point_count, 1),
            return_count_by_row=np.ones(point_count, dtype=np.int32),
            echo_rank_by_distance=np.zeros(point_count, dtype=np.int32),
            nearest_echo_distance_delta=np.zeros(point_count, dtype=np.float32),
            strongest_echo_intensity_delta=np.zeros(point_count, dtype=np.float32),
        )

    key_to_rows: dict[tuple[int, int], list[int]] = {}
    for row_index, key in enumerate(zip(slot.astype(np.int64), pixel.astype(np.int64), strict=False)):
        key_to_rows.setdefault((int(key[0]), int(key[1])), []).append(row_index)

    echo_ids = sorted({int(value) for value in block_id[np.isfinite(block_id)]})
    if not echo_ids:
        echo_ids = [-1]
    echo_to_col = {echo_id: index for index, echo_id in enumerate(echo_ids)}
    group_keys = np.asarray(list(key_to_rows), dtype=np.int64)
    row_indices = np.full((len(key_to_rows), len(echo_ids)), -1, dtype=np.int64)
    return_count = np.ones(point_count, dtype=np.int32)
    echo_rank = np.zeros(point_count, dtype=np.int32)
    nearest_delta = np.zeros(point_count, dtype=np.float32)
    strongest_delta = np.zeros(point_count, dtype=np.float32)

    for group_index, rows in enumerate(key_to_rows.values()):
        return_count[rows] = len(rows)
        sorted_by_distance = sorted(rows, key=lambda row: float(distance[row]))
        sorted_by_intensity = sorted(rows, key=lambda row: float(intensity[row]), reverse=True)
        strongest_intensity = float(intensity[sorted_by_intensity[0]])
        for rank, row in enumerate(sorted_by_distance):
            echo_rank[row] = rank
            current_block = int(block_id[row]) if np.isfinite(block_id[row]) else -1
            row_indices[group_index, echo_to_col[current_block]] = row
            if len(sorted_by_distance) > 1:
                nearest_delta[row] = min(
                    abs(float(distance[row]) - float(distance[other])) for other in sorted_by_distance if other != row
                )
            strongest_delta[row] = float(intensity[row]) - strongest_intensity

    return EchoGroups(
        row_indices=row_indices,
        echo_ids=np.asarray(echo_ids, dtype=np.int32),
        group_keys=group_keys,
        return_count_by_row=return_count,
        echo_rank_by_distance=echo_rank,
        nearest_echo_distance_delta=nearest_delta,
        strongest_echo_intensity_delta=strongest_delta,
    )


def _read_table(path: Path) -> tuple[list[list[str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first = ""
        for raw_line in handle:
            if raw_line.strip():
                first = raw_line.strip()
                break
        if not first:
            return [], []
        delimiter = _detect_delimiter(first)
        if delimiter is None:
            columns = _split_whitespace(first)
            rows = [_split_whitespace(line.strip()) for line in handle if line.strip()]
        else:
            reader = csv.reader(handle, delimiter=delimiter)
            columns = [part.strip() for part in next(csv.reader([first], delimiter=delimiter))]
            rows = [[part.strip() for part in row] for row in reader if any(part.strip() for part in row)]
    return rows, columns


def _detect_delimiter(header: str) -> str | None:
    if "\t" in header:
        return "\t"
    if "," in header:
        return ","
    return None


def _split_whitespace(value: str) -> list[str]:
    return [part for part in re.split(r"\s+", value.strip()) if part]


def _require_column(column_map: dict[str, int], name: str, path: Path) -> int:
    try:
        return column_map[name]
    except KeyError as exc:
        raise ValueError(f"HL320 CSV {path} has no required column {name!r}") from exc


def _parse_float(value: str, path: Path, line_number: int, column: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: column {column!r} is not numeric: {value!r}") from exc


def _field(frame: HL320Frame, name: str, *, fill: float = 0.0) -> np.ndarray:
    values = frame.fields.get(name)
    if values is None:
        return np.full((frame.point_count,), fill, dtype=np.float32)
    return values.astype(np.float32, copy=False)


def _normalize_u8(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    if out.size and np.nanmax(out) > 1.0:
        out /= 255.0
    return out
