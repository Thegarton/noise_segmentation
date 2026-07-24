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


def primary_returns_frame(frame: HL320Frame) -> HL320Frame:
    block_id = frame.fields.get("block_id")
    if block_id is None:
        return frame
    primary = np.isfinite(block_id) & (block_id.astype(np.int64) == 0)
    if not np.any(primary):
        raise ValueError(f"HL320 CSV {frame.path} has blockID column but no blockID == 0 rows")
    fields = {name: values[primary].copy() for name, values in frame.fields.items()}
    return HL320Frame(
        frame_id=frame.frame_id,
        path=frame.path,
        points=frame.points[primary].copy(),
        columns=frame.columns,
        fields=fields,
    )


def build_hl320_features(frame: HL320Frame, *, echo_frame: HL320Frame | None = None) -> np.ndarray:
    groups = group_echo_returns(frame)
    if echo_frame is not None:
        groups = project_echo_context_to_frame(frame, echo_frame)
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


def project_echo_context_to_frame(frame: HL320Frame, echo_frame: HL320Frame) -> EchoGroups:
    point_count = frame.point_count
    default_row_indices = np.arange(point_count, dtype=np.int64).reshape(point_count, 1)
    defaults = EchoGroups(
        row_indices=default_row_indices,
        echo_ids=np.asarray([-1], dtype=np.int32),
        group_keys=np.arange(point_count, dtype=np.int64).reshape(point_count, 1),
        return_count_by_row=np.ones(point_count, dtype=np.int32),
        echo_rank_by_distance=np.zeros(point_count, dtype=np.int32),
        nearest_echo_distance_delta=np.zeros(point_count, dtype=np.float32),
        strongest_echo_intensity_delta=np.zeros(point_count, dtype=np.float32),
    )

    slot = _field(frame, "slot")
    pixel = _field(frame, "pixel")
    if not (np.all(np.isfinite(slot)) and np.all(np.isfinite(pixel))):
        return defaults

    echo_slot = _field(echo_frame, "slot")
    echo_pixel = _field(echo_frame, "pixel")
    if not (np.all(np.isfinite(echo_slot)) and np.all(np.isfinite(echo_pixel))):
        return defaults

    valid_echo = valid_return_mask(echo_frame.points)
    echo_groups = group_echo_returns(echo_frame)
    echo_block_id = _field(echo_frame, "block_id", fill=-1.0)
    key_to_echo_rows: dict[tuple[int, int], list[int]] = {}
    for row_index, key in enumerate(zip(echo_slot.astype(np.int64), echo_pixel.astype(np.int64), strict=False)):
        if not valid_echo[row_index]:
            continue
        key_to_echo_rows.setdefault((int(key[0]), int(key[1])), []).append(row_index)

    block_id = _field(frame, "block_id", fill=0.0)
    return_count = defaults.return_count_by_row.copy()
    echo_rank = defaults.echo_rank_by_distance.copy()
    nearest_delta = defaults.nearest_echo_distance_delta.copy()
    strongest_delta = defaults.strongest_echo_intensity_delta.copy()
    row_indices = np.full((point_count, max(1, len(echo_groups.echo_ids))), -1, dtype=np.int64)

    for row_index, key in enumerate(zip(slot.astype(np.int64), pixel.astype(np.int64), strict=False)):
        echo_rows = key_to_echo_rows.get((int(key[0]), int(key[1])))
        if not echo_rows:
            continue
        desired_block = int(block_id[row_index]) if np.isfinite(block_id[row_index]) else 0
        chosen_echo_row = next(
            (echo_row for echo_row in echo_rows if int(echo_block_id[echo_row]) == desired_block),
            echo_rows[0],
        )
        return_count[row_index] = len(echo_rows)
        echo_rank[row_index] = echo_groups.echo_rank_by_distance[chosen_echo_row]
        nearest_delta[row_index] = echo_groups.nearest_echo_distance_delta[chosen_echo_row]
        strongest_delta[row_index] = echo_groups.strongest_echo_intensity_delta[chosen_echo_row]
        for column_index, echo_row in enumerate(echo_rows[: row_indices.shape[1]]):
            row_indices[row_index, column_index] = echo_row

    return EchoGroups(
        row_indices=row_indices,
        echo_ids=echo_groups.echo_ids,
        group_keys=np.stack([slot.astype(np.int64), pixel.astype(np.int64)], axis=1),
        return_count_by_row=return_count,
        echo_rank_by_distance=echo_rank,
        nearest_echo_distance_delta=nearest_delta,
        strongest_echo_intensity_delta=strongest_delta,
    )


def group_echo_returns(frame: HL320Frame) -> EchoGroups:
    point_count = frame.point_count
    slot = _field(frame, "slot")
    pixel = _field(frame, "pixel")
    block_id = _field(frame, "block_id", fill=-1.0)
    distance = _field(frame, "distance")
    intensity = frame.points[:, 3]
    valid_return = valid_return_mask(frame.points)

    if not (np.all(np.isfinite(slot)) and np.all(np.isfinite(pixel))):
        row_indices = np.arange(point_count, dtype=np.int64).reshape(point_count, 1)
        return EchoGroups(
            row_indices=row_indices,
            echo_ids=np.asarray([-1], dtype=np.int32),
            group_keys=np.arange(point_count, dtype=np.int64).reshape(point_count, 1),
            return_count_by_row=valid_return.astype(np.int32, copy=False),
            echo_rank_by_distance=np.zeros(point_count, dtype=np.int32),
            nearest_echo_distance_delta=np.zeros(point_count, dtype=np.float32),
            strongest_echo_intensity_delta=np.zeros(point_count, dtype=np.float32),
        )

    key_to_rows: dict[tuple[int, int], list[int]] = {}
    for row_index, key in enumerate(zip(slot.astype(np.int64), pixel.astype(np.int64), strict=False)):
        if not valid_return[row_index]:
            continue
        key_to_rows.setdefault((int(key[0]), int(key[1])), []).append(row_index)

    echo_ids = sorted({int(value) for value in block_id[valid_return & np.isfinite(block_id)]})
    if not echo_ids:
        echo_ids = [-1]
    echo_to_col = {echo_id: index for index, echo_id in enumerate(echo_ids)}
    group_keys = np.asarray(list(key_to_rows), dtype=np.int64)
    row_indices = np.full((len(key_to_rows), len(echo_ids)), -1, dtype=np.int64)
    return_count = np.zeros(point_count, dtype=np.int32)
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


def valid_return_mask(points: np.ndarray) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float32)[:, :3]
    finite = np.all(np.isfinite(xyz), axis=1)
    nonzero = np.linalg.norm(xyz, axis=1) > 1e-8
    return finite & nonzero


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
