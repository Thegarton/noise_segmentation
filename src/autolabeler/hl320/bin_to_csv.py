from __future__ import annotations

import csv
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HL320_BIN_CSV_COLUMNS = [
    "x",
    "y",
    "z",
    "azimuth",
    "vertical",
    "intensity",
    "slot",
    "pixel",
    "hcell",
    "vcell",
    "Cxd",
    "Cyd",
]

GZIP_HEADER_LEN = 16
BATCH_POINT_LEN = 52
ECHO_COUNT = 3
PIXELS_PER_SLOT = 128
SLOT_DATA_SIZE = BATCH_POINT_LEN * ECHO_COUNT * PIXELS_PER_SLOT


@dataclass(frozen=True)
class HL320BinFrameConversion:
    source_bin: Path
    output_csv: Path
    frame_id: str
    rows: int
    width: int
    height: int


@dataclass(frozen=True)
class HL320BinDirectoryConversion:
    bin_dir: Path
    output_dir: Path
    calibration_map: Path
    name_mode: str
    frames: list[HL320BinFrameConversion]
    manifest_path: Path


def convert_hl320_bin_dir_to_csv(
    *,
    bin_dir: str | Path,
    output_dir: str | Path,
    calibration_map: str | Path | None = None,
    name_mode: str = "sequential",
    overwrite: bool = False,
) -> HL320BinDirectoryConversion:
    bin_root = Path(bin_dir).expanduser().resolve()
    csv_root = Path(output_dir).expanduser().resolve()
    if not bin_root.is_dir():
        raise FileNotFoundError(f"HL320 bin directory does not exist: {bin_root}")
    if name_mode not in {"sequential", "stem"}:
        raise ValueError(f"name_mode must be 'sequential' or 'stem', got {name_mode!r}")
    bin_paths = sorted(bin_root.glob("*.bin"))
    if not bin_paths:
        raise FileNotFoundError(f"No .bin files found in {bin_root}")

    calibration_path = resolve_calibration_map(bin_root, calibration_map=calibration_map)
    calibration = read_hl320_calibration_map(calibration_path)
    prepare_csv_output_dir(csv_root, overwrite=overwrite)

    frames = []
    for index, bin_path in enumerate(bin_paths):
        frame_id = f"{index:06d}" if name_mode == "sequential" else bin_path.stem
        output_csv = csv_root / f"{frame_id}.csv"
        frames.append(
            convert_hl320_bin_to_csv(
                bin_path=bin_path,
                output_csv=output_csv,
                calibration=calibration,
                frame_id=frame_id,
            )
        )

    manifest = {
        "version": 1,
        "format": "hl320_bin_to_csv_v1",
        "bin_dir": str(bin_root),
        "output_dir": str(csv_root),
        "calibration_map": str(calibration_path),
        "name_mode": name_mode,
        "columns": HL320_BIN_CSV_COLUMNS,
        "frames": [
            {
                "frame_id": frame.frame_id,
                "source_bin": str(frame.source_bin),
                "output_csv": str(frame.output_csv),
                "rows": int(frame.rows),
                "width": int(frame.width),
                "height": int(frame.height),
            }
            for frame in frames
        ],
    }
    manifest_path = csv_root / "hl320_bin_to_csv_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return HL320BinDirectoryConversion(
        bin_dir=bin_root,
        output_dir=csv_root,
        calibration_map=calibration_path,
        name_mode=name_mode,
        frames=frames,
        manifest_path=manifest_path,
    )


def convert_hl320_bin_to_csv(
    *,
    bin_path: str | Path,
    output_csv: str | Path,
    calibration: dict[str, Any],
    frame_id: str | None = None,
) -> HL320BinFrameConversion:
    source = Path(bin_path).expanduser().resolve()
    destination = Path(output_csv).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"HL320 bin file does not exist: {source}")

    data = source.read_bytes()
    if len(data) < GZIP_HEADER_LEN:
        raise ValueError(f"HL320 bin file is too small: {source}")
    width = int.from_bytes(data[4:8], "little")
    height = int.from_bytes(data[8:12], "little")
    if width <= 0 or height <= 0:
        raise ValueError(f"HL320 bin file {source} has invalid width/height: {width}x{height}")

    rows = list(decode_hl320_bin_rows(data, width=width, height=height, calibration=calibration))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HL320_BIN_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return HL320BinFrameConversion(
        source_bin=source,
        output_csv=destination,
        frame_id=frame_id or destination.stem,
        rows=len(rows),
        width=width,
        height=height,
    )


def decode_hl320_bin_rows(
    data: bytes,
    *,
    width: int,
    height: int,
    calibration: dict[str, Any],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for slot_index in range(width):
        for pixel_index in range(height):
            offset = GZIP_HEADER_LEN + pixel_index * BATCH_POINT_LEN * ECHO_COUNT + slot_index * SLOT_DATA_SIZE
            if offset + BATCH_POINT_LEN > len(data):
                break
            x = struct.unpack_from("<f", data, offset)[0]
            y = struct.unpack_from("<f", data, offset + 4)[0]
            z = struct.unpack_from("<f", data, offset + 8)[0]
            azimuth = struct.unpack_from("<f", data, offset + 16)[0]
            vertical = struct.unpack_from("<f", data, offset + 20)[0]
            intensity = int.from_bytes(data[offset + 24 : offset + 28], "little")
            slot = int.from_bytes(data[offset + 32 : offset + 36], "little")
            pixel = int.from_bytes(data[offset + 36 : offset + 40], "little")
            hcell, vcell, cxd, cyd = lidar_to_camera_coord(slot, pixel, calibration)
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "azimuth": azimuth,
                    "vertical": vertical,
                    "intensity": intensity,
                    "slot": slot,
                    "pixel": pixel,
                    "hcell": hcell,
                    "vcell": vcell,
                    "Cxd": cxd,
                    "Cyd": cyd,
                }
            )
    return rows


def lidar_to_camera_coord(slot: int, pixel: int, calibration: dict[str, Any]) -> tuple[int, int, float, float]:
    vcell = 3 * (16 * (slot // 21) + (pixel // 8)) + 1
    hcell = 3 * ((slot % 21) * 8 + (pixel % 8)) + 1
    fov_offset_index = lut_v_index(vcell) + lut_h_index(hcell)

    calibration_root = calibration.get("calibration", calibration)
    horizontal = coefficient_list(calibration_root["HorizontalAngleCoef"][fov_offset_index])
    vertical = coefficient_list(calibration_root["VerticalAngleCoef"][fov_offset_index])
    if len(horizontal) != 8 or len(vertical) != 8:
        raise ValueError(f"Calibration coefficients must contain 8 values, got {len(horizontal)} and {len(vertical)}")

    lx = hcell - 503 / 2
    ly = vcell - 383 / 2
    radius_sq = lx**2 + ly**2
    k1, k2, k3, k4, k5, k6, k7, k8 = horizontal
    p1, p2, p3, p4, p5, p6, p7, p8 = vertical
    cxd = (
        k1 * lx * radius_sq
        + k2 * lx * (radius_sq**2)
        + k3 * lx * (radius_sq**3)
        + k4 * 2 * lx * ly
        + k5 * (radius_sq + 2 * (lx**2))
        + k6 * lx
        + k7 * ly
        + k8
    )
    cyd = (
        p1 * ly * radius_sq
        + p2 * ly * (radius_sq**2)
        + p3 * ly * (radius_sq**3)
        + p4 * 2 * lx * ly
        + p5 * (radius_sq + 2 * (ly**2))
        + p6 * ly
        + p7 * lx
        + p8
    )
    return int(hcell), int(vcell), float(cxd), float(cyd)


def lut_v_index(vcell: int) -> int:
    if 0 <= vcell <= 95:
        return 0
    if 96 <= vcell <= 191:
        return 6
    if 192 <= vcell <= 287:
        return 12
    if 288 <= vcell <= 383:
        return 18
    return 0


def lut_h_index(hcell: int) -> int:
    if 0 <= hcell <= 83:
        return 0
    if 84 <= hcell <= 167:
        return 1
    if 168 <= hcell <= 251:
        return 2
    if 252 <= hcell <= 335:
        return 3
    if 336 <= hcell <= 419:
        return 4
    if 420 <= hcell <= 503:
        return 5
    return 0


def coefficient_list(value: Any) -> list[float]:
    if isinstance(value, str):
        tokens = [token for token in re.split(r"[\s,]+", value.strip()) if token]
        return [float(token) for token in tokens]
    return [float(item) for item in value]


def resolve_calibration_map(bin_dir: Path, *, calibration_map: str | Path | None) -> Path:
    if calibration_map is not None:
        path = Path(calibration_map).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Calibration map does not exist: {path}")
        return path
    for directory in [bin_dir, *bin_dir.parents]:
        candidate = directory / "calibration_map.txt"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find calibration_map.txt in {bin_dir} or its parents. Pass --calibration-map.")


def read_hl320_calibration_map(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Calibration map is empty: {source}")
    if not text.startswith("{"):
        text = "{" + text + "}"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Calibration map is not valid JSON: {source}: {exc}") from exc
    calibration = payload.get("calibration", payload)
    if "HorizontalAngleCoef" not in calibration or "VerticalAngleCoef" not in calibration:
        raise ValueError(f"Calibration map {source} must contain HorizontalAngleCoef and VerticalAngleCoef")
    return payload


def prepare_csv_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"CSV output directory is not empty: {path}. Use --overwrite to replace it.")
            for item in path.iterdir():
                if item.is_dir():
                    import shutil

                    shutil.rmtree(item)
                else:
                    item.unlink()
    else:
        path.mkdir(parents=True)
