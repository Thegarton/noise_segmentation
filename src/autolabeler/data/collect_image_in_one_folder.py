from __future__ import annotations

from pathlib import Path

from PIL import Image


def collect_overlay_image_in_one_folder(dir_path: str | Path) -> None:
    _collect_images(
        root=Path(dir_path).expanduser().resolve(),
        source_name="overlay.jpg",
        target_name="all_overlay",
        suffix="overlay",
    )


def collect_combine_image_in_one_folder(dir_path: str | Path) -> None:
    _collect_images(
        root=Path(dir_path).expanduser().resolve(),
        source_name="preview.jpg",
        target_name="all_combine",
        suffix="combine",
    )


def _collect_images(
    *,
    root: Path,
    source_name: str,
    target_name: str,
    suffix: str,
) -> None:
    target_dir = root / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for frame_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if frame_dir.name in {"all_overlay", "all_combine"}:
            continue
        source_path = frame_dir / source_name
        if not source_path.is_file():
            continue
        with Image.open(source_path) as image:
            image.convert("RGB").save(target_dir / f"{frame_dir.name}_{suffix}.jpg")
