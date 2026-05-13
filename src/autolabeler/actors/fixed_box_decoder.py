from __future__ import annotations

from pathlib import Path

from ..data.schemas import Box3D


DEFAULT_FIXED_ACTOR_SIZES_WHL: dict[str, list[float]] = {
    "TRUCK_BUS": [10.0, 2.6, 3.2],
    "CAR": [4.2, 1.8, 1.6],
    "CYCLIST": [1.8, 0.7, 1.7],
    "MOTORCYCLE": [2.1, 0.8, 1.5],
    "PEDESTRIAN": [0.7, 0.7, 1.75],
}


def load_fixed_actor_sizes(config_path: str | None = None) -> dict[str, list[float]]:
    if config_path is None:
        config_path = str(Path("configs") / "classes.yaml")

    path = Path(config_path)
    if not path.exists():
        return {k: v[:] for k, v in DEFAULT_FIXED_ACTOR_SIZES_WHL.items()}

    sizes: dict[str, list[float]] = {}
    in_section = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("fixed_actor_sizes:"):
            in_section = True
            continue
        if in_section and not raw_line.startswith((" ", "\t")):
            break
        if not in_section or ":" not in line:
            continue

        name, value = line.strip().split(":", 1)
        value = value.strip()
        if not (value.startswith("[") and value.endswith("]")):
            continue
        nums = [float(x.strip()) for x in value[1:-1].split(",") if x.strip()]
        if len(nums) == 3:
            sizes[name.strip()] = nums

    merged = {k: v[:] for k, v in DEFAULT_FIXED_ACTOR_SIZES_WHL.items()}
    merged.update(sizes)
    return merged


class FixedBoxDecoder:
    def __init__(self, fixed_sizes: dict[str, list[float]] | None = None) -> None:
        self.fixed_sizes = fixed_sizes or load_fixed_actor_sizes()

    def decode(self, semantic_class: str, center: list[float], yaw: float) -> Box3D:
        size = self.fixed_sizes.get(semantic_class)
        if size is None:
            raise ValueError(f"No fixed actor size configured for {semantic_class}")
        return Box3D(center=[float(x) for x in center], size=[float(x) for x in size], yaw=float(yaw), box_type="fixed_actor")
