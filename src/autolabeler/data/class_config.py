from __future__ import annotations

from pathlib import Path


DEFAULT_CLASSES_CONFIG = str(Path("configs") / "classes.yaml")


def load_semantic_classes(config_path: str | None = None) -> dict[str, int]:
    sections = _parse_simple_yaml_sections(config_path or DEFAULT_CLASSES_CONFIG)
    classes = sections.get("semantic_classes", {})
    if not isinstance(classes, dict):
        raise ValueError("semantic_classes must be a mapping of class_name: id")
    return {str(name): int(class_id) for name, class_id in classes.items()}


def load_noise_groups(config_path: str | None = None) -> dict[str, list[str]]:
    sections = _parse_simple_yaml_sections(config_path or DEFAULT_CLASSES_CONFIG)
    groups = sections.get("noise_groups", {})
    if not isinstance(groups, dict):
        raise ValueError("noise_groups must be a mapping of group_name: [class_names]")
    return {str(name): [str(x) for x in values] for name, values in groups.items()}


def invert_class_mapping(class_to_id: dict[str, int]) -> dict[int, str]:
    return {class_id: name for name, class_id in class_to_id.items()}


def _parse_simple_yaml_sections(config_path: str) -> dict[str, dict]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Class config does not exist: {path}")

    sections: dict[str, dict] = {}
    current_section: str | None = None
    current_nested_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue

        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        line = line_without_comment.strip()

        if indent == 0 and line.endswith(":"):
            current_section = line[:-1]
            sections[current_section] = {}
            current_nested_key = None
            continue

        if current_section is None:
            continue

        section = sections[current_section]
        if indent == 2 and ":" in line:
            key, value = [x.strip() for x in line.split(":", 1)]
            if value == "":
                section[key] = []
                current_nested_key = key
                continue
            section[key] = _parse_scalar(value)
            current_nested_key = None
            continue

        if indent == 4 and line.startswith("- ") and current_nested_key is not None:
            section[current_nested_key].append(line[2:].strip())

    return sections


def _parse_scalar(value: str):
    if value.startswith("[") and value.endswith("]"):
        return [float(x.strip()) for x in value[1:-1].split(",") if x.strip()]
    try:
        return int(value)
    except ValueError:
        return value
