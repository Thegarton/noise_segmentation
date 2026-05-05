import json
from pathlib import Path


def write_versioned_snapshot(path: str, records: list[dict], version: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pseudo_label_version": version, "records": records}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)
