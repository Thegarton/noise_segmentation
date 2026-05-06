import json
from ..data.schemas import AutoLabelingResult


def export_jsonl(path: str, results: list[AutoLabelingResult]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_jsonable(), ensure_ascii=False) + "\n")
