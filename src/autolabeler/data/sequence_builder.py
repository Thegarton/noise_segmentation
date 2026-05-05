from __future__ import annotations

from .dataset_indexer import FrameRecord


def build_temporal_windows(records: list[FrameRecord], k_past: int = 2, k_future: int = 2) -> list[dict]:
    windows = []
    for i, rec in enumerate(records):
        past = records[max(0, i - k_past): i]
        future = records[i + 1: i + 1 + k_future]
        windows.append({"current": rec, "past": past, "future": future})
    return windows
