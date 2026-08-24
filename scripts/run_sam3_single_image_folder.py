#!/usr/bin/env python3
"""Compatibility facade for the modular SAM3 single-image pipeline."""

from __future__ import annotations

import sys
from functools import wraps
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolabeler.teachers.sam3_single_image import *
from autolabeler.teachers.sam3_single_image.pipeline import (
    process_image as _pipeline_process_image,
)


@wraps(_pipeline_process_image)
def process_image(*args, **kwargs):
    """Process one image while preserving patchable legacy helper hooks."""
    kwargs.setdefault("_apply_colour_correction", apply_simple_colour_correction_rgb)
    kwargs.setdefault("_save_image_outputs", save_image_outputs)
    return _pipeline_process_image(*args, **kwargs)
