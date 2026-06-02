"""Load external exemplar crops and optional page localization for export metadata."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from symbol_matching.models import BBox

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_EXEMPLAR_CROP = _REPO_ROOT / "Sample_Input" / "example_input_crop.png"


def load_exemplar_rgb(path: Path) -> np.ndarray:
    """Load a user-provided exemplar crop (HxWx3 uint8 RGB)."""
    if not path.is_file():
        raise FileNotFoundError(f"exemplar crop not found: {path}")
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB exemplar, got shape {rgb.shape}")
    if rgb.shape[0] < 4 or rgb.shape[1] < 4:
        raise ValueError(f"exemplar crop too small: {rgb.shape[1]}x{rgb.shape[0]}")
    return rgb


def ensure_exemplar_rgb(exemplar_rgb: np.ndarray) -> np.ndarray:
    if exemplar_rgb.ndim != 3 or exemplar_rgb.shape[2] != 3:
        raise ValueError(f"exemplar_rgb must be HxWx3 uint8 RGB, got shape {exemplar_rgb.shape}")
    if exemplar_rgb.flags["C_CONTIGUOUS"]:
        return exemplar_rgb
    return np.ascontiguousarray(exemplar_rgb)


def locate_exemplar_bbox(page_rgb: np.ndarray, exemplar_rgb: np.ndarray, max_side: int) -> BBox:
    """Estimate exemplar position on a page (export metadata only, not used for matching)."""
    page_h, page_w = page_rgb.shape[:2]
    ex_h, ex_w = exemplar_rgb.shape[:2]
    longest = max(page_h, page_w)
    scale = 1.0 if longest <= max_side else float(max_side) / float(longest)
    if scale < 1.0:
        page_work = cv2.resize(
            page_rgb,
            (max(1, int(round(page_w * scale))), max(1, int(round(page_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        ex_work = cv2.resize(
            exemplar_rgb,
            (max(1, int(round(ex_w * scale))), max(1, int(round(ex_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        page_work = page_rgb
        ex_work = exemplar_rgb
    page_gray = cv2.cvtColor(page_work, cv2.COLOR_RGB2GRAY)
    ex_gray = cv2.cvtColor(ex_work, cv2.COLOR_RGB2GRAY)
    if ex_gray.shape[0] > page_gray.shape[0] or ex_gray.shape[1] > page_gray.shape[1]:
        raise ValueError("exemplar crop is larger than the reference page")
    result = cv2.matchTemplate(page_gray, ex_gray, cv2.TM_CCOEFF_NORMED)
    _min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    inv = 1.0 / scale if scale < 1.0 else 1.0
    x1 = float(max_loc[0]) * inv
    y1 = float(max_loc[1]) * inv
    return BBox(x1=x1, y1=y1, x2=x1 + float(ex_w), y2=y1 + float(ex_h))
