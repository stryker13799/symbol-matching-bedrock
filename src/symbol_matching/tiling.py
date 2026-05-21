"""Shared tile grid utilities for page and ROI search."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

from symbol_matching.models import BBox


def tile_is_blank(tile_rgb: np.ndarray, max_mean: float, max_std: float) -> bool:
    gray = np.asarray(Image.fromarray(tile_rgb).convert("L"), dtype=np.float32)
    m = float(np.mean(gray))
    s = float(np.std(gray))
    return m >= max_mean and s <= max_std


def tile_origins(
    region_h: int,
    region_w: int,
    tile_size: int,
    overlap: int,
) -> List[Tuple[int, int]]:
    """Return (x, y) top-left origins covering a region with the given overlap."""
    if tile_size <= overlap:
        raise ValueError("tile_size must exceed overlap")
    if region_h <= 0 or region_w <= 0:
        return []
    stride = tile_size - overlap
    xs: List[int] = []
    x = 0
    while x + tile_size < region_w:
        xs.append(x)
        x += stride
    xs.append(max(0, region_w - tile_size))
    ys: List[int] = []
    y = 0
    while y + tile_size < region_h:
        ys.append(y)
        y += stride
    ys.append(max(0, region_h - tile_size))
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    return [(ox, oy) for oy in ys for ox in xs]


def clamp_bbox_to_image(bbox: BBox, width: int, height: int) -> BBox:
    x1 = float(max(0.0, min(bbox.x1, float(width - 1))))
    y1 = float(max(0.0, min(bbox.y1, float(height - 1))))
    x2 = float(max(x1 + 1.0, min(bbox.x2, float(width))))
    y2 = float(max(y1 + 1.0, min(bbox.y2, float(height))))
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def full_page_bbox(width: int, height: int) -> BBox:
    return BBox(x1=0.0, y1=0.0, x2=float(width), y2=float(height))
