"""Shared tile grid utilities for page and ROI search."""

from __future__ import annotations

import cv2
import numpy as np

from symbol_matching.models import BBox


def tile_is_blank_gray(tile_gray: np.ndarray, max_mean: float, max_std: float) -> bool:
    return float(tile_gray.mean()) >= max_mean and float(tile_gray.std()) <= max_std


def tile_is_blank(tile_rgb: np.ndarray, max_mean: float, max_std: float) -> bool:
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    return tile_is_blank_gray(gray, max_mean, max_std)


def tile_origins(
    region_h: int,
    region_w: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Return (x, y) top-left origins covering a region with the given overlap."""
    if tile_size <= overlap:
        raise ValueError("tile_size must exceed overlap")
    if region_h <= 0 or region_w <= 0:
        return []
    stride = tile_size - overlap
    xs: list[int] = []
    x = 0
    while x + tile_size < region_w:
        xs.append(x)
        x += stride
    xs.append(max(0, region_w - tile_size))
    ys: list[int] = []
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
