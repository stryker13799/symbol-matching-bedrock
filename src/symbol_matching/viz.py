"""Annotated overlays and crop saving."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from symbol_matching.models import BBox, MatchHit


def _score_to_rgb(score: float) -> Tuple[int, int, int]:
    clipped = max(0.0, min(1.0, float(score)))
    red = int(round(255.0 * (1.0 - clipped)))
    green = int(round(255.0 * clipped))
    return red, green, 0


def crop_rgb(image_rgb: np.ndarray, bbox: BBox) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x1 = int(max(0, np.floor(bbox.x1)))
    y1 = int(max(0, np.floor(bbox.y1)))
    x2 = int(min(w, np.ceil(bbox.x2)))
    y2 = int(min(h, np.ceil(bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty crop for bbox {bbox} on image {w}x{h}")
    return image_rgb[y1:y2, x1:x2].copy()


def save_png(image_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path, format="PNG")


def draw_region_proposals_on_page(
    page_rgb: np.ndarray,
    detections: Sequence[Tuple[BBox, float]],
    search_rois: Sequence[BBox],
) -> np.ndarray:
    """Overlay ONNX region detections (green) and merged search ROI (cyan)."""
    pil = Image.fromarray(page_rgb.copy())
    drawer = ImageDraw.Draw(pil)
    line_w = max(2, min(page_rgb.shape[:2]) // 600)
    for bbox, conf in detections:
        drawer.rectangle(
            [(bbox.x1, bbox.y1), (bbox.x2, bbox.y2)],
            outline=(80, 200, 80),
            width=line_w,
        )
        drawer.text(
            (bbox.x1 + 4, bbox.y1 + 4),
            f"drawing {conf:.2f}",
            fill=(80, 200, 80),
        )
    roi_w = max(3, line_w + 1)
    for roi in search_rois:
        drawer.rectangle(
            [(roi.x1, roi.y1), (roi.x2, roi.y2)],
            outline=(0, 200, 255),
            width=roi_w,
        )
        drawer.text(
            (roi.x1 + 6, max(0.0, roi.y1 - 18)),
            "search ROI",
            fill=(0, 200, 255),
        )
    return np.asarray(pil, dtype=np.uint8)


def draw_hits_on_page(page_rgb: np.ndarray, hits: List[MatchHit]) -> np.ndarray:
    """Return a copy of ``page_rgb`` with one labeled box per hit."""
    pil = Image.fromarray(page_rgb.copy())
    drawer = ImageDraw.Draw(pil)
    line_width = max(2, min(page_rgb.shape[:2]) // 600)
    for hit in hits:
        color = _score_to_rgb(hit.score)
        drawer.rectangle(
            [(hit.bbox.x1, hit.bbox.y1), (hit.bbox.x2, hit.bbox.y2)],
            outline=color,
            width=line_width,
        )
        drawer.text((hit.bbox.x1 + 4, hit.bbox.y1 + 4), f"{hit.score:.2f}", fill=color)
    return np.asarray(pil, dtype=np.uint8)
