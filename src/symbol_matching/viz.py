"""Annotated overlays and crop saving."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from symbol_matching.models import BBox, MatchHit


def _score_to_rgb(score: float) -> tuple[int, int, int]:
    clipped = max(0.0, min(1.0, float(score)))
    red = int(round(255.0 * (1.0 - clipped)))
    green = int(round(255.0 * clipped))
    return red, green, 0


def crop_rgb(image_rgb: np.ndarray, bbox: BBox) -> np.ndarray:
    """Return a view into ``image_rgb`` (caller must keep the parent array alive)."""
    h, w = image_rgb.shape[:2]
    x1 = int(max(0, np.floor(bbox.x1)))
    y1 = int(max(0, np.floor(bbox.y1)))
    x2 = int(min(w, np.ceil(bbox.x2)))
    y2 = int(min(h, np.ceil(bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty crop for bbox {bbox} on image {w}x{h}")
    return image_rgb[y1:y2, x1:x2]


def crop_rgb_owned(image_rgb: np.ndarray, bbox: BBox) -> np.ndarray:
    """Return a compact owned crop (releases dependency on the full page buffer)."""
    crop = crop_rgb(image_rgb, bbox)
    if crop.flags["C_CONTIGUOUS"] and crop.flags["OWNDATA"]:
        return crop
    return np.ascontiguousarray(crop)


def save_png(image_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not image_rgb.flags["C_CONTIGUOUS"]:
        image_rgb = np.ascontiguousarray(image_rgb)
    Image.fromarray(image_rgb).save(path, format="PNG", compress_level=1)


def draw_region_proposals_on_page(
    page_rgb: np.ndarray,
    detections: Sequence[tuple[BBox, float]],
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


def draw_hits_on_page(page_rgb: np.ndarray, hits: list[MatchHit]) -> np.ndarray:
    """Return a copy of ``page_rgb`` with one labeled box per hit."""
    if len(hits) == 0:
        return page_rgb.copy()
    out = np.ascontiguousarray(page_rgb.copy())
    line_width = max(2, min(page_rgb.shape[:2]) // 600)
    font_scale = max(0.35, line_width / 4.0)
    for hit in hits:
        color = _score_to_rgb(hit.score)
        bgr = (color[2], color[1], color[0])
        x1 = int(max(0, np.floor(hit.bbox.x1)))
        y1 = int(max(0, np.floor(hit.bbox.y1)))
        x2 = int(np.ceil(hit.bbox.x2))
        y2 = int(np.ceil(hit.bbox.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, line_width)
        cv2.putText(
            out,
            f"{hit.score:.2f}",
            (x1 + 4, y1 + 4 + int(12 * font_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            bgr,
            1,
            cv2.LINE_AA,
        )
    return out
