"""One-shot symbol matching via binary template matching with TTA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from symbol_matching.models import BBox, MatchHit
from symbol_matching.tiling import full_page_bbox, tile_is_blank, tile_origins


@dataclass(frozen=True)
class MatcherConfig:
    scales: Tuple[float, ...] = (0.85, 0.92, 1.0, 1.08, 1.18)
    rotations_deg: Tuple[int, ...] = (0, 90, 180, 270)
    score_threshold: float = 0.55
    nms_iou: float = 0.30
    max_hits_per_page: int = 200
    max_search_side: int = 3000
    max_candidates_per_variant: int = 500
    tile_size: int = 768
    tile_overlap: int = 192
    skip_blank_tiles: bool = True
    blank_tile_max_mean: float = 252.0
    blank_tile_max_std: float = 3.0


def _to_ink_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=10,
    )
    return binary


def _trim_to_ink(mask: np.ndarray, padding: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return mask, (0, 0)
    y1 = max(0, int(ys.min()) - padding)
    y2 = min(mask.shape[0], int(ys.max()) + 1 + padding)
    x1 = max(0, int(xs.min()) - padding)
    x2 = min(mask.shape[1], int(xs.max()) + 1 + padding)
    return mask[y1:y2, x1:x2], (x1, y1)


def _rotate_mask(mask: np.ndarray, degrees: int) -> np.ndarray:
    if degrees % 360 == 0:
        return mask
    if degrees == 90:
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(mask, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported rotation: {degrees}")


def _scale_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return mask
    new_w = max(1, int(round(mask.shape[1] * scale)))
    new_h = max(1, int(round(mask.shape[0] * scale)))
    return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _scale_for_page(page_h: int, page_w: int, max_side: int) -> float:
    longest = max(page_h, page_w)
    if longest <= max_side:
        return 1.0
    return float(max_side) / float(longest)


def _nms(boxes: List[BBox], scores: List[float], iou_threshold: float) -> List[int]:
    if len(boxes) == 0:
        return []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while len(order) > 0:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if boxes[i].iou(boxes[j]) < iou_threshold]
    return keep


def build_template_bank(
    exemplar_rgb: np.ndarray,
    config: MatcherConfig,
) -> List[Tuple[np.ndarray, int, float]]:
    mask = _to_ink_mask(exemplar_rgb)
    trimmed, _ = _trim_to_ink(mask, padding=2)
    if trimmed.size == 0 or trimmed.shape[0] < 4 or trimmed.shape[1] < 4:
        raise ValueError("exemplar contains no detectable ink; pick a tighter box")
    bank: List[Tuple[np.ndarray, int, float]] = []
    for rot in config.rotations_deg:
        rotated = _rotate_mask(trimmed, rot)
        for scale in config.scales:
            variant = _scale_mask(rotated, scale)
            if variant.shape[0] >= 4 and variant.shape[1] >= 4:
                bank.append((variant, rot, scale))
    return bank


def _match_variants_on_tile(
    tile_mask: np.ndarray,
    template_bank: Sequence[Tuple[np.ndarray, int, float]],
    config: MatcherConfig,
    page_scale: float,
    tile_offset_x: float,
    tile_offset_y: float,
    roi_offset_x: float,
    roi_offset_y: float,
) -> Tuple[List[BBox], List[float], List[str]]:
    work_h, work_w = tile_mask.shape
    boxes: List[BBox] = []
    scores: List[float] = []
    sources: List[str] = []
    inv = 1.0 / page_scale if page_scale < 1.0 else 1.0

    for tmpl, rot, scale in template_bank:
        scaled_tmpl = _scale_mask(tmpl, page_scale) if page_scale < 1.0 else tmpl
        th, tw = scaled_tmpl.shape
        if th < 4 or tw < 4 or th > work_h or tw > work_w:
            continue
        result = cv2.matchTemplate(tile_mask, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
        peak_radius = max(3, min(th, tw) // 3)
        kernel = np.ones((peak_radius, peak_radius), dtype=np.uint8)
        dilated = cv2.dilate(result, kernel)
        peak_mask = (result == dilated) & (result >= config.score_threshold)
        ys, xs = np.where(peak_mask)
        if ys.size == 0:
            continue
        cand_scores = result[ys, xs]
        if ys.size > config.max_candidates_per_variant:
            top_idx = np.argpartition(cand_scores, -config.max_candidates_per_variant)[
                -config.max_candidates_per_variant:
            ]
            ys, xs, cand_scores = ys[top_idx], xs[top_idx], cand_scores[top_idx]
        source_tag = f"template:rot{rot}:s{scale:.2f}"
        for y, x, sc in zip(ys.tolist(), xs.tolist(), cand_scores.tolist()):
            wx = float(x) + tile_offset_x
            wy = float(y) + tile_offset_y
            boxes.append(
                BBox(
                    x1=wx * inv + roi_offset_x,
                    y1=wy * inv + roi_offset_y,
                    x2=(wx + float(tw)) * inv + roi_offset_x,
                    y2=(wy + float(th)) * inv + roi_offset_y,
                )
            )
            scores.append(float(sc))
            sources.append(source_tag)
    return boxes, scores, sources


def match_exemplar_on_page(
    page_rgb: np.ndarray,
    template_bank: Sequence[Tuple[np.ndarray, int, float]],
    config: MatcherConfig,
    search_rois: Optional[List[BBox]] = None,
) -> List[MatchHit]:
    """Search inside region ROI(s) using tiled ink matching; skip near-blank tiles."""
    page_h, page_w = page_rgb.shape[:2]
    rois = search_rois if search_rois is not None else [full_page_bbox(page_w, page_h)]
    if len(rois) != 1:
        raise ValueError("template engine expects a single search ROI (union of regions)")
    roi = rois[0]
    rx1 = int(max(0, np.floor(roi.x1)))
    ry1 = int(max(0, np.floor(roi.y1)))
    rx2 = int(min(page_w, np.ceil(roi.x2)))
    ry2 = int(min(page_h, np.ceil(roi.y2)))
    search_rgb = page_rgb[ry1:ry2, rx1:rx2]
    if search_rgb.size == 0:
        return []
    roi_offset_x = float(rx1)
    roi_offset_y = float(ry1)

    page_scale = _scale_for_page(search_rgb.shape[0], search_rgb.shape[1], config.max_search_side)
    if page_scale < 1.0:
        work_rgb = cv2.resize(
            search_rgb,
            (
                max(1, int(round(search_rgb.shape[1] * page_scale))),
                max(1, int(round(search_rgb.shape[0] * page_scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work_rgb = search_rgb
    work_mask = _to_ink_mask(work_rgb)
    work_h, work_w = work_mask.shape

    if config.tile_size <= 0:
        raise ValueError("tile_size must be positive")
    overlap = min(config.tile_overlap, config.tile_size - 1)
    origins = tile_origins(work_h, work_w, config.tile_size, overlap)

    boxes: List[BBox] = []
    scores: List[float] = []
    sources: List[str] = []

    for ox, oy in origins:
        tile_mask = work_mask[oy : oy + config.tile_size, ox : ox + config.tile_size]
        if tile_mask.shape[0] < 4 or tile_mask.shape[1] < 4:
            continue
        if config.skip_blank_tiles:
            tile_rgb = work_rgb[oy : oy + config.tile_size, ox : ox + config.tile_size]
            if tile_is_blank(
                tile_rgb,
                config.blank_tile_max_mean,
                config.blank_tile_max_std,
            ):
                continue
        t_boxes, t_scores, t_sources = _match_variants_on_tile(
            tile_mask,
            template_bank,
            config,
            page_scale,
            float(ox),
            float(oy),
            roi_offset_x,
            roi_offset_y,
        )
        boxes.extend(t_boxes)
        scores.extend(t_scores)
        sources.extend(t_sources)

    if len(boxes) == 0:
        return []

    keep_indices = _nms(boxes, scores, config.nms_iou)
    keep_indices = keep_indices[: config.max_hits_per_page]
    return [
        MatchHit(page_id="", bbox=boxes[i], score=scores[i], source=sources[i])
        for i in keep_indices
    ]
