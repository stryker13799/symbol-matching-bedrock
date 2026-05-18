"""One-shot symbol matching via binary template matching with TTA.

The exemplar crop is binarized into an "ink" mask, then matched against each
page's ink mask at multiple scales and 0/90/180/270 rotations. This is a
deterministic baseline that works well for clean construction-drawing symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from symbol_matching.models import BBox, MatchHit


@dataclass(frozen=True)
class MatcherConfig:
    scales: Tuple[float, ...] = (0.85, 0.92, 1.0, 1.08, 1.18)
    rotations_deg: Tuple[int, ...] = (0, 90, 180, 270)
    score_threshold: float = 0.55
    nms_iou: float = 0.30
    max_hits_per_page: int = 200
    # Pages are downscaled so their longest side is at most this many pixels
    # before matching; templates scale to match.
    max_search_side: int = 3000
    # Per (rotation, scale) variant: cap candidates kept before global NMS.
    max_candidates_per_variant: int = 500


def _to_ink_mask(rgb: np.ndarray) -> np.ndarray:
    """Return a uint8 mask where 'ink' pixels are 255 and background is 0."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Adaptive threshold handles uneven background / scanned drawings.
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
    """Crop ``mask`` tightly around non-zero pixels with a small padding margin."""
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
    """Build (mask, rotation_deg, scale) variants of the exemplar.

    Returns variants that have a usable amount of ink content.
    """
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


def match_exemplar_on_page(
    page_rgb: np.ndarray,
    template_bank: Sequence[Tuple[np.ndarray, int, float]],
    config: MatcherConfig,
) -> List[MatchHit]:
    """Search one page for every template variant; return NMS-filtered hits."""
    page_h, page_w = page_rgb.shape[:2]
    page_scale = _scale_for_page(page_h, page_w, config.max_search_side)
    if page_scale < 1.0:
        work = cv2.resize(
            page_rgb,
            (max(1, int(round(page_w * page_scale))), max(1, int(round(page_h * page_scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = page_rgb
    work_mask = _to_ink_mask(work)
    work_h, work_w = work_mask.shape

    boxes: List[BBox] = []
    scores: List[float] = []
    sources: List[str] = []

    for tmpl, rot, scale in template_bank:
        scaled_tmpl = _scale_mask(tmpl, page_scale) if page_scale < 1.0 else tmpl
        th, tw = scaled_tmpl.shape
        if th < 4 or tw < 4 or th > work_h or tw > work_w:
            continue
        result = cv2.matchTemplate(work_mask, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
        # Keep only pixels that are both above-threshold and a local maximum,
        # which collapses each peak to one candidate point.
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
        inv = 1.0 / page_scale if page_scale < 1.0 else 1.0
        source_tag = f"template:rot{rot}:s{scale:.2f}"
        for y, x, sc in zip(ys.tolist(), xs.tolist(), cand_scores.tolist()):
            boxes.append(
                BBox(
                    x1=float(x) * inv,
                    y1=float(y) * inv,
                    x2=float(x + tw) * inv,
                    y2=float(y + th) * inv,
                )
            )
            scores.append(float(sc))
            sources.append(source_tag)

    if len(boxes) == 0:
        return []

    keep_indices = _nms(boxes, scores, config.nms_iou)
    keep_indices = keep_indices[: config.max_hits_per_page]
    return [
        MatchHit(page_id="", bbox=boxes[i], score=scores[i], source=sources[i])
        for i in keep_indices
    ]
