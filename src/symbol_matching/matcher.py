"""One-shot symbol matching via binary template matching with TTA."""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from symbol_matching.models import BBox, MatchHit
from symbol_matching.tiling import full_page_bbox, tile_is_blank_gray, tile_origins

_PEAK_KERNEL_CACHE: dict[int, np.ndarray] = {}
_RESERVED_CPU_CORES = 4


def max_parallel_workers() -> int:
    """Upper bound for process pools (leave cores free for OS / UI / GPU driver)."""
    cpus = os.cpu_count() or 1
    return max(1, cpus - _RESERVED_CPU_CORES)


@dataclass(frozen=True)
class MatcherConfig:
    scales: tuple[float, ...] = (0.85, 0.92, 1.0, 1.08, 1.18)
    rotations_deg: tuple[int, ...] = (0, 90, 180, 270)
    score_threshold: float = 0.55
    nms_iou: float = 0.30
    max_hits_per_page: int = 200
    max_search_side: int = 3000
    max_candidates_per_variant: int = 200
    max_candidates_per_tile: int = 120
    max_candidates_before_nms: int = 2500
    peak_size_divisor: int = 3
    tile_size: int = 768
    tile_overlap: int = 192
    skip_blank_tiles: bool = True
    blank_tile_max_mean: float = 252.0
    blank_tile_max_std: float = 3.0
    tile_workers: int = 1


@dataclass(frozen=True)
class _TileJob:
    tile_mask: np.ndarray
    ox: float
    oy: float
    page_scale: float
    roi_offset_x: float
    roi_offset_y: float


def resolve_parallel_workers(
    tile_workers: int,
    page_workers: int,
    num_pages: int,
) -> tuple[int, int]:
    """Return (tile_workers, page_workers) without nested process pools.

    For small page counts, prefer tile parallelism (large sheets, one page at a time).
    For many pages, prefer page parallelism.
    """
    tw = max(1, int(tile_workers))
    pw = max(1, int(page_workers))
    cap = max_parallel_workers()
    pw = min(pw, cap)
    tw = min(tw, cap)
    if pw > 1 and tw > 1:
        if num_pages <= 3:
            pw = 1
        else:
            tw = 1
    return tw, pw


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


def _trim_to_ink(mask: np.ndarray, padding: int) -> tuple[np.ndarray, tuple[int, int]]:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return mask, (0, 0)
    x, y, w, h = cv2.boundingRect(coords)
    y1 = max(0, y - padding)
    x1 = max(0, x - padding)
    y2 = min(mask.shape[0], y + h + padding)
    x2 = min(mask.shape[1], x + w + padding)
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


def _peak_dilate_kernel(peak_radius: int) -> np.ndarray:
    cached = _PEAK_KERNEL_CACHE.get(peak_radius)
    if cached is not None:
        return cached
    kernel = np.ones((peak_radius, peak_radius), dtype=np.uint8)
    _PEAK_KERNEL_CACHE[peak_radius] = kernel
    return kernel


def _scaled_template_bank(
    template_bank: Sequence[tuple[np.ndarray, int, float]],
    page_scale: float,
) -> list[tuple[np.ndarray, int, float]]:
    if page_scale >= 1.0:
        return list(template_bank)
    return [(_scale_mask(tmpl, page_scale), rot, scale) for tmpl, rot, scale in template_bank]


def _keep_top_indices(scores: Sequence[float], limit: int) -> list[int]:
    n = len(scores)
    if n <= limit:
        return list(range(n))
    score_arr = np.asarray(scores, dtype=np.float32)
    return np.argpartition(score_arr, -limit)[-limit:].tolist()


def _nms(boxes: list[BBox], scores: list[float], iou_threshold: float) -> list[int]:
    """Greedy NMS on xyxy boxes (fallback when OpenCV NMS is unavailable)."""
    if len(boxes) == 0:
        return []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    while len(order) > 0:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if boxes[i].iou(boxes[j]) < iou_threshold]
    return keep


def _nms_opencv(boxes: list[BBox], scores: list[float], iou_threshold: float) -> list[int]:
    n = len(boxes)
    if n == 0:
        return []
    rects = np.empty((n, 4), dtype=np.float32)
    for i, box in enumerate(boxes):
        rects[i, 0] = box.x1
        rects[i, 1] = box.y1
        rects[i, 2] = box.width()
        rects[i, 3] = box.height()
    score_arr = np.asarray(scores, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(
        rects.tolist(),
        score_arr.tolist(),
        score_threshold=0.0,
        nms_threshold=float(iou_threshold),
    )
    if indices is None or len(indices) == 0:
        return []
    flat = np.asarray(indices).reshape(-1)
    return [int(i) for i in flat]


def build_template_bank(
    exemplar_rgb: np.ndarray,
    config: MatcherConfig,
) -> list[tuple[np.ndarray, int, float]]:
    mask = _to_ink_mask(exemplar_rgb)
    trimmed, _ = _trim_to_ink(mask, padding=2)
    if trimmed.size == 0 or trimmed.shape[0] < 4 or trimmed.shape[1] < 4:
        raise ValueError("exemplar contains no detectable ink; pick a tighter box")
    bank: list[tuple[np.ndarray, int, float]] = []
    for rot in config.rotations_deg:
        rotated = _rotate_mask(trimmed, rot)
        for scale in config.scales:
            variant = _scale_mask(rotated, scale)
            if variant.shape[0] >= 4 and variant.shape[1] >= 4:
                bank.append((variant, rot, scale))
    return bank


def _match_variants_on_tile(
    tile_mask: np.ndarray,
    template_bank: Sequence[tuple[np.ndarray, int, float]],
    config: MatcherConfig,
    page_scale: float,
    tile_offset_x: float,
    tile_offset_y: float,
    roi_offset_x: float,
    roi_offset_y: float,
) -> tuple[list[BBox], list[float], list[str]]:
    work_h, work_w = tile_mask.shape
    boxes: list[BBox] = []
    scores: list[float] = []
    sources: list[str] = []
    inv = 1.0 / page_scale if page_scale < 1.0 else 1.0

    for tmpl, rot, scale in template_bank:
        th, tw = tmpl.shape
        if th < 4 or tw < 4 or th > work_h or tw > work_w:
            continue
        result = cv2.matchTemplate(tile_mask, tmpl, cv2.TM_CCOEFF_NORMED)
        div = max(2, int(config.peak_size_divisor))
        peak_radius = max(3, min(th, tw) // div)
        dilated = cv2.dilate(result, _peak_dilate_kernel(peak_radius))
        peak_mask = np.logical_and(
            np.equal(result, dilated),
            np.greater_equal(result, config.score_threshold),
        )
        ys, xs = np.nonzero(peak_mask)
        if ys.size == 0:
            continue
        cand_scores = result[ys, xs]
        cap = config.max_candidates_per_variant
        if ys.size > cap:
            top_idx = np.argpartition(cand_scores, -cap)[-cap:]
            ys = ys[top_idx]
            xs = xs[top_idx]
            cand_scores = cand_scores[top_idx]
        source_tag = f"template:rot{rot}:s{scale:.2f}"
        wx = xs.astype(np.float64) + tile_offset_x
        wy = ys.astype(np.float64) + tile_offset_y
        tw_f = float(tw)
        th_f = float(th)
        x1s = wx * inv + roi_offset_x
        y1s = wy * inv + roi_offset_y
        x2s = (wx + tw_f) * inv + roi_offset_x
        y2s = (wy + th_f) * inv + roi_offset_y
        for i in range(int(cand_scores.shape[0])):
            boxes.append(
                BBox(
                    x1=float(x1s[i]),
                    y1=float(y1s[i]),
                    x2=float(x2s[i]),
                    y2=float(y2s[i]),
                )
            )
            scores.append(float(cand_scores[i]))
            sources.append(source_tag)
    return boxes, scores, sources


def _run_tile_job(
    job: _TileJob,
    search_bank: list[tuple[np.ndarray, int, float]],
    config: MatcherConfig,
) -> tuple[list[BBox], list[float], list[str]]:
    t_boxes, t_scores, t_sources = _match_variants_on_tile(
        job.tile_mask,
        search_bank,
        config,
        job.page_scale,
        job.ox,
        job.oy,
        job.roi_offset_x,
        job.roi_offset_y,
    )
    return _truncate_candidate_lists(
        t_boxes,
        t_scores,
        t_sources,
        config.max_candidates_per_tile,
    )


def _tile_worker_entry(
    payload: tuple[_TileJob, list[tuple[np.ndarray, int, float]], MatcherConfig],
) -> tuple[list[BBox], list[float], list[str]]:
    job, search_bank, config = payload
    return _run_tile_job(job, search_bank, config)


def _collect_tile_jobs(
    work_mask: np.ndarray,
    work_gray: np.ndarray | None,
    origins: Sequence[tuple[int, int]],
    config: MatcherConfig,
    page_scale: float,
    roi_offset_x: float,
    roi_offset_y: float,
) -> list[_TileJob]:
    ts = config.tile_size
    jobs: list[_TileJob] = []
    for ox, oy in origins:
        tile_mask = work_mask[oy : oy + ts, ox : ox + ts]
        if tile_mask.shape[0] < 4 or tile_mask.shape[1] < 4:
            continue
        if config.skip_blank_tiles:
            if work_gray is not None:
                tile_gray = work_gray[oy : oy + ts, ox : ox + ts]
                if tile_is_blank_gray(
                    tile_gray,
                    config.blank_tile_max_mean,
                    config.blank_tile_max_std,
                ):
                    continue
            else:
                raise ValueError("work_gray is required when skip_blank_tiles is enabled")
        jobs.append(
            _TileJob(
                tile_mask=tile_mask,
                ox=float(ox),
                oy=float(oy),
                page_scale=page_scale,
                roi_offset_x=roi_offset_x,
                roi_offset_y=roi_offset_y,
            )
        )
    return jobs


def _match_tiles_parallel(
    jobs: list[_TileJob],
    search_bank: list[tuple[np.ndarray, int, float]],
    config: MatcherConfig,
) -> tuple[list[BBox], list[float], list[str]]:
    workers = max(1, int(config.tile_workers))
    if workers <= 1 or len(jobs) <= 1:
        boxes: list[BBox] = []
        scores: list[float] = []
        sources: list[str] = []
        for job in jobs:
            t_boxes, t_scores, t_sources = _run_tile_job(job, search_bank, config)
            boxes.extend(t_boxes)
            scores.extend(t_scores)
            sources.extend(t_sources)
        return boxes, scores, sources

    boxes = []
    scores = []
    sources = []
    payloads = [(job, search_bank, config) for job in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for t_boxes, t_scores, t_sources in pool.map(_tile_worker_entry, payloads):
            boxes.extend(t_boxes)
            scores.extend(t_scores)
            sources.extend(t_sources)
    return boxes, scores, sources


def _truncate_candidate_lists(
    boxes: list[BBox],
    scores: list[float],
    sources: list[str],
    limit: int,
) -> tuple[list[BBox], list[float], list[str]]:
    if len(boxes) <= limit:
        return boxes, scores, sources
    keep = _keep_top_indices(scores, limit)
    return (
        [boxes[i] for i in keep],
        [scores[i] for i in keep],
        [sources[i] for i in keep],
    )


def match_exemplar_on_page(
    page_rgb: np.ndarray,
    template_bank: Sequence[tuple[np.ndarray, int, float]],
    config: MatcherConfig,
    search_rois: list[BBox] | None = None,
) -> list[MatchHit]:
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
    search_bank = _scaled_template_bank(template_bank, page_scale)
    work_gray: np.ndarray | None = None
    if config.skip_blank_tiles:
        work_gray = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY)

    if config.tile_size <= 0:
        raise ValueError("tile_size must be positive")
    overlap = min(config.tile_overlap, config.tile_size - 1)
    origins = tile_origins(work_h, work_w, config.tile_size, overlap)

    tile_jobs = _collect_tile_jobs(
        work_mask,
        work_gray,
        origins,
        config,
        page_scale,
        roi_offset_x,
        roi_offset_y,
    )
    boxes, scores, sources = _match_tiles_parallel(tile_jobs, search_bank, config)

    if len(boxes) == 0:
        return []

    boxes, scores, sources = _truncate_candidate_lists(
        boxes,
        scores,
        sources,
        config.max_candidates_before_nms,
    )
    keep_indices = _nms_opencv(boxes, scores, config.nms_iou)
    keep_indices = keep_indices[: config.max_hits_per_page]
    return [
        MatchHit(page_id="", bbox=boxes[i], score=scores[i], source=sources[i])
        for i in keep_indices
    ]


def template_match_page_task(
    page_rgb: np.ndarray,
    template_bank: list[tuple[np.ndarray, int, float]],
    config: MatcherConfig,
    search_rois: list[BBox],
) -> list[MatchHit]:
    """Picklable entry point for per-page process pools."""
    return match_exemplar_on_page(
        page_rgb,
        template_bank,
        config,
        search_rois=search_rois,
    )


def _template_match_page_entry(
    payload: tuple[
        np.ndarray,
        list[tuple[np.ndarray, int, float]],
        MatcherConfig,
        list[BBox],
    ],
) -> list[MatchHit]:
    page_rgb, template_bank, config, search_rois = payload
    return template_match_page_task(page_rgb, template_bank, config, search_rois)
