"""Template matcher: parallel tile pass matches sequential pass."""

from __future__ import annotations

import numpy as np

from symbol_matching.matcher import MatcherConfig, build_template_bank, match_exemplar_on_page


def _draw_glyph(canvas: np.ndarray, x: int, y: int, size: int) -> None:
    yy, xx = np.ogrid[: canvas.shape[0], : canvas.shape[1]]
    disc = (xx - x) ** 2 + (yy - y) ** 2 <= (size // 2) ** 2
    canvas[disc] = 0


def test_parallel_tiles_match_sequential() -> None:
    page = np.full((900, 1200, 3), 255, dtype=np.uint8)
    _draw_glyph(page, 400, 450, 28)
    _draw_glyph(page, 900, 500, 28)
    exemplar = page[420:480, 370:430].copy()
    base = MatcherConfig(
        scales=(1.0,),
        rotations_deg=(0,),
        score_threshold=0.55,
        nms_iou=0.30,
        max_hits_per_page=20,
        tile_size=400,
        tile_overlap=64,
        skip_blank_tiles=True,
        tile_workers=1,
    )
    bank = build_template_bank(exemplar, base)
    seq_hits = match_exemplar_on_page(page, bank, base)
    par_cfg = MatcherConfig(
        scales=base.scales,
        rotations_deg=base.rotations_deg,
        score_threshold=base.score_threshold,
        nms_iou=base.nms_iou,
        max_hits_per_page=base.max_hits_per_page,
        max_search_side=base.max_search_side,
        max_candidates_per_variant=base.max_candidates_per_variant,
        max_candidates_per_tile=base.max_candidates_per_tile,
        max_candidates_before_nms=base.max_candidates_before_nms,
        peak_size_divisor=base.peak_size_divisor,
        tile_size=base.tile_size,
        tile_overlap=base.tile_overlap,
        skip_blank_tiles=base.skip_blank_tiles,
        blank_tile_max_mean=base.blank_tile_max_mean,
        blank_tile_max_std=base.blank_tile_max_std,
        tile_workers=2,
    )
    par_hits = match_exemplar_on_page(page, bank, par_cfg)
    assert len(seq_hits) == len(par_hits)
    seq_sorted = sorted(
        [(round(h.bbox.x1, 1), round(h.bbox.y1, 1), round(h.score, 3)) for h in seq_hits]
    )
    par_sorted = sorted(
        [(round(h.bbox.x1, 1), round(h.bbox.y1, 1), round(h.score, 3)) for h in par_hits]
    )
    assert seq_sorted == par_sorted
