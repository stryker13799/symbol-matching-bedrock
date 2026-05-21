"""Template matcher: tiled search with blank-tile skip."""

from __future__ import annotations

import numpy as np

from symbol_matching.matcher import MatcherConfig, build_template_bank, match_exemplar_on_page
from symbol_matching.models import BBox
from symbol_matching.tiling import full_page_bbox


def _draw_glyph(canvas: np.ndarray, x: int, y: int, size: int) -> None:
    yy, xx = np.ogrid[: canvas.shape[0], : canvas.shape[1]]
    disc = (xx - x) ** 2 + (yy - y) ** 2 <= (size // 2) ** 2
    canvas[disc] = 0


def test_tiled_search_skips_blank_margin() -> None:
    page = np.full((1200, 1600, 3), 255, dtype=np.uint8)
    _draw_glyph(page, 800, 600, 30)
    exemplar = page[585:645, 785:845].copy()
    config = MatcherConfig(
        scales=(1.0,),
        rotations_deg=(0,),
        score_threshold=0.55,
        nms_iou=0.30,
        max_hits_per_page=10,
        max_search_side=3000,
        tile_size=400,
        tile_overlap=64,
        skip_blank_tiles=True,
    )
    bank = build_template_bank(exemplar, config)
    roi = full_page_bbox(page.shape[1], page.shape[0])
    hits = match_exemplar_on_page(page, bank, config, search_rois=[roi])
    assert len(hits) >= 1
    cx = (hits[0].bbox.x1 + hits[0].bbox.x2) / 2.0
    cy = (hits[0].bbox.y1 + hits[0].bbox.y2) / 2.0
    assert abs(cx - 800.0) < 40.0
    assert abs(cy - 600.0) < 40.0


def test_region_roi_limits_template_search() -> None:
    page = np.full((400, 800, 3), 255, dtype=np.uint8)
    _draw_glyph(page, 100, 200, 25)
    _draw_glyph(page, 650, 200, 25)
    exemplar = page[175:235, 75:135].copy()
    config = MatcherConfig(
        scales=(1.0,),
        rotations_deg=(0,),
        score_threshold=0.55,
        nms_iou=0.30,
        max_hits_per_page=10,
        tile_size=256,
        tile_overlap=64,
        skip_blank_tiles=True,
    )
    bank = build_template_bank(exemplar, config)
    roi = BBox(x1=0.0, y1=0.0, x2=400.0, y2=400.0)
    hits = match_exemplar_on_page(page, bank, config, search_rois=[roi])
    assert len(hits) >= 1
    for hit in hits:
        cx = (hit.bbox.x1 + hit.bbox.x2) / 2.0
        assert cx < 450.0
