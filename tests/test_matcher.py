"""Smoke test: a planted exemplar is recovered across a synthetic page."""

from __future__ import annotations

import numpy as np

from symbol_matching.matcher import MatcherConfig, build_template_bank, match_exemplar_on_page


def _draw_glyph(canvas: np.ndarray, x: int, y: int, size: int) -> None:
    """Draw a simple high-contrast glyph (a filled disc with a stem)."""
    yy, xx = np.ogrid[: canvas.shape[0], : canvas.shape[1]]
    disc = (xx - x) ** 2 + (yy - y) ** 2 <= (size // 2) ** 2
    canvas[disc] = 0
    stem_y1 = max(0, y - size)
    canvas[stem_y1:y, x - 1 : x + 2] = 0


def test_finds_planted_instances() -> None:
    page = np.full((400, 600, 3), 255, dtype=np.uint8)
    centers = [(80, 100), (300, 100), (500, 250), (180, 320)]
    for cx, cy in centers:
        _draw_glyph(page, cx, cy, 30)

    exemplar = page[60:130, 50:110].copy()
    config = MatcherConfig(
        scales=(1.0,),
        rotations_deg=(0,),
        score_threshold=0.55,
        nms_iou=0.30,
        max_hits_per_page=50,
        max_search_side=3000,
    )
    bank = build_template_bank(exemplar, config)
    hits = match_exemplar_on_page(page, bank, config)

    assert len(hits) >= len(centers), f"expected >= {len(centers)} hits, got {len(hits)}"
    for cx, cy in centers:
        nearest = min(
            hits,
            key=lambda h: (((h.bbox.x1 + h.bbox.x2) / 2 - cx) ** 2
                           + ((h.bbox.y1 + h.bbox.y2) / 2 - cy) ** 2),
        )
        assert abs((nearest.bbox.x1 + nearest.bbox.x2) / 2 - cx) < 30
        assert abs((nearest.bbox.y1 + nearest.bbox.y2) / 2 - cy) < 30
