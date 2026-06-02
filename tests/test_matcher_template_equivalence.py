"""Template hits must be identical for sequential vs parallel tile workers."""

from __future__ import annotations

import json

import numpy as np

from symbol_matching.matcher import MatcherConfig, build_template_bank, match_exemplar_on_page
from symbol_matching.models import MatchHit


def _fingerprint(hits: list[MatchHit]) -> str:
    rows = sorted(
        (
            round(h.bbox.x1, 4),
            round(h.bbox.y1, 4),
            round(h.bbox.x2, 4),
            round(h.bbox.y2, 4),
            round(h.score, 6),
            h.source,
        )
        for h in hits
    )
    return json.dumps(rows, separators=(",", ":"))


def _draw_glyph(canvas: np.ndarray, x: int, y: int, size: int) -> None:
    yy, xx = np.ogrid[: canvas.shape[0], : canvas.shape[1]]
    disc = (xx - x) ** 2 + (yy - y) ** 2 <= (size // 2) ** 2
    canvas[disc] = 0


def _config(tile_workers: int) -> MatcherConfig:
    return MatcherConfig(
        scales=(1.0, 1.08),
        rotations_deg=(0, 90),
        score_threshold=0.55,
        nms_iou=0.30,
        max_hits_per_page=50,
        tile_size=400,
        tile_overlap=64,
        skip_blank_tiles=True,
        tile_workers=tile_workers,
    )


def test_parallel_tiles_same_fingerprint_as_sequential() -> None:
    page = np.full((900, 1200, 3), 255, dtype=np.uint8)
    _draw_glyph(page, 400, 450, 28)
    _draw_glyph(page, 900, 500, 28)
    _draw_glyph(page, 200, 700, 28)
    exemplar = page[420:480, 370:430].copy()
    bank = build_template_bank(exemplar, _config(1))
    seq = match_exemplar_on_page(page, bank, _config(1))
    par = match_exemplar_on_page(page, bank, _config(4))
    assert _fingerprint(seq) == _fingerprint(par)
    assert len(seq) == len(par)
