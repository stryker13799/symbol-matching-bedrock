"""Exemplar PNG loading and page localization."""

from __future__ import annotations

import numpy as np

from symbol_matching.exemplar import (
    DEFAULT_SAMPLE_EXEMPLAR_CROP,
    load_exemplar_rgb,
    locate_exemplar_bbox,
)
from symbol_matching.matcher import MatcherConfig, build_template_bank

_CROP = DEFAULT_SAMPLE_EXEMPLAR_CROP


def test_load_sample_exemplar_crop() -> None:
    rgb = load_exemplar_rgb(_CROP)
    assert rgb.shape[2] == 3
    bank = build_template_bank(rgb, MatcherConfig(scales=(1.0,), rotations_deg=(0,)))
    assert len(bank) >= 1


def test_locate_exemplar_on_synthetic_page() -> None:
    page = np.full((400, 600, 3), 255, dtype=np.uint8)
    yy, xx = np.ogrid[:400, :600]
    cx, cy = 250, 150
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= 18**2
    page[disc] = 0
    exemplar = page[110:184, 210:284].copy()
    bbox = locate_exemplar_bbox(page, exemplar, max_side=3000)
    assert abs((bbox.x1 + bbox.x2) / 2.0 - cx) < 25.0
    assert abs((bbox.y1 + bbox.y2) / 2.0 - cy) < 25.0


def test_locate_sample_crop_on_pdf_page() -> None:
    pdf = _CROP.parent / "17180_-_FULL_100_CD_SET_-_With_ADDENDUM_1_(1)_(dragged)_(3).pdf"
    if not pdf.is_file():
        return
    from symbol_matching.pdf import render_pdf

    exemplar = load_exemplar_rgb(_CROP)
    rendered = render_pdf(pdf, dpi=200, max_pages=1)
    bbox = locate_exemplar_bbox(rendered[0].image_rgb, exemplar, max_side=3000)
    assert bbox.width() >= 50.0
    assert bbox.height() >= 50.0
