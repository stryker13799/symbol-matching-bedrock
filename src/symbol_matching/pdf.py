"""Render PDF pages to RGB images and infer per-page metadata from text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import numpy as np

from symbol_matching.models import PageRecord


# Sheet reference patterns commonly found in title blocks (e.g. E-201, M2.1, P-301).
_SHEET_REF_PATTERNS = (
    re.compile(r"\b([AESMPCLIFT][A-Z]?-?\d{1,2}\.?\d{0,2})\b"),
    re.compile(r"\b([AESMPCLIFT]\d{2,4})\b"),
)

# Coarse pageType classification by keyword scan over page text.
_PAGE_TYPE_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("plumbing", ("PLUMBING", "SANITARY", "DOMESTIC WATER")),
    ("electrical_lighting", ("LIGHTING",)),
    ("electrical_power", ("POWER",)),
    ("electrical", ("ELECTRICAL",)),
    ("mechanical", ("MECHANICAL", "HVAC", "DUCTWORK")),
    ("architectural_rcp", ("REFLECTED CEILING",)),
    ("architectural", ("ARCHITECTURAL", "FLOOR PLAN", "ENLARGED PLAN")),
    ("structural", ("STRUCTURAL", "FRAMING", "FOUNDATION")),
    ("civil", ("CIVIL", "SITE PLAN")),
    ("legend", ("LEGEND", "SYMBOLS", "ABBREVIATIONS")),
)

# Sheet-reference discipline letter -> coarse pageType. This is the strongest
# signal because architects/engineers reserve the prefix for the discipline.
_DISCIPLINE_LETTER_TO_PAGE_TYPE: dict = {
    "A": "architectural",
    "S": "structural",
    "E": "electrical",
    "M": "mechanical",
    "P": "plumbing",
    "C": "civil",
    "L": "landscape",
    "I": "interiors",
    "F": "fire_protection",
    "T": "telecom",
}


@dataclass(frozen=True)
class RenderedPage:
    """A rendered PDF page with its image and metadata."""

    record: PageRecord
    image_rgb: np.ndarray


def _extract_sheet_ref(text_upper: str) -> str:
    for pat in _SHEET_REF_PATTERNS:
        match = pat.search(text_upper)
        if match is not None:
            return match.group(1)
    return ""


def _extract_page_name(text_upper: str) -> str:
    plan_match = re.search(
        r"((?:FIRST|SECOND|THIRD|FOURTH|FIFTH|GROUND|BASEMENT|ROOF|MEZZANINE|PARTIAL|OVERALL|ENLARGED)"
        r"[A-Z0-9 \-]{0,40}?(?:PLAN|PLANS|VIEW|DETAIL|SCHEDULE|ELEVATION|SECTION))",
        text_upper,
    )
    if plan_match is not None:
        return plan_match.group(1).strip()
    title_match = re.search(
        r"([A-Z][A-Z0-9 \-]{4,40}(?:PLAN|SCHEDULE|LEGEND|DETAILS|ELEVATIONS|SECTIONS))",
        text_upper,
    )
    if title_match is not None:
        return title_match.group(1).strip()
    return ""


def _classify_page_type(text_upper: str, sheet_ref: str) -> str:
    if sheet_ref != "":
        first = sheet_ref[0].upper()
        mapped = _DISCIPLINE_LETTER_TO_PAGE_TYPE.get(first)
        if mapped is not None:
            # For electrical, try to refine to lighting/power from the page text.
            if mapped == "electrical":
                if "LIGHTING" in text_upper:
                    return "electrical_lighting"
                if "POWER" in text_upper:
                    return "electrical_power"
            return mapped
    for label, keywords in _PAGE_TYPE_RULES:
        for kw in keywords:
            if kw in text_upper:
                return label
    return "unknown"


def _plan_family(page_name_upper: str, page_type: str) -> str:
    """Group pages that should match together for SCOPE_SIMILAR_NAME.

    For example, 'FIRST FLOOR POWER PLAN' and 'SECOND FLOOR POWER PLAN' should
    share a family. Strategy: strip the level/qualifier prefix and keep the
    discipline-specific suffix.
    """
    cleaned = re.sub(
        r"\b(FIRST|SECOND|THIRD|FOURTH|FIFTH|GROUND|BASEMENT|ROOF|MEZZANINE|"
        r"PARTIAL|OVERALL|ENLARGED|LEVEL\s+\d+|FLOOR)\b",
        "",
        page_name_upper,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned == "":
        return f"{page_type}:general"
    return f"{page_type}:{cleaned.lower()}"


def _pixmap_to_rgb(pix: "fitz.Pixmap") -> np.ndarray:
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr)


def render_pdf(pdf_path: Path, dpi: int, max_pages: int) -> List[RenderedPage]:
    """Render up to ``max_pages`` from ``pdf_path`` at ``dpi`` resolution.

    Each page also gets a coarse metadata record inferred from its text content.
    """
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[RenderedPage] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as doc:
        page_count = min(doc.page_count, max_pages)
        for idx in range(page_count):
            page = doc.load_page(idx)
            text_upper = page.get_text("text").upper()
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            rgb = _pixmap_to_rgb(pix)

            page_name = _extract_page_name(text_upper) or f"Page {idx + 1}"
            sheet_ref = _extract_sheet_ref(text_upper)
            page_type = _classify_page_type(text_upper, sheet_ref)
            record = PageRecord(
                id=f"p{idx + 1}",
                page_index=idx,
                page_name=page_name,
                sheet_ref=sheet_ref,
                page_type=page_type,
                plan_family=_plan_family(page_name.upper(), page_type),
                width=int(rgb.shape[1]),
                height=int(rgb.shape[0]),
            )
            pages.append(RenderedPage(record=record, image_rgb=rgb))
    return pages
