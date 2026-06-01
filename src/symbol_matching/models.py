"""Typed models for pages, geometry, matches, and export payloads."""

from __future__ import annotations

from pydantic import BaseModel


class BBox(BaseModel):
    """Axis-aligned box in pixel coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.width() * self.height()

    def iou(self, other: BBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area() + other.area() - inter
        if union <= 0.0:
            return 0.0
        return inter / union


class PageRecord(BaseModel):
    """One rendered drawing page plus its metadata."""

    id: str
    page_index: int
    page_name: str
    sheet_ref: str
    page_type: str
    plan_family: str
    width: int
    height: int


class MatchHit(BaseModel):
    """One detected instance on a page.

    ``score`` is the value used for sorting, JSON export, and overlay coloring.
    For ``template+dino`` it is DINO cosine; ``template_score`` holds the
    proposal-stage template correlation when applicable.
    """

    page_id: str
    bbox: BBox
    score: float
    source: str
    crop_path: str | None = None
    template_score: float | None = None
    dino_cosine: float | None = None


class CaptureExport(BaseModel):
    id: str
    page_id: str
    bbox_xyxy: list[float]
    crop_path: str
    score: float
    dino_cosine: float | None = None


class DrawingItemExport(BaseModel):
    id: str
    page_id: str
    page_name: str
    sheet_ref: str
    page_type: str
    bbox_xyxy: list[float]
    score: float
    source: str
    captures: list[CaptureExport]
    template_score: float | None = None
    dino_cosine: float | None = None


class RunExport(BaseModel):
    """Full JSON export for a single run."""

    reference_page_id: str
    reference_bbox_xyxy: list[float]
    scope: str
    searched_page_ids: list[str]
    model_path: str | None
    hits: list[DrawingItemExport]
