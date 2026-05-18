"""BBox geometry and export schema tests."""

from __future__ import annotations

from symbol_matching.models import BBox, CaptureExport, DrawingItemExport, RunExport


def test_iou_identical() -> None:
    a = BBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    assert abs(a.iou(a) - 1.0) < 1e-9


def test_iou_disjoint() -> None:
    a = BBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    b = BBox(x1=20.0, y1=0.0, x2=30.0, y2=10.0)
    assert a.iou(b) == 0.0


def test_iou_half_overlap() -> None:
    a = BBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    b = BBox(x1=5.0, y1=0.0, x2=15.0, y2=10.0)
    assert abs(a.iou(b) - (50.0 / 150.0)) < 1e-9


def test_export_json_roundtrip() -> None:
    run = RunExport(
        reference_page_id="p1",
        reference_bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
        scope="this_page",
        searched_page_ids=["p1"],
        model_path=None,
        hits=[
            DrawingItemExport(
                id="i1",
                page_id="p1",
                page_name="Test",
                sheet_ref="E201",
                page_type="electrical",
                bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
                score=0.9,
                source="template+dino:template:rot0:s1.00",
                template_score=0.8,
                dino_cosine=0.9,
                captures=[
                    CaptureExport(
                        id="c1",
                        page_id="p1",
                        bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
                        crop_path="/tmp/x.png",
                        score=0.9,
                        dino_cosine=0.9,
                    )
                ],
            )
        ],
    )
    restored = RunExport.model_validate_json(run.model_dump_json())
    assert restored.hits[0].bbox_xyxy == [1.0, 2.0, 3.0, 4.0]
    assert restored.hits[0].score == 0.9
    assert restored.hits[0].template_score == 0.8
    assert restored.hits[0].dino_cosine == 0.9
    assert restored.searched_page_ids == ["p1"]
    assert restored.reference_bbox_xyxy == [1.0, 2.0, 3.0, 4.0]
