"""ONNX region detector tests (parse logic + optional live model)."""

from __future__ import annotations

import numpy as np
import pytest

from symbol_matching.region_proposal import (
    TRAIN_IMGSZ,
    OnnxRegionDetector,
    _parse_yolo_onnx_output,
    default_region_onnx_path,
    infer_rgb_to_nchw_float,
    load_region_detector,
    preprocess_training_matched,
    RegionProposalConfig,
)


def test_infer_rgb_to_nchw_shape() -> None:
    rgb = np.zeros((TRAIN_IMGSZ, TRAIN_IMGSZ, 3), dtype=np.uint8)
    tensor = infer_rgb_to_nchw_float(rgb)
    assert tensor.shape == (1, 3, TRAIN_IMGSZ, TRAIN_IMGSZ)
    assert tensor.max() <= 1.0


def test_parse_yolo_onnx_output_filters_low_conf() -> None:
    preds = np.zeros((1, 2, 5), dtype=np.float32)
    preds[0, 0, :] = [320.0, 320.0, 200.0, 200.0, 0.9]
    preds[0, 1, :] = [100.0, 100.0, 50.0, 50.0, 0.1]
    hits = _parse_yolo_onnx_output(preds, conf=0.25, iou_threshold=0.45, max_detections=10)
    assert len(hits) == 1
    assert hits[0][4] == pytest.approx(0.9)


@pytest.mark.integration
def test_live_onnx_detector_on_blank_page() -> None:
    onnx_path = default_region_onnx_path()
    if not onnx_path.is_file():
        pytest.skip(f"ONNX weights missing: {onnx_path}")
    cfg = RegionProposalConfig(enabled=True, onnx_path=onnx_path, conf=0.25, ort_device="cuda")
    detector = load_region_detector(cfg)
    assert "CUDAExecutionProvider" in detector.active_providers
    page = np.full((360, 540, 3), 255, dtype=np.uint8)
    infer_rgb, _, _ = preprocess_training_matched(page)
    hits = detector.detect_640(infer_rgb, 0.25, 0.45, 50)
    assert isinstance(hits, list)


@pytest.mark.integration
def test_live_onnx_matches_exported_output_layout() -> None:
    onnx_path = default_region_onnx_path()
    if not onnx_path.is_file():
        pytest.skip(f"ONNX weights missing: {onnx_path}")
    det = OnnxRegionDetector(onnx_path, ort_device="cuda")
    assert det.active_providers[0] == "CUDAExecutionProvider"
    infer_rgb, _, _ = preprocess_training_matched(
        np.full((3600, 5400, 3), 240, dtype=np.uint8)
    )
    hits = det.detect_640(infer_rgb, 0.25, 0.45, 50)
    for x1, y1, x2, y2, score in hits:
        assert 0.0 <= x1 < x2 <= TRAIN_IMGSZ
        assert 0.0 <= y1 < y2 <= TRAIN_IMGSZ
        assert score >= 0.25
