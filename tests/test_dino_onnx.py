"""DINOv3 ONNX preprocess and optional live session tests."""

from __future__ import annotations

import numpy as np
import pytest

from symbol_matching.dinov3_rerank import (
    DINO_INPUT_SIZE,
    OnnxDinoEmbedder,
    default_dino_onnx_path,
    l2_normalize_rows,
    load_dinov3_embedder,
    preprocess_crops,
)


def test_preprocess_crops_shape_and_range() -> None:
    rgbs = [
        np.zeros((48, 64, 3), dtype=np.uint8),
        np.full((200, 180, 3), 128, dtype=np.uint8),
    ]
    tensor = preprocess_crops(rgbs)
    assert tensor.shape == (2, 3, DINO_INPUT_SIZE, DINO_INPUT_SIZE)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_l2_normalize_rows_unit_length() -> None:
    vecs = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    normed = l2_normalize_rows(vecs)
    norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-5)


@pytest.mark.integration
def test_live_dino_onnx_embedder() -> None:
    onnx_path = default_dino_onnx_path()
    if not onnx_path.is_file():
        pytest.skip(f"DINOv3 ONNX missing: {onnx_path}")
    embedder = load_dinov3_embedder(onnx_path, "cuda")
    assert "CUDAExecutionProvider" in embedder.active_providers
    rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
    emb = embedder.embed_batch([rgb, rgb])
    assert emb.shape == (2, 384)
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-4)


@pytest.mark.integration
def test_live_dino_onnx_matches_embedder_class() -> None:
    onnx_path = default_dino_onnx_path()
    if not onnx_path.is_file():
        pytest.skip(f"DINOv3 ONNX missing: {onnx_path}")
    det = OnnxDinoEmbedder(onnx_path, ort_device="cuda")
    assert det.active_providers[0] == "CUDAExecutionProvider"
    emb = det.embed_batch([np.zeros((32, 32, 3), dtype=np.uint8)])
    assert emb.shape == (1, 384)
