"""Second-stage reranking of template hits with DINOv3 cosine similarity (ONNX).

Uses the user exemplar crop as the query embedding and scores each candidate
crop from template matching. Template correlation is proposal-only; after
filtering by ``min_cosine``, ``MatchHit.score`` is the DINO cosine (same as
``dino_cosine``); ``template_score`` preserves the sourcing template value.

Export weights: ``python scripts/export_dino_onnx.py``
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from symbol_matching.models import MatchHit
from symbol_matching.ort_session import create_ort_session
from symbol_matching.viz import crop_rgb_owned

DINO_INPUT_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_SRC_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ONNX = _SRC_ROOT / "dinov3_weights" / "dinov3_vits16.onnx"


def default_dino_onnx_path() -> Path:
    env = os.environ.get("DINOV3_ONNX")
    if env:
        return Path(env)
    return _DEFAULT_ONNX


def preprocess_crops(rgbs: Sequence[np.ndarray]) -> np.ndarray:
    """Pack RGB uint8 crops into NCHW float32 ready for DINOv3 ONNX (B, 3, 224, 224)."""
    n = len(rgbs)
    if n == 0:
        raise ValueError("preprocess_crops requires at least one image")
    out = np.empty((n, 3, DINO_INPUT_SIZE, DINO_INPUT_SIZE), dtype=np.float32)
    for i, rgb in enumerate(rgbs):
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected HxWx3 uint8 RGB, got shape {rgb.shape}")
        h, w = rgb.shape[:2]
        interp = cv2.INTER_AREA if max(h, w) > DINO_INPUT_SIZE else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, (DINO_INPUT_SIZE, DINO_INPUT_SIZE), interpolation=interp)
        plane = resized.astype(np.float32) * (1.0 / 255.0)
        for c in range(3):
            out[i, c] = (plane[:, :, c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
    return out


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


class OnnxDinoEmbedder:
    """Batched DINOv3 embeddings via onnxruntime."""

    def __init__(self, onnx_path: Path, ort_device: str) -> None:
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"DINOv3 ONNX not found: {onnx_path}. Run: python scripts/export_dino_onnx.py"
            )
        self._session = create_ort_session(onnx_path, ort_device)
        self._input_name = self._session.get_inputs()[0].name
        self.active_providers: list[str] = list(self._session.get_providers())

    def embed_batch(self, rgbs: Sequence[np.ndarray]) -> np.ndarray:
        """Return L2-normalized float32 embeddings of shape ``(len(rgbs), dim)``."""
        if len(rgbs) == 0:
            return np.empty((0, 0), dtype=np.float32)
        tensor = preprocess_crops(rgbs)
        raw = self._session.run(None, {self._input_name: tensor})[0]
        return l2_normalize_rows(np.asarray(raw, dtype=np.float32))


def load_dinov3_embedder(onnx_path: Path, ort_device: str) -> OnnxDinoEmbedder:
    return OnnxDinoEmbedder(onnx_path, ort_device)


@dataclass(frozen=True)
class DinoRerankConfig:
    onnx_path: Path = _DEFAULT_ONNX
    ort_device: str = "cuda"
    batch_size: int = 32
    min_cosine: float = 0.55


def rerank_template_hits_on_page(
    page_rgb: np.ndarray,
    exemplar_rgb: np.ndarray,
    template_hits: list[MatchHit],
    embedder: OnnxDinoEmbedder,
    config: DinoRerankConfig,
) -> list[MatchHit]:
    """Filter ``template_hits`` by DINOv3 cosine vs ``exemplar_rgb``.

    ``score`` on returned hits is DINO cosine (for viz, sorting, and export);
    ``template_score`` is the template correlation that proposed each box.
    """
    if len(template_hits) == 0:
        return []

    exemplar_emb = embedder.embed_batch([exemplar_rgb])[0]

    crops: list[np.ndarray] = []
    valid: list[MatchHit] = []
    for h in template_hits:
        try:
            c = crop_rgb_owned(page_rgb, h.bbox)
        except ValueError:
            continue
        if c.size == 0:
            continue
        crops.append(c)
        valid.append(h)

    if len(crops) == 0:
        return []

    all_sims = np.empty(len(crops), dtype=np.float32)
    bs = max(1, int(config.batch_size))
    offset = 0
    for start in range(0, len(crops), bs):
        chunk = crops[start : start + bs]
        emb = embedder.embed_batch(chunk)
        sims = np.dot(emb, exemplar_emb)
        n = int(sims.shape[0])
        all_sims[offset : offset + n] = sims
        offset += n

    out: list[MatchHit] = []
    for h, sim in zip(valid, all_sims, strict=True):
        sim_f = float(sim)
        if sim_f < config.min_cosine:
            continue
        t_score = float(h.score)
        out.append(
            MatchHit(
                page_id=h.page_id,
                bbox=h.bbox,
                score=sim_f,
                source=f"template+dino:{h.source}",
                crop_path=h.crop_path,
                template_score=t_score,
                dino_cosine=sim_f,
            )
        )
    out.sort(key=lambda x: -x.score)
    return out
