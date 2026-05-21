"""Second-stage reranking of template hits with DINOv3 cosine similarity.

Uses the user exemplar crop as the query embedding and scores each candidate
crop from template matching. Template correlation is proposal-only; after
filtering by ``min_cosine``, ``MatchHit.score`` is the DINO cosine (same as
``dino_cosine``); ``template_score`` preserves the sourcing template value.

Model card: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from symbol_matching.models import BBox, MatchHit
from symbol_matching.viz import crop_rgb_owned

DEFAULT_DINOV3_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"

_bundle: Optional[Tuple[object, object]] = None
_bundle_key: Optional[str] = None


def resolve_hf_token(explicit_token: Optional[str]) -> Optional[str]:
    if explicit_token is not None and explicit_token.strip() != "":
        return explicit_token.strip()
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env is not None and env.strip() != "":
        return env.strip()
    return None


def release_dinov3_bundle() -> None:
    """Unload the cached DINOv3 model and free GPU memory if possible."""
    global _bundle, _bundle_key
    if _bundle is None:
        return
    import gc

    model, processor = _bundle
    del model, processor
    _bundle = None
    _bundle_key = None
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_dinov3_bundle(model_id: str, token: Optional[str]) -> Tuple[object, object]:
    """Load and cache ``(model, processor)``.

    Releases any cached SAM3 bundle first so only one large HF model owns the GPU.
    """
    global _bundle, _bundle_key
    key = f"{model_id}\0{token or ''}"
    if _bundle is not None and _bundle_key == key:
        return _bundle

    from symbol_matching.sam3 import release_sam3_bundle

    release_sam3_bundle()

    import torch
    from transformers import AutoImageProcessor, AutoModel

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    kwargs: dict = {}
    if token is not None:
        kwargs["token"] = token
    processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
    if torch.cuda.is_available():
        model = AutoModel.from_pretrained(model_id, device_map="auto", **kwargs)
    else:
        model = AutoModel.from_pretrained(model_id, **kwargs).to("cpu")
    model.eval()
    _bundle = (model, processor)
    _bundle_key = key
    return _bundle


def _pad_to_min_size(rgb: np.ndarray, min_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h >= min_side and w >= min_side:
        return rgb
    canvas = np.full((max(h, min_side), max(w, min_side), 3), 255, dtype=np.uint8)
    canvas[:h, :w] = rgb
    return canvas


def _embed_batch(
    rgbs: List[np.ndarray],
    model: object,
    processor: object,
    use_fp16: bool,
) -> np.ndarray:
    """Return L2-normalized float32 embeddings of shape ``(len(rgbs), dim)``."""
    import torch

    pil_list = [Image.fromarray(_pad_to_min_size(r, 32)) for r in rgbs]
    inputs = processor(images=pil_list, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    dtype = torch.float16 if use_fp16 and device.type == "cuda" else torch.float32
    with torch.inference_mode():
        if device.type == "cuda" and use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**inputs)
        else:
            out = model(**inputs)
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        pooled = out.last_hidden_state[:, 0, :]
    vec = pooled.float()
    vec = torch.nn.functional.normalize(vec, dim=1, eps=1e-12)
    return vec.detach().cpu().numpy().astype(np.float32, copy=False)


@dataclass(frozen=True)
class DinoRerankConfig:
    model_id: str = DEFAULT_DINOV3_MODEL_ID
    batch_size: int = 32
    min_cosine: float = 0.55
    use_fp16: bool = True


def rerank_template_hits_on_page(
    page_rgb: np.ndarray,
    exemplar_rgb: np.ndarray,
    template_hits: List[MatchHit],
    model: object,
    processor: object,
    config: DinoRerankConfig,
) -> List[MatchHit]:
    """Filter ``template_hits`` by DINOv3 cosine vs ``exemplar_rgb``.

    ``score`` on returned hits is DINO cosine (for viz, sorting, and export);
    ``template_score`` is the template correlation that proposed each box.
    """
    if len(template_hits) == 0:
        return []

    exemplar_emb = _embed_batch([exemplar_rgb], model, processor, config.use_fp16)[0]

    crops: List[np.ndarray] = []
    valid: List[MatchHit] = []
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
        emb = _embed_batch(chunk, model, processor, config.use_fp16)
        sims = np.dot(emb, exemplar_emb)
        n = int(sims.shape[0])
        all_sims[offset : offset + n] = sims
        offset += n

    out: List[MatchHit] = []
    for h, sim in zip(valid, all_sims):
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
