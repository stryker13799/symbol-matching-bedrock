"""Optional SAM 3 step: refine the user's box to a tight bbox around the symbol.

Transformers' SAM3 PCS API takes ``input_boxes`` on the same image, so we use it
only on the reference page to snap the user's loose rectangle to the actual
symbol's segmentation. The resulting tight crop is then used as the exemplar
for cross-page template matching.

This module imports torch/transformers lazily so the rest of the pipeline runs
without the optional ``[sam3]`` extra installed.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

from symbol_matching.models import BBox

DEFAULT_HF_MODEL_ID = "facebook/sam3"

_hf_bundle: Optional[tuple] = None
_hf_bundle_key: Optional[str] = None


def resolve_hf_token(explicit_token: Optional[str]) -> Optional[str]:
    if explicit_token is not None and explicit_token.strip() != "":
        return explicit_token.strip()
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env is not None and env.strip() != "":
        return env.strip()
    return None


def load_sam3_bundle(model_id: str, token: Optional[str]) -> tuple:
    """Load and cache (model, processor) for ``model_id``.

    Subsequent calls with the same key return the cached bundle.
    """
    global _hf_bundle, _hf_bundle_key
    key = f"{model_id}\0{token or ''}"
    if _hf_bundle is not None and _hf_bundle_key == key:
        return _hf_bundle

    import torch  # noqa: WPS433 (lazy import is intentional)
    from transformers import Sam3Model, Sam3Processor

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    kwargs: dict = {}
    if token is not None:
        kwargs["token"] = token
    if torch.cuda.is_available():
        model = Sam3Model.from_pretrained(model_id, device_map="auto", **kwargs)
    else:
        model = Sam3Model.from_pretrained(model_id, **kwargs).to("cpu")
    processor = Sam3Processor.from_pretrained(model_id, **kwargs)
    model.eval()
    _hf_bundle = (model, processor)
    _hf_bundle_key = key
    return _hf_bundle


def refine_exemplar_bbox(
    page_rgb: np.ndarray,
    user_bbox: BBox,
    model_id: str,
    hf_token: Optional[str],
    score_threshold: float,
) -> BBox:
    """Run SAM3 on the reference page with the user's box; return the best mask's bbox.

    If SAM3 returns no candidates above ``score_threshold``, the original
    ``user_bbox`` is returned unchanged.
    """
    import torch  # noqa: WPS433

    token = resolve_hf_token(hf_token)
    model, processor = load_sam3_bundle(model_id, token)
    device = next(model.parameters()).device

    pil = Image.fromarray(page_rgb)
    input_boxes = [[[user_bbox.x1, user_bbox.y1, user_bbox.x2, user_bbox.y2]]]
    inputs = processor(
        images=pil,
        input_boxes=input_boxes,
        input_boxes_labels=[[1]],
        return_tensors="pt",
    )
    tensor_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**tensor_inputs)
    target_size = (int(page_rgb.shape[0]), int(page_rgb.shape[1]))
    post = processor.image_processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        target_sizes=[target_size],
    )
    result = post[0]
    scores_t = result["scores"]
    boxes_t = result["boxes"]
    if int(scores_t.shape[0]) == 0:
        return user_bbox
    best = int(torch.argmax(scores_t).item())
    row = boxes_t[best].detach().float().cpu().tolist()
    return BBox(x1=float(row[0]), y1=float(row[1]), x2=float(row[2]), y2=float(row[3]))
