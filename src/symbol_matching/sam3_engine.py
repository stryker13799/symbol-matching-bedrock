"""SAM 3 cross-page matcher via the composite-and-tile few-shot trick.

The HuggingFace ``Sam3Processor`` only accepts ``input_boxes`` as a prompt on
the same image you pass in, so a box drawn on the reference page cannot be
sent verbatim against another page. To work around that limitation **without
modifying SAM 3**, we build a composite image per tile of the target page:

    +----------------------------+--------+
    |                            |        |
    |        target page tile    |EXEMPLAR|
    |        (native resolution) | crop   |
    |                            |        |
    +----------------------------+--------+

The composite is sized to SAM 3's native input (1008x1008 by default). We then
prompt SAM 3 with the exemplar's known bounding box inside the composite.
SAM 3 grounds the concept from the exemplar region and returns instance boxes
across the *entire* composite. We drop any detection that overlaps the
exemplar region and map the rest back to full-page coordinates, then run
non-maximum suppression across overlapping tiles.

This is a documented hack, not a blessed API path. It works because SAM 3 was
trained on image-exemplar prompts, but expect occasional noise — keep the
score threshold meaningful and use NMS aggressively. GPU + fp16 is strongly
preferred; on CPU a single page takes minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from symbol_matching.models import BBox, MatchHit
from symbol_matching.sam3 import load_sam3_bundle, resolve_hf_token

DEFAULT_SAM3_MODEL_ID = "facebook/sam3"


@dataclass(frozen=True)
class Sam3EngineConfig:
    tile_size: int = 768
    tile_overlap: int = 192
    composite_size: int = 1008
    exemplar_max_side: int = 200
    exemplar_pad_px: int = 12
    gutter_px: int = 16
    score_threshold: float = 0.40
    nms_iou: float = 0.30
    max_hits_per_page: int = 200
    use_fp16: bool = True
    # How many composites to pack into one ``model(**inputs)`` call. 3080 Ti
    # 12GB can usually fit 6–8 at 1008²; raise until OOM, then lower by one.
    batch_size: int = 8
    # Uniformly downscale the page (and exemplar) before tiling so the number
    # of tiles scales with *work resolution*, not raw 200 DPI megapixels.
    # 3200 long side → ~20 tiles/page instead of ~100+ at 7200×4800.
    max_page_infer_side: int = 3200
    # Skip tiles that are almost blank (saves SAM3 forwards on white margins).
    skip_blank_tiles: bool = True
    blank_tile_max_mean: float = 252.0
    blank_tile_max_std: float = 3.0


def _nms(boxes: List[BBox], scores: List[float], iou_threshold: float) -> List[int]:
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if boxes[i].iou(boxes[j]) < iou_threshold]
    return keep


def _resize_rgb_long_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return rgb
    scale = float(max_side) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return np.asarray(
        Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _uniform_work_scale(ph: int, pw: int, max_infer_side: int) -> float:
    """Return scale in (0, 1] such that max(ph, pw) * scale <= max_infer_side."""
    if max_infer_side <= 0:
        return 1.0
    longest = max(ph, pw)
    if longest <= max_infer_side:
        return 1.0
    return float(max_infer_side) / float(longest)


def _resize_rgb_uniform(rgb: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-9:
        return rgb
    if scale <= 0.0 or scale > 1.0:
        raise ValueError("scale must be in (0, 1] for downscale")
    h, w = rgb.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return np.asarray(
        Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _tile_is_blank(tile_rgb: np.ndarray, max_mean: float, max_std: float) -> bool:
    gray = np.asarray(Image.fromarray(tile_rgb).convert("L"), dtype=np.float32)
    m = float(np.mean(gray))
    s = float(np.std(gray))
    return m >= max_mean and s <= max_std


def _tile_origins(page_h: int, page_w: int, tile_size: int, overlap: int) -> List[Tuple[int, int]]:
    """Return (x, y) top-left origins covering the page with the given overlap."""
    if tile_size <= overlap:
        raise ValueError("tile_size must exceed overlap")
    stride = tile_size - overlap
    xs: List[int] = []
    x = 0
    while x + tile_size < page_w:
        xs.append(x)
        x += stride
    xs.append(max(0, page_w - tile_size))
    ys: List[int] = []
    y = 0
    while y + tile_size < page_h:
        ys.append(y)
        y += stride
    ys.append(max(0, page_h - tile_size))
    # Deduplicate (small pages collapse to single tile).
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    return [(x, y) for y in ys for x in xs]


def _build_composite(
    tile_rgb: np.ndarray,
    exemplar_rgb: np.ndarray,
    config: Sam3EngineConfig,
) -> Tuple[np.ndarray, BBox]:
    """Paste the tile + exemplar onto a white canvas. Return (canvas, exemplar_bbox)."""
    size = config.composite_size
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    th, tw = tile_rgb.shape[:2]
    eh, ew = exemplar_rgb.shape[:2]

    canvas[:th, :tw] = tile_rgb

    # Place exemplar in the top-right, separated by a gutter from the tile.
    ex_x = size - ew - config.exemplar_pad_px
    ex_y = config.exemplar_pad_px
    if ex_x <= tw + config.gutter_px:
        # Tile is too wide; fall back to bottom-right.
        ex_x = size - ew - config.exemplar_pad_px
        ex_y = size - eh - config.exemplar_pad_px
    canvas[ex_y : ex_y + eh, ex_x : ex_x + ew] = exemplar_rgb

    exemplar_bbox = BBox(
        x1=float(ex_x),
        y1=float(ex_y),
        x2=float(ex_x + ew),
        y2=float(ex_y + eh),
    )
    return canvas, exemplar_bbox


def _detect_on_composite_batch(
    composites: List[np.ndarray],
    exemplar_bboxes: List[BBox],
    model: object,
    processor: object,
    config: Sam3EngineConfig,
) -> List[List[Tuple[BBox, float]]]:
    """Run a batch of composites through SAM 3 in a single forward pass."""
    import torch  # local import: torch is an optional dep

    pil_images = [Image.fromarray(c) for c in composites]
    input_boxes = [
        [[bb.x1, bb.y1, bb.x2, bb.y2]] for bb in exemplar_bboxes
    ]
    input_boxes_labels = [[1] for _ in exemplar_bboxes]
    inputs = processor(
        images=pil_images,
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    tensor_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    target_sizes = [(int(c.shape[0]), int(c.shape[1])) for c in composites]
    autocast_dtype = torch.float16 if config.use_fp16 and device.type == "cuda" else None
    with torch.inference_mode():
        if autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                outputs = model(**tensor_inputs)
        else:
            outputs = model(**tensor_inputs)
    post = processor.image_processor.post_process_instance_segmentation(
        outputs,
        threshold=config.score_threshold,
        target_sizes=target_sizes,
    )
    batch_out: List[List[Tuple[BBox, float]]] = []
    for result in post:
        boxes_t = result["boxes"]
        scores_t = result["scores"]
        items: List[Tuple[BBox, float]] = []
        for i in range(int(boxes_t.shape[0])):
            row = boxes_t[i].detach().float().cpu().tolist()
            score = float(scores_t[i].detach().float().cpu())
            items.append(
                (
                    BBox(x1=float(row[0]), y1=float(row[1]), x2=float(row[2]), y2=float(row[3])),
                    score,
                )
            )
        batch_out.append(items)
    return batch_out


def match_exemplar_on_page_with_sam3(
    page_rgb: np.ndarray,
    exemplar_rgb: np.ndarray,
    config: Sam3EngineConfig,
    model: object,
    processor: object,
    progress_cb: Optional[callable] = None,
) -> List[MatchHit]:
    """Cross-page SAM 3 matching for one page using composite tiles."""
    import torch

    ph, pw = page_rgb.shape[:2]
    work_scale = _uniform_work_scale(ph, pw, config.max_page_infer_side)
    page_work = _resize_rgb_uniform(page_rgb, work_scale)
    exemplar_work = _resize_rgb_uniform(exemplar_rgb, work_scale)
    phw, pww = page_work.shape[:2]

    # Cap the exemplar so it leaves room on the composite for the tile.
    exemplar_display = _resize_rgb_long_side(exemplar_work, config.exemplar_max_side)
    eh, ew = exemplar_display.shape[:2]

    # Tile must fit alongside the exemplar within the composite.
    max_tile_from_composite = (
        config.composite_size - max(ew, eh) - config.gutter_px - 2 * config.exemplar_pad_px
    )
    effective_tile = min(config.tile_size, max_tile_from_composite)
    if effective_tile < 128:
        raise ValueError(
            "exemplar too large for the chosen composite_size; "
            "lower exemplar_max_side or raise composite_size"
        )

    origins = _tile_origins(phw, pww, effective_tile, min(config.tile_overlap, effective_tile - 1))
    # Filter blank tiles before counting batches (progress is meaningful).
    filtered_origins: List[Tuple[int, int]] = []
    for ox, oy in origins:
        tile = page_work[oy : oy + effective_tile, ox : ox + effective_tile]
        if tile.shape[0] < 4 or tile.shape[1] < 4:
            continue
        if config.skip_blank_tiles and _tile_is_blank(
            tile, config.blank_tile_max_mean, config.blank_tile_max_std
        ):
            continue
        filtered_origins.append((ox, oy))

    all_boxes: List[BBox] = []
    all_scores: List[float] = []

    batch_size = max(1, int(config.batch_size))
    n_batches = (len(filtered_origins) + batch_size - 1) // batch_size
    inv_work = 1.0 / work_scale

    for b_idx, start in enumerate(range(0, len(filtered_origins), batch_size)):
        chunk = filtered_origins[start : start + batch_size]
        composites: List[np.ndarray] = []
        ex_bboxes: List[BBox] = []
        tile_origins_chunk: List[Tuple[int, int]] = []
        for (tx, ty) in chunk:
            tile = page_work[ty : ty + effective_tile, tx : tx + effective_tile]
            composite, ex_bbox = _build_composite(tile, exemplar_display, config)
            composites.append(composite)
            ex_bboxes.append(ex_bbox)
            tile_origins_chunk.append((tx, ty))

        if not composites:
            continue

        if progress_cb is not None:
            progress_cb(b_idx + 1, n_batches, len(filtered_origins))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_detections = _detect_on_composite_batch(
            composites, ex_bboxes, model, processor, config
        )

        for detections, (tx, ty), ex_bbox in zip(batch_detections, tile_origins_chunk, ex_bboxes):
            for det_bbox, score in detections:
                if det_bbox.iou(ex_bbox) > 0.10:
                    continue
                if det_bbox.x2 < 0 or det_bbox.y2 < 0:
                    continue
                if det_bbox.x1 > effective_tile or det_bbox.y1 > effective_tile:
                    continue
                cx1 = float(max(0.0, det_bbox.x1))
                cy1 = float(max(0.0, det_bbox.y1))
                cx2 = float(min(float(effective_tile), det_bbox.x2))
                cy2 = float(min(float(effective_tile), det_bbox.y2))
                if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                    continue
                # Map from work pixels back to full-resolution page pixels.
                page_box = BBox(
                    x1=(cx1 + float(tx)) * inv_work,
                    y1=(cy1 + float(ty)) * inv_work,
                    x2=(cx2 + float(tx)) * inv_work,
                    y2=(cy2 + float(ty)) * inv_work,
                )
                all_boxes.append(page_box)
                all_scores.append(score)

    keep = _nms(all_boxes, all_scores, config.nms_iou)
    keep = keep[: config.max_hits_per_page]
    return [
        MatchHit(page_id="", bbox=all_boxes[i], score=all_scores[i], source="sam3:composite-tile")
        for i in keep
    ]


def load_sam3_engine(
    model_id: str,
    hf_token: Optional[str],
) -> Tuple[object, object]:
    """Resolve token and return the cached (model, processor) bundle."""
    return load_sam3_bundle(model_id, resolve_hf_token(hf_token))
