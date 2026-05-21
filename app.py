"""Streamlit POC: two-stage canvas (overview → zoomed ROI) → symbol matching."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from symbol_matching.matcher import MatcherConfig, max_parallel_workers
from symbol_matching.models import BBox, MatchHit, PageRecord
from symbol_matching.pdf import RenderedPage, render_pdf
from symbol_matching.pipeline import ENGINE_SAM3, ENGINE_TEMPLATE, ENGINE_TEMPLATE_DINO, run_matching
from symbol_matching.scope import (
    SCOPE_ALL_PAGES,
    SCOPE_PAGE_TYPE,
    SCOPE_SIMILAR_NAME,
    SCOPE_THIS_PAGE,
    select_pages_for_scope,
)
from symbol_matching.viz import crop_rgb, draw_hits_on_page

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_PDF = _PROJECT_ROOT / "Sample_Input" / "17180_-_FULL_100_CD_SET_-_With_ADDENDUM_1_(1)_(dragged)_(3).pdf"

_OVERVIEW_MAX_SIDE = 900   # full-page thumbnail for picking ROI
_ZOOM_MAX_SIDE = 900       # zoomed ROI for drawing the exemplar box
_CONTEXT_DISPLAY_PX = 600  # zoomed hit inspector panel width

_SCOPE_LABELS = {
    SCOPE_THIS_PAGE: "This page only",
    SCOPE_SIMILAR_NAME: "Similar page name / plan family",
    SCOPE_PAGE_TYPE: "Same pageType",
    SCOPE_ALL_PAGES: "All discovered pages",
}

_ENGINE_LABELS = {
    ENGINE_TEMPLATE: "Template (OpenCV, fast)",
    ENGINE_TEMPLATE_DINO: "Template + DINOv3 rerank",
    ENGINE_SAM3: "SAM3 composite-tile",
}


@st.cache_data(show_spinner=False)
def _render_pdf_cached(pdf_bytes: bytes, dpi: int, max_pages: int) -> List[RenderedPage]:
    tmp = _PROJECT_ROOT / "exports" / "_streamlit_tmp.pdf"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(pdf_bytes)
    return render_pdf(tmp, dpi=dpi, max_pages=max_pages)


def _preview_scale(h: int, w: int, max_side: int) -> float:
    longest = max(h, w)
    if longest <= max_side:
        return 1.0
    return float(max_side) / float(longest)


def _resize_rgb(rgb: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return rgb
    h, w = rgb.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return np.asarray(
        Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _last_rect_from_canvas(
    canvas_json: Optional[dict],
) -> Optional[Tuple[float, float, float, float]]:
    """Return (left, top, w, h) of the last drawn rect in canvas pixels, or None."""
    if canvas_json is None:
        return None
    rects = [o for o in canvas_json.get("objects", []) if o.get("type") == "rect"]
    if not rects:
        return None
    r = rects[-1]
    w = float(r.get("width", 0.0)) * float(r.get("scaleX", 1.0))
    h = float(r.get("height", 0.0)) * float(r.get("scaleY", 1.0))
    if w <= 0 or h <= 0:
        return None
    return float(r.get("left", 0.0)), float(r.get("top", 0.0)), w, h


def _canvas_rect_to_bbox(
    rect: Tuple[float, float, float, float],
    inv_scale: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> BBox:
    left, top, w, h = rect
    return BBox(
        x1=offset_x + left * inv_scale,
        y1=offset_y + top * inv_scale,
        x2=offset_x + (left + w) * inv_scale,
        y2=offset_y + (top + h) * inv_scale,
    )


def _context_crop_with_box(
    page_rgb: np.ndarray,
    hit: MatchHit,
    display_px: int,
) -> np.ndarray:
    """Return an annotated context crop around a hit, scaled for display."""
    bb = hit.bbox
    pad_x = max(bb.width() * 1.2, 40.0)
    pad_y = max(bb.height() * 1.2, 40.0)
    ctx = BBox(
        x1=max(0.0, bb.x1 - pad_x),
        y1=max(0.0, bb.y1 - pad_y),
        x2=min(float(page_rgb.shape[1]), bb.x2 + pad_x),
        y2=min(float(page_rgb.shape[0]), bb.y2 + pad_y),
    )
    ctx_rgb = crop_rgb(page_rgb, ctx)
    # Adjust the hit bbox to be relative to the context crop before annotating.
    adj = MatchHit(
        page_id=hit.page_id,
        bbox=BBox(
            x1=bb.x1 - ctx.x1,
            y1=bb.y1 - ctx.y1,
            x2=bb.x2 - ctx.x1,
            y2=bb.y2 - ctx.y1,
        ),
        score=hit.score,
        source=hit.source,
    )
    annotated = draw_hits_on_page(ctx_rgb, [adj])
    scale = min(4.0, float(display_px) / max(annotated.shape[1], 1))
    return _resize_rgb(annotated, scale)


def _show_results(state: Dict[str, Any], page_records: List[PageRecord]) -> None:
    hits: List[MatchHit] = state["hits"]
    export = state["export"]
    artifacts = state["artifacts"]
    rendered_map: Dict[str, RenderedPage] = state["rendered_map"]
    page_lookup = {p.id: p for p in page_records}

    st.markdown("---")
    st.success(f"Done — **{len(hits)} match(es)** across **{len(export.searched_page_ids)} page(s)**.")

    if hits and hits[0].dino_cosine is not None:
        st.caption(
            "Table **score** column is DINO cosine (template correlation is **t_score** only for proposals)."
        )

    rows = [
        {
            "score": round(h.score, 3),
            "page": h.page_id,
            "page_name": page_lookup[h.page_id].page_name if h.page_id in page_lookup else "",
            "t_score": round(h.template_score, 3) if h.template_score is not None else None,
            "dino": round(h.dino_cosine, 3) if h.dino_cosine is not None else None,
            "x1": round(h.bbox.x1, 1),
            "y1": round(h.bbox.y1, 1),
            "x2": round(h.bbox.x2, 1),
            "y2": round(h.bbox.y2, 1),
            "source": h.source,
        }
        for h in sorted(hits, key=lambda h: -h.score)
    ]
    st.dataframe(rows, use_container_width=True)

    st.download_button(
        "Download JSON export",
        data=export.model_dump_json(indent=2),
        file_name="symbol_match_export.json",
        mime="application/json",
    )

    # --- Hit Explorer ---
    st.subheader("Hit Explorer")
    st.caption(
        "Select a page to see the annotated overlay, "
        "then browse individual hits with the slider to zoom in."
    )

    hit_page_ids = sorted({h.page_id for h in hits})
    if not hit_page_ids:
        return

    sel_page_id = st.selectbox(
        "Page",
        hit_page_ids,
        format_func=lambda pid: (
            f"{pid} | {page_lookup[pid].sheet_ref or '-'} | {page_lookup[pid].page_name}"
            if pid in page_lookup else pid
        ),
        key="explorer_page",
    )

    if artifacts.region_overlays_dir is not None:
        region_path = artifacts.region_overlays_dir / f"{sel_page_id}_regions.png"
        if region_path.is_file():
            st.image(
                str(region_path),
                caption=f"ONNX region proposals — {sel_page_id} (green=detection, cyan=search ROI)",
                use_container_width=True,
            )

    overlay_path = artifacts.overlays_dir / f"{sel_page_id}.png"
    if overlay_path.is_file():
        st.image(str(overlay_path), caption=f"All hits — {sel_page_id}", use_container_width=True)

    page_hits = sorted(
        [h for h in hits if h.page_id == sel_page_id],
        key=lambda h: -h.score,
    )
    if not page_hits:
        return

    hit_idx = st.slider(
        f"Browse {len(page_hits)} hit(s) on {sel_page_id}  ·  use slider to zoom in",
        min_value=0,
        max_value=len(page_hits) - 1,
        value=0,
        key="explorer_hit_idx",
    )
    selected = page_hits[hit_idx]

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.caption(
            f"**Hit #{hit_idx + 1}/{len(page_hits)}** — "
            f"score `{selected.score:.3f}` · "
            f"bbox ({selected.bbox.x1:.0f}, {selected.bbox.y1:.0f}) → "
            f"({selected.bbox.x2:.0f}, {selected.bbox.y2:.0f})"
        )
        if sel_page_id in rendered_map:
            ctx = _context_crop_with_box(
                rendered_map[sel_page_id].image_rgb, selected, _CONTEXT_DISPLAY_PX
            )
            st.image(ctx, caption="Zoomed context (box + surroundings)", clamp=True)

    with right_col:
        st.caption("Raw hit crop")
        if selected.crop_path and Path(selected.crop_path).is_file():
            raw = np.asarray(Image.open(selected.crop_path), dtype=np.uint8)
            # Scale up small crops so they're easy to inspect.
            s = min(6.0, 300.0 / max(raw.shape[1], 1))
            st.image(_resize_rgb(raw, s), caption=f"Crop — {raw.shape[1]}×{raw.shape[0]} px", clamp=True)
        else:
            st.info("Crop file not found.")


def main() -> None:
    st.set_page_config(page_title="Symbol Matching POC", layout="wide")
    st.title("Symbol Matching POC")

    with st.sidebar:
        st.header("PDF input")
        uploaded = st.file_uploader("Drawing PDF", type=["pdf"])
        dpi = int(st.number_input("Render DPI", min_value=100, max_value=400, value=200, step=25))
        max_pages = int(st.number_input("Max pages", min_value=1, max_value=200, value=20))

        st.header("Matching")
        scope_value = st.selectbox(
            "Scope",
            options=list(_SCOPE_LABELS.keys()),
            format_func=lambda k: _SCOPE_LABELS[k],
            index=list(_SCOPE_LABELS.keys()).index(SCOPE_ALL_PAGES),
        )
        engine = st.selectbox(
            "Engine",
            options=[ENGINE_TEMPLATE, ENGINE_TEMPLATE_DINO, ENGINE_SAM3],
            index=0,
            format_func=lambda e: _ENGINE_LABELS[e],
            help=(
                "Template is fastest. Template+DINO: template threshold proposes boxes; "
                "DINO cosine is the reported score, overlays, and table sort (GPU). "
                "SAM3 is slowest."
            ),
        )
        nms_iou = float(st.slider("NMS IoU", 0.1, 0.9, 0.30, 0.05))
        max_hits = int(st.number_input("Max hits per page", min_value=1, max_value=500, value=200))

        st.header("Drawing region (YOLO)")
        use_yolo_regions = st.checkbox(
            "Limit search to YOLO drawing region (ONNX)",
            value=False,
            help="Lightweight ONNX region detector per page (training-matched preprocess). "
            "Search runs inside the merged ROI; SAM3 still skips near-blank tiles there.",
        )
        yolo_conf = float(st.slider("YOLO region conf", 0.10, 0.90, 0.25, 0.05))
        yolo_ort_device = st.selectbox("Region ONNX device", ["cuda", "cpu"], index=0)

        st.header("Search tiling (all engines)")
        search_tile = int(st.number_input("Tile size (work px)", 256, 1500, 768, 32))
        search_overlap = int(st.number_input("Tile overlap (work px)", 0, 600, 192, 16))
        search_skip_blank = st.checkbox("Skip near-blank tiles", value=True)

        parallel_cap = max_parallel_workers()
        template_tile_workers = 1
        template_page_workers = 1
        if engine in (ENGINE_TEMPLATE, ENGINE_TEMPLATE_DINO):
            st.subheader("Template CPU parallelism")
            template_tile_workers = int(
                st.number_input(
                    "Tile workers",
                    min_value=1,
                    max_value=parallel_cap,
                    value=min(4, parallel_cap),
                    step=1,
                    help="Parallel OpenCV template passes per page (capped at CPU count − 4).",
                )
            )
            template_page_workers = int(
                st.number_input(
                    "Page workers",
                    min_value=1,
                    max_value=parallel_cap,
                    value=min(2, parallel_cap),
                    step=1,
                    help="Parallel pages for template pass; disables tile workers when > 1.",
                )
            )

        if engine == ENGINE_TEMPLATE:
            min_score = float(st.slider("Score threshold", 0.30, 0.95, 0.55, 0.01))
            st.caption(
                "Lower thresholds (e.g. below 0.5) on weak symbols produce many "
                "correlation peaks and run longer; tighten the exemplar box or use "
            )
            use_rot4 = st.checkbox("Search 0/90/180/270 rotations", value=True)
            scale_min = float(st.slider("Scale min", 0.5, 1.0, 0.85, 0.01))
            scale_max = float(st.slider("Scale max", 1.0, 1.5, 1.18, 0.01))
            scale_steps = int(st.number_input("Scale steps", min_value=1, max_value=9, value=5))
        elif engine == ENGINE_TEMPLATE_DINO:
            min_score = float(
                st.slider(
                    "Template proposal threshold",
                    0.30,
                    0.95,
                    0.50,
                    0.01,
                    help="OpenCV matchTemplate — only sourcing; table and overlay colors use DINO cosine.",
                )
            )
            use_rot4 = st.checkbox("Search 0/90/180/270 rotations", value=True)
            scale_min = float(st.slider("Scale min", 0.5, 1.0, 0.85, 0.01))
            scale_max = float(st.slider("Scale max", 1.0, 1.5, 1.18, 0.01))
            scale_steps = int(st.number_input("Scale steps", min_value=1, max_value=9, value=5))
            dino_min_cosine = float(st.slider("DINO min cosine", 0.30, 0.95, 0.55, 0.01))
            dino_batch = int(st.number_input("DINO batch size", 4, 128, 32, 4))
            dino_fp16 = st.checkbox("DINO fp16 on CUDA", value=True)
        else:
            min_score = float(st.slider("SAM3 score threshold", 0.20, 0.90, 0.40, 0.01))
            sam3_exemplar_side = int(st.number_input("SAM3 exemplar max side", 80, 400, 200, 10))
            sam3_fp16 = st.checkbox("Use fp16 on CUDA", value=True)
            sam3_max_page_side = int(
                st.number_input(
                    "SAM3 max page side (work px, 0=native, slow)",
                    min_value=0,
                    max_value=8192,
                    value=3200,
                    step=128,
                    help="Downscale page+exemplar before tiling. Lower = fewer tiles, faster.",
                )
            )
            sam3_batch = int(st.number_input("SAM3 batch size", 1, 16, 8, 1))
            use_rot4 = True
            scale_min, scale_max, scale_steps = 0.85, 1.18, 5

        st.header("Optional SAM3 box refine")
        use_sam3 = st.checkbox(
            "Snap exemplar bbox using SAM3 (reference page only)", value=False
        )

    # --- Load PDF ---
    pdf_bytes: Optional[bytes] = None
    if uploaded is not None:
        pdf_bytes = uploaded.getvalue()
    elif _DEFAULT_PDF.is_file():
        pdf_bytes = _DEFAULT_PDF.read_bytes()
        st.caption(f"Using sample PDF: {_DEFAULT_PDF.name}")
    if pdf_bytes is None:
        st.info("Upload a PDF (or drop one into Sample_Input/) to begin.")
        return

    with st.spinner(f"Rendering PDF at {dpi} DPI..."):
        rendered = _render_pdf_cached(pdf_bytes, dpi=dpi, max_pages=max_pages)
    page_records = [rp.record for rp in rendered]

    ref_idx = st.selectbox(
        "Reference page",
        options=list(range(len(rendered))),
        format_func=lambda i: (
            f"{page_records[i].id} | {page_records[i].sheet_ref or '-'} | {page_records[i].page_name}"
        ),
    )
    ref_page = rendered[ref_idx]
    page_rgb = ref_page.image_rgb
    page_h, page_w = page_rgb.shape[:2]

    st.markdown(
        f"**Page type:** `{ref_page.record.page_type}`  ·  "
        f"**Plan family:** `{ref_page.record.plan_family}`  ·  "
        f"**Native size:** {page_w}×{page_h} px at {dpi} DPI"
    )

    # --- Stage 1: overview — pick ROI ---
    st.markdown("---")
    st.subheader("Step 1 — Pick region of interest")
    st.caption(
        "Draw a blue rectangle around the area containing the symbol. "
        "Loose is fine — you will zoom in next."
    )

    overview_scale = _preview_scale(page_h, page_w, _OVERVIEW_MAX_SIDE)
    overview_img = _resize_rgb(page_rgb, overview_scale)

    overview_canvas = st_canvas(
        fill_color="rgba(0, 100, 255, 0.10)",
        stroke_width=2,
        stroke_color="#0064ff",
        background_image=Image.fromarray(overview_img),
        update_streamlit=True,
        height=overview_img.shape[0],
        width=overview_img.shape[1],
        drawing_mode="rect",
        display_toolbar=True,
        key=f"overview_{ref_page.record.id}_{dpi}",
    )

    roi_rect = _last_rect_from_canvas(overview_canvas.json_data)
    if roi_rect is None:
        st.info("Draw a blue rectangle on the overview to zoom in.")
        # Still show persisted results from a previous run if available.
        if "hits" in st.session_state:
            _show_results(st.session_state, page_records)
        return

    roi_bbox_page = _canvas_rect_to_bbox(roi_rect, 1.0 / overview_scale)
    pad = max(20.0, (roi_bbox_page.width() + roi_bbox_page.height()) * 0.25)
    roi_padded = BBox(
        x1=max(0.0, roi_bbox_page.x1 - pad),
        y1=max(0.0, roi_bbox_page.y1 - pad),
        x2=min(float(page_w), roi_bbox_page.x2 + pad),
        y2=min(float(page_h), roi_bbox_page.y2 + pad),
    )

    # --- Stage 2: zoomed view — draw exemplar box ---
    st.markdown("---")
    st.subheader("Step 2 — Draw the exemplar box")
    st.caption(
        "The selected region is shown enlarged below. "
        "Draw a **red rectangle** tightly around the symbol you want to find."
    )

    roi_crop = crop_rgb(page_rgb, roi_padded)
    zoom_scale = _preview_scale(roi_crop.shape[0], roi_crop.shape[1], _ZOOM_MAX_SIDE)
    zoom_img = _resize_rgb(roi_crop, zoom_scale)

    zoom_key = (
        f"zoom_{ref_page.record.id}_{dpi}_"
        f"{roi_padded.x1:.0f}_{roi_padded.y1:.0f}_{roi_padded.x2:.0f}_{roi_padded.y2:.0f}"
    )

    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.caption(
            f"Zoom crop: {roi_crop.shape[1]}×{roi_crop.shape[0]} page px  "
            f"→ displayed at {zoom_img.shape[1]}×{zoom_img.shape[0]} ({zoom_scale:.2f}×)"
        )
        zoom_canvas = st_canvas(
            fill_color="rgba(220, 0, 0, 0.12)",
            stroke_width=2,
            stroke_color="#cc0000",
            background_image=Image.fromarray(zoom_img),
            update_streamlit=True,
            height=zoom_img.shape[0],
            width=zoom_img.shape[1],
            drawing_mode="rect",
            display_toolbar=True,
            key=zoom_key,
        )

    with right_col:
        ex_rect_preview = _last_rect_from_canvas(zoom_canvas.json_data)
        if ex_rect_preview is not None:
            ex_bbox_page = _canvas_rect_to_bbox(
                ex_rect_preview,
                inv_scale=1.0 / zoom_scale,
                offset_x=roi_padded.x1,
                offset_y=roi_padded.y1,
            )
            exemplar_crop = crop_rgb(page_rgb, ex_bbox_page)
            s = min(6.0, 400.0 / max(exemplar_crop.shape[1], 1))
            st.caption(
                f"Exemplar at page coords: "
                f"({ex_bbox_page.x1:.0f}, {ex_bbox_page.y1:.0f}) → "
                f"({ex_bbox_page.x2:.0f}, {ex_bbox_page.y2:.0f})  "
                f"({ex_bbox_page.width():.0f}×{ex_bbox_page.height():.0f} px)"
            )
            st.image(_resize_rgb(exemplar_crop, s), caption="Exemplar crop preview", clamp=True)
        else:
            st.info("Draw a red rectangle on the zoomed view.")

    ex_rect = _last_rect_from_canvas(zoom_canvas.json_data)
    if ex_rect is None:
        if "hits" in st.session_state:
            _show_results(st.session_state, page_records)
        return

    user_bbox = _canvas_rect_to_bbox(
        ex_rect,
        inv_scale=1.0 / zoom_scale,
        offset_x=roi_padded.x1,
        offset_y=roi_padded.y1,
    )

    searched = select_pages_for_scope(page_records, ref_page.record.id, scope_value)
    st.caption(f"Scope `{scope_value}` will search {len(searched)} page(s).")

    st.markdown("---")
    if not st.button("Run matching", type="primary"):
        if "hits" in st.session_state:
            _show_results(st.session_state, page_records)
        return

    scale_factors = tuple(float(s) for s in np.linspace(scale_min, scale_max, scale_steps))
    template_peak_div = 2 if min_score < 0.5 else 3
    config = MatcherConfig(
        scales=scale_factors,
        rotations_deg=(0, 90, 180, 270) if use_rot4 else (0,),
        score_threshold=min_score,
        nms_iou=nms_iou,
        max_hits_per_page=max_hits,
        peak_size_divisor=template_peak_div,
        tile_size=search_tile,
        tile_overlap=search_overlap,
        skip_blank_tiles=search_skip_blank,
        tile_workers=template_tile_workers,
    )

    refined_bbox: Optional[BBox] = None
    if use_sam3:
        from symbol_matching.sam3 import refine_exemplar_bbox

        with st.spinner("Refining exemplar with SAM3..."):
            refined_bbox = refine_exemplar_bbox(
                page_rgb,
                user_bbox,
                model_id="facebook/sam3",
                hf_token=None,
                score_threshold=0.5,
            )
        if engine != ENGINE_SAM3:
            from symbol_matching.sam3 import release_sam3_bundle

            release_sam3_bundle()

    out_dir = _PROJECT_ROOT / "exports" / "streamlit_run"

    from symbol_matching.region_proposal import (
        RegionProposalConfig,
        default_region_onnx_path,
    )

    region_cfg = RegionProposalConfig(
        enabled=use_yolo_regions,
        onnx_path=default_region_onnx_path(),
        conf=yolo_conf,
        ort_device=yolo_ort_device,
    )

    sam3_engine_cfg = None
    dino_engine_cfg = None
    if engine == ENGINE_TEMPLATE_DINO:
        from symbol_matching.dinov3_rerank import DinoRerankConfig

        dino_engine_cfg = DinoRerankConfig(
            batch_size=dino_batch,
            min_cosine=dino_min_cosine,
            use_fp16=dino_fp16,
        )
    if engine == ENGINE_SAM3:
        from symbol_matching.sam3_engine import Sam3EngineConfig

        sam3_engine_cfg = Sam3EngineConfig(
            tile_size=search_tile,
            tile_overlap=search_overlap,
            composite_size=1008,
            exemplar_max_side=sam3_exemplar_side,
            score_threshold=min_score,
            nms_iou=nms_iou,
            max_hits_per_page=max_hits,
            use_fp16=sam3_fp16,
            batch_size=sam3_batch,
            max_page_infer_side=sam3_max_page_side,
            skip_blank_tiles=search_skip_blank,
        )

    spinner_label = f"Matching across {len(searched)} page(s) with engine '{engine}'..."
    with st.spinner(spinner_label):
        hits, export, artifacts = run_matching(
            rendered=rendered,
            reference_page_id=ref_page.record.id,
            user_bbox=user_bbox,
            searched_pages=searched,
            matcher_config=config,
            output_dir=out_dir,
            scope_label=scope_value,
            refined_bbox=refined_bbox,
            engine=engine,
            sam3_engine_config=sam3_engine_cfg,
            dino_rerank_config=dino_engine_cfg,
            region_config=region_cfg,
            page_workers=template_page_workers,
        )

    st.session_state["hits"] = hits
    st.session_state["export"] = export
    st.session_state["artifacts"] = artifacts
    st.session_state["rendered_map"] = {rp.record.id: rp for rp in rendered}

    _show_results(st.session_state, page_records)


if __name__ == "__main__":
    main()
