"""Orchestrator: render PDF, refine exemplar, match, export."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from symbol_matching.matcher import (
    MatcherConfig,
    build_template_bank,
    match_exemplar_on_page,
)
from symbol_matching.models import (
    BBox,
    CaptureExport,
    DrawingItemExport,
    MatchHit,
    PageRecord,
    RunExport,
)
from symbol_matching.pdf import RenderedPage
from symbol_matching.region_proposal import (
    RegionProposalConfig,
    load_region_model,
    resolve_page_regions,
)
from symbol_matching.tiling import full_page_bbox
from symbol_matching.viz import (
    crop_rgb,
    draw_hits_on_page,
    draw_region_proposals_on_page,
    save_png,
)

ENGINE_TEMPLATE = "template"
ENGINE_TEMPLATE_DINO = "template+dino"
ENGINE_SAM3 = "sam3"
ALL_ENGINES = (ENGINE_TEMPLATE, ENGINE_TEMPLATE_DINO, ENGINE_SAM3)


@dataclass(frozen=True)
class RunArtifacts:
    """Files written by a single run."""

    export_path: Path
    crops_dir: Path
    overlays_dir: Path
    region_overlays_dir: Optional[Path]


def _clamp_bbox(bbox: BBox, page: PageRecord) -> BBox:
    x1 = float(max(0.0, min(bbox.x1, page.width - 1.0)))
    y1 = float(max(0.0, min(bbox.y1, page.height - 1.0)))
    x2 = float(max(x1 + 1.0, min(bbox.x2, float(page.width))))
    y2 = float(max(y1 + 1.0, min(bbox.y2, float(page.height))))
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _hits_to_drawing_items(
    page_hits: Sequence[Tuple[PageRecord, MatchHit, Path]],
) -> List[DrawingItemExport]:
    items: List[DrawingItemExport] = []
    for page, hit, crop_path in page_hits:
        item_id = f"item-{uuid.uuid4()}"
        cap = CaptureExport(
            id=f"cap-{uuid.uuid4()}",
            page_id=page.id,
            bbox_xyxy=[hit.bbox.x1, hit.bbox.y1, hit.bbox.x2, hit.bbox.y2],
            crop_path=str(crop_path.resolve()),
            score=hit.score,
            dino_cosine=hit.dino_cosine,
        )
        items.append(
            DrawingItemExport(
                id=item_id,
                page_id=page.id,
                page_name=page.page_name,
                sheet_ref=page.sheet_ref,
                page_type=page.page_type,
                bbox_xyxy=[hit.bbox.x1, hit.bbox.y1, hit.bbox.x2, hit.bbox.y2],
                score=hit.score,
                source=hit.source,
                captures=[cap],
                template_score=hit.template_score,
                dino_cosine=hit.dino_cosine,
            )
        )
    return items


def _persist_page_hits(
    page: PageRecord,
    page_rgb: np.ndarray,
    raw_hits: List[MatchHit],
    crops_dir: Path,
    overlays_dir: Path,
) -> Tuple[List[MatchHit], List[Tuple[PageRecord, MatchHit, Path]]]:
    enriched: List[Tuple[PageRecord, MatchHit, Path]] = []
    final_hits: List[MatchHit] = []
    for raw in raw_hits:
        crop_path = crops_dir / f"{page.id}_{uuid.uuid4().hex[:12]}.png"
        save_png(crop_rgb(page_rgb, raw.bbox), crop_path)
        hit = MatchHit(
            page_id=page.id,
            bbox=raw.bbox,
            score=raw.score,
            source=raw.source,
            crop_path=str(crop_path.resolve()),
            template_score=raw.template_score,
            dino_cosine=raw.dino_cosine,
        )
        final_hits.append(hit)
        enriched.append((page, hit, crop_path))
    overlay = draw_hits_on_page(page_rgb, [h for _, h, _ in enriched])
    save_png(overlay, overlays_dir / f"{page.id}.png")
    return final_hits, enriched


def run_matching(
    rendered: List[RenderedPage],
    reference_page_id: str,
    user_bbox: BBox,
    searched_pages: List[PageRecord],
    matcher_config: MatcherConfig,
    output_dir: Path,
    scope_label: str,
    refined_bbox: Optional[BBox] = None,
    engine: str = ENGINE_TEMPLATE,
    sam3_engine_config: Optional[object] = None,
    sam3_model_id: str = "facebook/sam3",
    sam3_hf_token: Optional[str] = None,
    dino_rerank_config: Optional[object] = None,
    dino_hf_token: Optional[str] = None,
    region_config: Optional[RegionProposalConfig] = None,
    progress_cb: Optional[callable] = None,
) -> Tuple[List[MatchHit], RunExport, RunArtifacts]:
    """Run end-to-end matching and write JSON + crops + annotated overlays.

    ``engine`` selects the matcher: ``template`` (OpenCV only), ``template+dino``
    (template proposals then DINOv3 cosine filter; hit ``score`` is DINO cosine),
    or ``sam3`` (composite-tile SAM 3). For ``sam3`` pass ``sam3_engine_config``;
    for ``template+dino`` pass ``dino_rerank_config``.
    """
    if engine not in ALL_ENGINES:
        raise ValueError(f"unsupported engine: {engine}; choose from {ALL_ENGINES}")

    page_index: Dict[str, RenderedPage] = {p.record.id: p for p in rendered}
    if reference_page_id not in page_index:
        raise ValueError(f"reference page id not in rendered set: {reference_page_id}")

    ref_page = page_index[reference_page_id]
    exemplar_bbox = _clamp_bbox(refined_bbox or user_bbox, ref_page.record)
    exemplar_crop = crop_rgb(ref_page.image_rgb, exemplar_bbox)

    crops_dir = output_dir / "crops"
    overlays_dir = output_dir / "overlays"
    crops_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    region_overlays_dir: Optional[Path] = None
    if region_config is not None and region_config.enabled:
        region_overlays_dir = output_dir / "region_overlays"
        region_overlays_dir.mkdir(parents=True, exist_ok=True)

    all_hits: List[MatchHit] = []
    enriched: List[Tuple[PageRecord, MatchHit, Path]] = []
    model_path: str

    r_cfg = region_config if region_config is not None else RegionProposalConfig(enabled=False)
    region_detector: Optional[object] = None
    if r_cfg.enabled:
        region_detector = load_region_model(r_cfg)
        model_path_suffix = f"+onnx-region:{r_cfg.onnx_path.name}"
    else:
        model_path_suffix = ""

    def _page_regions(page_rgb: np.ndarray) -> Tuple[List[Tuple[BBox, float]], List[BBox]]:
        if not r_cfg.enabled:
            page_h, page_w = page_rgb.shape[:2]
            return [], [full_page_bbox(page_w, page_h)]
        return resolve_page_regions(page_rgb, r_cfg, region_detector)

    def _save_region_overlay(page: PageRecord, page_rgb: np.ndarray) -> None:
        if region_overlays_dir is None:
            return
        scored, search_rois = _page_regions(page_rgb)
        overlay = draw_region_proposals_on_page(page_rgb, scored, search_rois)
        save_png(overlay, region_overlays_dir / f"{page.id}_regions.png")

    if engine == ENGINE_TEMPLATE:
        template_bank = build_template_bank(exemplar_crop, matcher_config)
        model_path = f"opencv:matchTemplate:binary-ink{model_path_suffix}"
        for page in searched_pages:
            if page.id not in page_index:
                continue
            page_rgb = page_index[page.id].image_rgb
            _save_region_overlay(page, page_rgb)
            _, search_rois = _page_regions(page_rgb)
            if progress_cb is not None and r_cfg.enabled:
                progress_cb(f"template: page {page.id} — ONNX region ROI + tiled search")
            raw_hits = match_exemplar_on_page(
                page_rgb, template_bank, matcher_config, search_rois=search_rois
            )
            page_hits, page_enriched = _persist_page_hits(
                page, page_rgb, raw_hits, crops_dir, overlays_dir
            )
            all_hits.extend(page_hits)
            enriched.extend(page_enriched)
    elif engine == ENGINE_TEMPLATE_DINO:
        from symbol_matching.dinov3_rerank import (
            DinoRerankConfig,
            load_dinov3_bundle,
            rerank_template_hits_on_page,
        )

        template_bank = build_template_bank(exemplar_crop, matcher_config)
        d_cfg = dino_rerank_config if dino_rerank_config is not None else DinoRerankConfig()
        dino_model, dino_processor = load_dinov3_bundle(d_cfg.model_id, dino_hf_token)
        model_path = f"opencv:matchTemplate+{d_cfg.model_id}:cosine-rerank{model_path_suffix}"
        for page in searched_pages:
            if page.id not in page_index:
                continue
            page_rgb = page_index[page.id].image_rgb
            _save_region_overlay(page, page_rgb)
            _, search_rois = _page_regions(page_rgb)
            if progress_cb is not None:
                progress_cb(
                    f"template+dino: page {page.id} ({page.sheet_ref}) — "
                    f"{'ONNX ROI + ' if r_cfg.enabled else ''}tiled template pass…"
                )
            raw_template = match_exemplar_on_page(
                page_rgb, template_bank, matcher_config, search_rois=search_rois
            )
            if progress_cb is not None:
                progress_cb(
                    f"  {page.id}: {len(raw_template)} template hit(s) → DINOv3 rerank…"
                )
            raw_reranked = rerank_template_hits_on_page(
                page_rgb,
                exemplar_crop,
                raw_template,
                dino_model,
                dino_processor,
                d_cfg,
            )
            raw_reranked = raw_reranked[: matcher_config.max_hits_per_page]
            page_hits, page_enriched = _persist_page_hits(
                page, page_rgb, raw_reranked, crops_dir, overlays_dir
            )
            all_hits.extend(page_hits)
            enriched.extend(page_enriched)
    elif engine == ENGINE_SAM3:
        from symbol_matching.sam3_engine import (
            Sam3EngineConfig,
            load_sam3_engine,
            match_exemplar_on_page_with_sam3,
        )

        if sam3_engine_config is None:
            sam3_engine_config = Sam3EngineConfig()
        model, processor = load_sam3_engine(sam3_model_id, sam3_hf_token)
        model_path = (
            f"huggingface:{sam3_model_id}:composite-tile"
            f"+fp16={sam3_engine_config.use_fp16}{model_path_suffix}"
        )
        for page in searched_pages:
            if page.id not in page_index:
                continue
            page_rgb = page_index[page.id].image_rgb
            _save_region_overlay(page, page_rgb)
            _, search_rois = _page_regions(page_rgb)
            if progress_cb is not None:
                progress_cb(
                    f"sam3: page {page.id} ({page.sheet_ref}) — "
                    f"work_side≤{sam3_engine_config.max_page_infer_side}, "
                    f"batch={sam3_engine_config.batch_size}"
                    f"{', ONNX ROI' if r_cfg.enabled else ''}"
                )
            raw_hits = match_exemplar_on_page_with_sam3(
                page_rgb=page_rgb,
                exemplar_rgb=exemplar_crop,
                config=sam3_engine_config,
                model=model,
                processor=processor,
                search_rois=search_rois,
                progress_cb=(
                    (
                        lambda b_done, b_total, n_tiles, pid=page.id: progress_cb(
                            f"  {pid}: SAM3 batch {b_done}/{b_total} ({n_tiles} non-blank tiles)"
                        )
                    )
                    if progress_cb is not None else None
                ),
            )
            page_hits, page_enriched = _persist_page_hits(
                page, page_rgb, raw_hits, crops_dir, overlays_dir
            )
            all_hits.extend(page_hits)
            enriched.extend(page_enriched)
    else:
        raise ValueError(f"unsupported engine: {engine}")

    export = RunExport(
        reference_page_id=ref_page.record.id,
        reference_bbox_xyxy=[exemplar_bbox.x1, exemplar_bbox.y1, exemplar_bbox.x2, exemplar_bbox.y2],
        scope=scope_label,
        searched_page_ids=[p.id for p in searched_pages],
        model_path=model_path,
        hits=_hits_to_drawing_items(enriched),
    )

    export_path = output_dir / "symbol_match_export.json"
    export_path.write_text(export.model_dump_json(indent=2), encoding="utf-8")

    artifacts = RunArtifacts(
        export_path=export_path,
        crops_dir=crops_dir,
        overlays_dir=overlays_dir,
        region_overlays_dir=region_overlays_dir,
    )
    return all_hits, export, artifacts
