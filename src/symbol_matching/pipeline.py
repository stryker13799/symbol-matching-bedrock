"""Orchestrator: render PDF, match, export."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from symbol_matching.matcher import (
    MatcherConfig,
    _template_match_page_entry,
    build_template_bank,
    match_exemplar_on_page,
    resolve_parallel_workers,
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
ALL_ENGINES = (ENGINE_TEMPLATE, ENGINE_TEMPLATE_DINO)


@dataclass(frozen=True)
class RunArtifacts:
    """Files written by a single run."""

    export_path: Path
    crops_dir: Path
    overlays_dir: Path
    region_overlays_dir: Path | None


def _clamp_bbox(bbox: BBox, page: PageRecord) -> BBox:
    x1 = float(max(0.0, min(bbox.x1, page.width - 1.0)))
    y1 = float(max(0.0, min(bbox.y1, page.height - 1.0)))
    x2 = float(max(x1 + 1.0, min(bbox.x2, float(page.width))))
    y2 = float(max(y1 + 1.0, min(bbox.y2, float(page.height))))
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _hits_to_drawing_items(
    page_hits: Sequence[tuple[PageRecord, MatchHit, Path]],
) -> list[DrawingItemExport]:
    items: list[DrawingItemExport] = []
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


def _run_template_page_pass(
    page_jobs: list[tuple[PageRecord, np.ndarray, list[BBox]]],
    template_bank: Sequence[tuple[np.ndarray, int, float]],
    match_cfg: MatcherConfig,
    page_match_cfg: MatcherConfig,
    page_workers: int,
    progress_cb: Callable[[str], None] | None,
    progress_label: str,
) -> list[tuple[PageRecord, np.ndarray, list[MatchHit]]]:
    """Template match each page; tile-parallel or page-parallel, not both."""
    bank_list = list(template_bank)
    if page_workers <= 1:
        results: list[tuple[PageRecord, np.ndarray, list[MatchHit]]] = []
        for page, page_rgb, search_rois in page_jobs:
            if progress_cb is not None:
                progress_cb(f"{progress_label}: page {page.id}")
            raw_hits = match_exemplar_on_page(
                page_rgb, bank_list, match_cfg, search_rois=search_rois
            )
            results.append((page, page_rgb, raw_hits))
        return results

    payloads = [
        (page_rgb, bank_list, page_match_cfg, search_rois) for _, page_rgb, search_rois in page_jobs
    ]
    raw_by_index: dict[int, list[MatchHit]] = {}
    with ProcessPoolExecutor(max_workers=page_workers) as pool:
        future_to_idx = {
            pool.submit(_template_match_page_entry, payloads[i]): i for i in range(len(payloads))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            raw_by_index[idx] = future.result()
            if progress_cb is not None:
                page = page_jobs[idx][0]
                progress_cb(f"{progress_label}: finished page {page.id}")
    return [(page_jobs[i][0], page_jobs[i][1], raw_by_index[i]) for i in range(len(page_jobs))]


def _persist_page_hits(
    page: PageRecord,
    page_rgb: np.ndarray,
    raw_hits: list[MatchHit],
    crops_dir: Path,
    overlays_dir: Path,
) -> tuple[list[MatchHit], list[tuple[PageRecord, MatchHit, Path]]]:
    enriched: list[tuple[PageRecord, MatchHit, Path]] = []
    final_hits: list[MatchHit] = []
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
    rendered: list[RenderedPage],
    reference_page_id: str,
    user_bbox: BBox,
    searched_pages: list[PageRecord],
    matcher_config: MatcherConfig,
    output_dir: Path,
    scope_label: str,
    refined_bbox: BBox | None = None,
    engine: str = ENGINE_TEMPLATE,
    dino_rerank_config: object | None = None,
    region_config: RegionProposalConfig | None = None,
    progress_cb: callable | None = None,
    page_workers: int = 1,
) -> tuple[list[MatchHit], RunExport, RunArtifacts]:
    """Run end-to-end matching and write JSON + crops + annotated overlays.

    ``engine`` is ``template`` (OpenCV only) or ``template+dino`` (template proposals
    then DINOv3 ONNX cosine filter). For ``template+dino`` pass ``dino_rerank_config``.
    """
    if engine not in ALL_ENGINES:
        raise ValueError(f"unsupported engine: {engine}; choose from {ALL_ENGINES}")

    tile_workers, effective_page_workers = resolve_parallel_workers(
        matcher_config.tile_workers, page_workers
    )
    match_cfg = matcher_config
    if tile_workers != matcher_config.tile_workers:
        match_cfg = replace(matcher_config, tile_workers=tile_workers)
    page_match_cfg = match_cfg
    if effective_page_workers > 1:
        page_match_cfg = replace(match_cfg, tile_workers=1)

    page_index: dict[str, RenderedPage] = {p.record.id: p for p in rendered}
    if reference_page_id not in page_index:
        raise ValueError(f"reference page id not in rendered set: {reference_page_id}")

    ref_page = page_index[reference_page_id]
    exemplar_bbox = _clamp_bbox(refined_bbox or user_bbox, ref_page.record)
    exemplar_crop = crop_rgb(ref_page.image_rgb, exemplar_bbox)

    crops_dir = output_dir / "crops"
    overlays_dir = output_dir / "overlays"
    crops_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    region_overlays_dir: Path | None = None
    if region_config is not None and region_config.enabled:
        region_overlays_dir = output_dir / "region_overlays"
        region_overlays_dir.mkdir(parents=True, exist_ok=True)

    all_hits: list[MatchHit] = []
    enriched: list[tuple[PageRecord, MatchHit, Path]] = []
    model_path: str

    r_cfg = region_config if region_config is not None else RegionProposalConfig(enabled=False)
    region_detector: object | None = None
    if r_cfg.enabled:
        region_detector = load_region_model(r_cfg)
        model_path_suffix = f"+onnx-region:{r_cfg.onnx_path.name}"
    else:
        model_path_suffix = ""

    region_cache: dict[str, tuple[list[tuple[BBox, float]], list[BBox]]] = {}

    def _page_regions(
        page_id: str, page_rgb: np.ndarray
    ) -> tuple[list[tuple[BBox, float]], list[BBox]]:
        cached = region_cache.get(page_id)
        if cached is not None:
            return cached
        if not r_cfg.enabled:
            page_h, page_w = page_rgb.shape[:2]
            result: tuple[list[tuple[BBox, float]], list[BBox]] = (
                [],
                [full_page_bbox(page_w, page_h)],
            )
        else:
            result = resolve_page_regions(page_rgb, r_cfg, region_detector)
        region_cache[page_id] = result
        return result

    def _save_region_overlay(page: PageRecord, page_rgb: np.ndarray) -> None:
        if region_overlays_dir is None:
            return
        scored, search_rois = _page_regions(page.id, page_rgb)
        overlay = draw_region_proposals_on_page(page_rgb, scored, search_rois)
        save_png(overlay, region_overlays_dir / f"{page.id}_regions.png")

    dino_embedder: object | None = None
    if engine == ENGINE_TEMPLATE_DINO:
        from symbol_matching.dinov3_rerank import DinoRerankConfig, load_dinov3_embedder

        d_cfg = dino_rerank_config if dino_rerank_config is not None else DinoRerankConfig()
        dino_embedder = load_dinov3_embedder(d_cfg.onnx_path, d_cfg.ort_device)
        model_path = f"opencv:matchTemplate+onnx-dinov3:cosine-rerank{model_path_suffix}"

    if engine == ENGINE_TEMPLATE:
        template_bank = build_template_bank(exemplar_crop, match_cfg)
        model_path = f"opencv:matchTemplate:binary-ink{model_path_suffix}"
        page_jobs: list[tuple[PageRecord, np.ndarray, list[BBox]]] = []
        for page in searched_pages:
            if page.id not in page_index:
                continue
            page_rgb = page_index[page.id].image_rgb
            _save_region_overlay(page, page_rgb)
            _, search_rois = _page_regions(page.id, page_rgb)
            page_jobs.append((page, page_rgb, search_rois))
        template_results = _run_template_page_pass(
            page_jobs,
            template_bank,
            match_cfg,
            page_match_cfg,
            effective_page_workers,
            progress_cb,
            "template",
        )
        for page, page_rgb, raw_hits in template_results:
            page_hits, page_enriched = _persist_page_hits(
                page, page_rgb, raw_hits, crops_dir, overlays_dir
            )
            all_hits.extend(page_hits)
            enriched.extend(page_enriched)
    elif engine == ENGINE_TEMPLATE_DINO:
        from symbol_matching.dinov3_rerank import DinoRerankConfig, rerank_template_hits_on_page

        template_bank = build_template_bank(exemplar_crop, match_cfg)
        d_cfg = dino_rerank_config if dino_rerank_config is not None else DinoRerankConfig()
        dino_page_jobs: list[tuple[PageRecord, np.ndarray, list[BBox]]] = []
        for page in searched_pages:
            if page.id not in page_index:
                continue
            page_rgb = page_index[page.id].image_rgb
            _save_region_overlay(page, page_rgb)
            _, search_rois = _page_regions(page.id, page_rgb)
            dino_page_jobs.append((page, page_rgb, search_rois))
        template_results = _run_template_page_pass(
            dino_page_jobs,
            template_bank,
            match_cfg,
            page_match_cfg,
            effective_page_workers,
            progress_cb,
            "template+dino",
        )
        for page, page_rgb, raw_template in template_results:
            if progress_cb is not None:
                progress_cb(f"  {page.id}: {len(raw_template)} template hit(s) → DINOv3 rerank…")
            raw_reranked = rerank_template_hits_on_page(
                page_rgb,
                exemplar_crop,
                raw_template,
                dino_embedder,
                d_cfg,
            )
            raw_reranked = raw_reranked[: match_cfg.max_hits_per_page]
            page_hits, page_enriched = _persist_page_hits(
                page, page_rgb, raw_reranked, crops_dir, overlays_dir
            )
            all_hits.extend(page_hits)
            enriched.extend(page_enriched)
    else:
        raise ValueError(f"unsupported engine: {engine}")

    if region_detector is not None:
        del region_detector
    if dino_embedder is not None:
        del dino_embedder

    export = RunExport(
        reference_page_id=ref_page.record.id,
        reference_bbox_xyxy=[
            exemplar_bbox.x1,
            exemplar_bbox.y1,
            exemplar_bbox.x2,
            exemplar_bbox.y2,
        ],
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
