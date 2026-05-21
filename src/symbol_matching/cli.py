"""``symbol-match`` CLI: PDF in, JSON + annotated overlays out."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import click

from symbol_matching.matcher import MatcherConfig
from symbol_matching.models import BBox
from symbol_matching.pdf import render_pdf
from symbol_matching.pipeline import (
    ALL_ENGINES,
    ENGINE_SAM3,
    ENGINE_TEMPLATE,
    ENGINE_TEMPLATE_DINO,
    run_matching,
)
from symbol_matching.scope import ALL_SCOPES, SCOPE_ALL_PAGES, select_pages_for_scope


def _parse_bbox(raw: str) -> BBox:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise click.BadParameter("bbox must be 'x1,y1,x2,y2'")
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise click.BadParameter(f"non-numeric bbox value: {exc}") from exc
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise click.BadParameter("bbox must satisfy x2>x1 and y2>y1")
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _parse_scales(raw: str) -> Tuple[float, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if len(parts) == 0:
        raise click.BadParameter("scales must contain at least one value")
    return tuple(float(p) for p in parts)


def _parse_rotations(raw: str) -> Tuple[int, ...]:
    if raw == "rot4":
        return (0, 90, 180, 270)
    if raw in ("0", "none"):
        return (0,)
    parts = [int(p.strip()) for p in raw.split(",") if p.strip() != ""]
    return tuple(parts)


@click.command(name="symbol-match")
@click.option("--pdf", "pdf_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--reference-page", "reference_page", type=int, required=True,
              help="1-indexed page number containing the symbol exemplar.")
@click.option("--bbox", "bbox_raw", type=str, required=True,
              help="Exemplar box in rendered-page pixel coords: 'x1,y1,x2,y2'.")
@click.option("--scope", "scope_label", type=click.Choice(ALL_SCOPES), default=SCOPE_ALL_PAGES,
              show_default=True)
@click.option("--output-dir", "output_dir", type=click.Path(path_type=Path),
              default=Path("exports/cli_run"), show_default=True)
@click.option("--dpi", type=int, default=200, show_default=True)
@click.option("--max-pages", type=int, default=20, show_default=True)
@click.option("--max-search-side", type=int, default=3000, show_default=True)
@click.option("--min-score", type=float, default=0.40, show_default=True,
              help="Lower for higher recall.")
@click.option("--nms-iou", type=float, default=0.30, show_default=True)
@click.option("--max-hits-per-page", type=int, default=50, show_default=True)
@click.option("--scales", "scales_raw", type=str, default="0.85,0.92,1.0,1.08,1.18",
              show_default=True)
@click.option("--rotations", "rotations_raw", type=str, default="rot4", show_default=True,
              help="'rot4' for 0/90/180/270, '0' for none, or 'a,b,c'.")
@click.option("--engine", "engine", type=click.Choice(ALL_ENGINES), default=ENGINE_TEMPLATE_DINO,
              show_default=True, help="template: OpenCV only. template+dino: template then "
              "DINOv3 cosine rerank (GPU recommended). sam3: composite-tile SAM3.")
@click.option("--use-sam3-refine", is_flag=True, default=False,
              help="Use SAM3 on the reference page to snap the box to the symbol.")
@click.option("--sam3-model", default="facebook/sam3", show_default=True)
@click.option("--sam3-score", default=0.4, type=float, show_default=True,
              help="SAM3 detection score threshold (applies to both refine and engine).")
@click.option("--sam3-tile", default=768, type=int, show_default=True,
              help="Tile size in page pixels for the SAM3 engine.")
@click.option("--sam3-overlap", default=192, type=int, show_default=True,
              help="Tile overlap in page pixels for the SAM3 engine.")
@click.option("--sam3-exemplar-side", default=200, type=int, show_default=True,
              help="Exemplar crop is resized so its longest side is at most this many composite pixels.")
@click.option("--sam3-composite", default=1008, type=int, show_default=True,
              help="Composite canvas size; SAM3 native input is 1008.")
@click.option("--sam3-fp16/--sam3-fp32", default=True, show_default=True,
              help="Use float16 autocast on CUDA for the SAM3 engine.")
@click.option("--sam3-max-page-side", default=3200, type=int, show_default=True,
              help="Downscale page+exemplar so longest side is at most this (fewer tiles). "
              "Use 0 to disable (native resolution, very slow on large sheets).")
@click.option("--sam3-batch", default=8, type=int, show_default=True,
              help="Composites per SAM3 forward (raise until OOM on your GPU).")
@click.option("--sam3-no-skip-blank", is_flag=True, default=False,
              help="Run SAM3 on every tile including near-white margins.")
@click.option("--yolo-regions/--no-yolo-regions", default=True, show_default=True,
              help="Restrict search to YOLO 'drawing' region(s) per page (training-matched preprocess).")
@click.option("--yolo-onnx", type=click.Path(path_type=Path), default=None,
              help="Path to drawing-region ONNX model. Default: src/drawing_region_yolo_model/weights.onnx")
@click.option("--yolo-conf", default=0.25, type=float, show_default=True,
              help="Drawing-region detection confidence threshold.")
@click.option("--yolo-padding-frac", default=0.02, type=float, show_default=True,
              help="Pad merged drawing ROI by this fraction of page width/height.")
@click.option("--yolo-ort-device", default="cuda", show_default=True,
              help="ONNX Runtime device for region model: cuda or cpu.")
@click.option("--dino-model", default="facebook/dinov3-vits16-pretrain-lvd1689m", show_default=True,
              help="DINOv3 model id for --engine template+dino.")
@click.option("--dino-min-cosine", default=0.55, type=float, show_default=True,
              help="Min cosine similarity vs user exemplar to keep a template hit.")
@click.option("--dino-batch", default=32, type=int, show_default=True,
              help="Crops per DINOv3 forward.")
@click.option("--dino-fp16/--dino-fp32", default=True, show_default=True,
              help="Use float16 autocast on CUDA for DINOv3.")
@click.option("--hf-token", default=None, type=str,
              help="Hugging Face token; defaults to $HF_TOKEN if unset.")
def main(
    pdf_path: Path,
    reference_page: int,
    bbox_raw: str,
    scope_label: str,
    output_dir: Path,
    dpi: int,
    max_pages: int,
    max_search_side: int,
    min_score: float,
    nms_iou: float,
    max_hits_per_page: int,
    scales_raw: str,
    rotations_raw: str,
    engine: str,
    use_sam3_refine: bool,
    sam3_model: str,
    sam3_score: float,
    sam3_tile: int,
    sam3_overlap: int,
    sam3_exemplar_side: int,
    sam3_composite: int,
    sam3_fp16: bool,
    sam3_max_page_side: int,
    sam3_batch: int,
    sam3_no_skip_blank: bool,
    yolo_regions: bool,
    yolo_onnx: Optional[Path],
    yolo_conf: float,
    yolo_padding_frac: float,
    yolo_ort_device: str,
    dino_model: str,
    dino_min_cosine: float,
    dino_batch: int,
    dino_fp16: bool,
    hf_token: Optional[str],
) -> None:
    """Run one-shot symbol matching on a PDF drawing set."""
    user_bbox = _parse_bbox(bbox_raw)
    config = MatcherConfig(
        scales=_parse_scales(scales_raw),
        rotations_deg=_parse_rotations(rotations_raw),
        score_threshold=min_score,
        nms_iou=nms_iou,
        max_hits_per_page=max_hits_per_page,
        max_search_side=max_search_side,
        tile_size=sam3_tile,
        tile_overlap=sam3_overlap,
        skip_blank_tiles=not sam3_no_skip_blank,
    )

    click.echo(f"Rendering {pdf_path.name} at {dpi} DPI (max {max_pages} pages)...")
    rendered = render_pdf(pdf_path, dpi=dpi, max_pages=max_pages)
    click.echo(f"Rendered {len(rendered)} page(s).")
    for rp in rendered:
        click.echo(
            f"  {rp.record.id}: {rp.record.sheet_ref or '-'} | "
            f"{rp.record.page_name} | type={rp.record.page_type}"
        )

    if reference_page < 1 or reference_page > len(rendered):
        raise click.BadParameter(
            f"reference-page must be between 1 and {len(rendered)} (got {reference_page})"
        )
    ref_page_record = rendered[reference_page - 1].record

    searched = select_pages_for_scope(
        [rp.record for rp in rendered], ref_page_record.id, scope_label
    )
    click.echo(
        f"Scope '{scope_label}' selects {len(searched)} page(s): "
        f"{', '.join(p.id for p in searched)}"
    )

    refined_bbox = None
    if use_sam3_refine:
        from symbol_matching.sam3 import refine_exemplar_bbox

        click.echo("Refining exemplar bbox with SAM3...")
        refined_bbox = refine_exemplar_bbox(
            rendered[reference_page - 1].image_rgb,
            user_bbox,
            model_id=sam3_model,
            hf_token=hf_token,
            score_threshold=sam3_score,
        )
        click.echo(
            f"  refined bbox: ({refined_bbox.x1:.0f},{refined_bbox.y1:.0f}) -> "
            f"({refined_bbox.x2:.0f},{refined_bbox.y2:.0f})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    from symbol_matching.region_proposal import (
        RegionProposalConfig,
        default_region_onnx_path,
    )

    onnx_path = yolo_onnx if yolo_onnx is not None else default_region_onnx_path()
    region_cfg = RegionProposalConfig(
        enabled=yolo_regions,
        onnx_path=onnx_path,
        conf=yolo_conf,
        padding_frac=yolo_padding_frac,
        ort_device=yolo_ort_device,
    )
    if yolo_regions:
        click.echo(
            f"Drawing-region ONNX: {onnx_path}, conf={yolo_conf}, "
            f"padding_frac={yolo_padding_frac}, ort_device={yolo_ort_device}"
        )

    sam3_engine_cfg = None
    dino_engine_cfg = None
    if engine == ENGINE_TEMPLATE_DINO:
        from symbol_matching.dinov3_rerank import DinoRerankConfig

        dino_engine_cfg = DinoRerankConfig(
            model_id=dino_model,
            batch_size=dino_batch,
            min_cosine=dino_min_cosine,
            use_fp16=dino_fp16,
        )
        click.echo(
            f"Engine 'template+dino': DINOv3={dino_model}, min_cosine={dino_min_cosine}, "
            f"batch={dino_batch}, fp16={dino_fp16}"
        )
    if engine == ENGINE_SAM3:
        from symbol_matching.sam3_engine import Sam3EngineConfig

        sam3_engine_cfg = Sam3EngineConfig(
            tile_size=sam3_tile,
            tile_overlap=sam3_overlap,
            composite_size=sam3_composite,
            exemplar_max_side=sam3_exemplar_side,
            score_threshold=sam3_score,
            nms_iou=nms_iou,
            max_hits_per_page=max_hits_per_page,
            use_fp16=sam3_fp16,
            batch_size=sam3_batch,
            max_page_infer_side=sam3_max_page_side,
            skip_blank_tiles=not sam3_no_skip_blank,
        )
        click.echo(
            f"Engine 'sam3': tile={sam3_tile}, overlap={sam3_overlap}, composite={sam3_composite}, "
            f"exemplar_max_side={sam3_exemplar_side}, batch={sam3_batch}, "
            f"max_page_side={sam3_max_page_side or 'native'}, fp16={sam3_fp16}, score>={sam3_score}"
        )

    hits, export, artifacts = run_matching(
        rendered=rendered,
        reference_page_id=ref_page_record.id,
        user_bbox=user_bbox,
        searched_pages=searched,
        matcher_config=config,
        output_dir=output_dir,
        scope_label=scope_label,
        refined_bbox=refined_bbox,
        engine=engine,
        sam3_engine_config=sam3_engine_cfg,
        sam3_model_id=sam3_model,
        sam3_hf_token=hf_token,
        progress_cb=lambda msg: click.echo(msg),
        dino_rerank_config=dino_engine_cfg,
        dino_hf_token=hf_token,
        region_config=region_cfg,
    )

    click.echo(f"Done. {len(hits)} match(es) across {len(export.searched_page_ids)} page(s).")
    click.echo(f"  JSON:     {artifacts.export_path}")
    click.echo(f"  Crops:    {artifacts.crops_dir}")
    click.echo(f"  Overlays: {artifacts.overlays_dir}")
    if artifacts.region_overlays_dir is not None:
        click.echo(f"  Regions:  {artifacts.region_overlays_dir}")


if __name__ == "__main__":
    main()
