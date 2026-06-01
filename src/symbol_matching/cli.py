"""``symbol-match`` CLI: PDF in, JSON + annotated overlays out."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import click

from symbol_matching.matcher import MatcherConfig, max_parallel_workers
from symbol_matching.models import BBox
from symbol_matching.pdf import render_pdf
from symbol_matching.pipeline import (
    ALL_ENGINES,
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
@click.option("--tile-workers", type=int, default=0, show_default=True,
              help="Template tile process pool size; 0 uses min(4, CPU count − 4).")
@click.option("--page-workers", type=int, default=0, show_default=True,
              help="Template page process pool size; 0 uses min(2, CPU count − 4). "
              "Page parallelism disables tile parallelism.")
@click.option("--scales", "scales_raw", type=str, default="0.85,0.92,1.0,1.08,1.18",
              show_default=True)
@click.option("--rotations", "rotations_raw", type=str, default="rot4", show_default=True,
              help="'rot4' for 0/90/180/270, '0' for none, or 'a,b,c'.")
@click.option("--engine", "engine", type=click.Choice(ALL_ENGINES), default=ENGINE_TEMPLATE_DINO,
              show_default=True, help="template: OpenCV only. template+dino: template then "
              "DINOv3 ONNX cosine rerank (GPU).")
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
@click.option("--dino-onnx", type=click.Path(path_type=Path), default=None,
              help="DINOv3 ONNX path. Default: src/dinov3_weights/dinov3_vits16.onnx")
@click.option("--dino-ort-device", default="cuda", show_default=True,
              help="ONNX Runtime device for DINOv3: cuda or cpu.")
@click.option("--dino-min-cosine", default=0.55, type=float, show_default=True,
              help="Min cosine similarity vs user exemplar to keep a template hit.")
@click.option("--dino-batch", default=32, type=int, show_default=True,
              help="Crops per DINOv3 ONNX forward.")
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
    tile_workers: int,
    page_workers: int,
    scales_raw: str,
    rotations_raw: str,
    engine: str,
    yolo_regions: bool,
    yolo_onnx: Optional[Path],
    yolo_conf: float,
    yolo_padding_frac: float,
    yolo_ort_device: str,
    dino_onnx: Optional[Path],
    dino_ort_device: str,
    dino_min_cosine: float,
    dino_batch: int,
) -> None:
    """Run one-shot symbol matching on a PDF drawing set."""
    user_bbox = _parse_bbox(bbox_raw)
    parallel_cap = max_parallel_workers()
    resolved_tile_workers = (
        min(parallel_cap, tile_workers) if tile_workers > 0 else min(4, parallel_cap)
    )
    resolved_page_workers = (
        min(parallel_cap, page_workers) if page_workers > 0 else min(2, parallel_cap)
    )
    config = MatcherConfig(
        scales=_parse_scales(scales_raw),
        rotations_deg=_parse_rotations(rotations_raw),
        score_threshold=min_score,
        nms_iou=nms_iou,
        max_hits_per_page=max_hits_per_page,
        max_search_side=max_search_side,
        tile_workers=resolved_tile_workers,
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

    dino_engine_cfg = None
    if engine == ENGINE_TEMPLATE_DINO:
        from symbol_matching.dinov3_rerank import DinoRerankConfig, default_dino_onnx_path

        dino_path = dino_onnx if dino_onnx is not None else default_dino_onnx_path()
        dino_engine_cfg = DinoRerankConfig(
            onnx_path=dino_path,
            ort_device=dino_ort_device,
            batch_size=dino_batch,
            min_cosine=dino_min_cosine,
        )
        click.echo(
            f"Engine 'template+dino': DINOv3 ONNX={dino_path}, min_cosine={dino_min_cosine}, "
            f"batch={dino_batch}, ort_device={dino_ort_device}"
        )

    hits, export, artifacts = run_matching(
        rendered=rendered,
        reference_page_id=ref_page_record.id,
        user_bbox=user_bbox,
        searched_pages=searched,
        matcher_config=config,
        output_dir=output_dir,
        scope_label=scope_label,
        engine=engine,
        progress_cb=lambda msg: click.echo(msg),
        dino_rerank_config=dino_engine_cfg,
        region_config=region_cfg,
        page_workers=resolved_page_workers,
    )

    click.echo(f"Done. {len(hits)} match(es) across {len(export.searched_page_ids)} page(s).")
    click.echo(f"  JSON:     {artifacts.export_path}")
    click.echo(f"  Crops:    {artifacts.crops_dir}")
    click.echo(f"  Overlays: {artifacts.overlays_dir}")
    if artifacts.region_overlays_dir is not None:
        click.echo(f"  Regions:  {artifacts.region_overlays_dir}")


if __name__ == "__main__":
    main()
