"""Run drawing-region ONNX on Sample_Input PDF pages and save overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

from symbol_matching.pdf import _pixmap_to_rgb
from symbol_matching.region_proposal import (
    RegionProposalConfig,
    default_region_onnx_path,
    load_region_detector,
    preprocess_training_matched,
    resolve_page_regions,
)
from symbol_matching.viz import draw_region_proposals_on_page


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=default_region_onnx_path(),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "Sample_Input"
        / "17180_-_FULL_100_CD_SET_-_With_ADDENDUM_1_(1)_(dragged)_(3).pdf",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "exports"
        / "yolo_region_proposals_train_matched",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def _render_page_rgb(pdf_path: Path, page_index: int, dpi: int) -> tuple[np.ndarray, int, int]:
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        rgb = _pixmap_to_rgb(pix)
    return rgb, int(rgb.shape[1]), int(rgb.shape[0])


def main() -> None:
    args = _parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {args.onnx}. Run: python scripts/export_region_onnx.py"
        )
    if not args.pdf.is_file():
        raise FileNotFoundError(f"pdf not found: {args.pdf}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = RegionProposalConfig(enabled=True, onnx_path=args.onnx, conf=args.conf)
    detector = load_region_detector(cfg)

    with fitz.open(args.pdf) as doc:
        page_count = min(doc.page_count, args.max_pages)

    summary_lines: list[str] = [f"onnx={args.onnx} conf={args.conf} dpi={args.dpi}"]
    for page_index in range(page_count):
        page_rgb, page_w, page_h = _render_page_rgb(args.pdf, page_index, args.dpi)
        scored, search_rois = resolve_page_regions(page_rgb, cfg, detector)
        gray640, _, _ = preprocess_training_matched(page_rgb)
        overlay = draw_region_proposals_on_page(page_rgb, scored, search_rois)
        boxes_page = [bbox for bbox, _ in scored]
        stem = f"page_{page_index + 1:03d}"
        Image.fromarray(page_rgb).save(args.out_dir / f"{stem}_input.png")
        Image.fromarray(overlay).save(args.out_dir / f"{stem}_regions.png")
        infer_viz = np.dstack([gray640, gray640, gray640])
        Image.fromarray(infer_viz).save(args.out_dir / f"{stem}_infer_640_gray.png")

        page_area = float(page_w * page_h)
        covered = sum(b.area() for b in boxes_page)
        summary_lines.append(
            f"page {page_index + 1}: {len(boxes_page)} region(s), "
            f"coverage {100.0 * covered / page_area:.1f}% ({page_w}x{page_h})"
        )
        for i, bb in enumerate(boxes_page):
            summary_lines.append(
                f"  [{i}] xyxy=({bb.x1:.0f},{bb.y1:.0f},{bb.x2:.0f},{bb.y2:.0f}) "
                f"area={bb.area() / page_area * 100:.1f}%"
            )

    summary_path = args.out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))
    print(f"Wrote overlays to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
