"""Drawing-region proposals via lightweight ONNX (training-matched preprocess)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from symbol_matching.models import BBox
from symbol_matching.ort_session import create_ort_session
from symbol_matching.tiling import clamp_bbox_to_image, full_page_bbox

TRAIN_IMGSZ = 640
_SRC_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_DIR = _SRC_ROOT / "drawing_region_yolo_model"
_DEFAULT_WEIGHTS = _DEFAULT_MODEL_DIR / "weights.pt"
_DEFAULT_ONNX = _DEFAULT_MODEL_DIR / "weights.onnx"


@dataclass(frozen=True)
class RegionProposalConfig:
    enabled: bool = False
    onnx_path: Path = _DEFAULT_ONNX
    weights_path: Path = _DEFAULT_WEIGHTS
    conf: float = 0.25
    padding_frac: float = 0.02
    min_padding_px: float = 32.0
    merge_nms_iou: float = 0.50
    fallback_full_page: bool = True
    iou_threshold: float = 0.45
    max_detections: int = 50
    ort_device: str = "cuda"


class RegionDetector(Protocol):
    def detect_640(
        self,
        infer_rgb: np.ndarray,
        conf: float,
        iou_threshold: float,
        max_detections: int,
    ) -> List[Tuple[float, float, float, float, float]]:
        ...


def default_region_weights_path() -> Path:
    env = os.environ.get("YOLO_REGION_WEIGHTS")
    if env:
        return Path(env)
    return _DEFAULT_WEIGHTS


def default_region_onnx_path() -> Path:
    env = os.environ.get("YOLO_REGION_ONNX")
    if env:
        return Path(env)
    weights = default_region_weights_path()
    if weights.suffix == ".pt":
        return weights.with_suffix(".onnx")
    return _DEFAULT_ONNX


def preprocess_training_matched(page_rgb: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Grayscale, resize 640x640 stretch (not letterbox).

    Returns a single-channel ``(640, 640)`` uint8 image (not 3-channel RGB).
    """
    page_h, page_w = page_rgb.shape[:2]
    gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
    stretched = cv2.resize(
        gray, (TRAIN_IMGSZ, TRAIN_IMGSZ), interpolation=cv2.INTER_LINEAR
    )
    scale_x = float(TRAIN_IMGSZ) / float(page_w)
    scale_y = float(TRAIN_IMGSZ) / float(page_h)
    return stretched, scale_x, scale_y


def gray640_to_nchw(gray: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Write normalized NCHW into ``out`` (shape ``(1, 3, 640, 640)``), reusing plane 0."""
    if gray.shape != (TRAIN_IMGSZ, TRAIN_IMGSZ):
        raise ValueError(f"expected gray {(TRAIN_IMGSZ, TRAIN_IMGSZ)}, got {gray.shape}")
    if out.shape != (1, 3, TRAIN_IMGSZ, TRAIN_IMGSZ):
        raise ValueError(f"expected out {(1, 3, TRAIN_IMGSZ, TRAIN_IMGSZ)}, got {out.shape}")
    plane = out[0, 0]
    np.multiply(gray, 1.0 / 255.0, out=plane, casting="unsafe")
    out[0, 1][:] = plane
    out[0, 2][:] = plane
    return out


def infer_rgb_to_nchw_float(infer_rgb: np.ndarray) -> np.ndarray:
    """Pack grayscale or RGB uint8 into float NCHW ``(1, 3, H, W)``."""
    if infer_rgb.ndim == 2:
        buf = np.zeros((1, 3, TRAIN_IMGSZ, TRAIN_IMGSZ), dtype=np.float32)
        return gray640_to_nchw(infer_rgb, buf)
    if infer_rgb.ndim != 3 or infer_rgb.shape[2] != 3:
        raise ValueError(f"expected HxW or HxWx3 uint8, got shape {infer_rgb.shape}")
    buf = np.zeros((1, 3, TRAIN_IMGSZ, TRAIN_IMGSZ), dtype=np.float32)
    np.divide(infer_rgb[:, :, 0], 255.0, out=buf[0, 0], casting="unsafe")
    buf[0, 1][:] = buf[0, 0]
    buf[0, 2][:] = buf[0, 0]
    return buf


def map_boxes_stretch_to_page(
    boxes: Sequence[Tuple[float, float, float, float, float]],
    scale_x: float,
    scale_y: float,
) -> List[BBox]:
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError("scale factors must be positive")
    inv_x = 1.0 / scale_x
    inv_y = 1.0 / scale_y
    mapped: List[BBox] = []
    for x1, y1, x2, y2, score in boxes:
        mapped.append(
            BBox(
                x1=float(x1) * inv_x,
                y1=float(y1) * inv_y,
                x2=float(x2) * inv_x,
                y2=float(y2) * inv_y,
            )
        )
    return mapped


def _box_iou(a: BBox, b: BBox) -> float:
    return a.iou(b)


def _nms_xyxy(
    boxes: List[Tuple[float, float, float, float]],
    scores: List[float],
    iou_threshold: float,
) -> List[int]:
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        remaining: List[int] = []
        for j in order:
            ax1, ay1, ax2, ay2 = boxes[i]
            bx1, by1, bx2, by2 = boxes[j]
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            iw = max(0.0, inter_x2 - inter_x1)
            ih = max(0.0, inter_y2 - inter_y1)
            inter = iw * ih
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - inter
            iou = inter / union if union > 0.0 else 0.0
            if iou < iou_threshold:
                remaining.append(j)
        order = remaining
    return keep


def _xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    half_w = w / 2.0
    half_h = h / 2.0
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def _parse_yolo_onnx_output(
    output: np.ndarray,
    conf: float,
    iou_threshold: float,
    max_detections: int,
) -> List[Tuple[float, float, float, float, float]]:
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 3:
        # Ultralytics ONNX: (1, 4+nc, num_anchors) e.g. (1, 5, 8400).
        if arr.shape[2] >= 100 and arr.shape[1] < arr.shape[2]:
            arr = np.transpose(arr, (0, 2, 1))
        preds = arr[0]
    elif arr.ndim == 2:
        preds = arr if arr.shape[0] > arr.shape[1] else arr.T
    else:
        raise ValueError(f"unexpected ONNX output shape: {arr.shape}")

    if preds.shape[1] < 5:
        return []

    cx = preds[:, 0]
    cy = preds[:, 1]
    bw = preds[:, 2]
    bh = preds[:, 3]
    if preds.shape[1] == 5:
        scores_arr = preds[:, 4]
    else:
        scores_arr = np.max(preds[:, 4:], axis=1)

    valid = (scores_arr >= conf) & (bw > 0.0) & (bh > 0.0)
    if not np.any(valid):
        return []

    cx = cx[valid]
    cy = cy[valid]
    bw = bw[valid]
    bh = bh[valid]
    scores_arr = scores_arr[valid]

    half_w = bw * 0.5
    half_h = bh * 0.5
    x1 = np.clip(cx - half_w, 0.0, float(TRAIN_IMGSZ))
    y1 = np.clip(cy - half_h, 0.0, float(TRAIN_IMGSZ))
    x2 = np.clip(cx + half_w, 0.0, float(TRAIN_IMGSZ))
    y2 = np.clip(cy + half_h, 0.0, float(TRAIN_IMGSZ))
    size_ok = (x2 - x1 >= 2.0) & (y2 - y1 >= 2.0)
    if not np.any(size_ok):
        return []

    x1 = x1[size_ok]
    y1 = y1[size_ok]
    x2 = x2[size_ok]
    y2 = y2[size_ok]
    scores_arr = scores_arr[size_ok]

    boxes_xyxy = list(zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist()))
    scores_list = scores_arr.tolist()
    keep = _nms_xyxy(boxes_xyxy, scores_list, iou_threshold)[:max_detections]
    return [
        (boxes_xyxy[i][0], boxes_xyxy[i][1], boxes_xyxy[i][2], boxes_xyxy[i][3], scores_list[i])
        for i in keep
    ]


class OnnxRegionDetector:
    def __init__(self, onnx_path: Path, ort_device: str) -> None:
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"ONNX region model not found: {onnx_path}. "
                f"Run: python scripts/export_region_onnx.py"
            )
        self._session = create_ort_session(onnx_path, ort_device)
        self._input_name = self._session.get_inputs()[0].name
        self.active_providers: List[str] = list(self._session.get_providers())
        self._input_nchw = np.zeros(
            (1, 3, TRAIN_IMGSZ, TRAIN_IMGSZ), dtype=np.float32
        )

    def detect_640(
        self,
        infer_rgb: np.ndarray,
        conf: float,
        iou_threshold: float,
        max_detections: int,
    ) -> List[Tuple[float, float, float, float, float]]:
        if infer_rgb.ndim == 2:
            gray640_to_nchw(infer_rgb, self._input_nchw)
        elif infer_rgb.ndim == 3:
            np.divide(
                infer_rgb[:, :, 0], 255.0, out=self._input_nchw[0, 0], casting="unsafe"
            )
            self._input_nchw[0, 1][:] = self._input_nchw[0, 0]
            self._input_nchw[0, 2][:] = self._input_nchw[0, 0]
        else:
            raise ValueError(f"expected HxW or HxWx3 uint8, got shape {infer_rgb.shape}")
        outputs = self._session.run(None, {self._input_name: self._input_nchw})
        return _parse_yolo_onnx_output(outputs[0], conf, iou_threshold, max_detections)


def load_region_detector(config: RegionProposalConfig) -> RegionDetector:
    onnx_path = config.onnx_path if config.onnx_path.is_file() else default_region_onnx_path()
    return OnnxRegionDetector(onnx_path, config.ort_device)


def load_region_model(config: RegionProposalConfig) -> RegionDetector:
    """Alias used by pipeline (returns ONNX detector, not Ultralytics)."""
    return load_region_detector(config)


def _nms_boxes(boxes: List[BBox], scores: List[float], iou_threshold: float) -> List[int]:
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _box_iou(boxes[i], boxes[j]) < iou_threshold]
    return keep


def union_bbox_with_padding(
    boxes: Sequence[BBox],
    page_w: int,
    page_h: int,
    padding_frac: float,
    min_padding_px: float,
) -> BBox:
    if len(boxes) == 0:
        raise ValueError("union_bbox_with_padding requires at least one box")
    x1 = min(b.x1 for b in boxes)
    y1 = min(b.y1 for b in boxes)
    x2 = max(b.x2 for b in boxes)
    y2 = max(b.y2 for b in boxes)
    pad_x = max(min_padding_px, float(page_w) * padding_frac)
    pad_y = max(min_padding_px, float(page_h) * padding_frac)
    return clamp_bbox_to_image(
        BBox(x1=x1 - pad_x, y1=y1 - pad_y, x2=x2 + pad_x, y2=y2 + pad_y),
        page_w,
        page_h,
    )


def detect_drawing_regions_scored(
    page_rgb: np.ndarray,
    detector: RegionDetector,
    config: RegionProposalConfig,
) -> List[Tuple[BBox, float]]:
    gray640, scale_x, scale_y = preprocess_training_matched(page_rgb)
    raw = detector.detect_640(
        gray640,
        config.conf,
        config.iou_threshold,
        config.max_detections,
    )
    boxes = map_boxes_stretch_to_page(raw, scale_x, scale_y)
    scored = [(boxes[i], float(raw[i][4])) for i in range(len(boxes))]
    if len(scored) <= 1:
        return scored
    box_list = [b for b, _ in scored]
    scores = [s for _, s in scored]
    keep = _nms_boxes(box_list, scores, config.merge_nms_iou)
    return [scored[i] for i in keep]


def detect_drawing_regions(
    page_rgb: np.ndarray,
    detector: RegionDetector,
    config: RegionProposalConfig,
) -> List[BBox]:
    return [bbox for bbox, _ in detect_drawing_regions_scored(page_rgb, detector, config)]


def resolve_page_regions(
    page_rgb: np.ndarray,
    config: RegionProposalConfig,
    detector: Optional[RegionDetector],
) -> Tuple[List[Tuple[BBox, float]], List[BBox]]:
    """Return ONNX detections (bbox, conf) and page-space search ROIs for matching."""
    page_h, page_w = page_rgb.shape[:2]
    full = full_page_bbox(page_w, page_h)
    if not config.enabled:
        return [], [full]
    if detector is None:
        raise ValueError("region proposal enabled but detector is None")
    scored = detect_drawing_regions_scored(page_rgb, detector, config)
    regions = [bbox for bbox, _ in scored]
    if len(regions) == 0:
        if config.fallback_full_page:
            return [], [full]
        return [], []
    union = union_bbox_with_padding(
        regions,
        page_w,
        page_h,
        config.padding_frac,
        config.min_padding_px,
    )
    return scored, [union]


def resolve_page_search_rois(
    page_rgb: np.ndarray,
    config: RegionProposalConfig,
    detector: Optional[RegionDetector],
) -> List[BBox]:
    """Return page-space ROIs to search (single union for template; SAM3 uses same list)."""
    _, search_rois = resolve_page_regions(page_rgb, config, detector)
    return search_rois
