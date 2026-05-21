"""Drawing-region proposals via lightweight ONNX (training-matched preprocess)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np
from PIL import Image

from symbol_matching.models import BBox
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
    """Roboflow v6: grayscale, resize 640x640 stretch (not letterbox)."""
    page_h, page_w = page_rgb.shape[:2]
    gray = np.asarray(Image.fromarray(page_rgb).convert("L"), dtype=np.uint8)
    gray_pil = Image.fromarray(gray)
    stretched = gray_pil.resize((TRAIN_IMGSZ, TRAIN_IMGSZ), Image.Resampling.BILINEAR)
    infer_rgb = np.stack([np.asarray(stretched, dtype=np.uint8)] * 3, axis=-1)
    scale_x = float(TRAIN_IMGSZ) / float(page_w)
    scale_y = float(TRAIN_IMGSZ) / float(page_h)
    return infer_rgb, scale_x, scale_y


def infer_rgb_to_nchw_float(infer_rgb: np.ndarray) -> np.ndarray:
    chw = np.transpose(infer_rgb.astype(np.float32), (2, 0, 1)) / 255.0
    return np.expand_dims(chw, axis=0)


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
    arr = np.asarray(output)
    if arr.ndim == 3:
        # Ultralytics ONNX: (1, 4+nc, num_anchors) e.g. (1, 5, 8400).
        if arr.shape[2] >= 100 and arr.shape[1] < arr.shape[2]:
            arr = np.transpose(arr, (0, 2, 1))
        preds = arr[0]
    elif arr.ndim == 2:
        preds = arr if arr.shape[0] > arr.shape[1] else arr.T
    else:
        raise ValueError(f"unexpected ONNX output shape: {arr.shape}")

    boxes_xyxy: List[Tuple[float, float, float, float]] = []
    scores: List[float] = []
    for row in preds:
        if row.shape[0] < 5:
            continue
        if row.shape[0] == 5:
            cx, cy, bw, bh, score = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
        else:
            cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            score = float(np.max(row[4:]))
        if score < conf or bw <= 0.0 or bh <= 0.0:
            continue
        x1, y1, x2, y2 = _xywh_to_xyxy(cx, cy, bw, bh)
        x1 = max(0.0, min(float(TRAIN_IMGSZ), x1))
        y1 = max(0.0, min(float(TRAIN_IMGSZ), y1))
        x2 = max(0.0, min(float(TRAIN_IMGSZ), x2))
        y2 = max(0.0, min(float(TRAIN_IMGSZ), y2))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            continue
        boxes_xyxy.append((x1, y1, x2, y2))
        scores.append(score)

    keep = _nms_xyxy(boxes_xyxy, scores, iou_threshold)[:max_detections]
    return [(boxes_xyxy[i][0], boxes_xyxy[i][1], boxes_xyxy[i][2], boxes_xyxy[i][3], scores[i]) for i in keep]


def _preload_ort_cuda_dlls() -> None:
    """Load CUDA/cuDNN DLLs on Windows before creating a CUDA ORT session."""
    if sys.platform != "win32":
        return
    import onnxruntime as ort

    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        preload()


def _ort_session_providers(ort_device: str) -> List[str | Tuple[str, dict]]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    device = ort_device.strip().lower()
    if device in ("cuda", "gpu", "0"):
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. "
                "Install onnxruntime-gpu: uv pip install onnxruntime-gpu"
            )
        _preload_ort_cuda_dlls()
        return [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
    if device == "cpu":
        return ["CPUExecutionProvider"]
    raise ValueError(f"unsupported ort_device: {ort_device!r}; use 'cuda' or 'cpu'")


def _create_ort_session(onnx_path: Path, ort_device: str) -> object:
    import onnxruntime as ort

    providers = _ort_session_providers(ort_device)
    return ort.InferenceSession(str(onnx_path), providers=providers)


class OnnxRegionDetector:
    def __init__(self, onnx_path: Path, ort_device: str) -> None:
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"ONNX region model not found: {onnx_path}. "
                f"Run: python scripts/export_region_onnx.py"
            )
        self._session = _create_ort_session(onnx_path, ort_device)
        self._input_name = self._session.get_inputs()[0].name
        self.active_providers: List[str] = list(self._session.get_providers())

    def detect_640(
        self,
        infer_rgb: np.ndarray,
        conf: float,
        iou_threshold: float,
        max_detections: int,
    ) -> List[Tuple[float, float, float, float, float]]:
        tensor = infer_rgb_to_nchw_float(infer_rgb)
        outputs = self._session.run(None, {self._input_name: tensor})
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
    infer_rgb, scale_x, scale_y = preprocess_training_matched(page_rgb)
    raw = detector.detect_640(
        infer_rgb,
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
