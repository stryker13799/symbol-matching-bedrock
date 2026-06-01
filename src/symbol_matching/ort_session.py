"""Shared ONNX Runtime session creation (CUDA preload, graph opts, quiet logs)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import onnxruntime as ort


def preload_ort_cuda_dlls() -> None:
    """Load CUDA/cuDNN DLLs on Windows before creating a CUDA ORT session."""
    if sys.platform != "win32":
        return
    preload_fn = getattr(ort, "preload_dlls", None)
    if preload_fn is not None:
        preload_fn()


def ort_session_providers(ort_device: str) -> List[str | Tuple[str, dict]]:
    available = set(ort.get_available_providers())
    device = ort_device.strip().lower()
    if device in ("cuda", "gpu", "0"):
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. "
                "Install onnxruntime-gpu: uv pip install onnxruntime-gpu"
            )
        preload_ort_cuda_dlls()
        return [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
    if device == "cpu":
        return ["CPUExecutionProvider"]
    raise ValueError(f"unsupported ort_device: {ort_device!r}; use 'cuda' or 'cpu'")


def create_ort_session_options() -> ort.SessionOptions:
    """Graph optimizations at load time; suppress ORT info/warning log spam."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    return options


def create_ort_session(onnx_path: Path, ort_device: str) -> ort.InferenceSession:
    providers = ort_session_providers(ort_device)
    options = create_ort_session_options()
    return ort.InferenceSession(str(onnx_path), options, providers=providers)
