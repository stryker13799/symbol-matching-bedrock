"""Release cached PyTorch model bundles and CUDA allocator memory."""

from __future__ import annotations

import gc
from typing import Optional


def _flush_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def release_all_gpu_bundles() -> None:
    """Drop SAM3 and DINOv3 caches and encourage GPU memory reclaim."""
    from symbol_matching.dinov3_rerank import release_dinov3_bundle
    from symbol_matching.sam3 import release_sam3_bundle

    release_sam3_bundle()
    release_dinov3_bundle()
    gc.collect()
    _flush_cuda_cache()


def release_gpu_bundles_except(keep: Optional[str]) -> None:
    """Release cached bundles except ``keep`` ('sam3' | 'dino' | None releases all)."""
    if keep == "sam3":
        from symbol_matching.dinov3_rerank import release_dinov3_bundle

        release_dinov3_bundle()
    elif keep == "dino":
        from symbol_matching.sam3 import release_sam3_bundle

        release_sam3_bundle()
    else:
        release_all_gpu_bundles()
        return
    gc.collect()
    _flush_cuda_cache()
