"""Export DINOv3 ViT-S/16 (HF) to ONNX for onnxruntime-gpu inference.

Requires the ``export`` extra (torch, transformers, onnx) and HF_TOKEN for gated
weights. Reads ``HF_TOKEN`` from the environment or repo-root ``.env``.

Usage:
    python scripts/export_dino_onnx.py
    python scripts/export_dino_onnx.py --output src/dinov3_weights/dinov3_vits16.onnx
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from transformers import AutoModel

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_OUTPUT = _REPO_ROOT / "src" / "dinov3_weights" / "dinov3_vits16.onnx"
INPUT_SIZE = 224
DEFAULT_ONNX_OPSET = 18
MAX_LEGACY_EXPORT_OPSET = 18


def load_repo_dotenv() -> None:
    """Load ``.env`` from the repo root into ``os.environ`` (existing vars win)."""
    dotenv_path = _REPO_ROOT / ".env"
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "" or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key == "":
            continue
        if key not in os.environ:
            os.environ[key] = value


def resolve_hf_token() -> str | None:
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env is not None and env.strip() != "":
        return env.strip()
    return None


def validate_export_opset(opset: int) -> int:
    if opset > MAX_LEGACY_EXPORT_OPSET:
        raise ValueError(
            f"opset {opset} is not supported by legacy torch.onnx.export "
            f"(max {MAX_LEGACY_EXPORT_OPSET}). Use --opset {MAX_LEGACY_EXPORT_OPSET} or "
            "a torch.export/dynamo pipeline for higher opsets."
        )
    return opset


class DinoEmbeddingWrapper(nn.Module):
    """Export pooler_output (384-d) for cosine rerank."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values)
        pooled = out.pooler_output
        if pooled is None:
            pooled = out.last_hidden_state[:, 0, :]
        return pooled


def export_dino_onnx(model_id: str, output_path: Path, opset: int) -> Path:
    token = resolve_hf_token()
    kwargs: dict = {}
    if token is not None:
        kwargs["token"] = token

    # Default HF attention is SDPA (faster on GPU). Legacy ONNX trace supports it;
    # graph optimizations run at inference via ort_session.create_ort_session.
    backbone = AutoModel.from_pretrained(model_id, **kwargs)
    backbone.eval()
    wrapper = DinoEmbeddingWrapper(backbone)
    wrapper.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load(str(output_path)))
    return output_path


def main() -> None:
    load_repo_dotenv()
    parser = argparse.ArgumentParser(description="Export DINOv3 ViT-S/16 to ONNX.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_ONNX_OPSET,
        help=f"ONNX opset for legacy export (max {MAX_LEGACY_EXPORT_OPSET}).",
    )
    args = parser.parse_args()

    if resolve_hf_token() is None:
        raise RuntimeError(
            "HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required for gated DINOv3 weights. "
            "Set it in the environment or repo-root .env."
        )

    opset = validate_export_opset(args.opset)
    out = export_dino_onnx(args.model_id, args.output, opset)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
