"""SentenceTransformer construction tuned for lower host RAM (threads, dtype, low_cpu_mem_usage)."""

from __future__ import annotations

import os

from sentence_transformers import SentenceTransformer


def apply_torch_thread_env() -> None:
    """Cap BLAS/thread pools; full core count often spikes RAM during attention/linear layers."""
    import torch

    intra = os.environ.get("ATLAS_TORCH_INTRAOP_THREADS")
    inter = os.environ.get("ATLAS_TORCH_INTEROP_THREADS")
    if intra is not None:
        torch.set_num_threads(max(1, int(intra)))
    if inter is not None:
        torch.set_num_interop_threads(max(1, int(inter)))


def make_sentence_transformer(model_id: str) -> SentenceTransformer:
    apply_torch_thread_env()
    import torch

    dtype_s = (os.environ.get("ATLAS_EMBED_TORCH_DTYPE") or "").strip().lower()
    model_kwargs: dict = {"low_cpu_mem_usage": True}
    if dtype_s in ("float16", "fp16"):
        model_kwargs["torch_dtype"] = torch.float16
    elif dtype_s in ("bfloat16", "bf16"):
        model_kwargs["torch_dtype"] = torch.bfloat16

    device = (os.environ.get("ATLAS_EMBED_DEVICE") or "").strip() or None

    return SentenceTransformer(
        model_id,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        device=device,
    )
