"""Lazy pipeline exports keep Wan and Diffusers Hunyuan dependencies isolated."""

from __future__ import annotations


__all__ = [
    "CausalTokenMergingWanPipeline",
    "TokenMergingWanPipeline",
    "load_hunyuan_token_merging_pipeline",
]


def __getattr__(name):
    if name == "CausalTokenMergingWanPipeline":
        from .causal_token_merging_wan import CausalTokenMergingWanPipeline

        return CausalTokenMergingWanPipeline
    if name == "TokenMergingWanPipeline":
        from .token_merging_wan import TokenMergingWanPipeline

        return TokenMergingWanPipeline
    if name == "load_hunyuan_token_merging_pipeline":
        from .token_merging_hunyuan import load_hunyuan_token_merging_pipeline

        return load_hunyuan_token_merging_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
