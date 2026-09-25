"""Experimental SparsePR adapter for native Wan2.1 self-attention."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


_SPARSEPR_CONFIG_SIGNATURE: str | None = None


def _initialize_sparsepr(options: dict[str, Any]):
    global _SPARSEPR_CONFIG_SIGNATURE

    from sparsepr.adapters import wan22

    signature = json.dumps(options, sort_keys=True)
    if signature == _SPARSEPR_CONFIG_SIGNATURE:
        return wan22

    from sparsepr.adapters.wan22 import Wan22TI2VSparseConfig
    from sparsepr.kmeans_utils import (
        set_flashinfer_backend,
        set_flashinfer_workspace_cache_enabled,
    )

    config = Wan22TI2VSparseConfig(
        pattern="N8_custom_v6",
        num_q_centroids=int(options.get("num_q_centroids", 300)),
        num_k_centroids=int(options.get("num_k_centroids", 1000)),
        n8_target_density=float(options.get("target_density", 0.22)),
        n8_probe_rows=int(options.get("probe_rows", 64)),
        n8_repair_rank=int(options.get("repair_rank", 16)),
        n8_role_q_rank=int(options.get("role_q_rank", 64)),
        n8_role_k_rank=int(options.get("role_k_rank", 48)),
        n8_role_v_rank=int(options.get("role_v_rank", 16)),
        n8_metric_sample_tokens=int(options.get("metric_sample_tokens", 2048)),
        kmeans_iter_init=int(options.get("kmeans_iter_init", 25)),
        kmeans_iter_step=int(options.get("kmeans_iter_step", 2)),
        n8_selector_policy=str(options.get("selector_policy", "attention_mass")),
        n8_flashinfer_backend=str(options.get("flashinfer_backend", "auto")),
        n8_compile_role_kv_projection=bool(
            options.get("compile_role_kv_projection", False)
        ),
        logging_file=None,
    )
    wan22._CONFIG = config
    set_flashinfer_backend(config.n8_flashinfer_backend)
    set_flashinfer_workspace_cache_enabled(True)
    _SPARSEPR_CONFIG_SIGNATURE = signature
    return wan22


def _append_record(record: dict[str, Any]) -> None:
    raw_path = os.environ.get("SPARSEPR_WAN21_LOG", "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def sparsepr_wan21_attention(
    owner: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layer_idx: int,
    options: dict[str, Any],
) -> torch.Tensor:
    """Run the official SparsePR N8 core on prepared Wan2.1 Q/K/V tensors.

    Inputs and output use Wan's ``[batch, tokens, heads, head_dim]`` layout.
    The online routing state is owned by each self-attention module.
    """

    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "SparsePR Wan2.1 requires matching Q/K/V shapes; got "
            f"{query.shape}, {key.shape}, and {value.shape}."
        )
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError(
            "SparsePR Wan2.1 currently requires batch-one BLHD tensors; got "
            f"{query.shape}."
        )

    wan22 = _initialize_sparsepr(options)
    call_index = int(getattr(owner, "_sparsepr_call_index", 0))
    reset_every_calls = int(options.get("reset_every_calls", 0))
    if reset_every_calls > 0 and call_index % reset_every_calls == 0:
        owner._n8_cores = {}
        owner._n8_core = None

    owner._svg_cfg_branch = 0
    owner._svg_layer_idx = int(layer_idx)
    owner._svg_generation_step = (
        call_index % reset_every_calls if reset_every_calls > 0 else call_index
    )
    owner._svg_model_name = "wan2.1-t2v-14b-stepdistill"

    source_dtype = value.dtype
    query_bhld = query.transpose(1, 2).contiguous()
    key_bhld = key.transpose(1, 2).contiguous()
    value_bhld = value.transpose(1, 2).contiguous()
    output_bhld, density, timings = wan22._sparse_n8_attention_bhld(
        owner,
        query_bhld,
        key_bhld,
        value_bhld,
        None,
    )
    owner._sparsepr_call_index = call_index + 1

    _append_record(
        {
            "record_type": "sparsepr_wan21_attention",
            "layer": int(layer_idx),
            "call_index": call_index,
            "generation_step": int(owner._svg_generation_step),
            "target_density": float(options.get("target_density", 0.22)),
            "reported_density": float(density),
            "tokens": int(query.shape[1]),
            "heads": int(query.shape[2]),
            "head_dim": int(query.shape[3]),
            **timings,
        }
    )
    return output_bhld.transpose(1, 2).to(source_dtype).contiguous()
