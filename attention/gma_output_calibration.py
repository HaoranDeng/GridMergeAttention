"""Projection-folded calibration utilities for all-layer GMA.

The optional capture path collects per-layer, per-head sufficient statistics
for the ridge problem

    min_T ||Y_gma T - Y_dense||_F^2 + lambda ||T - I||_F^2.

Only sampled dense-query outputs are evaluated. The original fitted transforms
are folded into the existing output projection.  The optional query-conditioned
variant instead fits

    Y_dense ~= Y_gma T_y + Q T_q + b,

and evaluates the two small per-head transforms before the existing output
projection.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path

import torch
import torch.nn.functional as F


_CAPTURE_STATE: dict[int, dict[str, torch.Tensor | int]] = {}
_TIMESTEP_CAPTURE_STATE: dict[
    int, dict[int, dict[str, torch.Tensor | int]]
] = {}
_LAYER_CALLS: dict[int, int] = {}
_REGISTERED = False


def _capture_path() -> str | None:
    return os.environ.get("GMA_OUTPUT_CALIBRATION_CAPTURE_PATH")


def apply_query_conditioned_gma_calibration(
    approximate: torch.Tensor,
    query: torch.Tensor,
    parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
):
    """Apply regular per-head GMA/query transforms in token-major layout."""

    gma_transform, query_transform, bias = parameters
    calibrated = torch.einsum(
        "bshd,hde->bshe", approximate, gma_transform
    )
    calibrated.add_(
        torch.einsum("bshd,hde->bshe", query, query_transform)
    )
    calibrated.add_(bias[None, None, :, :])
    return calibrated


def _register_flush():
    global _REGISTERED
    if not _REGISTERED:
        atexit.register(flush_gma_output_calibration_capture)
        _REGISTERED = True


@torch.inference_mode()
def maybe_capture_gma_output_calibration(
    approximate: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layer_idx: int,
    timestep: int | None = None,
):
    """Accumulate sampled dense-vs-GMA normal equations when requested."""

    if not _capture_path():
        return
    _register_flush()
    if approximate.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("GMA calibration capture expects matching [B,S,H,D] tensors")
    if approximate.shape != query.shape or query.size(0) != 1:
        raise ValueError("GMA calibration capture currently requires batch size one")

    layer_idx = int(layer_idx)
    call_idx = _LAYER_CALLS.get(layer_idx, 0)
    _LAYER_CALLS[layer_idx] = call_idx + 1
    sample_count = min(
        int(os.environ.get("GMA_OUTPUT_CALIBRATION_SAMPLES", "256")),
        query.size(1),
    )
    dense_query_chunk = int(
        os.environ.get("GMA_OUTPUT_CALIBRATION_DENSE_QUERY_CHUNK", "64")
    )
    base_seed = int(os.environ.get("GMA_OUTPUT_CALIBRATION_SEED", "20260822"))
    generator = torch.Generator(device="cpu").manual_seed(
        base_seed + 1009 * layer_idx + 9176 * call_idx
    )
    indices = torch.randperm(
        query.size(1), generator=generator
    )[:sample_count].sort().values.to(query.device)

    query_h = query.transpose(1, 2)
    key_h = key.transpose(1, 2)
    value_h = value.transpose(1, 2)
    dense_chunks = []
    for start in range(0, sample_count, dense_query_chunk):
        chunk = indices[start : start + dense_query_chunk]
        dense_chunks.append(
            F.scaled_dot_product_attention(
                query_h.index_select(2, chunk),
                key_h,
                value_h,
                scale=query.size(-1) ** -0.5,
            )
        )
    dense = torch.cat(dense_chunks, dim=2).squeeze(0).float()
    gma = approximate.index_select(1, indices).transpose(1, 2).squeeze(0).float()

    by_timestep = os.environ.get("GMA_OUTPUT_CALIBRATION_BY_TIMESTEP") == "1"
    if by_timestep and timestep is None:
        raise ValueError("timestep-conditioned GMA capture requires a timestep")
    query_conditioned = (
        os.environ.get("GMA_OUTPUT_CALIBRATION_QUERY_CONDITIONED") == "1"
    )
    stats_dtype = (
        torch.float32 if by_timestep or query_conditioned else torch.float64
    )
    layer_states = (
        _TIMESTEP_CAPTURE_STATE.setdefault(int(timestep), {})
        if by_timestep
        else _CAPTURE_STATE
    )
    if query_conditioned:
        sampled_query = (
            query.index_select(1, indices).transpose(1, 2).squeeze(0).float()
        )
        features = torch.cat((gma, sampled_query), dim=-1)
        ztz = torch.einsum("hnd,hne->hde", features, features).to(
            stats_dtype
        ).cpu()
        zty = torch.einsum("hnd,hne->hde", features, dense).to(
            stats_dtype
        ).cpu()
        zsum = features.sum(dim=1).to(stats_dtype).cpu()
        ysum = dense.sum(dim=1).to(stats_dtype).cpu()
        state = layer_states.setdefault(
            layer_idx,
            {
                "ztz": torch.zeros_like(ztz),
                "zty": torch.zeros_like(zty),
                "zsum": torch.zeros_like(zsum),
                "ysum": torch.zeros_like(ysum),
                "samples": 0,
                "calls": 0,
            },
        )
        state["ztz"].add_(ztz)
        state["zty"].add_(zty)
        state["zsum"].add_(zsum)
        state["ysum"].add_(ysum)
    else:
        xtx = torch.einsum("hnd,hne->hde", gma, gma).to(stats_dtype).cpu()
        xty = torch.einsum("hnd,hne->hde", gma, dense).to(stats_dtype).cpu()
        state = layer_states.setdefault(
            layer_idx,
            {
                "xtx": torch.zeros_like(xtx),
                "xty": torch.zeros_like(xty),
                "samples": 0,
                "calls": 0,
            },
        )
        state["xtx"].add_(xtx)
        state["xty"].add_(xty)
    state["samples"] = int(state["samples"]) + sample_count
    state["calls"] = int(state["calls"]) + 1


def flush_gma_output_calibration_capture():
    """Persist current sufficient statistics atomically."""

    path_text = _capture_path()
    if not path_text or (not _CAPTURE_STATE and not _TIMESTEP_CAPTURE_STATE):
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    first_state = next(
        iter(
            next(iter(_TIMESTEP_CAPTURE_STATE.values())).values()
            if _TIMESTEP_CAPTURE_STATE
            else _CAPTURE_STATE.values()
        )
    )
    query_conditioned = "ztz" in first_state
    payload = {
        "version": 3 if query_conditioned else (2 if _TIMESTEP_CAPTURE_STATE else 1),
        "objective": (
            "min ||Y_gma T_y + Q T_q + b - Y_dense||^2 + ridge-to-[I,0]"
            if query_conditioned
            else "min ||Y_gma T - Y_dense||^2 + lambda ||T-I||^2"
        ),
        "query_conditioned": query_conditioned,
        "layer_stats": {str(key): value for key, value in _CAPTURE_STATE.items()},
    }
    if _TIMESTEP_CAPTURE_STATE:
        payload["timestep_layer_stats"] = {
            str(timestep): {
                str(layer_idx): state for layer_idx, state in layer_states.items()
            }
            for timestep, layer_states in _TIMESTEP_CAPTURE_STATE.items()
        }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    print(
        "[gma_output_calibration] saved "
        f"{sum(len(states) for states in _TIMESTEP_CAPTURE_STATE.values()) or len(_CAPTURE_STATE)} "
        f"layer statistics to {path}",
        flush=True,
    )
