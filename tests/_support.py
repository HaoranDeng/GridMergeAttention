"""Shared test helpers and independent reference calculations.

The helpers in this file deliberately avoid importing the future causal
implementation at module-import time.  This lets ``unittest`` collect the
whole suite before the implementation exists and produces focused failures
that explain which contract is still missing.
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest import mock

import torch


def import_module_cpu_safely(module_name: str):
    """Import Wan modules without requiring a GPU merely for a default argument.

    ``wan.modules.t5`` currently evaluates ``torch.cuda.current_device()`` at
    class-definition time. No CUDA operation is needed by these CPU unit
    tests, so temporarily returning device index zero keeps module discovery
    independent of GPU availability. Real model construction is never mocked.
    """

    if torch.cuda.is_available():
        return importlib.import_module(module_name)
    with mock.patch.object(torch.cuda, "current_device", return_value=0):
        return importlib.import_module(module_name)


def require_symbol(module_name: str, symbol_name: str) -> Any:
    """Return a required implementation symbol or fail with a useful message."""

    try:
        module = import_module_cpu_safely(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise AssertionError(
            f"Could not import {module_name!r}. The causal Self-Forcing port "
            "has not provided this module yet."
        ) from exc

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise AssertionError(
            f"{module_name!r} must export {symbol_name!r} for the causal "
            "Self-Forcing token-merging path."
        ) from exc


def gridmerge_one_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    height: int,
    width: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Slow, independent reference for default one-token GridMerge.

    This reference intentionally supports different temporal lengths for Q
    and K/V.  That is the shape used by cached Self-Forcing inference: Q is
    only the current temporal block while K/V contains the complete visible
    prefix.  The reference covers the production defaults ``f_tied=1``,
    ``h_reduce=1``, ``w_reduce=1``, and ``q_init='ith'``.
    """

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise AssertionError("q/k/v must have shape [batch, sequence, heads, dim].")
    if k.shape != v.shape:
        raise AssertionError("k and v must have identical shapes in this reference.")
    if q.size(0) != k.size(0) or q.shape[2:] != k.shape[2:]:
        raise AssertionError("q and k/v batch, head, and head_dim dimensions must match.")

    frame_tokens = height * width
    if q.size(1) % frame_tokens or k.size(1) % frame_tokens:
        raise AssertionError("q and k/v lengths must contain complete spatial frames.")

    batch, _, heads, dim = q.shape
    query_frames = q.size(1) // frame_tokens
    key_frames = k.size(1) // frame_tokens
    q = q.reshape(batch, query_frames, height, width, heads, dim)
    k = k.reshape(batch, key_frames, height, width, heads, dim)
    v = v.reshape(batch, key_frames, height, width, heads, dim)

    if scale is None:
        scale = dim**-0.5
    scale_sqrt = scale**0.5
    q = q * scale_sqrt
    k = k * scale_sqrt

    # For q_init='ith', the query row at spatial row k is the seed used to
    # compress the columns of key row k. Query/key frame counts may differ.
    merge_logits = torch.einsum("bakjhd,bfklhd->bhafkjl", q, k).float()
    shifted_logits = merge_logits - merge_logits.amax(dim=-1, keepdim=True)
    merge_weights = torch.softmax(shifted_logits, dim=-1).to(k.dtype)
    merge_logprob = shifted_logits - torch.logsumexp(
        shifted_logits, dim=-1, keepdim=True
    )

    merged_keys = torch.einsum("bhafkjl,bfklhd->bafjkhd", merge_weights, k)
    merged_values = torch.einsum("bhafkjl,bfklhe->bafjkhe", merge_weights, v)

    routing_bias = (merge_weights * merge_logprob).sum(dim=-1, keepdim=True)
    routing_bias = routing_bias.transpose(-2, -3)
    routing_logits = torch.einsum("bafjkhd,baijhd->bhafjki", merged_keys, q)
    routing_logits = routing_logits - routing_bias

    # Flatten the visible (key frame, key row) pairs, normalize over them,
    # then reconstruct the output for only the current query frames.
    routing_logits = routing_logits.permute(0, 1, 2, 4, 6, 3, 5)
    routing_weights = torch.softmax(
        routing_logits.reshape(
            batch,
            heads,
            query_frames,
            width,
            height,
            key_frames * height,
        ).float(),
        dim=-1,
    ).to(q.dtype)
    routing_weights = routing_weights.reshape(
        batch,
        heads,
        query_frames,
        width,
        height,
        key_frames,
        height,
    ).permute(0, 1, 2, 5, 3, 6, 4)

    output = torch.einsum("bhafjki,bafjkhe->baijhe", routing_weights, merged_values)
    return output.reshape(batch, query_frames * frame_tokens, heads, dim)


def assert_state_dict_schemas_match(
    test_case,
    first: torch.nn.Module,
    second: torch.nn.Module,
) -> None:
    """Assert that two model classes can consume the same strict checkpoint."""

    first_state = first.state_dict()
    second_state = second.state_dict()
    test_case.assertEqual(set(first_state), set(second_state))
    for key in first_state:
        test_case.assertEqual(
            first_state[key].shape,
            second_state[key].shape,
            msg=f"state_dict shape mismatch for {key}",
        )
