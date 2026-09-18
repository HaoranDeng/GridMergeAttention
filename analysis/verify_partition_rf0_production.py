"""Verify the real RF0 partition production port against the measured override.

Two existing captures, all 40 heads and all 21 frames. The first operator
call has NO monkeypatch. A second, observation-only call records the real
selector's unchanged arguments and returned table; a third replays the old
four-warp/owner-logp override. No seed table is substituted into attention.
This is a parity diagnostic, not a latency benchmark or video experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "analysis"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import benchmark_rf0_seed_attention as benchmark

MODE = benchmark.MODES["partition_c16"]
EXPECTED_OPTIONS = {"candidate_count": 16, "candidate_geometry_count": 16,
                    "block_j": 4, "probe_count": 16, "global_probe_count": 256,
                    "gs_num_warps": 4, "reuse_owner_logp": True}
QKV_SHAPE = (1, 75600, 40, 128)
SEED_SHAPE = (1, 1, 21, 45, 80, 40, 4)


def require_production_options(options):
    """Reject the historical eight-warp dispatcher BEFORE adding any override."""
    for key, expected in EXPECTED_OPTIONS.items():
        if key not in options or options[key] != expected:
            raise AssertionError(f"Production port missing/wrong {key}: {options.get(key)!r}; expected {expected!r}")
    if options["reuse_owner_logp"] is not True:
        raise AssertionError("Production must explicitly pass reuse_owner_logp=True")
    if set(options) - set(EXPECTED_OPTIONS) - {"output_dtype"}:
        raise AssertionError("Unexpected selector overrides/workspaces in production dispatch")


def _bitwise_equal(torch, left, right):
    return bool(left.shape == right.shape and left.dtype == right.dtype and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def _attention(token_module, q, k, v):
    return token_module._hard_c4_attention(
        q, k, v, h=45, w=80, h_reduce=1, w_reduce=1, sm_scale=128 ** -0.5,
        query_outer_chunk=21, head_chunk=8, hard_route_backend="hard_c4", seed_mode=MODE,
        exact_intra_frame_attention=False, exact_neighbor_window_columns=0, exact_query_rows=0,
        first_frame_full_attention=False, full_attention_anchor_time_count=0,
        full_attention_anchor_row_count=0, full_attention_anchor_scale=1.0).contiguous()


def _check_output(torch, output):
    if tuple(output.shape) != QKV_SHAPE or output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise AssertionError("Expected contiguous BF16 complete output [1,75600,40,128]")
    if not bool(torch.isfinite(output).all()):
        raise AssertionError("Nonfinite complete attention output")


def _observe(torch, token_module, real_selector, q, k, v, *, apply_old_override):
    calls, tables = [], []
    expected_q = q.reshape(1, 21, 45, 80, 40, 128)[:, 0:1]
    expected_k = k.reshape(1, 21, 45, 80, 40, 128)

    def recording_selector(*inputs, **received):
        if calls:
            raise AssertionError("Expected exactly one complete 40-head selector call")
        # This gate precedes the old override, so an old production checkout
        # cannot pass by having its eight-warps default silently overwritten.
        require_production_options(received)
        if len(inputs) != 3 or inputs[2] != 128 ** -0.5 or received.get("output_dtype") != torch.uint8:
            raise AssertionError("Unexpected selector scale, dtype or positional arguments")
        if inputs[0].shape != expected_q.shape or not torch.equal(inputs[0], expected_q):
            raise AssertionError("Selector must receive the original RF0 query, all 40 heads")
        if inputs[1].shape != expected_k.shape or not torch.equal(inputs[1], expected_k):
            raise AssertionError("Selector must receive unchanged K for all 21 frames and 40 heads")
        effective = received | benchmark.PARTITION_OVERRIDE if apply_old_override else dict(received)
        calls.append({"received": {key: str(value) if key == "output_dtype" else value for key, value in received.items()},
                      "effective": {key: str(value) if key == "output_dtype" else value for key, value in effective.items()},
                      "query_shape": list(inputs[0].shape), "key_shape": list(inputs[1].shape),
                      "original_rf0_query_verified": True, "key_tensor_verified": True})
        table = real_selector(*inputs, **effective)
        if tuple(table.shape) != SEED_SHAPE or table.dtype != torch.uint8:
            raise AssertionError("Unexpected native real-selector C4 table")
        rows = torch.arange(45, device=table.device).view(1, 1, 1, 45, 1, 1)
        if not bool((table < 45).all()) or not bool((table[..., 0] == rows).all()):
            raise AssertionError("Real seed table has invalid rows or lost the fixed row anchor")
        if not bool((table.int().sort(dim=-1).values.diff(dim=-1) > 0).all()):
            raise AssertionError("Real seed table must contain four distinct rows")
        tables.append(table.detach().clone())  # Observation only: return the untouched real table.
        return table

    with patch.object(token_module, "partition_mass_seed_rows_triton", recording_selector):
        output = _attention(token_module, q, k, v)
    if len(calls) != 1 or len(tables) != 1:
        raise AssertionError("Production bypassed the real full-head partition selector")
    _check_output(torch, output)
    return output, tables[0], {"selector_calls": len(calls), "calls": calls,
                               "old_override_applied": apply_old_override}


def run(args, identity):
    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import attention.token_merging as token_module
    from attention.partition_mass_seed_triton import partition_mass_seed_rows_triton
    from evaluate_entropy_seed_oracle import _load_grid

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if token_module.partition_mass_seed_rows_triton is not partition_mass_seed_rows_triton:
        raise AssertionError("Production selector was already wrapped before verification")
    payload = identity | {"partition_mode": MODE, "current_production_port_verified": False,
        "settings": {"qkv": [str(path) for path in args.qkv], "output": str(args.output), "device": str(device)},
        "device": torch.cuda.get_device_name(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "expected_received_production_options": EXPECTED_OPTIONS, "old_measured_override": benchmark.PARTITION_OVERRIDE,
        "scope": "Two captures, complete 40-head/all-21-frame operator. No video; no fake or reused seed table.",
        "comparison": "Unpatched production vs observation-only production vs historical override applied to real selector",
        "sdpa_backend": "External PyTorch SDPA forced FLASH_ATTENTION, same BF16 QKV and output layout",
        "attention_timing": {"mean_attention_ms": None, "mean_sdpa_attention_ms": None, "speedup_vs_sdpa": None,
                             "status": "unavailable: parity-only, includes compilation and table-observation overhead; not a timing run"},
        "purity": "Exclusive single-GPU checking/monitoring is owned by the external wrapper, not this verifier",
        "records": {}, "complete": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save()
    with torch.inference_mode():
        for path in args.qkv:
            info = path.stat()
            record = {"capture": {"path": str(path.resolve()), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns},
                      "current_production_port_verified": False}
            payload["records"][str(path)] = record
            save()
            try:
                host = _load_grid(str(path), list(range(40)))
                record["capture"]["stored_tensor_dtypes"] = [str(grid.dtype) for grid in host]
                record["capture"]["conversion"] = "Same stored tensors converted to BF16 for all paths and SDPA"
                if any(tuple(grid.shape) != (40, 21, 45, 80, 128) for grid in host):
                    raise ValueError("Expected all 40 heads, 21x45x80 spatial grid, D128")
                q, k, v = tuple(grid.permute(1, 2, 3, 0, 4).reshape(QKV_SHAPE).contiguous()
                                .to(device=device, dtype=torch.bfloat16) for grid in host)
                del host
                # No patch whatsoever is active for this first formal call.
                production = _attention(token_module, q, k, v)
                _check_output(torch, production)
                traced, real_table, record["production_dispatch_observation"] = _observe(
                    torch, token_module, partition_mass_seed_rows_triton, q, k, v, apply_old_override=False)
                record["unpatched_vs_observed_output_bitwise_equal"] = _bitwise_equal(torch, production, traced)
                del traced
                override, old_table, record["old_override_dispatch_observation"] = _observe(
                    torch, token_module, partition_mass_seed_rows_triton, q, k, v, apply_old_override=True)
                record["production_vs_override_seed_bitwise_equal"] = _bitwise_equal(torch, real_table, old_table)
                record["production_vs_override_output_bitwise_equal"] = _bitwise_equal(torch, production, override)
                record["output_shape"] = list(production.shape)
                record["output_dtype"] = str(production.dtype)
                record["seed_shape"] = list(real_table.shape)
                save()
                if not all(record[key] for key in ("unpatched_vs_observed_output_bitwise_equal",
                            "production_vs_override_seed_bitwise_equal", "production_vs_override_output_bitwise_equal")):
                    raise AssertionError("Strict bitwise production/observation/old-override parity failed")
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    dense = functional.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0,
                        is_causal=False, scale=128 ** -0.5).transpose(1, 2).contiguous()
                _check_output(torch, dense)
                record["production_vs_sdpa_40heads"] = benchmark.output_metrics(production, dense)
                record["old_override_vs_sdpa_40heads"] = benchmark.output_metrics(override, dense)
                record["current_production_port_verified"] = True
                save()
                print(path.parent.name, json.dumps(record), flush=True)
                del q, k, v, production, override, real_table, old_table, dense
                torch.cuda.empty_cache()
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                save()
                raise
    payload.update(complete=True, current_production_port_verified=True)
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv", nargs=2, type=Path, default=[Path(path) for path in benchmark.DEFAULT_CAPTURES])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing verification")
    if len({str(path.resolve()) for path in args.qkv}) != 2 or not all(path.is_file() for path in args.qkv):
        parser.error("two distinct existing captures are required")
    benchmark.check_capture_environment(os.environ)
    identity = benchmark.provenance()
    identity["source_sha256"]["analysis/verify_partition_rf0_production.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run(args, identity)


if __name__ == "__main__":
    main()
