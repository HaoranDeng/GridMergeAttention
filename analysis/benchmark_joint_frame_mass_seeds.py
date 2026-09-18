"""Native joint-frame seed parity and complete 40-head attention benchmark.

Run only on an externally verified exclusive, clean H200. Frozen QKV is read
once. Four selected heads first validate native seeds against the Torch screen;
headline timing always uses all 40 heads. Each timed GCA call recomputes its
selector and includes merge, route and contiguous output, never a cached table.
Compilation, reference construction and context setup are outside CUDA events.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attention.entropy_mass_seed_triton import entropy_mass_seed_rows_triton
from attention.joint_frame_mass_seed_triton import joint_frame_mass_seed_rows_triton
from attention.token_merging import _hard_c4_attention
from evaluate_entropy_seed_oracle import _load_grid
from screen_partition_population_seeds import custom_tables


MODES = {
    "rf10": "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2",
    "mass_original": "entropy_mass_span_c32_p16_g256_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2",
    "joint_c16": "entropy_joint_mass_span_c16_p16_g256_j4_t1_u16_cg16_sq1_qm_fp8score_skv_nl_ml2",
    "joint_c32": "entropy_joint_mass_span_c32_p16_g256_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2",
}
REFERENCE_NAMES = {
    "mass_original": "gs_256",
    "joint_c16": "gs_joint2_c16_256",
    "joint_c32": "gs_joint2_c32_256",
}
PARITY_HEADS = [0, 10, 20, 30]
SCALE = 128 ** -0.5


def _tokens(grid):
    return grid.permute(1, 2, 3, 0, 4).reshape(1, 75600, grid.shape[0], 128).contiguous()


def _gca(q, k, v, mode):
    return _hard_c4_attention(
        q, k, v, h=45, w=80, h_reduce=1, w_reduce=1,
        sm_scale=SCALE, query_outer_chunk=21, head_chunk=8,
        hard_route_backend="hard_c4", exact_intra_frame_attention=False,
        exact_neighbor_window_columns=0, exact_query_rows=0,
        first_frame_full_attention=False, full_attention_anchor_time_count=0,
        full_attention_anchor_row_count=0, full_attention_anchor_scale=1.0,
        seed_mode=mode,
    ).contiguous()


def _dense(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=0.0, is_causal=False, scale=SCALE,
    ).transpose(1, 2).contiguous()


def _context(name):
    return sdpa_kernel(SDPBackend.FLASH_ATTENTION) if name == "sdpa" else nullcontext()


def _byte_equal(left, right):
    return bool(left.shape == right.shape and left.dtype == right.dtype and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
    ))


def _difference(output, reference):
    delta = output.float() - reference.float()
    error = delta.square()
    return {
        "finite": bool(torch.isfinite(output).all()),
        "bitwise_equal": _byte_equal(output, reference),
        "mse": float(error.mean()),
        "max_abs_error": float(delta.abs().max()),
        "relative_l2": float((error.sum() / reference.float().square().sum().clamp_min(1e-20)).sqrt()),
        "frame_mse": error.reshape(1, 21, 45 * 80, -1).mean((0, 2, 3)).tolist(),
    }


def _seed_invariants(table, heads):
    expected = (1, 1, 21, 45, 80, heads, 4)
    if tuple(table.shape) != expected:
        raise AssertionError(f"seed shape {tuple(table.shape)} != {expected}")
    row = torch.arange(45, device=table.device).view(1, 1, 1, 45, 1, 1)
    ordered = table.int().sort(dim=-1).values
    return {
        "shape": list(table.shape), "dtype": str(table.dtype),
        "min_row": int(table.min()), "max_row": int(table.max()),
        "rows_in_range": bool(((table >= 0) & (table < 45)).all()),
        "matched_anchor_first": bool((table[..., 0] == row).all()),
        "distinct_rows": bool((ordered.diff(dim=-1) > 0).all()),
    }


def _seed_agreement(native, reference):
    shared = (native[..., :, None] == reference[..., None, :]).any(-1).float().sum(-1)
    return {
        "bitwise_equal": _byte_equal(native, reference),
        "slot_agreement": float((native == reference).float().mean()),
        "mean_shared_seeds": float(shared.mean()),
        "same_set_fraction": float((shared == 4).float().mean()),
    }


def _native_table(name, query_grid, key_grid):
    if name == "mass_original":
        return entropy_mass_seed_rows_triton(
            query_grid[:, 10:11].contiguous(), key_grid, SCALE,
            candidate_count=32, candidate_geometry_count=45, block_j=4,
            probe_count=16, global_probe_count=256, gs_num_warps=4,
            native_low_precision_dot=False, tf32_low_precision_dot=False,
        )
    count = 16 if name == "joint_c16" else 32
    return joint_frame_mass_seed_rows_triton(
        query_grid[:, [5, 15]].contiguous(), key_grid, SCALE,
        candidate_count=count, candidate_geometry_count=16 if count == 16 else 45,
        block_j=4, probe_count=16, global_probe_count=256,
        gs_num_warps=4, native_low_precision_dot=False,
    )


def _table_output(q, k, v, table):
    # This injection is ONLY used for untimed parity, never headline timing.
    with patch("attention.token_merging.hellinger_span_seed_rows_triton", return_value=table):
        return _gca(q, k, v, MODES["rf10"])


def _dispatch_output(name, q, k, v, query_grid):
    joint = name.startswith("joint_")
    selector = joint_frame_mass_seed_rows_triton if joint else entropy_mass_seed_rows_triton
    symbol = "joint_frame_mass_seed_rows_triton" if joint else "entropy_mass_seed_rows_triton"
    with patch(f"attention.token_merging.{symbol}", wraps=selector) as spy:
        output = _gca(q, k, v, MODES[name])
    if spy.call_count != 1:
        raise AssertionError(f"{name}: expected one full-head selector dispatch, got {spy.call_count}")
    positional, keyword = spy.call_args
    expected_query = query_grid[:, [5, 15]] if joint else query_grid[:, 10:11]
    if not torch.equal(positional[0], expected_query):
        raise AssertionError(f"{name}: production did not select the expected original query frames")
    if keyword.get("native_low_precision_dot", False) or keyword.get("tf32_low_precision_dot", False):
        raise AssertionError("this benchmark requires the original TF32x3 normalizer")
    expected_count = 16 if name == "joint_c16" else 32
    expected_geometry = 16 if name == "joint_c16" else 45
    if keyword.get("candidate_count", 16 if joint else 32) != expected_count:
        raise AssertionError(f"{name}: unexpected candidate count")
    if keyword.get("candidate_geometry_count", 16 if joint else 45) != expected_geometry:
        raise AssertionError(f"{name}: unexpected candidate geometry")
    return output, {
        "selector_calls": spy.call_count, "query_shape": list(positional[0].shape),
        "key_shape": list(positional[1].shape), "original_router_frames_verified": True,
    }


def _parity(host_grids, device, record, save, block_chunk):
    grids = tuple(grid[PARITY_HEADS].to(device=device, dtype=torch.bfloat16) for grid in host_grids)
    q, k, v = tuple(_tokens(grid) for grid in grids)
    query_grid = q.reshape(1, 21, 45, 80, 4, 128)
    key_grid = k.reshape(1, 21, 45, 80, 4, 128)
    tables = custom_tables(grids[0], grids[1], list(REFERENCE_NAMES.values()),
                           chunk=block_chunk, probes=16, router="rf10")
    for name, reference_name in REFERENCE_NAMES.items():
        native = _native_table(name, query_grid, key_grid)
        reference = tables[reference_name]
        invariants = _seed_invariants(native, 4)
        comparison = _seed_agreement(native, reference)
        item = {"native_invariants": invariants, "seed_vs_torch": comparison}
        record["parity_4heads"][name] = item
        item["seed_gate_passed"] = comparison["mean_shared_seeds"] >= 3.98
        save()
        if not all(invariants[key] for key in ("rows_in_range", "matched_anchor_first", "distinct_rows")):
            raise AssertionError(f"{name}: seed invariant failed")
        if not item["seed_gate_passed"]:
            raise AssertionError(f"{name}: native/Torch seed agreement below 3.98/4")
        native_output, dispatch = _dispatch_output(name, q, k, v, query_grid)
        item["dispatch"] = dispatch
        injected_output = _table_output(q, k, v, native)
        item["dispatch_vs_native_table"] = _difference(native_output, injected_output)
        del injected_output
        reference_output = _table_output(q, k, v, reference)
        item["output_vs_torch_table"] = _difference(native_output, reference_output)
        save()
        print("parity_4heads", name, json.dumps(item), flush=True)
        if not item["dispatch_vs_native_table"]["bitwise_equal"]:
            raise AssertionError(f"{name}: dispatched operator differs from explicit native table")
        if not item["output_vs_torch_table"]["finite"]:
            raise AssertionError(f"{name}: nonfinite output")
        del native, reference, native_output, reference_output


def _benchmark(functions, args, record, save, device):
    names = list(functions)
    for name in names:
        with _context(name):
            for _ in range(args.warmup):
                output = functions[name]()
        del output
    torch.cuda.synchronize(device)
    samples = {name: [] for name in names}
    record["timing_orders"] = []
    for repeat in range(args.repeats):
        order = names if repeat % 2 == 0 else names[::-1]
        record["timing_orders"].append(order)
        for name in order:
            # Backend context and event construction are outside the events.
            with _context(name):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    output = functions[name]()
                end.record()
                end.synchronize()
            samples[name].append(start.elapsed_time(end) / args.iterations)
            del output
            record["attention_latency"][name] = {
                "mean_attention_ms": float(np.mean(samples[name])),
                "std_ms": float(np.std(samples[name])), "samples_ms": samples[name],
                "calls_per_sample": args.iterations,
            }
            save()
            print("attention_latency", repeat, name, samples[name][-1], flush=True)
    sdpa = record["attention_latency"]["sdpa"]["mean_attention_ms"]
    rf10 = record["attention_latency"]["rf10"]["mean_attention_ms"]
    mass = record["attention_latency"]["mass_original"]["mean_attention_ms"]
    for item in record["attention_latency"].values():
        item["speedup_vs_sdpa"] = sdpa / item["mean_attention_ms"]
        item["delta_vs_rf10_ms"] = item["mean_attention_ms"] - rf10
        item["delta_vs_mass_original_ms"] = item["mean_attention_ms"] - mass
    save()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reference-block-chunk", type=int, default=21)
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.warmup, args.reference_block_chunk) < 1:
        raise ValueError("iterations, repeats, warmup and reference block chunk must be positive")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    result = {
        "settings": {**vars(args), "qkv": [str(path) for path in args.qkv], "output": str(args.output)},
        "device": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "parity_heads": PARITY_HEADS,
        "timing_heads": list(range(40)), "dtype": "torch.bfloat16", "modes": MODES,
        "scope": "complete selector + merge + route + contiguous output; excludes model projections/RoPE",
        "sdpa_backend": "PyTorch SDPA forced FLASH_ATTENTION",
        "output_layout": "contiguous [1,75600,40,128] for every timed method",
        "normalizer": "original BF16-to-FP32 TF32x3; native/TF32-single optimizations disabled",
        "algorithm": {"rf10_router": [10], "joint_routers": [5, 15], "seed_count": 4,
                      "local_probes": 16, "global_probes": 256, "block_j": 4,
                      "joint_c16_bank": [16, 16], "joint_c32_bank": [32, 45],
                      "joint_projection": "independent_per_frame",
                      "joint_mass": "one_common_frame_candidate_max", "gs_num_warps": 4},
        "records": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    save()
    for path in args.qkv:
        info = path.stat()
        record = {"capture": {"path": str(path.resolve()), "bytes": info.st_size,
                              "mtime_ns": info.st_mtime_ns},
                  "parity_4heads": {}, "output_vs_sdpa_40heads": {}, "attention_latency": {}}
        result["records"][str(path)] = record
        save()
        host_grids = _load_grid(str(path), list(range(40)))
        if any(tuple(grid.shape) != (40, 21, 45, 80, 128) for grid in host_grids):
            raise ValueError("every capture must contain all 40 heads with canonical Wan720p shape")
        _parity(host_grids, device, record, save, args.reference_block_chunk)
        torch.cuda.synchronize(device)
        q, k, v = tuple(_tokens(grid).to(device=device, dtype=torch.bfloat16) for grid in host_grids)
        del host_grids
        functions = {name: partial(_gca, q, k, v, mode) for name, mode in MODES.items()}
        functions["sdpa"] = partial(_dense, q, k, v)
        with _context("sdpa"):
            dense_output = functions["sdpa"]()
        for name in MODES:
            output = functions[name]()
            if tuple(output.shape) != tuple(q.shape) or not output.is_contiguous():
                raise AssertionError(f"{name}: output layout mismatch")
            record["output_vs_sdpa_40heads"][name] = _difference(output, dense_output)
            del output
            save()
            if not record["output_vs_sdpa_40heads"][name]["finite"]:
                raise AssertionError(f"{name}: nonfinite full-head output")
        del dense_output
        quality = record["output_vs_sdpa_40heads"]
        for item in quality.values():
            item["mse_change_vs_rf10"] = item["mse"] / quality["rf10"]["mse"] - 1
            item["mse_change_vs_mass_original"] = item["mse"] / quality["mass_original"]["mse"] - 1
        save()
        _benchmark(functions, args, record, save, device)
        print(str(path), json.dumps(record, indent=2), flush=True)
        del q, k, v, functions
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
