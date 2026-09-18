"""Experimental RF0 first-medoid screen; no production/config changes.

Two frozen captures, all 40 heads, 21 frames. Exact independent four-head
seed-table parity is mandatory before timing; any mismatch stops the run.
Both defaults are L20; they do not cover the known L33/head30 weak point.
The external launcher, not this script, must enforce clean single-GPU use.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import benchmark_rf0_seed_attention as base
from verify_partition_rf0_production import require_production_options

METHODS = {name: base.MODES[name] for name in ("rf0", "partition_c16")}
METHODS["medoid_c16"] = base.MODES["partition_c16"]  # Isolated selector patch, NOT a new production mode.
PARITY_HEADS, PARITY_CHUNK = (0, 10, 20, 30), 256
ITERATIONS, REPEATS, WARMUP = base.ITERATIONS, base.REPEATS, base.WARMUP
EXTRA_SOURCES = ("analysis/benchmark_rf0_medoid_seed_attention.py", "analysis/verify_partition_rf0_production.py",
                 "attention/partition_mass_medoid_seed_triton.py", "analysis/partition_variational_selector.py")


def selector_adapter(selector, observer=None):
    def dispatch(*inputs, **options):
        require_production_options(options)  # Fail BEFORE calling; never repair old eight-warp kwargs.
        table = selector(*inputs, **options)
        if observer is not None:
            observer(table, inputs, options)
        return table  # Actual native result, never a cached/reference replacement.
    return dispatch


def validate_seed_array(table, heads, *, anchored):
    import numpy as np
    expected = (1, 1, 21, 45, 80, heads, 4)
    if table.shape != expected or table.dtype != np.uint8:
        raise AssertionError(f"Expected uint8 native table {expected}")
    if (table >= 45).any() or (np.diff(np.sort(table.astype(np.int16), axis=-1), axis=-1) <= 0).any():
        raise AssertionError("Seed rows must be in range and pairwise distinct")
    matches = table[..., 0] == np.arange(45).reshape(1, 1, 1, 45, 1, 1)
    if anchored and not matches.all():
        raise AssertionError("Existing anchored method lost its row anchor")
    return {"shape": list(table.shape), "dtype": str(table.dtype), "range_and_distinctness_valid": True,
            "anchor_required": anchored, "first_row_anchor_fraction": float(matches.mean()),
            "sha256_raw_uint8": hashlib.sha256(table.tobytes()).hexdigest()}


def parity_counts(native, reference):
    import numpy as np
    if native.shape != reference.shape or native.shape[-1] != 4:
        raise ValueError("Parity arrays must have equal [...,4] shapes")
    return {"blocks": int(np.prod(native.shape[:-1])), "ordered_slot_mismatches": int((native != reference).sum()),
            "ordered_tuple_mismatches": int((native != reference).any(-1).sum()),
            "first_seed_mismatches": int((native[..., 0] != reference[..., 0]).sum()),
            "set_mismatches": int((np.sort(native, axis=-1) != np.sort(reference, axis=-1)).any(-1).sum()),
            "exact_table_equal": bool(np.array_equal(native, reference))}


def block_coordinates(index):
    return index % 4, (index // 4) % 80, (index // 320) % 45, index // 14400


def mismatch_scores(logits, importance, positions, first_cost):
    import torch
    from partition_variational_selector import partition_variational_gap
    result = [{"scores": first_cost.tolist(), "direction": "min",
               "best_second_margin": float(first_cost.sort().values.diff()[0])}]
    for step in range(1, 4):
        gap = partition_variational_gap(logits, positions[:step])
        score = importance * (-torch.expm1(-gap))
        score[positions[:step]] = -torch.inf
        top = score.topk(2).values
        result.append({"scores": [float(v) if torch.isfinite(v) else None for v in score],
                       "direction": "max", "best_second_margin": float(top[0] - top[1])})
    return result


def semantic_parity(q, k, artifact):
    import numpy as np
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    from attention.entropy_mass_seed_triton import global_probe_log_norm_triton
    from attention.partition_mass_medoid_seed_triton import (
        partition_mass_medoid_seed_rows_triton, partition_mass_medoid_seeds_reference)
    q4 = q.reshape(1, 21, 45, 80, 40, 128)[:, :1, :, :, PARITY_HEADS, :].contiguous()
    k4 = k.reshape(1, 21, 45, 80, 40, 128)[:, :, :, :, PARITY_HEADS, :].contiguous()
    norm = global_probe_log_norm_triton(q4, k4, scale=128 ** -0.5, global_probe_count=256)
    native = partition_mass_medoid_seed_rows_triton(q4, k4, 128 ** -0.5, global_normalizer=norm)
    actual = native.cpu().numpy()
    validate_seed_array(actual, 4, anchored=False)
    expected = np.empty_like(actual)
    bank, _ = multiscale_candidate_table(45, 16, q.device, geometry_count=16)
    bank = bank.long()
    flat_actual, flat_expected = actual.reshape(-1, 4), expected.reshape(-1, 4)
    examples, prior_tf32 = [], torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, len(flat_actual), PARITY_CHUNK):
            index = torch.arange(start, min(start + PARITY_CHUNK, len(flat_actual)), device=q.device)
            head, column, row, frame = block_coordinates(index)
            rows = bank[row]
            query = q4[0, 0, rows, column[:, None], head[:, None]].float()
            probes = torch.arange(16, device=q.device) * 5
            keys = k4[0, frame[:, None], row[:, None], probes[None, :], head[:, None]].float()
            logits = torch.bmm(query, keys.transpose(1, 2)) * (128 ** -0.5)
            local_lse = torch.logsumexp(logits, dim=-1)
            mass = local_lse - norm[0, 0, rows, column[:, None], head[:, None]] * 0.6931471805599453
            importance = (mass - mass.amax(-1, keepdim=True)).exp()
            chosen, diagnostic = partition_mass_medoid_seeds_reference(logits, importance, return_diagnostics=True)
            selected_rows = rows.gather(1, chosen).to(torch.uint8).cpu().numpy()
            flat_expected[start:start + len(index)] = selected_rows
            bad = np.flatnonzero((flat_actual[start:start + len(index)] != selected_rows).any(-1))
            for offset in bad[:max(0, 8 - len(examples))]:
                native_rows = torch.as_tensor(flat_actual[start + offset].astype(np.int64), device=q.device)
                native_match = rows[offset, :, None] == native_rows[None, :]
                native_in_bank = bool((native_match.sum(0) == 1).all())
                native_positions = native_match.long().argmax(0)
                examples.append({"coordinate_frame_row_column_original_head": [int(frame[offset]), int(row[offset]),
                    int(column[offset]), PARITY_HEADS[int(head[offset])]], "native_rows": native_rows.tolist(),
                    "reference_rows": selected_rows[offset].tolist(), "native_rows_in_candidate_bank": native_in_bank,
                    "native_prefix_recomputed_scores": mismatch_scores(logits[offset], importance[offset], native_positions,
                                                                        diagnostic["first_medoid_cost"][offset]) if native_in_bank else None,
                    "reference_prefix_recomputed_scores": mismatch_scores(logits[offset], importance[offset], chosen[offset],
                                                                           diagnostic["first_medoid_cost"][offset])})
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_tf32
    np.savez_compressed(artifact, native=actual, independent_reference=expected)
    result = parity_counts(actual, expected) | {"heads": list(PARITY_HEADS), "frames": 21,
        "artifact": str(artifact), "mismatch_examples": examples, "gate": "exact ordered table equality; zero mismatches",
        "reference": "Independent Torch explicit-pair reverse KL + independently recomputed partition deficit",
        "score_scope": "Actual BF16 QKV, FP32 non-TF32 bmm and natural-unit reference scores; not Triton register dumps",
        "normalizer_scope": "Shared original native FP32 TF32x3 G256; selector semantics checked conditional on it"}
    return result, actual


def attention(token, q, k, v, mode):
    return token._hard_c4_attention(q, k, v, h=45, w=80, h_reduce=1, w_reduce=1, sm_scale=128 ** -0.5,
        query_outer_chunk=21, head_chunk=8, hard_route_backend="hard_c4", seed_mode=mode,
        exact_intra_frame_attention=False, exact_neighbor_window_columns=0, exact_query_rows=0,
        first_frame_full_attention=False, full_attention_anchor_time_count=0,
        full_attention_anchor_row_count=0, full_attention_anchor_scale=1.0).contiguous()


def run(args, identity):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import attention.token_merging as token
    from attention.partition_mass_medoid_seed_triton import partition_mass_medoid_seed_rows_triton
    from evaluate_entropy_seed_oracle import _load_grid
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    payload = identity | {"complete": False, "records": {}, "methods": METHODS, "router_frame": 0,
        "settings": {"qkv": [str(p) for p in args.qkv], "iterations": ITERATIONS, "repeats": REPEATS, "warmup": WARMUP},
        "device": torch.cuda.get_device_name(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "qkv_shape": [1, 75600, 40, 128], "dtype": "BF16",
        "selector_contract": "C16/G16/P16 Uniform16/G256/J4, actual four warps + owner-logp reuse; no kwargs repaired",
        "hellinger_contract": "Existing RF0 C32/G45/P16 Uniform16 J4 production path",
        "normalizer": "Original FP32 TF32x3; no precision override", "head_chunk": 8, "query_outer_chunk": 21,
        "scope": "Complete selection + unchanged SQ1 merge + full route + contiguous output; excludes projections/RoPE",
        "medoid_scope": "Experimental selector-only patch; no production mode/config is changed; anchored partition stays unpatched",
        "sdpa_backend": "External PyTorch SDPA forced FLASH_ATTENTION, same QKV/layout/dtype",
        "timing_scope": "Interleaved 4x5 calls after 2 warmups; compilation, observations, reference and context setup excluded",
        "purity": "External clean single-GPU launcher/monitor required; script does not certify exclusivity",
        "limitations": "Two frozen L20 captures only; known L33/head30 weakness is not covered. Passing correctness/runtime cannot promote quality. Output error is not video PSNR, SSIM or LPIPS; not all-layer trajectory coverage"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    medoid_dispatch = selector_adapter(partition_mass_medoid_seed_rows_triton)
    def context(name):
        if name == "sdpa":
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        return patch.object(token, "partition_mass_seed_rows_triton", medoid_dispatch) if name == "medoid_c16" else nullcontext()
    def dense(q, k, v):
        return functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=False, scale=128 ** -0.5).transpose(1, 2).contiguous()
    save()
    with torch.inference_mode():
        for number, path in enumerate(args.qkv):
            record = {"complete": False, "capture": {"path": str(path.resolve()), "bytes": path.stat().st_size,
                      "mtime_ns": path.stat().st_mtime_ns}, "output_vs_sdpa_40heads": {}, "attention_latency": {}, "timing_orders": []}
            payload["records"][str(path)] = record
            save()
            try:
                host = _load_grid(str(path), list(range(40)))
                if any(tuple(t.shape) != (40, 21, 45, 80, 128) for t in host):
                    raise ValueError("Require full 40-head, 21-frame, 45x80, D128 captures")
                record["capture"]["stored_dtypes"] = [str(t.dtype) for t in host]
                record["capture"]["conversion"] = "All stored tensors converted to BF16 for every method and reference"
                q, k, v = [t.permute(1, 2, 3, 0, 4).reshape(1, 75600, 40, 128).contiguous().to(device, torch.bfloat16) for t in host]
                del host
                record["semantic_parity"], small_table = semantic_parity(q, k, sidecar(args.output, number, "parity"))
                save()
                if not record["semantic_parity"]["exact_table_equal"]:
                    raise AssertionError("Semantic parity pending/failed: exact table mismatch; timing forbidden")
                functions = {name: partial(attention, token, q, k, v, mode) for name, mode in METHODS.items()}
                functions["sdpa"] = partial(dense, q, k, v)
                with context("sdpa"):
                    reference = functions["sdpa"]()
                tables, record["seed_tables"] = {}, {}
                for name in METHODS:
                    symbol = "hellinger_span_seed_rows_triton" if name == "rf0" else "partition_mass_seed_rows_triton"
                    real = partition_mass_medoid_seed_rows_triton if name == "medoid_c16" else getattr(token, symbol)
                    observations = []
                    def observer(table, inputs, options):
                        if observations or not torch.equal(inputs[0], q.reshape(1, 21, 45, 80, 40, 128)[:, :1]):
                            raise AssertionError("Require one full-head selector with the actual RF0 query")
                        if not torch.equal(inputs[1], k.reshape(1, 21, 45, 80, 40, 128)):
                            raise AssertionError("Selector did not receive unchanged all-frame K")
                        array = table.cpu().numpy().copy()
                        observations.append(validate_seed_array(array, 40, anchored=name != "medoid_c16"))
                        tables[name] = array
                        observations[0]["received_options"] = {key: str(val) if key == "output_dtype" else val for key, val in options.items()}
                    def recording(*inputs, **options):
                        result = real(*inputs, **options)
                        observer(result, inputs, options)
                        return result
                    wrapped = recording if name == "rf0" else selector_adapter(real, observer)
                    with patch.object(token, symbol, wrapped):
                        output = functions[name]()
                    if len(observations) != 1 or output.shape != q.shape or output.dtype != q.dtype or not output.is_contiguous():
                        raise AssertionError("Invalid selector observation or complete output")
                    record["seed_tables"][name] = observations[0]
                    record["output_vs_sdpa_40heads"][name] = base.output_metrics(output, reference)
                    del output
                record["four_vs_forty_head_medoid_table_equal"] = bool(np.array_equal(tables["medoid_c16"][..., PARITY_HEADS, :], small_table))
                table_path = sidecar(args.output, number, "tables")
                np.savez_compressed(table_path, **tables)
                record["native_seed_tables_artifact"] = str(table_path)
                if not record["four_vs_forty_head_medoid_table_equal"]:
                    raise AssertionError("Four/full-head medoid table mismatch; timing forbidden")
                del reference, tables, small_table
                for item in record["output_vs_sdpa_40heads"].values():
                    baseline_mse = record["output_vs_sdpa_40heads"]["rf0"]["mse"]
                    item["mse_change_vs_rf0"] = item["mse"] / baseline_mse - 1 if baseline_mse else None
                for name, function in functions.items():
                    with context(name):
                        for _ in range(WARMUP):
                            output = function()
                    del output
                torch.cuda.synchronize(device)
                samples = {name: [] for name in functions}
                for repeat in range(REPEATS):
                    order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
                    record["timing_orders"].append(order)
                    for name in order:
                        with context(name):
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(ITERATIONS):
                                output = functions[name]()
                            end.record()
                            end.synchronize()
                        del output
                        samples[name].append(start.elapsed_time(end) / ITERATIONS)
                        record["attention_latency"][name] = base.latency_stats(samples[name])
                        save()
                record["attention_latency"] = base.relative_timings(record["attention_latency"])
                record["complete"] = True
                save()
                del q, k, v, functions, function
                torch.cuda.empty_cache()
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                save()
                raise
    payload["complete"] = True
    save()


def sidecar(output, number, kind):
    return output.with_name(f"{output.stem}_capture{number}_{kind}.npz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv", nargs=2, type=Path, default=[Path(p) for p in base.DEFAULT_CAPTURES])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if any(p.exists() for p in [args.output] + [sidecar(args.output, n, k) for n in range(2) for k in ("parity", "tables")]):
        parser.error("Refusing to overwrite output or seed-table artifacts")
    if len({p.resolve() for p in args.qkv}) != 2 or not all(p.is_file() for p in args.qkv):
        parser.error("Require two distinct existing captures")
    base.check_capture_environment(os.environ)
    identity = base.provenance()
    identity["source_sha256"].update({p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in EXTRA_SOURCES})
    run(args, identity)


if __name__ == "__main__":
    main()
