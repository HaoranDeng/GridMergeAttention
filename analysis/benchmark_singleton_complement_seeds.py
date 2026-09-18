"""Bounded singleton-complement parity and timing on one full-40-head capture.

Run only after the Torch quality screen passes and a clean GPU is available.
All GCA variants retain SQ1 and the original merge/full-route implementation.
The optional four-head Torch-table check and all compilation are untimed.
Dormant: the 2026-09-04 Torch screen regressed on both captures, so this script
was prepared but not run. That candidate does not qualify for benchmarking.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import json
from pathlib import Path
import sys
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEED_METHODS = ("rf10", "mass_original", "bulk_original", "bulk_native", "bulk_tf32")
SEED_GATE = 3.98


def _selector_options(name: str) -> dict:
    if name not in SEED_METHODS[1:]:
        raise ValueError(f"not an entropy-mass selector: {name}")
    return {
        "candidate_count": 32, "candidate_geometry_count": 45,
        "block_j": 4, "probe_count": 16, "global_probe_count": 256,
        "gs_num_warps": 4, "gs_num_stages": 1, "weight_mode": "mass",
        "norm_block_q": 32, "norm_num_warps": 8,
        "singleton_complement": name.startswith("bulk_"),
        "native_low_precision_dot": name == "bulk_native",
        "tf32_low_precision_dot": name == "bulk_tf32",
    }


def _configured_selector(selector, options):
    def configured(*inputs, **kwargs):
        # Override even explicit dispatcher kwargs; functools.partial defaults
        # alone would allow them to silently defeat the requested experiment.
        kwargs.update(options)
        return selector(*inputs, **kwargs)

    return configured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--parity", action="store_true",
                        help="Untimed gs_bulk_256 Torch parity on heads 0,10,20,30")
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.warmup) < 1:
        parser.error("iterations, repeats, and warmup must be positive")
    import torch

    with torch.inference_mode():
        _run(args)


def _run(args) -> None:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from attention.adaptive_seed_triton import hellinger_span_seed_rows_triton
    from attention.entropy_mass_seed_triton import (
        entropy_mass_seed_rows_triton, global_probe_log_norm_triton,
    )
    from benchmark_mass_normalizer_precision import MASS, DOT_OPTIONS, _agreement, _difference
    from benchmark_mass_selector_launch import _summary, _timed
    from evaluate_entropy_seed_oracle import _load_grid
    from screen_partition_population_seeds import BASE_MODE, custom_tables, production

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    grids = _load_grid(args.qkv, list(range(40)))
    q, k, v = [
        grid.permute(1, 2, 3, 0, 4).reshape(1, 75600, 40, 128)
        .contiguous().to(device=device, dtype=torch.bfloat16)
        for grid in grids
    ]
    del grids
    if any(tuple(t.shape) != (1, 75600, 40, 128) for t in (q, k, v)):
        raise ValueError("expected full 40-head 21x45x80 Wan QKV")
    native_q = q.reshape(1, 21, 45, 80, 40, 128)
    keys = k.reshape(1, 21, 45, 80, 40, 128)
    router = native_q[:, 10:11].contiguous()
    scale = 128**-0.5
    result = {
        "settings": {**vars(args), "output": str(args.output)},
        "device": torch.cuda.get_device_name(device),
        "qkv_shape": list(q.shape), "dtype": str(q.dtype), "completed": False,
        "fixed_algorithm": {
            "router_frame": 10, "candidates": 32, "candidate_geometry": 45,
            "uniform_probes": 16, "global_probes": 256, "sq1": True,
            "head_chunk": 8, "query_outer_chunk": 21,
            "rf10_mode": BASE_MODE, "mass_mode": MASS,
        },
        "selector_options": {name: _selector_options(name) for name in SEED_METHODS[1:]},
        "normalizer_dot_options": DOT_OPTIONS,
        "normalizer_scope": "40 heads, global Uniform256 log2 partition normalizer",
        "selector_scope": "40 heads, normalizer plus full C4 seed selection",
        "attention_scope": (
            "40-head complete operator including selector, original merge/full route, "
            "contiguous output; excludes model projections/RoPE and video generation"
        ),
        "sdpa_backend": "forced FLASH_ATTENTION; external reference only",
        "output_layout": "contiguous [B,N,H,D] for all complete operators",
        "timing": "CUDA events; warmed, interleaved reversed order; context setup excluded",
        "parity_caveat": "Native/TF32 lowering may change rounding; bitwise equality is measured, not assumed",
        "parity": {}, "normalizer_latency": {}, "selector_latency": {}, "attention_latency": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def require(condition, message):
        if not condition:
            result["failure"] = message
            save()
            raise RuntimeError(message)

    def invariant(table, heads):
        expected = (1, 1, 21, 45, 80, heads, 4)
        require(tuple(table.shape) == expected and table.dtype == torch.uint8,
                f"invalid native C4 table shape/dtype: {table.shape}, {table.dtype}")
        rows = torch.arange(45, device=device).view(1, 1, 1, 45, 1, 1)
        return {
            "rows_in_range": bool((table < 45).all()),
            "anchor_first": bool((table[..., 0] == rows).all()),
            "four_distinct_rows": bool((table.to(torch.int16).sort(-1).values.diff(dim=-1) > 0).all()),
        }

    def seed(name, query=router, key=keys):
        if name == "rf10":
            return hellinger_span_seed_rows_triton(
                query, key, scale, alpha=0.2, candidate_count=32,
                candidate_geometry_count=45, top_probes=0, uniform_probes=16,
                probe_anchor_count=2, block_j=4, group_tile=16,
                output_dtype=torch.uint8, gs_num_warps=4,
            )
        return entropy_mass_seed_rows_triton(
            query, key, scale, output_dtype=torch.uint8, **_selector_options(name),
        )

    def normalizer(mode):
        return global_probe_log_norm_triton(
            router, keys, scale, global_probe_count=256, block_q=32,
            num_warps=8, num_stages=1, **DOT_OPTIONS[mode],
        )

    def context(name):
        if name in SEED_METHODS[1:]:
            return patch(
                "attention.token_merging.entropy_mass_seed_rows_triton",
                new=_configured_selector(entropy_mass_seed_rows_triton, _selector_options(name)),
            )
        if name == "sdpa":
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        return nullcontext()

    def dense():
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=False, scale=scale,
        ).transpose(1, 2).contiguous()

    attention = {
        name: (lambda mode=BASE_MODE if name == "rf10" else MASS:
               production(q, k, v, mode=mode).contiguous())
        for name in SEED_METHODS
    }
    attention["sdpa"] = dense

    if args.parity:
        head_ids = torch.tensor([0, 10, 20, 30], device=device)
        small_native = [
            tensor.reshape(1, 21, 45, 80, 40, 128).index_select(4, head_ids).contiguous()
            for tensor in (q, k, v)
        ]
        q4, k4, v4 = small_native
        qg4, kg4 = [tensor[0].permute(3, 0, 1, 2, 4).contiguous() for tensor in (q4, k4)]
        reference = custom_tables(qg4, kg4, ["gs_bulk_256"], chunk=21, probes=16)["gs_bulk_256"]
        candidate = seed("bulk_original", q4[:, 10:11].contiguous(), k4)
        check = _agreement(candidate, reference)
        check["invariants"] = invariant(candidate, 4)
        tensors4 = [tensor.reshape(1, 75600, 4, 128) for tensor in small_native]
        reference_output = production(*tensors4, table=reference).contiguous()
        candidate_output = production(*tensors4, table=candidate).contiguous()
        check["output_vs_torch_table"] = _difference(candidate_output, reference_output)
        check["reference_output_finite"] = bool(torch.isfinite(reference_output).all())
        check["gate_passed"] = (
            check["mean_shared_seeds"] >= SEED_GATE and all(check["invariants"].values())
            and check["output_vs_torch_table"]["finite"] and check["reference_output_finite"]
        )
        check["scope"] = "heads 0,10,20,30, all frames; gs_bulk_256 Torch reference; untimed"
        result["parity"]["bulk_original_vs_torch_4heads"] = check
        save()
        require(check["gate_passed"], "four-head Torch parity failed before timing")
        del small_native, q4, k4, v4, qg4, kg4, tensors4, reference, candidate
        del reference_output, candidate_output

    original_norm = normalizer("original")
    mass_table, bulk_table = seed("mass_original"), seed("bulk_original")
    off_probe = torch.arange(80, device=device) % 5 != 0
    unaffected = _agreement(bulk_table[..., off_probe, :, :], mass_table[..., off_probe, :, :])
    unaffected.update({"query_column_count": 64, "query_column_fraction": 0.8})
    result["parity"]["bulk_vs_mass_nonprobe_columns"] = unaffected
    for name, table in (("mass_original", mass_table), ("bulk_original", bulk_table)):
        result["parity"][f"{name}_invariants"] = invariant(table, 40)
        require(all(result["parity"][f"{name}_invariants"].values()), f"{name} seed invariants failed")
    save()
    require(unaffected["bitwise_equal"], "bulk_original changed seeds in non-Uniform16 columns; inspect before timing")
    for mode in ("native", "tf32"):
        norm, table = normalizer(mode), seed(f"bulk_{mode}")
        result["parity"][f"normalizer_bulk_{mode}_vs_bulk_original"] = _difference(norm, original_norm)
        check = _agreement(table, bulk_table)
        check["invariants"] = invariant(table, 40)
        check["gate_passed"] = check["mean_shared_seeds"] >= SEED_GATE and all(check["invariants"].values())
        result["parity"][f"seeds_bulk_{mode}_vs_bulk_original"] = check
        save()
        require(check["gate_passed"] and bool(torch.isfinite(norm).all()), f"bulk_{mode} seed/normalizer parity gate failed")
        del norm, table
    require(bool(torch.isfinite(original_norm).all()), "original normalizer is nonfinite")
    del original_norm, mass_table, bulk_table

    with context("bulk_original"):
        original_output = attention["bulk_original"]()
    require(bool(torch.isfinite(original_output).all()), "bulk_original attention output is nonfinite")
    for mode in ("native", "tf32"):
        with context(f"bulk_{mode}"):
            output = attention[f"bulk_{mode}"]()
        check = _difference(output, original_output)
        result["parity"][f"attention_bulk_{mode}_vs_bulk_original"] = check
        result["parity"][f"all_bulk_{mode}_bitwise_equal"] = all(
            result["parity"][f"{stage}_bulk_{mode}_vs_bulk_original"]["bitwise_equal"]
            for stage in ("normalizer", "seeds", "attention")
        )
        save()
        require(check["finite"], f"bulk_{mode} complete attention output is nonfinite")
        del output
    del original_output

    def benchmark(functions, section, context_factory):
        for name, function in functions.items():
            with context_factory(name):
                for _ in range(args.warmup):
                    output = function()
            require(bool(torch.isfinite(output).all()), f"nonfinite warmup output: {section}/{name}")
            if section == "attention_latency":
                require(tuple(output.shape) == tuple(q.shape) and output.is_contiguous(),
                        f"unexpected complete output layout: {name}")
            del output
        torch.cuda.synchronize(device)
        names, samples = list(functions), {name: [] for name in functions}
        for repeat in range(args.repeats):
            for name in names if repeat % 2 == 0 else names[::-1]:
                # Set dispatch and Flash backend before _timed creates events.
                with context_factory(name):
                    elapsed, output = _timed(functions[name], args.iterations)
                del output
                samples[name].append(elapsed)
                result[section][name] = _summary(samples[name])
                print(f"{section} repeat={repeat} {name}: {elapsed:.6f} ms", flush=True)
                save()

    benchmark({mode: partial(normalizer, mode) for mode in DOT_OPTIONS},
              "normalizer_latency", lambda _: nullcontext())
    benchmark({name: partial(seed, name) for name in SEED_METHODS},
              "selector_latency", lambda _: nullcontext())
    benchmark(attention, "attention_latency", context)
    for section in ("selector_latency", "attention_latency"):
        rf10_ms = result[section]["rf10"]["mean_ms"]
        for value in result[section].values():
            value["delta_vs_rf10_ms"] = value["mean_ms"] - rf10_ms
            value["ratio_vs_rf10"] = value["mean_ms"] / rf10_ms
    sdpa_ms = result["attention_latency"]["sdpa"]["mean_ms"]
    for name, value in result["attention_latency"].items():
        value["speedup_vs_sdpa"] = sdpa_ms / value["mean_ms"]
        value["no_slower_than_rf10_mean"] = value["ratio_vs_rf10"] <= 1.0
    result["completed"] = True
    save()
    print(json.dumps({"parity": result["parity"], "attention_latency": result["attention_latency"],
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
