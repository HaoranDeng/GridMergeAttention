"""Full-40-head launch sweep for the partition-mass C4 selector.

This script varies execution settings only: J1/2/4/8 and 4/8 warps. Every
partition launch is checked against default J4/8 seed rows before promotion.
The best eligible selector is then timed through the unchanged complete GCA
operator, interleaved with RF10 and entropy-mass Hellinger-Span. Compilation
and temporary dispatch-context setup are excluded from CUDA-event timings.
``--reuse-owner-logp`` tests a mathematically equivalent buffer/gather removal
at the same eight launches, keeping the original J4/8 as a measured control.
``--cpu-selfcheck`` checks the reuse identity without importing Torch/Triton.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from typing import Callable
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MASS_MODE = "entropy_mass_span_c32_p16_g256_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2"
PARTITION_MODE = "entropy_partition_mass_c32_p16_g256_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2"
DEFAULT_NAME = "partition_j4_w8"
SEED_GATE = 3.98


def _cpu_owner_logp_selfcheck() -> dict:
    """Check the exact masked-cell identity, not Triton compilation or speed."""
    import numpy as np

    groups, candidates, probes = 4, 32, 16
    index = np.arange(groups * candidates * probes).reshape(groups, candidates, probes)
    ordinary = 3.0 * np.sin(index / 17.0) + np.cos(index / 31.0)
    duplicate = ordinary.copy()
    duplicate[:, 1::2] = duplicate[:, ::2]
    extreme = np.full_like(ordinary, -1200.0)
    for group in range(groups):
        extreme[group, np.arange(candidates), (np.arange(candidates) + group) % probes] = 0
    fixtures = {"uniform": np.zeros_like(ordinary), "ordinary": ordinary,
                "duplicate": duplicate, "extreme": extreme}
    anchors = np.array([0, 7, 19, 31])
    batch = np.arange(groups)
    empty_cells = 0

    def select(logp, importance, reuse):
        nonlocal empty_cells
        selected = np.zeros((groups, 4), dtype=np.int64)
        selected[:, 0] = anchors
        selected_logp = np.zeros((groups, 4, probes), dtype=logp.dtype)
        selected_logp[:, 0] = logp[batch, anchors]
        owner = np.zeros((groups, probes), dtype=np.int64)
        owner_logp = selected_logp[:, 0].copy()
        used = np.arange(candidates)[None, :] == anchors[:, None]
        history = []
        for step in range(1, 4):
            represented = np.zeros((groups, candidates), dtype=logp.dtype)
            for cell_slot in range(step):
                cell = owner == cell_slot
                stored = selected_logp[:, cell_slot]
                assert np.array_equal(stored[cell], owner_logp[cell])
                seed_logp = owner_logp if reuse else stored
                active = cell.any(axis=1)
                if reuse:
                    empty_cells += int((~active).sum())
                cell_max = np.where(cell, seed_logp, -np.inf).max(axis=1)
                cell_max = np.where(active, cell_max, 0)
                # Eager evaluation outside the mask may overflow; masked
                # values cannot enter the weights or the conditional bound.
                with np.errstate(over="ignore", under="ignore"):
                    exponential = np.where(cell, np.exp2(seed_logp - cell_max[:, None]), 0)
                    denominator = np.where(active, exponential.sum(axis=1), 1)
                    weight = exponential / denominator[:, None]
                    log_weight = np.where(
                        cell, seed_logp - cell_max[:, None] - np.log2(denominator)[:, None], 0
                    )
                    entropy = -(weight * log_weight).sum(axis=1)
                    bound = (logp * weight[:, None, :]).sum(axis=2) + entropy[:, None]
                    represented += np.where(active[:, None], np.exp2(bound), 0)
                assert np.isfinite(represented).all()
                history.append(weight.copy())
            history.append(represented.copy())
            score = np.where(~used, importance * np.maximum(1 - represented, 0), -1)
            pick = score.argmax(axis=1)
            selected[:, step] = pick
            used[batch, pick] = True
            if step < 3:
                new_logp = logp[batch, pick]
                selected_logp[:, step] = new_logp
                owner = np.where(new_logp > owner_logp, step, owner)
                owner_logp = np.maximum(owner_logp, new_logp)
                history.append(owner.copy())
        assert np.array_equal(selected[:, 0], anchors)
        assert (np.diff(np.sort(selected, axis=1), axis=1) > 0).all()
        return selected, history

    for dtype in (np.float32, np.float64):
        for logits in fixtures.values():
            logits = logits.astype(dtype)
            shifted = logits - logits.max(axis=2, keepdims=True)
            with np.errstate(under="ignore"):
                logp = shifted - np.log2(np.exp2(shifted).sum(axis=2, keepdims=True))
            importance = np.linspace(0.1, 1.0, groups * candidates, dtype=dtype).reshape(groups, candidates)
            original, original_history = select(logp, importance, False)
            reused, reused_history = select(logp, importance, True)
            assert np.array_equal(original, reused)
            assert len(original_history) == len(reused_history)
            assert all(np.array_equal(a, b) for a, b in zip(original_history, reused_history))
    assert empty_cells > 0
    return {
        "passed": True, "fixtures": list(fixtures), "dtypes": ["float32", "float64"],
        "groups_checked": groups * len(fixtures) * 2, "empty_cells_checked": empty_cells,
        "conditional_weights_bounds_owners_and_c4": "exactly equal",
        "scope": "NumPy algebra only; no Triton/GPU parity or performance claim",
    }


def _seed_invariants(table: torch.Tensor) -> dict:
    import torch

    if tuple(table.shape) != (1, 1, 21, 45, 80, 40, 4):
        raise ValueError(f"unexpected native C4 seed shape: {table.shape}")
    if table.dtype != torch.uint8:
        raise TypeError(f"expected uint8 seed rows, got {table.dtype}")
    key_rows = torch.arange(45, device=table.device).view(1, 1, 1, 45, 1, 1)
    ordered = table.to(torch.int16).sort(dim=-1).values
    return {
        "rows_in_range": bool((table < 45).all()),
        "anchor_first": bool((table[..., 0] == key_rows).all()),
        "four_distinct_rows": bool((ordered.diff(dim=-1) > 0).all()),
    }


def _output_difference(output: torch.Tensor, reference: torch.Tensor) -> dict:
    import torch

    delta = output.float() - reference.float()
    denominator = reference.float().square().sum().clamp_min(1e-30)
    return {
        "finite": bool(torch.isfinite(output).all()),
        "relative_l2": float((delta.square().sum() / denominator).sqrt()),
        "max_abs_error": float(delta.abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reuse-owner-logp", action="store_true")
    parser.add_argument("--cpu-selfcheck", action="store_true")
    args = parser.parse_args()
    if args.cpu_selfcheck:
        print(json.dumps(_cpu_owner_logp_selfcheck(), indent=2))
        return
    if args.qkv is None or args.output is None:
        parser.error("--qkv and --output are required unless --cpu-selfcheck is used")
    if min(args.iterations, args.repeats, args.warmup) < 1:
        raise ValueError("iterations, repeats, and warmup must be positive")

    import torch

    with torch.inference_mode():
        _benchmark(args)


def _benchmark(args) -> None:
    import torch

    from attention.adaptive_seed_triton import hellinger_span_seed_rows_triton
    from attention.entropy_mass_seed_triton import entropy_mass_seed_rows_triton
    from attention.partition_mass_seed_triton import partition_mass_seed_rows_triton
    from benchmark_mass_selector_launch import _seed_agreement, _summary, _timed
    from evaluate_entropy_seed_oracle import _load_grid
    from screen_partition_population_seeds import BASE_MODE, production

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    grids = _load_grid(args.qkv, list(range(40)))
    q, k, v = [
        grid.permute(1, 2, 3, 0, 4).reshape(1, 75600, 40, 128)
        .contiguous().to(device=device, dtype=torch.bfloat16)
        for grid in grids
    ]
    del grids
    if any(tuple(tensor.shape) != (1, 75600, 40, 128) for tensor in (q, k, v)):
        raise ValueError("expected full 40-head 21x45x80 Wan QKV")
    router = q.reshape(1, 21, 45, 80, 40, 128)[:, 10:11]
    keys = k.reshape(1, 21, 45, 80, 40, 128)
    scale = 128**-0.5

    def rf10_selector() -> torch.Tensor:
        return hellinger_span_seed_rows_triton(
            router.contiguous(), keys.contiguous(), scale,
            alpha=0.2, candidate_count=32, candidate_geometry_count=45,
            top_probes=0, uniform_probes=16, probe_anchor_count=2,
            block_j=4, group_tile=16, output_dtype=torch.uint8, gs_num_warps=4,
        )

    def mass_selector() -> torch.Tensor:
        return entropy_mass_seed_rows_triton(
            router.contiguous(), keys.contiguous(), scale,
            candidate_count=32, candidate_geometry_count=45,
            block_j=4, probe_count=16, global_probe_count=256,
            output_dtype=torch.uint8, gs_num_warps=4, weight_mode="mass",
        )

    def partition_selector(
        block_j: int = 4, warps: int = 8, reuse_owner_logp: bool = False
    ) -> torch.Tensor:
        return partition_mass_seed_rows_triton(
            router.contiguous(), keys.contiguous(), scale,
            candidate_count=32, candidate_geometry_count=45,
            block_j=block_j, probe_count=16, global_probe_count=256,
            output_dtype=torch.uint8, gs_num_warps=warps,
            reuse_owner_logp=reuse_owner_logp,
        )

    launches = {
        f"partition_j{block_j}_w{warps}" + ("_owner" if args.reuse_owner_logp else ""): {
            "block_j": block_j, "gs_num_warps": warps,
            "reuse_owner_logp": args.reuse_owner_logp,
        }
        for block_j in (1, 2, 4, 8) for warps in (4, 8)
    }
    if args.reuse_owner_logp:
        # Keep a measured, eligible original implementation control without
        # doubling the launch sweep. It remains the parity reference too.
        launches[DEFAULT_NAME] = {
            "block_j": 4, "gs_num_warps": 8, "reuse_owner_logp": False,
        }
    selectors: dict[str, Callable] = {
        "rf10": rf10_selector, "mass_span": mass_selector,
    }
    for name, launch in launches.items():
        selectors[name] = (
            lambda j=launch["block_j"], w=launch["gs_num_warps"], r=launch["reuse_owner_logp"]:
            partition_selector(j, w, r)
        )
    result = {
        "settings": {**vars(args), "output": str(args.output)},
        "device": torch.cuda.get_device_name(device),
        "qkv_shape": list(q.shape), "dtype": str(q.dtype),
        "fixed_algorithm": {
            "router_frame": 10, "candidates": 32, "candidate_geometry": 45,
            "local_probes": 16, "global_probes": 256,
            "head_chunk": 8, "query_outer_chunk": 21, "sq1": True,
            "rf10_mode": BASE_MODE, "mass_span_mode": MASS_MODE,
            "partition_mode": PARTITION_MODE,
            "partition_reference_launch": DEFAULT_NAME,
            "minimum_mean_shared_seeds": SEED_GATE,
        },
        "selector_scope": "40 heads, normalization plus complete seed selector",
        "attention_scope": (
            "40-head complete GCA operator: selector, merge, route, contiguous output; "
            "excludes model projections and RoPE"
        ),
        "timing": "CUDA events; compilation/warmup/context setup excluded; alternating order",
        "selector_launches": launches,
        "selector_parity": {}, "selector_latency": {},
        "attention_parity": {}, "attention_latency": {}, "errors": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def sweep(functions: dict[str, Callable], section: str, context_factory) -> None:
        samples = {name: [] for name in functions}
        names = list(functions)
        for repeat in range(args.repeats):
            order = names if repeat % 2 == 0 else names[::-1]
            for name in order:
                # Dispatcher overrides are installed before event creation/recording.
                with context_factory(name):
                    latency, output = _timed(functions[name], args.iterations)
                del output
                samples[name].append(latency)
                result[section][name] = _summary(samples[name])
                print(f"{section} repeat={repeat} {name}: {latency:.6f} ms", flush=True)
                save()

    rf10_table = rf10_selector()
    default_table = partition_selector()
    torch.cuda.synchronize(device)
    active = {}
    for name, function in selectors.items():
        try:
            for _ in range(args.warmup):
                output = function()
            torch.cuda.synchronize(device)
            invariants = _seed_invariants(output)
            parity = {
                "vs_rf10": _seed_agreement(output, rf10_table),
                "vs_partition_default": _seed_agreement(output, default_table),
                "exact_match_partition_default": bool((output == default_table).all()),
                "invariants": invariants,
            }
            parity["eligible"] = all(invariants.values()) and (
                name not in launches
                or parity["vs_partition_default"]["mean_shared_seeds"] >= SEED_GATE
            )
            result["selector_parity"][name] = parity
            if all(invariants.values()):
                active[name] = function
            print(f"warm {name}: {json.dumps(parity)}", flush=True)
            del output
        except Exception as exc:
            result["errors"][name] = f"{type(exc).__name__}: {exc}"
            print(f"failed {name}: {exc}", flush=True)
        save()
    if "rf10" not in active or "mass_span" not in active:
        raise RuntimeError("RF10 and mass-span control selectors must both pass")
    sweep(active, "selector_latency", lambda _: nullcontext())
    eligible = [
        name for name in launches
        if name in result["selector_latency"]
        and result["selector_parity"][name]["eligible"]
    ]
    if not eligible:
        save()
        raise RuntimeError("no partition launch met the 3.98/4 seed-agreement gate")
    best_name = min(eligible, key=lambda name: result["selector_latency"][name]["mean_ms"])
    best_launch = launches[best_name]
    result["best_partition_selector"] = best_name
    result["best_partition_launch"] = best_launch
    baseline_ms = result["selector_latency"]["rf10"]["mean_ms"]
    for item in result["selector_latency"].values():
        item["delta_vs_rf10_ms"] = item["mean_ms"] - baseline_ms
        item["ratio_vs_rf10"] = item["mean_ms"] / baseline_ms
    del rf10_table, default_table
    save()

    def configured_partition(*inputs, **kwargs):
        # Explicit incoming dispatcher kwargs must be overwritten, not merely
        # supplied as functools.partial defaults.
        kwargs.update(best_launch)
        return partition_mass_seed_rows_triton(*inputs, **kwargs)

    def attention_context(name: str):
        if name == "partition_best":
            return patch(
                "attention.token_merging.partition_mass_seed_rows_triton",
                new=configured_partition,
            )
        return nullcontext()

    attention = {
        "rf10": lambda: production(q, k, v, mode=BASE_MODE).contiguous(),
        "mass_span": lambda: production(q, k, v, mode=MASS_MODE).contiguous(),
        "partition_best": lambda: production(q, k, v, mode=PARTITION_MODE).contiguous(),
    }
    if best_name != DEFAULT_NAME:
        attention["partition_default"] = (
            lambda: production(q, k, v, mode=PARTITION_MODE).contiguous()
        )
    default_output = production(q, k, v, mode=PARTITION_MODE).contiguous()
    for name, function in attention.items():
        with attention_context(name):
            for _ in range(args.warmup):
                output = function()
        torch.cuda.synchronize(device)
        if name.startswith("partition"):
            parity = _output_difference(output, default_output)
        else:
            parity = {"finite": bool(torch.isfinite(output).all())}
        result["attention_parity"][name] = parity
        del output
        save()
        if not parity["finite"]:
            raise RuntimeError(f"nonfinite complete attention output for {name}")
    del default_output
    sweep(attention, "attention_latency", attention_context)
    baseline_ms = result["attention_latency"]["rf10"]["mean_ms"]
    for item in result["attention_latency"].values():
        item["delta_vs_rf10_ms"] = item["mean_ms"] - baseline_ms
        item["ratio_vs_rf10"] = item["mean_ms"] / baseline_ms
    result["best_partition_seed_gate_passed"] = True
    result["best_partition_no_slower_than_rf10_mean"] = (
        result["attention_latency"]["partition_best"]["mean_ms"] <= baseline_ms
    )
    save()
    print(json.dumps({
        "best_partition_launch": best_launch,
        "attention_latency": result["attention_latency"],
        "output": str(args.output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
