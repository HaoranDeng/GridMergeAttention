"""Interleaved, full-40-head launch sweep for the entropy-mass selector.

Only launch parameters vary: Uniform16, C32/G45, global P256, RF10, and
the mass objective remain fixed. Compilation is excluded from all timings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attention.adaptive_seed_triton import hellinger_span_seed_rows_triton
from attention.entropy_mass_seed_triton import (
    entropy_mass_seed_rows_triton,
    global_probe_log_norm_triton,
)
from evaluate_entropy_seed_oracle import _load_grid


def _summary(values: list[float]) -> dict:
    return {
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values)),
        "samples_ms": values,
    }


def _seed_agreement(output: torch.Tensor, reference: torch.Tensor) -> dict:
    shared = (output[..., :, None] == reference[..., None, :]).any(-1)
    return {
        "slot_agreement": float((output == reference).float().mean()),
        "mean_shared_seeds": float(shared.float().sum(-1).mean()),
    }


def _timed(function: Callable, iterations: int) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output = function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations, output


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.warmup) < 1:
        raise ValueError("iterations, repeats, and warmup must be positive")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    qg, kg, vg = _load_grid(args.qkv, list(range(40)))

    def native(grid: torch.Tensor) -> torch.Tensor:
        return grid.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous().to(
            device=device, dtype=torch.bfloat16
        )

    q_full, k = native(qg), native(kg)
    del qg, kg, vg
    if tuple(q_full.shape) != (1, 21, 45, 80, 40, 128) or k.shape != q_full.shape:
        raise ValueError(f"expected full 40-head Wan Q/K grids; got {q_full.shape}, {k.shape}")
    q = q_full[:, 10:11]
    scale = 128**-0.5

    def baseline() -> torch.Tensor:
        return hellinger_span_seed_rows_triton(
            q.contiguous(), k.contiguous(), scale,
            alpha=0.2, candidate_count=32, candidate_geometry_count=45,
            top_probes=0, uniform_probes=16, probe_anchor_count=2,
            block_j=4, group_tile=16, output_dtype=torch.uint8, gs_num_warps=4,
        )

    def mass(block_j=4, warps=8, norm_block_q=32, norm_warps=8) -> torch.Tensor:
        return entropy_mass_seed_rows_triton(
            q.contiguous(), k.contiguous(), scale,
            candidate_count=32, candidate_geometry_count=45,
            block_j=block_j, probe_count=16, global_probe_count=256,
            output_dtype=torch.uint8, gs_num_warps=warps, weight_mode="mass",
            norm_block_q=norm_block_q, norm_num_warps=norm_warps,
        )

    def normalizer(block_q=32, warps=8) -> torch.Tensor:
        return global_probe_log_norm_triton(
            q.contiguous(), k.contiguous(), scale,
            global_probe_count=256, block_q=block_q, num_warps=warps,
        )

    default_name = "mass_j4_w8"
    selectors: dict[str, Callable] = {"rf10": baseline}
    launches: dict[str, dict] = {"rf10": {"block_j": 4, "gs_num_warps": 4}}
    for block_j in (2, 4, 8):
        for warps in (4, 8):
            name = f"mass_j{block_j}_w{warps}"
            selectors[name] = lambda j=block_j, w=warps: mass(j, w)
            launches[name] = {
                "block_j": block_j, "gs_num_warps": warps,
                "norm_block_q": 32, "norm_num_warps": 8,
            }
    normalizers = {
        f"norm_q{block_q}_w{warps}": (
            lambda b=block_q, w=warps: normalizer(b, w)
        )
        for block_q in (16, 32, 64) for warps in (4, 8)
    }
    result = {
        "settings": {**vars(args), "output": str(args.output)},
        "device": torch.cuda.get_device_name(device),
        "q_shape": list(q.shape), "k_shape": list(k.shape),
        "selector_scope": "40 heads; q.contiguous + normalizer + complete seed selector",
        "normalizer_scope": "40 heads; q.contiguous + global probe log normalizer",
        "fixed_algorithm": {
            "router_frame": 10, "candidates": 32, "geometry": 45,
            "uniform_probes": 16, "global_probes": 256,
            "mass_default_reference": default_name,
        },
        "selector_launches": launches,
        "selector_parity": {}, "normalizer_parity": {},
        "selector_latency": {}, "normalizer_latency": {}, "errors": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # Retain only the two seed references and one small normalizer, keeping
    # unrelated outputs out of the timed working set.
    baseline_table = baseline()
    default_table = mass()
    default_norm = normalizer()
    torch.cuda.synchronize(device)

    def compile_and_check(functions: dict[str, Callable], kind: str) -> list[str]:
        active = []
        for name, function in functions.items():
            try:
                for _ in range(args.warmup):
                    output = function()
                torch.cuda.synchronize(device)
                if kind == "selector":
                    parity = {
                        "vs_rf10": _seed_agreement(output, baseline_table),
                        "vs_mass_default": _seed_agreement(output, default_table),
                    }
                    parity["eligible"] = name == "rf10" or (
                        parity["vs_mass_default"]["mean_shared_seeds"] >= 3.98
                    )
                else:
                    delta = output.float() - default_norm.float()
                    parity = {
                        "finite": bool(torch.isfinite(output).all()),
                        "max_abs_error_vs_default": float(delta.abs().max()),
                        "rmse_vs_default": float(delta.square().mean().sqrt()),
                    }
                result[f"{kind}_parity"][name] = parity
                active.append(name)
                print(f"warm {name}: {json.dumps(parity)}", flush=True)
                del output
            except Exception as exc:
                result["errors"][name] = f"{type(exc).__name__}: {exc}"
                print(f"failed {name}: {exc}", flush=True)
            save()
        return active

    def sweep(functions: dict[str, Callable], names: list[str], kind: str) -> None:
        samples = {name: [] for name in names}
        for repeat in range(args.repeats):
            order = names if repeat % 2 == 0 else names[::-1]
            for name in order:
                value, output = _timed(functions[name], args.iterations)
                samples[name].append(value)
                del output
                result[f"{kind}_latency"][name] = _summary(samples[name])
                print(f"{kind} repeat={repeat} {name}: {value:.6f} ms", flush=True)
                save()

    selector_names = compile_and_check(selectors, "selector")
    norm_names = compile_and_check(normalizers, "normalizer")
    sweep(selectors, selector_names, "selector")
    sweep(normalizers, norm_names, "normalizer")
    baseline_ms = result["selector_latency"]["rf10"]["mean_ms"]
    for value in result["selector_latency"].values():
        value["delta_vs_rf10_ms"] = value["mean_ms"] - baseline_ms
        value["ratio_vs_rf10"] = value["mean_ms"] / baseline_ms
    eligible = [
        name for name in selector_names if name != "rf10"
        and result["selector_parity"][name]["eligible"]
    ]
    if not eligible:
        save()
        raise RuntimeError("no mass launch met the 3.98/4 seed-agreement gate")
    best_selector = min(eligible, key=lambda name: result["selector_latency"][name]["mean_ms"])
    finite_norms = [name for name in norm_names if result["normalizer_parity"][name]["finite"]]
    if not finite_norms:
        save()
        raise RuntimeError("no finite normalizer launch")
    best_norm = min(finite_norms, key=lambda name: result["normalizer_latency"][name]["mean_ms"])
    norm_parts = best_norm.split("_")
    norm_q, norm_w = int(norm_parts[1][1:]), int(norm_parts[2][1:])
    launch = launches[best_selector]
    combined = lambda: mass(launch["block_j"], launch["gs_num_warps"], norm_q, norm_w)
    result["best_selector_default_norm"] = best_selector
    result["best_normalizer"] = best_norm
    result["combined_launch"] = {
        **launch, "norm_block_q": norm_q, "norm_num_warps": norm_w,
    }
    # Confirm the composed launch end-to-end against RF10/default in the same
    # process rather than adding the two isolated timing gains.
    confirmation = {"rf10": baseline, "mass_default": mass, "mass_combined": combined}
    confirmation_names = compile_and_check(confirmation, "selector")
    original_latency = result["selector_latency"]
    result["selector_latency"] = {}
    sweep(confirmation, confirmation_names, "selector")
    result["confirmation_latency"] = result["selector_latency"]
    result["selector_latency"] = original_latency
    confirmation_baseline = result["confirmation_latency"]["rf10"]["mean_ms"]
    for value in result["confirmation_latency"].values():
        value["delta_vs_rf10_ms"] = value["mean_ms"] - confirmation_baseline
        value["ratio_vs_rf10"] = value["mean_ms"] / confirmation_baseline
    result["combined_eligible"] = result["selector_parity"].get("mass_combined", {}).get("eligible", False)
    save()
    print(json.dumps({
        "combined_launch": result["combined_launch"],
        "combined_eligible": result["combined_eligible"],
        "confirmation_latency": result["confirmation_latency"],
        "output": str(args.output),
    }, indent=2), flush=True)
    if not result["combined_eligible"]:
        raise RuntimeError("combined launch failed the 3.98/4 seed-agreement gate")


if __name__ == "__main__":
    main()
