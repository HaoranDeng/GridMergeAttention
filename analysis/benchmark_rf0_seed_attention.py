"""Fixed RF0 complete-attention benchmark; run only after the trajectory screen.

Four methods are imported from the active stream, with the same partition
four-warp/owner-logp override. Each capture uses all 40 heads and 21 frames.
Compilation, warmup and context setup are outside alternating CUDA-event
timings. An external launcher owns exclusive single-GPU purity checks.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "analysis"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from screen_native_seed_trajectory import MODES, check_capture_environment

CAPTURE_ROOT = "/local2/denghaoran/GridMergeAttention/experiments/learned_seed_capture_20260822"
DEFAULT_CAPTURES = [f"{CAPTURE_ROOT}/call{call}_holdout/qkv.pt.npz" for call in ("020", "100")]
ITERATIONS, REPEATS, WARMUP = 5, 4, 2
PARTITION_OVERRIDE = {"gs_num_warps": 4, "reuse_owner_logp": True}
CONFIGS = {
    "rf0": "token_merging_wan_720p_fisher_c4_exact_capacity_fewstep.yaml",
    "mass_c32": "token_merging_wan_720p_entropy_mass_span_rf0_fewstep.yaml",
    "mass_c16": "token_merging_wan_720p_entropy_mass_span_c16_rf0_fewstep.yaml",
    "partition_c16": "token_merging_wan_720p_entropy_partition_mass_c16_rf0_fewstep.yaml",
}
SOURCES = ("analysis/benchmark_rf0_seed_attention.py", "analysis/screen_native_seed_trajectory.py",
           "analysis/evaluate_entropy_seed_oracle.py", "attention/token_merging.py",
           "attention/adaptive_seed_pytorch.py", "attention/adaptive_seed_triton.py",
           "attention/entropy_mass_seed_triton.py", "attention/partition_mass_seed_triton.py",
           "attention/singleton_c4_wan21_14b_720p_triton.py", "attention/hard_c4_wan21_14b_720p_triton.py")


def validate_modes(modes):
    if set(modes) != set(CONFIGS):
        raise ValueError("Require exactly the four fixed RF0 stream methods")
    for name, mode in modes.items():
        if "_j4_t1_rf0_u16_" not in mode or "sq1" in mode or "singleton" in mode:
            raise ValueError(f"{name}: require explicit exact-C4 J4/T1/RF0/Uniform16 without singleton")
        if any(marker in mode for marker in ("joint", "_lyrc", "_lpf", "_ts3")):
            raise ValueError(f"{name}: no temporal mixture or per-layer schedule is allowed")


def latency_stats(samples):
    if not samples or any(not 0 < value < float("inf") for value in samples):
        raise ValueError("latency samples must be nonempty, finite and positive")
    return {"mean_attention_ms": statistics.mean(samples), "std_ms": statistics.pstdev(samples),
            "samples_ms": list(samples), "calls_per_sample": ITERATIONS}


def relative_timings(timing):
    rf0, sdpa = (timing[name]["mean_attention_ms"] for name in ("rf0", "sdpa"))
    return {name: item | {"ratio_vs_rf0": item["mean_attention_ms"] / rf0,
                         "speedup_vs_sdpa": sdpa / item["mean_attention_ms"]}
            for name, item in timing.items()}


def provenance():
    validate_modes(MODES)
    configs = {}
    for name, filename in CONFIGS.items():
        path = ROOT / "configs" / filename
        content = path.read_bytes()
        matches = re.findall(r"^\s*seed_mode:\s*([^\r\n#]+)", content.decode("utf-8"), re.MULTILINE)
        if len(matches) != 1 or matches[0].strip().strip("\"'") != MODES[name]:
            raise ValueError(f"active config and stream disagree for {name}")
        configs[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest()}
    return {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
            "configs": configs}


def output_metrics(output, reference):
    import torch
    if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(reference).all()):
        raise ValueError("nonfinite complete attention output/reference")
    error = (output.float() - reference.float()).square()
    error_sum, energy = float(error.sum()), float(reference.float().square().sum())
    return {"mse": float(error.mean()), "relative_l2": (error_sum / energy) ** 0.5 if energy else None,
            "numel": error.numel(), "squared_error_sum": error_sum, "reference_squared_sum": energy,
            "frame_mse": error.reshape(1, 21, 3600, 40, 128).mean((0, 2, 3, 4)).tolist(),
            "head_mse": error.mean((0, 1, 3)).tolist(), "finite": True}


def run(args, identity):
    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import attention.token_merging as token_module
    from attention.partition_mass_seed_triton import partition_mass_seed_rows_triton
    from evaluate_entropy_seed_oracle import _load_grid

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    payload = identity | {
        "settings": {"qkv": [str(path) for path in args.qkv], "output": str(args.output), "device": str(device),
                     "iterations": ITERATIONS, "repeats": REPEATS, "warmup": WARMUP},
        "device": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "methods": dict(MODES), "qkv_shape": [1, 75600, 40, 128], "dtype": "torch.bfloat16",
        "router_frame": 0, "head_chunk": 8, "query_outer_chunk": 21,
        "partition_launch_override": PARTITION_OVERRIDE,
        "partition_override_note": "Explicit same-as-stream override; config-only production must be validated separately",
        "normalizer": "original FP32 TF32x3; no low-precision normalizer override",
        "scope": "Complete selector + unchanged merge + route + contiguous output; excludes projections/RoPE",
        "sdpa_backend": "PyTorch SDPA forced FLASH_ATTENTION; same QKV/layout/dtype",
        "timing": "Alternating order; all compilation/warmup/context setup outside CUDA events; no cached seed tables",
        "purity": "External wrapper must verify and monitor one exclusively clean GPU; not certified by this script",
        "limitations": "Two frozen captures, not all-layer trajectory coverage or video PSNR",
        "records": {}, "complete": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    def partition_override(*inputs, **kwargs):
        return partition_mass_seed_rows_triton(*inputs, **(kwargs | PARTITION_OVERRIDE))
    def context(name):
        if name == "sdpa":
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        if name == "partition_c16":
            return patch.object(token_module, "partition_mass_seed_rows_triton", partition_override)
        return nullcontext()
    def attention(q, k, v, mode):
        return token_module._hard_c4_attention(
            q, k, v, h=45, w=80, h_reduce=1, w_reduce=1, sm_scale=128 ** -0.5,
            query_outer_chunk=21, head_chunk=8, hard_route_backend="hard_c4", seed_mode=mode,
            exact_intra_frame_attention=False, exact_neighbor_window_columns=0, exact_query_rows=0,
            first_frame_full_attention=False, full_attention_anchor_time_count=0,
            full_attention_anchor_row_count=0, full_attention_anchor_scale=1.0).contiguous()
    def dense(q, k, v):
        return functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0,
            is_causal=False, scale=128 ** -0.5).transpose(1, 2).contiguous()
    save()
    with torch.inference_mode():
        for path in args.qkv:
            info = path.stat()
            record = {"capture": {"path": str(path.resolve()), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns},
                      "output_vs_sdpa_40heads": {}, "attention_latency": {}, "timing_orders": []}
            payload["records"][str(path)] = record
            save()
            try:
                host = _load_grid(str(path), list(range(40)))
                record["capture"]["stored_tensor_dtypes"] = [str(grid.dtype) for grid in host]
                record["capture"]["conversion"] = "Stored capture tensors are converted to BF16 for both methods and SDPA"
                if any(tuple(grid.shape) != (40, 21, 45, 80, 128) for grid in host):
                    raise ValueError("Expected full 40-head 21x45x80x128 capture")
                q, k, v = tuple(grid.permute(1, 2, 3, 0, 4).reshape(1, 75600, 40, 128)
                                .contiguous().to(device=device, dtype=torch.bfloat16) for grid in host)
                del host
                functions = {name: partial(attention, q, k, v, mode) for name, mode in MODES.items()}
                functions["sdpa"] = partial(dense, q, k, v)
                with context("sdpa"):
                    reference = functions["sdpa"]()
                for name in MODES:
                    with context(name):
                        output = functions[name]()
                    if output.shape != q.shape or output.dtype != q.dtype or not output.is_contiguous():
                        raise ValueError(f"{name}: wrong output shape/dtype/layout")
                    record["output_vs_sdpa_40heads"][name] = output_metrics(output, reference)
                    del output
                    save()
                del reference
                for item in record["output_vs_sdpa_40heads"].values():
                    base = record["output_vs_sdpa_40heads"]["rf0"]["mse"]
                    item["mse_change_vs_rf0"] = item["mse"] / base - 1 if base else None
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
                        record["attention_latency"][name] = latency_stats(samples[name])
                        save()
                        print(path.parent.name, repeat, name, samples[name][-1], flush=True)
                record["attention_latency"] = relative_timings(record["attention_latency"])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv", nargs=2, type=Path, default=[Path(path) for path in DEFAULT_CAPTURES])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing benchmark")
    if len({str(path.resolve()) for path in args.qkv}) != 2:
        parser.error("two distinct existing captures are required")
    check_capture_environment(os.environ)
    identity = provenance()  # Validate/source-hash before loading Torch or touching CUDA.
    run(args, identity)


if __name__ == "__main__":
    main()
