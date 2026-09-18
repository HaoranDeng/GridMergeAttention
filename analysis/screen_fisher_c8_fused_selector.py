"""Validate and time a fused FP32 Fisher C8 selector on one clean H200."""

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.launch_quality_first_videos import gpu_processes, now, require_clean


DEFAULT_CAPTURE = Path(
    "/local2/denghaoran/GridMergeAttention/experiments/"
    "gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--fused-source", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=10)
    args = parser.parse_args()
    if min(args.warmup, args.blocks) < 1:
        parser.error("warmup and blocks must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    initial = require_clean(args.gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=initial["gpu"]["uuid"],
        OMP_NUM_THREADS="4",
        MAX_JOBS="2",
        TORCH_CUDA_ARCH_LIST="9.0a",
        TORCH_EXTENSIONS_DIR=str(args.output.parent / "torch_extensions"),
        TRITON_CACHE_DIR=str(args.output.parent / "triton_cache"),
    )

    stop = threading.Event()
    monitor_errors = []
    monitor_samples = []
    result = {
        "method": "fisher_geometric_c8_fp32_fused_selector_screen",
        "capacity": 8,
        "singleton_count": 0,
        "initial_gpu": initial,
        "capture": str(args.capture),
    }

    def monitor():
        while not stop.is_set():
            try:
                processes = gpu_processes(initial["gpu"]["uuid"])
                monitor_samples.append({"utc": now(), "processes": processes})
                if any(row["pid"] != os.getpid() for row in processes):
                    monitor_errors.append("foreign GPU process")
            except Exception as error:
                monitor_errors.append(repr(error))
            stop.wait(2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def save():
        result.update(
            monitor_errors=monitor_errors,
            monitor_samples=monitor_samples,
        )
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    try:
        ctypes.CDLL(
            "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
            mode=ctypes.RTLD_GLOBAL,
        )
        import torch

        from attention.affine_hellinger_seed_triton import (
            fisher_geometric_c8_seed_rows_triton,
        )
        from native_fisher.fused_c8_selector import NativeFusedFisherC8Selector
        from native_fisher.selector import NativeFisherSelector

        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        payload = torch.load(args.capture, map_location="cpu", weights_only=True)
        q = payload["q"].reshape(1, 21, 45, 80, 40, 128)[:, :1].cuda()
        k = payload["k"].reshape(1, 21, 45, 80, 40, 128).cuda()
        del payload

        separate = NativeFisherSelector(8, layout="probe4")
        fused = NativeFusedFisherC8Selector(source_path=args.fused_source)
        result["sources"] = {
            "separate": hashlib.sha256(separate.source_path.read_bytes()).hexdigest(),
            "fused": hashlib.sha256(fused.source_path.read_bytes()).hexdigest(),
        }
        result["fused_resources"] = fused.resources()
        save()

        reference = fisher_geometric_c8_seed_rows_triton(q, k).clone()
        separate_seeds = separate(q, k).clone()
        fused_seeds = fused(q, k).clone()

        def comparison(actual, expected):
            mismatch = actual != expected
            count = int(mismatch.sum())
            row = {
                "indices": expected.numel(),
                "mismatch_count": count,
                "mismatch_fraction": count / expected.numel(),
                "per_slot_mismatches": mismatch.reshape(-1, 8).sum(0).tolist(),
            }
            if count:
                row["first_mismatches"] = mismatch.nonzero()[:10].cpu().tolist()
            return row

        result["fused_vs_separate"] = comparison(fused_seeds, separate_seeds)
        result["fused_vs_fp64"] = comparison(fused_seeds, reference)
        result["separate_vs_fp64"] = comparison(separate_seeds, reference)
        save()

        zero_q = torch.zeros_like(q)
        result["zero_query_ties"] = {
            "fused_vs_separate": torch.equal(fused(zero_q, k), separate(zero_q, k)),
            "fused_vs_fp64": torch.equal(
                fused(zero_q, k), fisher_geometric_c8_seed_rows_triton(zero_q, k)
            ),
        }
        del zero_q
        save()

        calls = {"fused": lambda: fused(q, k), "separate": lambda: separate(q, k)}
        for _ in range(args.warmup):
            for call in calls.values():
                call()
        torch.cuda.synchronize()

        samples = {name: [] for name in calls}
        for block in range(args.blocks):
            order = (
                ("fused", "separate", "separate", "fused")
                if block % 2 == 0
                else ("separate", "fused", "fused", "separate")
            )
            for name in order:
                if monitor_errors:
                    raise RuntimeError(monitor_errors)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                calls[name]()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))

        result["timing"] = {
            name: {
                "mean_ms": statistics.fmean(values),
                "median_ms": statistics.median(values),
                "samples_ms": values,
            }
            for name, values in samples.items()
        }
        result["valid"] = (
            not monitor_errors
            and result["fused_vs_separate"]["mismatch_count"] == 0
            and all(result["zero_query_ties"].values())
        )
        save()
        print(json.dumps(result["timing"]), flush=True)
    except BaseException as error:
        result.update(valid=False, error=repr(error))
        save()
        raise
    finally:
        stop.set()
        thread.join(10)
        save()


if __name__ == "__main__":
    main()
