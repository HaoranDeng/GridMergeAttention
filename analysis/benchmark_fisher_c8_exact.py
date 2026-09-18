"""Validate and time FP32 geometric-Fisher exact-capacity C8 on a clean H200."""

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
    parser.add_argument(
        "--selector-backend",
        choices=("fused", "native"),
        default="fused",
    )
    parser.add_argument(
        "--selector-source",
        type=Path,
        default=None,
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument(
        "--libstdcxx",
        type=Path,
        default=Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6"),
        help="system C++ runtime to load before PyTorch extensions",
    )
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
        TORCH_EXTENSIONS_DIR=str(args.output.parent / "torch_extensions"),
        TRITON_CACHE_DIR=str(args.output.parent / "triton_cache"),
    )

    stop = threading.Event()
    monitor_errors = []
    monitor_samples = []

    def monitor():
        while not stop.is_set():
            try:
                processes = gpu_processes(initial["gpu"]["uuid"])
                monitor_samples.append({"utc": now(), "processes": processes})
                if any(row["pid"] != os.getpid() for row in processes):
                    monitor_errors.append("foreign GPU process")
            except Exception as error:
                monitor_errors.append(repr(error))
            stop.wait(3)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    result = {
        "method": "fisher_geometric_c8_rf0_fp32",
        "capacity": 8,
        "singleton_count": 0,
        "scope": "complete H40 exact-capacity C8 call; FP32 selector, pack, merge, and route included",
        "initial_gpu": initial,
        "capture": str(args.capture),
    }

    def save():
        result.update(
            monitor_errors=monitor_errors,
            monitor_samples=monitor_samples,
        )
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    try:
        if not args.libstdcxx.is_file():
            raise FileNotFoundError(args.libstdcxx)
        ctypes.CDLL(str(args.libstdcxx), mode=ctypes.RTLD_GLOBAL)
        result["libstdcxx_preload"] = str(args.libstdcxx)
        import torch

        from analysis.fisher_c8_exact import ExactFisherC8Attention
        from attention.affine_hellinger_seed_triton import (
            fisher_geometric_c8_seed_rows_triton,
        )
        from kernels.best import create_c8_r2

        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        payload = torch.load(args.capture, map_location="cpu", weights_only=True)
        q, k, v = [
            payload[name].reshape(1, 21, 45, 80, 40, 128).cuda()
            for name in ("q", "k", "v")
        ]
        del payload

        retained = create_c8_r2()
        native = ExactFisherC8Attention(
            backend=retained,
            selector_backend=args.selector_backend,
            selector_source=args.selector_source,
        )
        triton = ExactFisherC8Attention(backend=retained, selector_backend="triton")
        result["metadata"] = native.metadata()
        result["selector_source_sha256"] = hashlib.sha256(
            native.selector_source.read_bytes()
        ).hexdigest()
        result["resources"] = native.resources()
        save()

        reference_seeds = triton.select(q, k).clone()
        native_seeds = native.select(q, k).clone()
        mismatch = native_seeds != reference_seeds
        mismatch_count = int(mismatch.sum())
        result["seed_gate"] = {
            "indices": reference_seeds.numel(),
            "mismatch_count": mismatch_count,
            "mismatch_fraction": mismatch_count / reference_seeds.numel(),
            "per_slot_mismatches": mismatch.reshape(-1, 8).sum(0).tolist(),
            "reference_arithmetic": "fp64",
            "candidate_arithmetic": "fp32",
        }
        if result["seed_gate"]["mismatch_count"]:
            result["seed_gate"]["first_mismatches"] = mismatch.nonzero()[:10].cpu().tolist()
        save()
        zero_q = torch.zeros_like(q[:, :1])
        zero_native = native.selector(zero_q, k).clone()
        zero_reference = fisher_geometric_c8_seed_rows_triton(zero_q, k)
        result["zero_query_ties_exact"] = torch.equal(zero_native, zero_reference)
        del zero_q, zero_native, zero_reference
        save()
        if not result["zero_query_ties_exact"]:
            raise AssertionError("native selector changed the leftmost tie rule")

        outputs = {
            "fisher_fp32": torch.empty_like(q),
            "fisher_fp64_reference": torch.empty_like(q),
            "current_c8": torch.empty_like(q),
            "back_half": torch.empty_like(q),
        }

        def calls(name):
            if name == "fisher_fp32":
                return native(q, k, v, output=outputs[name])
            if name == "fisher_fp64_reference":
                return triton(q, k, v, output=outputs[name])
            if name == "current_c8":
                return retained(q, k, v, output=outputs[name])
            if name == "back_half":
                return native.run_with_seeds(
                    q, k, v, reference_seeds, output=outputs[name]
                )
            raise KeyError(name)

        def correctness(actual, expected):
            difference = actual.float() - expected.float()
            return {
                "bitwise_equal": torch.equal(actual, expected),
                "finite": bool(torch.isfinite(actual).all()),
                "max_abs": float(difference.abs().max()),
                "relative_l2": float(
                    (difference.square().sum() / expected.float().square().sum()).sqrt()
                ),
            }

        actual = calls("fisher_fp32").clone()
        expected = calls("fisher_fp64_reference").clone()
        result["complete_output_gate"] = correctness(actual, expected)
        del actual, expected
        save()
        if not result["complete_output_gate"]["finite"]:
            raise AssertionError(result["complete_output_gate"])

        def one_timing(name):
            if monitor_errors:
                raise RuntimeError(monitor_errors)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            calls(name)
            end.record()
            end.synchronize()
            return start.elapsed_time(end)

        names = ("fisher_fp32", "fisher_fp64_reference", "current_c8")
        for _ in range(args.warmup):
            for name in names:
                calls(name)
        torch.cuda.synchronize()

        samples = {name: [] for name in names}
        result["interleaved_blocks"] = []
        for block in range(args.blocks):
            order = names if block % 2 == 0 else tuple(reversed(names))
            rows = []
            for name in order:
                milliseconds = one_timing(name)
                samples[name].append(milliseconds)
                rows.append({"method": name, "ms": milliseconds})
            result["interleaved_blocks"].append(rows)
            save()

        result["complete_timing"] = {
            name: {
                "mean_ms": statistics.fmean(values),
                "median_ms": statistics.median(values),
                "samples_ms": values,
            }
            for name, values in samples.items()
        }

        def repeated_timing(function, repeats=5):
            values = []
            for _ in range(repeats):
                if monitor_errors:
                    raise RuntimeError(monitor_errors)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                function()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end))
            return {"mean_ms": statistics.fmean(values), "samples_ms": values}

        result["stage_timing"] = {
            "candidate_fp32_selector": repeated_timing(lambda: native.select(q, k)),
            "triton_fp64_selector": repeated_timing(lambda: triton.select(q, k)),
            "exact_c8_back_half": repeated_timing(lambda: calls("back_half")),
        }
        result["timing_protocol"] = (
            f"{args.warmup} warmups per method; {args.blocks} alternating-order "
            "complete calls per method; stage timings are five calls"
        )
        result["valid"] = not monitor_errors
        save()
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
