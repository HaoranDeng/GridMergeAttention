"""Screen exact-capacity Fisher C8 selector layouts on one clean H200.

The probe-logit kernel is shared by every variant. This script compares FP32
and FP64 Gram--Schmidt implementations and measures their end-to-end selector
time. FP64 candidates must match the Triton reference for all 24,192,000 seed
indices.  The production FP32 candidate reports its drift explicitly.  Every
candidate must preserve the zero-logit leftmost tie rule.
"""

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
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--libstdcxx",
        type=Path,
        default=Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6"),
    )
    args = parser.parse_args()
    if min(args.warmup, args.repeats) < 1:
        parser.error("warmup and repeats must be positive")
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
        "method": "exact-capacity Fisher C8 selector layout screen",
        "capacity": 8,
        "singleton_count": 0,
        "scope": "selector only; shared probe logits plus FP32/FP64 Gram-Schmidt",
        "initial_gpu": initial,
        "capture": str(args.capture),
        "variants": [],
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
        if not args.libstdcxx.is_file():
            raise FileNotFoundError(args.libstdcxx)
        ctypes.CDLL(str(args.libstdcxx), mode=ctypes.RTLD_GLOBAL)

        import torch
        from torch.utils.cpp_extension import load

        from attention.affine_hellinger_seed_triton import (
            fisher_geometric_c8_seed_rows_triton,
        )
        from native_fisher.selector import NativeFisherSelector, _logits

        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        payload = torch.load(args.capture, map_location="cpu", weights_only=True)
        q = payload["q"].reshape(1, 21, 45, 80, 40, 128)[:, :1].cuda()
        k = payload["k"].reshape(1, 21, 45, 80, 40, 128).cuda()
        del payload

        register_source = ROOT / "native_fisher/fisher_gs_register_distribute.cu"
        halving_source = ROOT / "native_fisher/fisher_c8_gs_register_halving.cu"
        distributed_source = ROOT / "native_fisher/fisher_gs_distributed.cu"
        candidate_source = ROOT / "native_fisher/fisher_gs.cu"
        selector = NativeFisherSelector(
            8,
            layout="probe4",
            source_path=register_source,
        )

        def load_extension(source):
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            extension = load(
                name=f"fisher_c8_screen_{digest[:16]}",
                sources=[str(source)],
                extra_cuda_cflags=["-O3", "--ptxas-options=-v"],
                verbose=False,
            )
            return extension, digest

        halving, halving_digest = load_extension(halving_source)
        distributed, distributed_digest = load_extension(distributed_source)
        candidate, candidate_digest = load_extension(candidate_source)
        register_digest = hashlib.sha256(register_source.read_bytes()).hexdigest()
        result["sources"] = {
            "register_distribute": register_digest,
            "register_halving": halving_digest,
            "distributed": distributed_digest,
            "candidate": candidate_digest,
        }
        save()

        def logits():
            _logits[(45 * 20, 40)](
                q,
                k,
                selector.candidates,
                selector.logits,
                num_warps=4,
                num_stages=1,
            )

        variants = [
            ("register_distribute_p4", selector.extension, 4),
            ("register_halving_fp32_p4", halving, 4),
            ("register_distribute_p8", selector.extension, 8),
            ("register_distribute_p16", selector.extension, 16),
            ("distributed_p4", distributed, 4),
            ("distributed_p8", distributed, 8),
            ("distributed_p16", distributed, 16),
            ("candidate", candidate, None),
        ]

        def run_gs(extension, probe_lanes):
            call_args = (
                selector.logits,
                selector.candidates,
                selector.anchors,
                selector.seeds,
                128**-0.5,
            )
            if probe_lanes is None:
                extension.run(*call_args)
            else:
                extension.run(*call_args, probe_lanes)
            return selector.seeds

        def timed(function):
            for _ in range(args.warmup):
                function()
            torch.cuda.synchronize()
            values = []
            for _ in range(args.repeats):
                if monitor_errors:
                    raise RuntimeError(monitor_errors)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                function()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end))
            return {
                "mean_ms": statistics.fmean(values),
                "median_ms": statistics.median(values),
                "samples_ms": values,
            }

        reference = fisher_geometric_c8_seed_rows_triton(q, k).clone()
        logits()
        result["logits_timing"] = timed(logits)

        for name, extension, probe_lanes in variants:
            run_gs(extension, probe_lanes)
            mismatch = selector.seeds != reference
            row = {
                "name": name,
                "probe_lanes": probe_lanes,
                "seed_indices": reference.numel(),
                "mismatch_count": int(mismatch.sum()),
                "per_slot_mismatches": mismatch.reshape(-1, 8).sum(0).tolist(),
            }
            if row["mismatch_count"]:
                row["first_mismatches"] = mismatch.nonzero()[:10].cpu().tolist()
            row["gs_timing"] = timed(lambda e=extension, p=probe_lanes: run_gs(e, p))
            row["selector_timing"] = timed(
                lambda e=extension, p=probe_lanes: (logits(), run_gs(e, p))
            )
            result["variants"].append(row)
            save()
            print("VARIANT", json.dumps(row), flush=True)

        selector.logits.zero_()
        run_gs(selector.extension, 4)
        zero_reference = selector.seeds.clone()
        for row, (_, extension, probe_lanes) in zip(result["variants"], variants):
            run_gs(extension, probe_lanes)
            row["zero_logits_ties_exact"] = torch.equal(
                selector.seeds, zero_reference
            )
        result["valid"] = (
            not monitor_errors
            and all(
                row["mismatch_count"] == 0
                for row in result["variants"]
                if row["name"] != "register_halving_fp32_p4"
            )
            and all(row["zero_logits_ties_exact"] for row in result["variants"])
        )
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
