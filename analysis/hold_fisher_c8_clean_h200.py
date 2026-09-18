"""Claim one clean H200 and continuously screen exact Fisher-C8 kernels.

The process waits without touching CUDA until the requested physical GPU is
clean.  After claiming it, the real layer-0 Q/K/V tensors and all discovered
selector and merge candidates stay resident while bounded rolling timing
statistics are written to disk.  New candidate sources are discovered without
releasing the CUDA context, so kernel iteration does not leave a gap in GPU
ownership.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.launch_quality_first_videos import gpu_processes, gpu_state


DEFAULT_CAPTURE = Path(
    "/local2/denghaoran/GridMergeAttention/experiments/"
    "gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt"
)
DEFAULT_SOURCE_GLOB = "native_fisher/fisher_c8_selector_fused_v*.cu"
DEFAULT_BASELINE_SOURCE = Path("native_fisher/fisher_c8_selector_fused_v2_dbuf.cu")
DEFAULT_MERGE_SOURCE_GLOB = "native_fisher/fisher_c8_merge_*.cu"
DEFAULT_ROUTE_SOURCE_GLOB = "native_fisher/fisher_c8_route_*.cu"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def clean_snapshot(index: int) -> dict | None:
    state = gpu_state(index)
    if (
        "H200" not in state["name"]
        or state["memory_mib"] > 100
        or state["utilization_percent"] != 0
    ):
        return None
    processes = gpu_processes(state["uuid"])
    if processes:
        return None
    return {"utc": now(), "gpu": state, "compute_processes": processes}


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RollingTiming:
    def __init__(self, recent_limit: int = 512):
        self.count = 0
        self.total_ms = 0.0
        self.minimum_ms = float("inf")
        self.maximum_ms = 0.0
        self.recent = deque(maxlen=recent_limit)

    def add(self, elapsed_ms: float) -> None:
        self.count += 1
        self.total_ms += elapsed_ms
        self.minimum_ms = min(self.minimum_ms, elapsed_ms)
        self.maximum_ms = max(self.maximum_ms, elapsed_ms)
        self.recent.append(elapsed_ms)

    def record(self) -> dict:
        if not self.count:
            return {"count": 0}
        record = {
            "count": self.count,
            "mean_ms": self.total_ms / self.count,
            "min_ms": self.minimum_ms,
            "max_ms": self.maximum_ms,
            "recent_count": len(self.recent),
            "recent_median_ms": statistics.median(self.recent),
            "recent_mean_ms": statistics.fmean(self.recent),
        }
        if len(self.recent) > 1:
            stdev = statistics.stdev(self.recent)
            sem = stdev / len(self.recent) ** 0.5
            record.update(
                recent_stdev_ms=stdev,
                recent_sem_ms=sem,
                recent_t_vs_zero=record["recent_mean_ms"] / sem if sem else None,
            )
        return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--source-glob", default=DEFAULT_SOURCE_GLOB)
    parser.add_argument("--baseline-source", type=Path, default=DEFAULT_BASELINE_SOURCE)
    parser.add_argument("--merge-source-glob", default=DEFAULT_MERGE_SOURCE_GLOB)
    parser.add_argument("--route-source-glob", default=DEFAULT_ROUTE_SOURCE_GLOB)
    parser.add_argument("--combined-merge-source", type=Path)
    parser.add_argument("--combined-route-source", type=Path)
    parser.add_argument(
        "--combined-overlap",
        action="store_true",
        help="double-buffer and overlap the combined candidate's merge/route chunks",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--stable-gap-seconds", type=float, default=0.1)
    parser.add_argument("--scan-seconds", type=float, default=30.0)
    parser.add_argument("--save-seconds", type=float, default=15.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--claim-leave-free-gib", type=float, default=12.0)
    parser.add_argument("--claim-chunk-gib", type=float, default=8.0)
    parser.add_argument("--backend-headroom-gib", type=float, default=42.0)
    parser.add_argument("--steady-leave-free-gib", type=float, default=4.0)
    parser.add_argument("--full-every-rounds", type=int, default=20)
    parser.add_argument("--stage-every-full", type=int, default=10)
    args = parser.parse_args()
    if (
        not args.gpus
        or any(index < 0 for index in args.gpus)
        or min(args.poll_seconds, args.stable_gap_seconds, args.scan_seconds, args.save_seconds) <= 0
        or args.stable_checks < 2
        or args.warmup < 1
        or args.claim_leave_free_gib <= 0
        or args.claim_chunk_gib <= 0
        or args.backend_headroom_gib <= args.claim_leave_free_gib
        or args.steady_leave_free_gib <= 0
        or args.steady_leave_free_gib >= args.claim_leave_free_gib
        or args.full_every_rounds < 1
        or args.stage_every_full < 1
    ):
        parser.error("invalid GPU, polling, timing, warmup, or reservation setting")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU indices must be unique")
    if (args.combined_merge_source is None) != (args.combined_route_source is None):
        parser.error("combined merge and route sources must be provided together")
    if args.combined_overlap and args.combined_merge_source is None:
        parser.error("combined overlap requires combined merge and route sources")
    if args.combined_overlap and args.backend_headroom_gib < 75:
        parser.error("combined overlap requires at least 75 GiB of backend headroom")
    if not args.capture.is_file():
        parser.error(f"capture does not exist: {args.capture}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "resident_state.json"
    events_path = args.output_dir / "resident_events.jsonl"
    lock_path = args.output_dir / "resident.lock"
    lock_stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another resident process owns {lock_path}") from error
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(f"{os.getpid()}\n")
    lock_stream.flush()

    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())

    state = {
        "method": "fisher_geometric_c8_fp32_fused_resident_complete_screen",
        "capacity": 8,
        "singleton_count": 0,
        "arithmetic": "fp32",
        "pid": os.getpid(),
        "host": os.uname().nodename,
        "gpu_indices": args.gpus,
        "capture": str(args.capture),
        "source_glob": args.source_glob,
        "baseline_source": str(args.baseline_source),
        "merge_source_glob": args.merge_source_glob,
        "route_source_glob": args.route_source_glob,
        "combined_merge_source": str(args.combined_merge_source) if args.combined_merge_source else None,
        "combined_route_source": str(args.combined_route_source) if args.combined_route_source else None,
        "combined_overlap": args.combined_overlap,
        "status": "starting",
        "candidates": {},
        "merge_candidates": {},
        "route_candidates": {},
        "combined_candidates": {},
        "contamination_events": [],
    }

    def event(kind: str, **fields) -> None:
        record = {"utc": now(), "event": kind, **fields}
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
        state.update(status=kind, updated_utc=record["utc"])
        write_json(state_path, state)

    event("waiting_for_clean_gpu")
    initial = None
    last_wait_event = 0.0
    while not stopped.is_set():
        candidate = None
        query_failures = []
        for index in args.gpus:
            try:
                candidate = clean_snapshot(index)
            except Exception as error:
                query_failures.append({"gpu": index, "error": repr(error)})
                continue
            if candidate is not None:
                break
        if candidate is None:
            if time.monotonic() - last_wait_event >= 60:
                event("waiting_for_clean_gpu", query_failures=query_failures)
                last_wait_event = time.monotonic()
            stopped.wait(args.poll_seconds)
            continue
        snapshots = [candidate]
        stable = True
        for _ in range(1, args.stable_checks):
            if stopped.wait(args.stable_gap_seconds):
                stable = False
                break
            snapshot = clean_snapshot(candidate["gpu"]["index"])
            if snapshot is None or snapshot["gpu"]["uuid"] != candidate["gpu"]["uuid"]:
                stable = False
                break
            snapshots.append(snapshot)
        if stable:
            initial = candidate
            state["claim_snapshots"] = snapshots
            break
        event("clean_candidate_lost")

    if initial is None:
        event("stopped_before_claim")
        return 130

    uuid = initial["gpu"]["uuid"]
    state["selected_gpu_index"] = initial["gpu"]["index"]
    state["selected_gpu_uuid"] = uuid
    os.environ.update(
        CUDA_VISIBLE_DEVICES=uuid,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        OMP_NUM_THREADS="4",
        MAX_JOBS="2",
        TORCH_CUDA_ARCH_LIST="9.0a",
        TORCH_EXTENSIONS_DIR=str(args.output_dir / "torch_extensions"),
        TRITON_CACHE_DIR=str(args.output_dir / "triton_cache"),
    )
    event("claiming_gpu", gpu=initial["gpu"])

    # Import and establish CUDA allocations only after the physical device is
    # identified.  Reserve most otherwise-free memory in bounded chunks before
    # CPU loading or compilation.  This closes the launch window and prevents a
    # large unrelated service from being placed on the same GPU while leaving
    # enough headroom for the real selector inputs and extension workspaces.
    import torch

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    reservations = []
    leave_free_bytes = int(args.claim_leave_free_gib * 1024**3)
    chunk_bytes = int(args.claim_chunk_gib * 1024**3)
    while True:
        free_bytes, _ = torch.cuda.mem_get_info()
        reservable = free_bytes - leave_free_bytes
        if reservable <= 0:
            break
        allocation_bytes = min(reservable, chunk_bytes)
        reservations.append(
            torch.empty(allocation_bytes, device="cuda", dtype=torch.uint8)
        )
    torch.cuda.synchronize()
    reserved_bytes = sum(tensor.numel() for tensor in reservations)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    event(
        "cuda_memory_claimed",
        claim_reserved_bytes=reserved_bytes,
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        leave_free_gib=args.claim_leave_free_gib,
    )

    payload = torch.load(args.capture, map_location="cpu", weights_only=True)
    q_full = payload["q"].reshape(1, 21, 45, 80, 40, 128).cuda()
    q = q_full[:, :1]
    k = payload["k"].reshape(1, 21, 45, 80, 40, 128).cuda()
    v = payload["v"].reshape(1, 21, 45, 80, 40, 128).cuda()
    del payload
    event(
        "inputs_resident",
        q_shape=list(q.shape),
        k_shape=list(k.shape),
        v_shape=list(v.shape),
        allocated_bytes=torch.cuda.memory_allocated(),
        claim_reserved_bytes=reserved_bytes,
    )

    monitor_lock = threading.Lock()
    foreign_processes = []

    def monitor() -> None:
        nonlocal foreign_processes
        contaminated = False
        while not stopped.is_set():
            try:
                processes = gpu_processes(uuid)
                current = [row for row in processes if row["pid"] != os.getpid()]
            except Exception as error:
                current = [{"monitor_error": repr(error)}]
            with monitor_lock:
                foreign_processes = current
            if current and not contaminated:
                state["contamination_events"].append({"utc": now(), "processes": current})
                event("foreign_process_detected", processes=current)
                contaminated = True
            elif not current and contaminated:
                event("exclusive_again")
                contaminated = False
            stopped.wait(2.0)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    from native_fisher.fused_c8_selector import NativeFusedFisherC8Selector

    selectors = {}
    timings = {}
    next_scan = 0.0
    next_save = 0.0
    baseline_key = None
    baseline_seeds = None
    zero_q = None
    attempted_digests = set()
    attempted_merge_digests = set()
    attempted_route_digests = set()
    backend = None
    full_output = None
    reference_output = None
    complete_timing = RollingTiming()
    stage_timings = {
        name: RollingTiming()
        for name in ("instrumented_total", "selector", "pack", "merge_total", "route_total")
    }
    merge_chunk_timings = [RollingTiming() for _ in range(8)]
    route_chunk_timings = [RollingTiming() for _ in range(8)]
    merge_candidates = {}
    merge_timings = {}
    merge_stage_timings = {}
    merge_pair_timings = {}
    production_merge_key = None
    route_candidates = {}
    route_timings = {}
    route_stage_timings = {}
    route_pair_timings = {}
    production_route_key = None
    combined_candidates = {}
    combined_timings = {}
    combined_stage_timings = {}
    combined_pair_timings = {}
    production_combined_key = None
    combined_registered = False
    selector_rounds = 0
    complete_calls = 0

    def save_state(status: str = "running") -> None:
        with monitor_lock:
            current_foreign = list(foreign_processes)
        state.update(
            status=status,
            updated_utc=now(),
            exclusive=not current_foreign,
            foreign_processes=current_foreign,
            candidates={
                key: record | {"timing": timings[key].record()}
                for key, record in state["candidates"].items()
            },
            full_timing={
                "complete_uninstrumented": complete_timing.record(),
                "stages": {name: timing.record() for name, timing in stage_timings.items()},
                "merge_chunks": [timing.record() for timing in merge_chunk_timings],
                "route_chunks": [timing.record() for timing in route_chunk_timings],
            },
            merge_candidates={
                key: record
                | (
                    {"complete_uninstrumented": merge_timings[key].record()}
                    | (
                        {
                            "paired_saving_vs_production_ms": merge_pair_timings[
                                key
                            ].record()
                        }
                        if key in merge_pair_timings
                        else {}
                    )
                    | {
                        "stages": {
                            name: timing.record()
                            for name, timing in merge_stage_timings[key]["totals"].items()
                        },
                        "merge_chunks": [
                            timing.record() for timing in merge_stage_timings[key]["merge_chunks"]
                        ],
                        "route_chunks": [
                            timing.record() for timing in merge_stage_timings[key]["route_chunks"]
                        ],
                    }
                    if key in merge_timings
                    else {}
                )
                for key, record in state["merge_candidates"].items()
            },
            route_candidates={
                key: record
                | (
                    {"complete_uninstrumented": route_timings[key].record()}
                    | (
                        {
                            "paired_saving_vs_production_ms": route_pair_timings[
                                key
                            ].record()
                        }
                        if key in route_pair_timings
                        else {}
                    )
                    | {
                        "stages": {
                            name: timing.record()
                            for name, timing in route_stage_timings[key]["totals"].items()
                        },
                        "merge_chunks": [
                            timing.record() for timing in route_stage_timings[key]["merge_chunks"]
                        ],
                        "route_chunks": [
                            timing.record() for timing in route_stage_timings[key]["route_chunks"]
                        ],
                    }
                    if key in route_timings
                    else {}
                )
                for key, record in state["route_candidates"].items()
            },
            combined_candidates={
                key: record
                | (
                    {"complete_uninstrumented": combined_timings[key].record()}
                    | (
                        {
                            "paired_saving_vs_production_ms": combined_pair_timings[
                                key
                            ].record()
                        }
                        if key in combined_pair_timings
                        else {}
                    )
                    | {
                        "stages": {
                            name: timing.record()
                            for name, timing in combined_stage_timings[key]["totals"].items()
                        },
                        "merge_chunks": [
                            timing.record()
                            for timing in combined_stage_timings[key]["merge_chunks"]
                        ],
                        "route_chunks": [
                            timing.record()
                            for timing in combined_stage_timings[key]["route_chunks"]
                        ],
                    }
                    if key in combined_timings
                    else {}
                )
                for key, record in state["combined_candidates"].items()
            },
        )
        write_json(state_path, state)

    def discover() -> None:
        nonlocal baseline_key, baseline_seeds, zero_q
        baseline_source = args.baseline_source
        if not baseline_source.is_absolute():
            baseline_source = ROOT / baseline_source
        sources = [baseline_source]
        sources.extend(path for path in sorted(ROOT.glob(args.source_glob)) if path != baseline_source)
        for source in sources:
            digest = source_digest(source)
            key = f"{source.stem}:{digest[:12]}"
            if digest in attempted_digests:
                continue
            attempted_digests.add(digest)
            try:
                selector = NativeFusedFisherC8Selector(source_path=source)
                for _ in range(args.warmup):
                    selector(q, k)
                torch.cuda.synchronize()
                seeds = selector(q, k).clone()
                torch.cuda.synchronize()
                if baseline_seeds is None:
                    baseline_key = key
                    baseline_seeds = seeds
                    zero_q = torch.zeros_like(q)
                mismatch = seeds != baseline_seeds
                zero_tie = torch.equal(selector(zero_q, k), selectors[baseline_key](zero_q, k)) if selectors else True
                selectors[key] = selector
                timings[key] = RollingTiming()
                state["candidates"][key] = {
                    "source": str(source.relative_to(ROOT)),
                    "source_sha256": digest,
                    "resources": selector.resources(),
                    "baseline": key == baseline_key,
                    "baseline_key": baseline_key,
                    "mismatch_count": int(mismatch.sum()),
                    "indices": mismatch.numel(),
                    "zero_query_tie": zero_tie,
                    "loaded_utc": now(),
                }
                event(
                    "candidate_loaded",
                    candidate=key,
                    mismatch_count=int(mismatch.sum()),
                    zero_query_tie=zero_tie,
                )
            except Exception as error:
                state.setdefault("load_failures", {})[key] = {
                    "utc": now(),
                    "source": str(source),
                    "error": repr(error),
                }
                event("candidate_load_failed", candidate=key, error=repr(error))

    def call_complete(
        *, profile: bool, merge_module=None, route_module=None, overlap_runner=None
    ) -> dict:
        selector = selectors[baseline_key]
        merger = backend.merge if merge_module is None else merge_module
        router = backend.route if route_module is None else route_module
        if not profile:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            seeds = selector(q, k)
            backend.merge.pack(q_full, backend.packed_q)
            if overlap_runner is None:
                for head in range(0, 40, 5):
                    merger.run(
                        backend.packed_q, k, v, seeds,
                        backend.mk, backend.mv, backend.bias,
                        head, 5, 105, 128 ** -0.5, backend.empty, 0, 1,
                    )
                    router.run(
                        q_full, backend.mk, backend.mv, backend.bias,
                        full_output, head, 5, 128 ** -0.5,
                    )
            else:
                overlap_runner.run(q_full, k, v, seeds, full_output, 128 ** -0.5)
            end.record()
            end.synchronize()
            return {"complete": start.elapsed_time(end)}

        if overlap_runner is not None:
            raise ValueError("per-stage profiling is undefined for overlapped execution")

        boundaries = [torch.cuda.Event(enable_timing=True) for _ in range(19)]
        boundaries[0].record()
        seeds = selector(q, k)
        boundaries[1].record()
        backend.merge.pack(q_full, backend.packed_q)
        boundaries[2].record()
        cursor = 3
        for head in range(0, 40, 5):
            merger.run(
                backend.packed_q, k, v, seeds,
                backend.mk, backend.mv, backend.bias,
                head, 5, 105, 128 ** -0.5, backend.empty, 0, 1,
            )
            boundaries[cursor].record()
            cursor += 1
            router.run(
                q_full, backend.mk, backend.mv, backend.bias,
                full_output, head, 5, 128 ** -0.5,
            )
            boundaries[cursor].record()
            cursor += 1
        boundaries[-1].synchronize()
        merge_chunks = []
        route_chunks = []
        cursor = 2
        for _ in range(8):
            merge_chunks.append(boundaries[cursor].elapsed_time(boundaries[cursor + 1]))
            cursor += 1
            route_chunks.append(boundaries[cursor].elapsed_time(boundaries[cursor + 1]))
            cursor += 1
        return {
            "instrumented_total": boundaries[0].elapsed_time(boundaries[-1]),
            "selector": boundaries[0].elapsed_time(boundaries[1]),
            "pack": boundaries[1].elapsed_time(boundaries[2]),
            "merge_chunks": merge_chunks,
            "route_chunks": route_chunks,
            "merge_total": sum(merge_chunks),
            "route_total": sum(route_chunks),
        }

    def initialize_full_backend() -> None:
        nonlocal backend, full_output, reference_output, reservations, reserved_bytes
        nonlocal production_merge_key, production_route_key, production_combined_key
        # Compile/load every extension while the initial hard reservation is
        # still in place.  Releasing 42 GiB before a cold compile leaves a
        # minutes-long placement window in which an unrelated job can enter.
        from kernels.best.c8_r2 import gma_c8_r2 as c8_r2_module

        c8_r2_module._extension("gma_c8_cuda_probe")
        c8_r2_module._extension("b1_gs_regpipe_v0")
        c8_r2_module._extension(c8_r2_module.MERGE_SOURCE, True)
        c8_r2_module._extension("gma_c8_route_pipeline", True)
        event("full_backend_extensions_ready")

        target_free_bytes = int(args.backend_headroom_gib * 1024**3)
        while reservations and torch.cuda.mem_get_info()[0] < target_free_bytes:
            reservation = reservations.pop()
            del reservation
            torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < target_free_bytes:
            raise RuntimeError(
                f"cannot create C8 back half: {free_bytes} free bytes below "
                f"{target_free_bytes} required headroom"
            )
        reserved_bytes = sum(tensor.numel() for tensor in reservations)
        event(
            "full_backend_headroom_ready",
            claim_reserved_bytes=reserved_bytes,
            free_bytes=free_bytes,
            total_bytes=total_bytes,
        )
        from kernels.best import create_c8_r2

        backend = create_c8_r2()
        full_output = torch.empty_like(q_full)
        state["back_half"] = {
            "implementation": "kernels.best.c8_r2",
            "merge_tiles": 105,
            "resources": backend.resources(),
        }
        for _ in range(2):
            call_complete(profile=False)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(full_output).all()):
            raise RuntimeError("resident exact-C8 complete output is non-finite")
        reference_output = full_output.clone()
        production_source = ROOT / "kernels/best/c8_r2/gma_c8_d10a_fused_argmax_v0.cu"
        production_digest = source_digest(production_source)
        production_key = f"production:{production_digest[:12]}"
        production_merge_key = production_key
        merge_candidates[production_key] = backend.merge
        merge_timings[production_key] = RollingTiming()
        merge_stage_timings[production_key] = new_merge_stage_timings()
        state["merge_candidates"][production_key] = {
            "source": str(production_source.relative_to(ROOT)),
            "source_sha256": production_digest,
            "baseline": True,
            "eligible": True,
            "finite": True,
            "relative_l2": 0.0,
            "max_abs": 0.0,
            "mismatch_count": 0,
            "resources": backend.resources(),
            "loaded_utc": now(),
        }
        production_route_source = ROOT / "kernels/best/c8_r2/gma_c8_route_pipeline.cu"
        production_route_digest = source_digest(production_route_source)
        production_route_key = f"production_route:{production_route_digest[:12]}"
        route_candidates[production_route_key] = backend.route
        route_timings[production_route_key] = RollingTiming()
        route_stage_timings[production_route_key] = new_merge_stage_timings()
        state["route_candidates"][production_route_key] = {
            "source": str(production_route_source.relative_to(ROOT)),
            "source_sha256": production_route_digest,
            "baseline": True,
            "eligible": True,
            "finite": True,
            "relative_l2": 0.0,
            "max_abs": 0.0,
            "mismatch_count": 0,
            "loaded_utc": now(),
        }
        if args.combined_merge_source is not None:
            production_combined_key = "production_combined"
            combined_candidates[production_combined_key] = (
                backend.merge,
                backend.route,
                None,
            )
            combined_timings[production_combined_key] = RollingTiming()
            combined_stage_timings[production_combined_key] = new_merge_stage_timings()
            state["combined_candidates"][production_combined_key] = {
                "merge_source": str(production_source.relative_to(ROOT)),
                "merge_source_sha256": production_digest,
                "route_source": str(production_route_source.relative_to(ROOT)),
                "route_source_sha256": production_route_digest,
                "baseline": True,
                "eligible": True,
                "finite": True,
                "relative_l2": 0.0,
                "max_abs": 0.0,
                "mismatch_count": 0,
                "loaded_utc": now(),
            }
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        state["claim_reserved_bytes"] = reserved_bytes
        state["resident_allocated_bytes"] = torch.cuda.memory_allocated()
        event(
            "full_backend_resident",
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            claim_reserved_bytes=reserved_bytes,
            resources=backend.resources(),
        )

    def new_merge_stage_timings() -> dict:
        return {
            "totals": {
                name: RollingTiming()
                for name in ("instrumented_total", "selector", "pack", "merge_total", "route_total")
            },
            "merge_chunks": [RollingTiming() for _ in range(8)],
            "route_chunks": [RollingTiming() for _ in range(8)],
        }

    def compare_output(actual, expected, chunk_elements: int = 8 * 1024 * 1024) -> dict:
        flat_actual = actual.reshape(-1)
        flat_expected = expected.reshape(-1)
        diff_sq = 0.0
        reference_sq = 0.0
        maximum = 0.0
        mismatches = 0
        finite = True
        for start in range(0, flat_actual.numel(), chunk_elements):
            stop = min(start + chunk_elements, flat_actual.numel())
            a = flat_actual[start:stop]
            b = flat_expected[start:stop]
            finite = finite and bool(torch.isfinite(a).all())
            mismatches += int((a != b).sum())
            delta = a.float() - b.float()
            maximum = max(maximum, float(delta.abs().max()))
            diff_sq += float(torch.sum(delta * delta, dtype=torch.float64))
            bf = b.float()
            reference_sq += float(torch.sum(bf * bf, dtype=torch.float64))
        return {
            "finite": finite,
            "mismatch_count": mismatches,
            "relative_l2": (diff_sq / reference_sq) ** 0.5,
            "max_abs": maximum,
        }

    def discover_merges() -> None:
        from native_fisher.fused_c8_merge import NativeFusedFisherC8Merge

        for source in sorted(ROOT.glob(args.merge_source_glob)):
            digest = source_digest(source)
            key = f"{source.stem}:{digest[:12]}"
            if digest in attempted_merge_digests:
                continue
            attempted_merge_digests.add(digest)
            try:
                merger = NativeFusedFisherC8Merge(source)
                sample = call_complete(profile=False, merge_module=merger)
                del sample
                torch.cuda.synchronize()
                comparison = compare_output(full_output, reference_output)
                eligible = (
                    comparison["finite"]
                    and comparison["relative_l2"] <= 5e-4
                    and comparison["max_abs"] <= 0.125
                )
                state["merge_candidates"][key] = {
                    "source": str(source.relative_to(ROOT)),
                    "source_sha256": digest,
                    "baseline": False,
                    "eligible": eligible,
                    **comparison,
                    "resources": merger.resources(),
                    "loaded_utc": now(),
                }
                if eligible:
                    merge_candidates[key] = merger
                    merge_timings[key] = RollingTiming()
                    merge_stage_timings[key] = new_merge_stage_timings()
                    merge_pair_timings[key] = RollingTiming()
                event("merge_candidate_loaded", candidate=key, eligible=eligible, **comparison)
            except Exception as error:
                state.setdefault("merge_load_failures", {})[key] = {
                    "utc": now(),
                    "source": str(source),
                    "error": repr(error),
                }
                event("merge_candidate_load_failed", candidate=key, error=repr(error))

    def discover_routes() -> None:
        from native_fisher.fused_c8_route import NativeFusedFisherC8Route

        for source in sorted(ROOT.glob(args.route_source_glob)):
            digest = source_digest(source)
            key = f"{source.stem}:{digest[:12]}"
            if digest in attempted_route_digests:
                continue
            attempted_route_digests.add(digest)
            try:
                router = NativeFusedFisherC8Route(source)
                sample = call_complete(profile=False, route_module=router)
                del sample
                torch.cuda.synchronize()
                comparison = compare_output(full_output, reference_output)
                eligible = (
                    comparison["finite"]
                    and comparison["relative_l2"] <= 5e-4
                    and comparison["max_abs"] <= 0.125
                )
                state["route_candidates"][key] = {
                    "source": str(source.relative_to(ROOT)),
                    "source_sha256": digest,
                    "baseline": False,
                    "eligible": eligible,
                    **comparison,
                    "loaded_utc": now(),
                }
                if eligible:
                    route_candidates[key] = router
                    route_timings[key] = RollingTiming()
                    route_stage_timings[key] = new_merge_stage_timings()
                    route_pair_timings[key] = RollingTiming()
                event("route_candidate_loaded", candidate=key, eligible=eligible, **comparison)
            except Exception as error:
                state.setdefault("route_load_failures", {})[key] = {
                    "utc": now(),
                    "source": str(source),
                    "error": repr(error),
                }
                event("route_candidate_load_failed", candidate=key, error=repr(error))

    def register_combined() -> None:
        nonlocal combined_registered
        if combined_registered or args.combined_merge_source is None:
            return
        combined_registered = True
        from native_fisher.fused_c8_merge import NativeFusedFisherC8Merge
        from native_fisher.fused_c8_route import NativeFusedFisherC8Route

        merge_source = args.combined_merge_source
        route_source = args.combined_route_source
        if not merge_source.is_absolute():
            merge_source = ROOT / merge_source
        if not route_source.is_absolute():
            route_source = ROOT / route_source
        merge_digest = source_digest(merge_source)
        route_digest = source_digest(route_source)
        key = f"combined:{merge_digest[:12]}+{route_digest[:12]}"
        if args.combined_overlap:
            key += "+overlap2s"
        try:
            merger = NativeFusedFisherC8Merge(merge_source)
            router = NativeFusedFisherC8Route(route_source)
            overlap_runner = None
            if args.combined_overlap:
                from kernels.best.c8_r2.overlap import MergeRouteOverlap

                overlap_runner = MergeRouteOverlap(backend, merger, router)
            call_complete(
                profile=False,
                merge_module=merger,
                route_module=router,
                overlap_runner=overlap_runner,
            )
            torch.cuda.synchronize()
            comparison = compare_output(full_output, reference_output)
            eligible = (
                comparison["finite"]
                and comparison["relative_l2"] <= 5e-4
                and comparison["max_abs"] <= 0.125
            )
            state["combined_candidates"][key] = {
                "merge_source": str(merge_source.relative_to(ROOT)),
                "merge_source_sha256": merge_digest,
                "route_source": str(route_source.relative_to(ROOT)),
                "route_source_sha256": route_digest,
                "baseline": False,
                "eligible": eligible,
                **comparison,
                "merge_resources": merger.resources(),
                "schedule": "two_stream_double_buffer" if overlap_runner else "serial",
                "overlap_extra_bytes": overlap_runner.extra_bytes if overlap_runner else 0,
                "loaded_utc": now(),
            }
            if eligible:
                combined_candidates[key] = (merger, router, overlap_runner)
                combined_timings[key] = RollingTiming()
                combined_stage_timings[key] = new_merge_stage_timings()
                combined_pair_timings[key] = RollingTiming()
            event("combined_candidate_loaded", candidate=key, eligible=eligible, **comparison)
        except Exception as error:
            state.setdefault("combined_load_failures", {})[key] = {
                "utc": now(),
                "merge_source": str(merge_source),
                "route_source": str(route_source),
                "error": repr(error),
            }
            event("combined_candidate_load_failed", candidate=key, error=repr(error))

    def tighten_steady_claim() -> None:
        nonlocal reservations, reserved_bytes
        target_free_bytes = int(args.steady_leave_free_gib * 1024**3)
        steady_chunk_bytes = 1024**3
        while True:
            free_bytes, _ = torch.cuda.mem_get_info()
            reservable = free_bytes - target_free_bytes
            if reservable <= 0:
                break
            allocation_bytes = min(reservable, steady_chunk_bytes)
            try:
                reservations.append(
                    torch.empty(allocation_bytes, device="cuda", dtype=torch.uint8)
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
        torch.cuda.synchronize()
        reserved_bytes = sum(tensor.numel() for tensor in reservations)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        state["claim_reserved_bytes"] = reserved_bytes
        state["resident_allocated_bytes"] = torch.cuda.memory_allocated()
        event(
            "steady_memory_claimed",
            claim_reserved_bytes=reserved_bytes,
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            leave_free_gib=args.steady_leave_free_gib,
        )

    try:
        while not stopped.is_set():
            moment = time.monotonic()
            if moment >= next_scan:
                discover()
                if backend is not None:
                    discover_merges()
                    discover_routes()
                    register_combined()
                next_scan = moment + args.scan_seconds
            if not selectors:
                save_state("waiting_for_candidate")
                stopped.wait(1.0)
                continue
            if backend is None:
                initialize_full_backend()
                discover_merges()
                discover_routes()
                register_combined()
                # An overlapped candidate allocates a second 30.5-GiB compressed
                # workspace.  Construct it before reservations consume the
                # remaining headroom, then tighten the one-GPU ownership claim.
                tighten_steady_claim()

            keys = list(selectors)
            for key in keys:
                selector = selectors[key]
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                selector(q, k)
                end.record()
                end.synchronize()
                with monitor_lock:
                    exclusive = not foreign_processes
                if exclusive:
                    timings[key].add(start.elapsed_time(end))
                if stopped.is_set():
                    break

            selector_rounds += 1
            if selector_rounds % args.full_every_rounds == 0 and not stopped.is_set():
                complete_calls += 1
                merge_items = list(merge_candidates.items())
                if complete_calls % 2 == 0:
                    merge_items.reverse()
                round_samples = {}
                for merge_key, merger in merge_items:
                    sample = call_complete(profile=False, merge_module=merger)
                    round_samples[merge_key] = sample["complete"]
                    with monitor_lock:
                        exclusive = not foreign_processes
                    if exclusive:
                        merge_timings[merge_key].add(sample["complete"])
                        if state["merge_candidates"][merge_key]["baseline"]:
                            complete_timing.add(sample["complete"])
                    if complete_calls % args.stage_every_full == 0:
                        profile = call_complete(profile=True, merge_module=merger)
                        with monitor_lock:
                            exclusive = not foreign_processes
                        if exclusive:
                            candidate_stages = merge_stage_timings[merge_key]
                            for name in candidate_stages["totals"]:
                                candidate_stages["totals"][name].add(profile[name])
                            for index, milliseconds in enumerate(profile["merge_chunks"]):
                                candidate_stages["merge_chunks"][index].add(milliseconds)
                            for index, milliseconds in enumerate(profile["route_chunks"]):
                                candidate_stages["route_chunks"][index].add(milliseconds)
                            if state["merge_candidates"][merge_key]["baseline"]:
                                for name in stage_timings:
                                    stage_timings[name].add(profile[name])
                                for index, milliseconds in enumerate(profile["merge_chunks"]):
                                    merge_chunk_timings[index].add(milliseconds)
                                for index, milliseconds in enumerate(profile["route_chunks"]):
                                    route_chunk_timings[index].add(milliseconds)
                if production_merge_key in round_samples:
                    production_ms = round_samples[production_merge_key]
                    with monitor_lock:
                        exclusive = not foreign_processes
                    if exclusive:
                        for merge_key, candidate_ms in round_samples.items():
                            if merge_key in merge_pair_timings:
                                merge_pair_timings[merge_key].add(
                                    production_ms - candidate_ms
                                )

                route_items = list(route_candidates.items())
                if complete_calls % 2 == 0:
                    route_items.reverse()
                route_round_samples = {}
                for route_key, router in route_items:
                    sample = call_complete(profile=False, route_module=router)
                    route_round_samples[route_key] = sample["complete"]
                    with monitor_lock:
                        exclusive = not foreign_processes
                    if exclusive:
                        route_timings[route_key].add(sample["complete"])
                    if complete_calls % args.stage_every_full == 0:
                        profile = call_complete(profile=True, route_module=router)
                        with monitor_lock:
                            exclusive = not foreign_processes
                        if exclusive:
                            candidate_stages = route_stage_timings[route_key]
                            for name in candidate_stages["totals"]:
                                candidate_stages["totals"][name].add(profile[name])
                            for index, milliseconds in enumerate(profile["merge_chunks"]):
                                candidate_stages["merge_chunks"][index].add(milliseconds)
                            for index, milliseconds in enumerate(profile["route_chunks"]):
                                candidate_stages["route_chunks"][index].add(milliseconds)
                if production_route_key in route_round_samples:
                    production_ms = route_round_samples[production_route_key]
                    with monitor_lock:
                        exclusive = not foreign_processes
                    if exclusive:
                        for route_key, candidate_ms in route_round_samples.items():
                            if route_key in route_pair_timings:
                                route_pair_timings[route_key].add(
                                    production_ms - candidate_ms
                                )

                combined_items = list(combined_candidates.items())
                if complete_calls % 2 == 0:
                    combined_items.reverse()
                combined_round_samples = {}
                for combined_key, (merger, router, overlap_runner) in combined_items:
                    sample = call_complete(
                        profile=False,
                        merge_module=merger,
                        route_module=router,
                        overlap_runner=overlap_runner,
                    )
                    combined_round_samples[combined_key] = sample["complete"]
                    with monitor_lock:
                        exclusive = not foreign_processes
                    if exclusive:
                        combined_timings[combined_key].add(sample["complete"])
                    if (
                        overlap_runner is None
                        and complete_calls % args.stage_every_full == 0
                    ):
                        profile = call_complete(
                            profile=True,
                            merge_module=merger,
                            route_module=router,
                        )
                        with monitor_lock:
                            exclusive = not foreign_processes
                        if exclusive:
                            candidate_stages = combined_stage_timings[combined_key]
                            for name in candidate_stages["totals"]:
                                candidate_stages["totals"][name].add(profile[name])
                            for index, milliseconds in enumerate(profile["merge_chunks"]):
                                candidate_stages["merge_chunks"][index].add(milliseconds)
                            for index, milliseconds in enumerate(profile["route_chunks"]):
                                candidate_stages["route_chunks"][index].add(milliseconds)
                if production_combined_key in combined_round_samples:
                    production_ms = combined_round_samples[production_combined_key]
                    with monitor_lock:
                        exclusive = not foreign_processes
                    if exclusive:
                        for combined_key, candidate_ms in combined_round_samples.items():
                            if combined_key in combined_pair_timings:
                                combined_pair_timings[combined_key].add(
                                    production_ms - candidate_ms
                                )

            moment = time.monotonic()
            if moment >= next_save:
                save_state()
                next_save = moment + args.save_seconds
    except BaseException as error:
        state["fatal_error"] = repr(error)
        save_state("fatal_error")
        raise
    finally:
        stopped.set()
        monitor_thread.join(10)
        save_state("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
