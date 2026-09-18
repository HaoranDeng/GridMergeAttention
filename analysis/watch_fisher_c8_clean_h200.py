"""Wait for one exclusively clean H200, then run the exact Fisher-C8 benchmark.

This watcher is intentionally outside a cluster scheduler.  It never starts on
a GPU with a foreign compute process, nonzero utilization, or more than the
driver's minimal allocation.  The benchmark repeats the same guard and keeps
monitoring the selected GPU.  A run contaminated after launch is retained as
an invalid attempt and the watcher resumes polling.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.launch_quality_first_videos import gpu_processes, gpu_state


DEFAULT_CAPTURE = Path(
    "/local2/denghaoran/GridMergeAttention/experiments/"
    "gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def clean_snapshot(index: int) -> dict | None:
    """Return an identified clean H200 snapshot, otherwise ``None``."""
    state = gpu_state(index)
    processes = gpu_processes(state["uuid"])
    if (
        "H200" not in state["name"]
        or processes
        or state["memory_mib"] > 100
        or state["utilization_percent"] != 0
    ):
        return None
    return {"utc": now(), "gpu": state, "compute_processes": processes}


def retryable_failure(receipt_path: Path) -> bool:
    """Only retry a launch race or a run invalidated by a foreign process."""
    if not receipt_path.exists():
        return True
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    errors = receipt.get("monitor_errors") or []
    return bool(errors) and all("foreign GPU process" in str(item) for item in errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--stable-gap-seconds", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=4)
    args = parser.parse_args()
    if (
        not args.gpus
        or any(index < 0 for index in args.gpus)
        or args.poll_seconds <= 0
        or args.stable_checks < 2
        or args.stable_gap_seconds <= 0
        or min(args.warmup, args.blocks) < 1
    ):
        parser.error("invalid GPU, polling, stability, or timing setting")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU indices must be unique")
    if not args.capture.is_file():
        parser.error(f"capture does not exist: {args.capture}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "watcher_state.json"
    events_path = args.output_dir / "watcher_events.jsonl"
    lock_path = args.output_dir / "watcher.lock"
    lock_stream = lock_path.open("a+", encoding="utf-8")
    try:
        import fcntl
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, BlockingIOError) as error:
        raise RuntimeError(f"another watcher owns {lock_path}") from error
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(f"{os.getpid()}\n")
    lock_stream.flush()

    stopped = False
    active_child: subprocess.Popen | None = None

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
        if active_child is not None and active_child.poll() is None:
            active_child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def event(kind: str, **fields) -> None:
        record = {"utc": now(), "event": kind, **fields}
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
        write_json(state_path, record | {"pid": os.getpid()})

    attempt = 0
    last_wait_event = 0.0
    event("started", gpus=args.gpus, poll_seconds=args.poll_seconds)
    while not stopped:
        candidate = None
        failures = []
        order = args.gpus[attempt % len(args.gpus):] + args.gpus[:attempt % len(args.gpus)]
        for index in order:
            try:
                candidate = clean_snapshot(index)
            except Exception as error:
                failures.append({"gpu": index, "error": repr(error)})
                continue
            if candidate is not None:
                break

        if candidate is None:
            monotonic = time.monotonic()
            if monotonic - last_wait_event >= 60:
                event("waiting", query_failures=failures)
                last_wait_event = monotonic
            time.sleep(args.poll_seconds)
            continue

        index = candidate["gpu"]["index"]
        uuid = candidate["gpu"]["uuid"]
        stable = True
        snapshots = [candidate]
        for _ in range(1, args.stable_checks):
            time.sleep(args.stable_gap_seconds)
            try:
                snapshot = clean_snapshot(index)
            except Exception as error:
                event("candidate_query_failed", gpu=index, error=repr(error))
                stable = False
                break
            if snapshot is None or snapshot["gpu"]["uuid"] != uuid:
                stable = False
                break
            snapshots.append(snapshot)
        if not stable:
            event("candidate_lost", gpu=index, uuid=uuid)
            time.sleep(args.poll_seconds)
            continue

        attempt += 1
        receipt = args.output_dir / f"layer00_attempt{attempt:03d}_gpu{index}.json"
        log = receipt.with_suffix(".log")
        command = [
            sys.executable,
            str(ROOT / "analysis/benchmark_fisher_c8_exact.py"),
            "--gpu", str(index),
            "--capture", str(args.capture),
            "--output", str(receipt),
            "--warmup", str(args.warmup),
            "--blocks", str(args.blocks),
        ]
        event(
            "launching",
            attempt=attempt,
            gpu=index,
            uuid=uuid,
            stable_snapshots=snapshots,
            receipt=str(receipt),
            log=str(log),
        )
        with log.open("xb") as stream:
            active_child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            while active_child.poll() is None and not stopped:
                time.sleep(1)
            if stopped and active_child.poll() is None:
                active_child.terminate()
            return_code = active_child.wait()
            active_child = None

        if stopped:
            event("stopped", attempt=attempt, gpu=index, return_code=return_code)
            return 130
        if return_code == 0:
            event("complete", attempt=attempt, gpu=index, receipt=str(receipt), log=str(log))
            return 0
        if retryable_failure(receipt):
            event(
                "retrying_after_contention",
                attempt=attempt,
                gpu=index,
                return_code=return_code,
                receipt=str(receipt),
                log=str(log),
            )
            time.sleep(args.poll_seconds)
            continue
        event(
            "fatal_child_failure",
            attempt=attempt,
            gpu=index,
            return_code=return_code,
            receipt=str(receipt),
            log=str(log),
        )
        return return_code or 1

    event("stopped_without_launch")
    return 130


if __name__ == "__main__":
    raise SystemExit(main())
