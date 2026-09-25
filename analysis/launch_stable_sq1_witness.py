"""Own one clean H200 for the bounded, off-path stable SQ1 witness control.

Requires an already completed primary attribution; never generates videos,
queues work, edits production, or accepts incomplete diagnostic evidence.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import diagnose_sq1_fixed_seed_witness as base
from analysis import launch_quality_first_videos as guard

ARTIFACTS = ("results.json", "stable_kernel.py", "stable_observer_kernel.py", "control_arrays.npz")
OWN_SOURCES = ("analysis/launch_stable_sq1_witness.py", "analysis/validate_stable_sq1_witness.py",
               "analysis/stable_sq1_bulk_kernel.py")


def account_devices():
    """Count all compute devices owned by this uid, not only our subprocess."""
    rows = csv.reader(io.StringIO(guard.smi("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits")))
    devices = set()
    for row in rows:
        if not row:
            continue
        base.require(len(row) == 2, "malformed account GPU inventory")
        uuid, raw_pid = (value.strip() for value in row)
        try:
            uid = (Path("/proc") / str(int(raw_pid))).stat().st_uid
        except FileNotFoundError:
            continue  # A process disappearing is not a new device claim.
        if uid == os.getuid():
            devices.add(uuid)
    return sorted(devices)


def account_limit():
    hostname = socket.gethostname().split(".")[0]
    base.require(hostname in ("scai6", "scai7"), "control is authorized only on scai6/scai7")
    return 2 if hostname == "scai6" else 1


def source_hashes():
    return base.source_hashes() | {name: base.digest(ROOT / name) for name in OWN_SOURCES}


def verify_sources(output, hashes):
    base.require(source_hashes() == hashes, "live control source changed")
    base.require(all(base.digest(output / "sources" / name) == sha for name, sha in hashes.items()),
                 "frozen control source changed")


def parent(args):
    from analysis import validate_stable_sq1_witness as control
    base.require(not args.output.exists(), "fresh control output required; no resume")
    initial = guard.require_clean(args.gpu)
    limit = account_limit()
    existing = account_devices()
    base.require(len(existing) < limit, "account GPU cap has no remaining slot")
    control.read_completed_primary(args.primary, old_root=args.old_root)
    hashes = source_hashes()
    args.output.mkdir(parents=True)
    for name, sha in hashes.items():
        target = args.output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
        base.require(base.digest(target) == sha, "control source copy mismatch")
    verify_sources(args.output, hashes)
    process, failure, status, artifacts = None, None, None, None
    foreign, polls, previous = [], 0, {}
    started = guard.now()

    def interrupted(number, _frame):
        raise KeyboardInterrupt(f"received signal {number}")

    try:
        previous = {number: signal.signal(number, interrupted) for number in (signal.SIGTERM, signal.SIGINT)}
        immediate = guard.require_clean(args.gpu, expected_uuid=initial["gpu"]["uuid"])
        base.require(len(account_devices()) < limit, "account GPU cap reached during preparation")
        base.write_json(args.output / "startup.json", dict(parent_pid=os.getpid(), initial=initial,
            immediately_before_spawn=immediate, source_sha256=hashes, source_root=str(ROOT),
            account_gpu_limit=limit, initial_account_devices=existing, monitor_interval_seconds=5,
            primary=str(args.primary), parent_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
        environment = guard.child_environment(args.output, initial["gpu"]["uuid"])
        environment["STABLE_SQ1_WITNESS_PARENT"] = str(os.getpid())
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--gpu", str(args.gpu),
                   "--primary", str(args.primary), "--output", str(args.output), "--old-root", str(args.old_root)]
        with (args.output / "worker.log").open("x", encoding="utf-8") as log, \
                (args.output / "gpu_monitor.jsonl").open("x", encoding="utf-8") as monitor:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            base.write_json(args.output / "process.json", dict(parent_pid=os.getpid(), worker_pid=process.pid,
                command=command, cwd=str(ROOT), spawned_utc=guard.now(),
                cuda_visible_devices=environment["CUDA_VISIBLE_DEVICES"]))
            while True:
                active = guard.gpu_processes(initial["gpu"]["uuid"])
                owned = account_devices()
                polls += 1
                monitor.write(json.dumps(dict(utc=guard.now(), compute_processes=active,
                                              account_devices=owned)) + "\n")
                monitor.flush()
                foreign = [record for record in active if record["pid"] != process.pid]
                base.require(not foreign, "foreign process detected; stop only own child")
                base.require(len(owned) <= limit, "account GPU cap exceeded; stop only own child")
                if process.poll() is not None:
                    break
                time.sleep(5)
            status = process.wait(timeout=15)
            base.require(status == 0, f"stable control worker exit {status}")
            report = json.loads((args.output / "results.json").read_text())
            control.validate_completed_result(report)
            verify_sources(args.output, hashes)
            base.require(base.digest(args.output / "control_arrays.npz") == report["arrays"]["sha256"],
                         "control arrays changed")
            artifacts = {name: base.digest(args.output / name) for name in ARTIFACTS}
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        try:
            guard.stop_own_child(process)
            if process is not None:
                status = process.returncode
        except BaseException as error:
            failure = (failure + "; " if failure else "") + f"cleanup: {error}"
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
        completion = dict(valid=failure is None and status == 0 and artifacts is not None and not foreign,
            started_utc=started, completed_utc=guard.now(), worker_exit_code=status,
            parent_pid=os.getpid(), worker_pid=process.pid if process else None,
            requested_gpu=args.gpu, gpu_uuid=initial["gpu"]["uuid"], account_gpu_limit=limit,
            foreign=foreign, failure=failure, monitor_interval_seconds=5, monitor_polls=polls,
            verified_artifacts=artifacts, timing_benchmark=False, video_quality_result=False)
        base.write_json(args.output / "completion.json", completion)
    base.require(completion["valid"], f"stable control invalid: {failure}; inspect worker.log")
    print(json.dumps(completion, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, default=base.OLD_ROOT)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.primary, args.output, args.old_root = (path.resolve() for path in (args.primary, args.output, args.old_root))
    if args.worker:
        from analysis import validate_stable_sq1_witness as control
        control.worker(args.primary, args.output, args.old_root)
    else:
        parent(args)


if __name__ == "__main__":
    main()
