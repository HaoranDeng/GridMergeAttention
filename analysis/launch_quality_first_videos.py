"""Single-clean-H200 parent guard for the isolated quality-first video worker.

Never queues, resumes, or chooses another GPU. Checks the requested physical
GPU before preparing a fresh snapshot and again immediately before spawning
the frozen worker. Polls compute processes every five seconds; contamination,
monitor failure, SIGINT, or SIGTERM stops only this launcher's own child.
Account-wide caps apply by default: scai6 <=2, scai7 <=1. Explicit task scope
owns one clean GPU while preserving unrelated account jobs on other devices.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis import run_quality_first_videos as worker
from analysis.quality_first_mode_contract import VIDEO_MODES, method_contract

POLL_SECONDS = 5
LAUNCHER_SOURCE = "analysis/launch_quality_first_videos.py"


def now():
    return datetime.now(timezone.utc).isoformat()


def smi(*arguments):
    return subprocess.check_output(["nvidia-smi", *arguments], text=True, timeout=10).strip()


def gpu_state(gpu):
    rows = list(csv.reader(io.StringIO(smi(
        "-i", str(gpu), "--query-gpu=index,name,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"))))
    worker.require(len(rows) == 1 and len(rows[0]) == 5, "GPU query must identify exactly one device")
    index, name, uuid, memory, utilization = (value.strip() for value in rows[0])
    return {"index": int(index), "name": name, "uuid": uuid,
            "memory_mib": int(memory), "utilization_percent": int(utilization)}


def gpu_processes(uuid):
    value = smi("-i", uuid, "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader,nounits")
    records = []
    for row in csv.reader(io.StringIO(value)):
        if not row:
            continue
        worker.require(len(row) >= 3, "malformed GPU-process query")
        records.append({"pid": int(row[0].strip()), "memory_mib": row[1].strip(),
                        "process_name": ",".join(row[2:]).strip()})
    return records


def account_devices():
    """Deduplicated compute GPUs for this uid, including unrelated tasks."""
    rows = csv.reader(io.StringIO(smi("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits")))
    devices = set()
    for row in rows:
        if not row:
            continue
        worker.require(len(row) == 2, "malformed account GPU inventory")
        uuid, raw_pid = (value.strip() for value in row)
        try:
            uid = (Path("/proc") / str(int(raw_pid))).stat().st_uid
        except FileNotFoundError:
            continue
        if uid == os.getuid():
            devices.add(uuid)
    return sorted(devices)


def account_limit():
    hostname = socket.gethostname().split(".")[0]
    worker.require(hostname in ("scai6", "scai7"), "new video runs are authorized only on scai6/scai7")
    return 2 if hostname == "scai6" else 1


def require_clean(gpu, *, expected_uuid=None):
    worker.require(type(gpu) is int and gpu >= 0, "exactly one nonnegative physical GPU index is required")
    state = gpu_state(gpu)
    worker.require(state["index"] == gpu and "H200" in state["name"] and state["uuid"].startswith("GPU-"),
                   "requested device is not a single identified H200")
    if expected_uuid is not None:
        worker.require(state["uuid"] == expected_uuid, "GPU identity changed during preparation")
    processes = gpu_processes(state["uuid"])
    worker.require(not processes and 0 <= state["memory_mib"] <= 100 and state["utilization_percent"] == 0,
                   "GPU is not exclusively clean; refusing to queue, wait, or launch")
    return {"utc": now(), "gpu": state, "compute_processes": processes}


def child_environment(run_dir, uuid, inherited=None):
    environment = dict(os.environ if inherited is None else inherited)
    run_dir = Path(run_dir).resolve()
    library = str(Path(sys.prefix) / "lib")
    environment.update({
        "CUDA_VISIBLE_DEVICES": uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
        "WAN_SDPA_BACKEND": "flash", "TOKEN_MERGING_FORCE_TORCH_RMSNORM": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TRITON_CACHE_DIR": str(run_dir / "triton_cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(run_dir / "torchinductor_cache"),
        "FLASHINFER_WORKSPACE_BASE": str(run_dir / "flashinfer_workspace"),
        "FLASHINFER_CACHE_DIR": str(run_dir / "flashinfer_cache"),
        "PYTHONPYCACHEPREFIX": str(run_dir / "python_cache"),
        "LD_LIBRARY_PATH": library + (os.pathsep + environment["LD_LIBRARY_PATH"]
                                       if environment.get("LD_LIBRARY_PATH") else ""),
    })
    return environment


def freeze_launcher(run_dir):
    """Add this untracked/new parent source before the snapshot is executed."""
    run_dir = Path(run_dir)
    destination = run_dir / "source" / LAUNCHER_SOURCE
    original = Path(__file__).resolve()
    original_sha = worker.digest(original)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with destination.open("xb") as stream:
            stream.write(original.read_bytes())
    worker.require(original_sha == worker.digest(original) == worker.digest(destination), "launcher freeze mismatch")
    path = run_dir / "source_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    record = {"path": LAUNCHER_SOURCE, "sha256": original_sha, "bytes": destination.stat().st_size}
    existing = [item for item in provenance["files"] if item["path"] == LAUNCHER_SOURCE]
    worker.require(not existing or existing == [record], "launcher provenance conflicts")
    if not existing:
        provenance["files"].append(record)
        worker.write_json(path, provenance)
    return record


def stop_own_child(process):
    """Never signals a discovered/foreign PID or a process group."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def verify_completion(run_dir, method, checkpoint):
    run_dir = Path(run_dir)
    worker.require((run_dir / "generation.exitcode").read_text().strip() == "0", "worker generation exit code is not zero")
    audit_path = run_dir / "dispatch_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    prompt_indices = worker.planned_prompt_indices(worker.verify_snapshot(run_dir, method, checkpoint))
    num_videos = len(prompt_indices)
    total_calls = num_videos * 160
    if method in worker.C10_METHODS:
        proof = json.loads((run_dir / "native_validation.json").read_text())
        worker.require(proof.get("valid") is True and proof.get("c") == 10
                       and proof.get("physical_capacity") == 16
                       and proof.get("c4_prefix_exact") is True
                       and proof.get("nine_pivots_checked") is True,
                       "C10 native validation missing")
        if method == worker.FP8_C10_METHOD:
            worker.require(proof.get("fp8_kv", {}).get("valid") is True, "FP8 KV native proof missing")
            worker.require(audit["first_call_metadata"]["intermediate_kv"]["scaled_fp8_kv"] is True, "FP8 adapter not active")
    if method == "fisher_geometric_c8_rf0":
        proof = json.loads((run_dir / "native_validation.json").read_text())
        worker.require(proof.get("valid") is True and proof.get("c") == 8
                       and proof.get("c4_prefix_exact") is True
                       and proof.get("seven_pivots_checked") is True,
                       "C8 native validation missing")
    expected = {"complete": True, "videos_observed": num_videos, "self_attention_calls": total_calls,
                "core_calls": total_calls, "selector_calls": total_calls, "merge_route_head_chunk_calls": total_calls * worker.axis_chunks(method),
                "per_layer_calls": [num_videos * 4] * 40, "declaration": method_contract(method), "error": None}
    if method in worker.SYNTHETIC_METHODS:
        expected["synthetic_seed_kernel_calls"] = (total_calls * 5)
    if method in worker.MEAN_METHODS:
        expected["mean_seed_constructions"] = total_calls
    stable = method in worker.STABLE_SQ1_METHODS
    if stable:
        expected["stable_merge_kernel_calls"] = (total_calls * 5)
    spatial = method in worker.AXIS_METHODS
    if spatial:
        chunk_calls = total_calls * worker.axis_chunks(method)
        expected.update(stable_merge_kernel_calls=chunk_calls, stable_merge_wrapper_calls=chunk_calls,
                        spatial_route_kernel_calls=chunk_calls, synthetic_seed_kernel_calls=0, mean_seed_constructions=0)
    for key, value in expected.items():
        worker.require(audit.get(key) == value, f"incomplete dispatch audit: {key}")
    stable_receipt = None
    if stable:
        stable_receipt = audit["callable_identities"]["stable_merge_kernel"]
        stable_path = run_dir / "stable_kernel.py"
        worker.require(Path(stable_receipt["path"]).resolve() == stable_path.resolve()
                       and worker.digest(stable_path) == stable_receipt["sha256"],
                       "stable merge generated source changed")
    if spatial:
        operator = audit.get(worker.axis_payload_key(method), {})
        worker.require(operator.get("layout") == method_contract(method)["layout"]
                       and operator.get("dispatch_counts") == dict(core_calls=total_calls, selector_calls=total_calls,
                                                                   merge_wrapper_calls=chunk_calls, stable_kernel_calls=chunk_calls)
                       and operator.get("route_kernel_calls") == chunk_calls
                       and operator.get("dense_fallback") is False and operator.get("fp8") is False,
                       "spatial axis/source/call coverage changed")
        worker.verify_spatial_sources(operator.get("source_receipt"), run_dir / "source", method)
        if method in (*worker.TEMPORAL_METHODS, *worker.CAPACITY_METHODS):
            worker.require(operator.get("physical_reference_frame") == 0 and operator.get("head_chunk") == 40 // worker.axis_chunks(method),
                           "physical RF0/head chunk changed")
        worker.validate_capacity_metadata(operator, method)
        proof_path = run_dir / "native_validation.json"
        worker.validate_axis_proof(json.loads(proof_path.read_text()), run_dir / "source", method)
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        worker.require(manifest["native_validation"]["sha256"] == worker.digest(proof_path), "spatial native proof changed")
        worker.require(not (run_dir / "stable_kernel.py").exists() and "generated_stable_kernel" not in audit,
                       "spatial run must not claim old generated full80 merge kernel")
    runtime_path = run_dir / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    worker.require(runtime["checkpoint_weights"] == "raw_state_dict", "checkpoint branch mismatch")
    records = runtime["records"]
    worker.require(len(records) == num_videos and [row["prompt_index"] for row in records] == prompt_indices, "runtime prompt coverage mismatch")
    worker.require(all(row["attention_calls"] == row["full_attention_calls"] == 160 for row in records), "runtime attention counts mismatch")
    worker.verify_snapshot(run_dir, method, checkpoint)
    return {"dispatch_audit_sha256": worker.digest(audit_path), "runtime_sha256": worker.digest(runtime_path),
            "generation_exitcode_sha256": worker.digest(run_dir / "generation.exitcode"),
            "source_provenance_sha256": worker.digest(run_dir / "source_provenance.json")} | (
                {"stable_kernel_sha256": stable_receipt["sha256"]} if stable else {}) | (
                {"native_validation_sha256": worker.digest(run_dir / "native_validation.json")} if spatial else {})


def launch(gpu, run_dir, method, checkpoint, *, spatial_native_proof=None, gpu_scope="account", num_videos=20, prompt_start=0):
    run_dir = Path(run_dir).resolve()
    worker.require(not run_dir.exists(), "fresh run-dir required; no resume or overwrite")
    if method == worker.SPATIAL_TRANSPOSE_METHOD:
        worker.require(spatial_native_proof is not None and Path(spatial_native_proof).is_absolute(),
                       "spatial method requires explicit prior native-proof path")
    if method in worker.AXIS_METHODS and spatial_native_proof is not None:
        worker.require(Path(spatial_native_proof).is_absolute(), "native proof path must be absolute")
        spatial_native_proof = Path(spatial_native_proof).resolve(strict=True)
        worker.validate_axis_proof(json.loads(spatial_native_proof.read_text()), ROOT, method)
    elif method not in worker.AXIS_METHODS:
        worker.require(spatial_native_proof is None, "spatial proof given for another method")
    # Dirty initial state fails before any snapshot or subprocess is created.
    initial = require_clean(gpu)
    worker.require(gpu_scope in ("account", "task"), "invalid GPU ownership scope")
    # Task scope still owns exactly one exclusively clean GPU and monitors
    # foreign processes there; unrelated existing account jobs are untouched.
    limit = account_limit() if gpu_scope == "account" else None
    existing = account_devices()
    worker.require(limit is None or len(existing) < limit, "account GPU cap has no remaining slot; refusing to queue")
    last_account_devices = existing
    worker.asset_preflight(checkpoint)
    process = None
    previous_signals = {}
    status = None
    foreign = []
    failure = None
    monitor_polls = 0
    completion_artifacts = None
    started = now()

    def interrupted(number, _frame):
        raise KeyboardInterrupt(f"received signal {number}")

    try:
        previous_signals = {number: signal.signal(number, interrupted) for number in (signal.SIGTERM, signal.SIGINT)}
        worker.prepare(run_dir, method, checkpoint, num_videos=num_videos, prompt_start=prompt_start)
        launcher_identity = freeze_launcher(run_dir)
        worker.verify_snapshot(run_dir, method, checkpoint)
        immediate = require_clean(gpu, expected_uuid=initial["gpu"]["uuid"])
        last_account_devices = account_devices()
        worker.require(limit is None or len(last_account_devices) < limit, "account GPU cap reached during preparation")
        startup = {"parent_pid": os.getpid(), "requested_gpu": gpu, "initial": initial,
                   "immediately_before_spawn": immediate, "monitor_interval_seconds": POLL_SECONDS,
                   "account_gpu_limit": limit, "gpu_scope": gpu_scope,
                   "task_gpu_limit": 1, "initial_account_devices": existing,
                   "immediate_account_devices": last_account_devices,
                   "launcher_source": launcher_identity, "method": method + "_bf16score"}
        worker.write_json(run_dir / "startup.json", startup, exclusive=True)
        frozen_worker = run_dir / "source/analysis/run_quality_first_videos.py"
        command = [sys.executable, "-u", str(frozen_worker), "--method", method,
                   "--run-dir", str(run_dir), "--checkpoint", str(Path(checkpoint).resolve()),
                   "--execute-prepared", "--parent-purity-managed"]
        if spatial_native_proof is not None:
            command += ["--spatial-native-proof", str(spatial_native_proof)]
        environment = child_environment(run_dir, initial["gpu"]["uuid"])
        with (run_dir / "worker.log").open("x", encoding="utf-8") as log, \
             (run_dir / "purity.log").open("x", encoding="utf-8") as purity, \
             (run_dir / "gpu_monitor.jsonl").open("x", encoding="utf-8") as monitor:
            process = subprocess.Popen(command, cwd=run_dir / "source", env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            worker.write_json(run_dir / "process.json", {
                "parent_pid": os.getpid(), "worker_pid": process.pid, "spawned_utc": now(),
                "command": command, "cwd": str(run_dir / "source"),
                "environment_overrides": {name: environment.get(name) for name in (
                    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "WAN_SDPA_BACKEND", "LD_LIBRARY_PATH",
                    "TOKEN_MERGING_FORCE_TORCH_RMSNORM", "PYTORCH_CUDA_ALLOC_CONF", "TRITON_CACHE_DIR",
                    "TORCHINDUCTOR_CACHE_DIR", "PYTHONPYCACHEPREFIX", "FLASHINFER_WORKSPACE_BASE", "FLASHINFER_CACHE_DIR")},
            }, exclusive=True)
            try:
                while True:
                    active = gpu_processes(initial["gpu"]["uuid"])
                    last_account_devices = account_devices()
                    monitor_polls += 1
                    monitor.write(json.dumps({"utc": now(), "compute_processes": active,
                                              "account_devices": last_account_devices}) + "\n")
                    monitor.flush()
                    foreign = [record for record in active if record["pid"] != process.pid]
                    if foreign:
                        purity.write(json.dumps({"utc": now(), "foreign": foreign}) + "\n")
                        purity.flush()
                        raise RuntimeError("foreign GPU process observed; timing invalid")
                    worker.require(limit is None or len(last_account_devices) <= limit,
                                   "account GPU cap exceeded; stop only own child")
                    if process.poll() is not None:
                        break
                    time.sleep(POLL_SECONDS)
                status = process.wait(timeout=15)
                worker.require(status == 0, f"worker exited with code {status}")
                completion_artifacts = verify_completion(run_dir, method, checkpoint)
            except BaseException as error:
                purity.write(json.dumps({"utc": now(), "guard_failure": f"{type(error).__name__}: {error}"}) + "\n")
                purity.flush()
                raise
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        # A second termination signal must not interrupt cleanup of our child.
        for number in previous_signals:
            signal.signal(number, signal.SIG_IGN)
        try:
            stop_own_child(process)
            if process is not None:
                status = process.returncode
        except BaseException as error:
            failure = (failure + "; " if failure else "") + f"cleanup failed: {type(error).__name__}: {error}"
        finally:
            for number, previous in previous_signals.items():
                signal.signal(number, previous)
        valid = failure is None and status == 0 and completion_artifacts is not None and not foreign
        receipt = {"valid": valid, "timing_valid": valid, "started_utc": started, "completed_utc": now(),
                   "parent_pid": os.getpid(), "worker_pid": process.pid if process is not None else None,
                   "worker_exit_code": status, "gpu_uuid": initial["gpu"]["uuid"], "requested_gpu": gpu,
                   "method": method + "_bf16score", "monitor_interval_seconds": POLL_SECONDS,
                   "monitor_polls": monitor_polls, "foreign": foreign, "failure": failure,
                   "account_gpu_limit": limit, "gpu_scope": gpu_scope,
                   "task_gpu_limit": 1, "last_account_devices": last_account_devices,
                   "verified_artifacts": completion_artifacts,
                   "timing_caveat": "five-second sampled process exclusivity; worker attention includes metadata-check overhead"}
        if run_dir.exists():
            worker.write_json(run_dir / "completion.json", receipt)
    worker.require(receipt["valid"], f"generation guard failed: {failure}; inspect completion.json/worker.log")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--method", choices=tuple(VIDEO_MODES), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-videos", type=int, default=20)
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--spatial-native-proof", type=Path)
    parser.add_argument("--gpu-scope", choices=("account", "task"), default="account",
                        help="task owns one clean GPU while allowing unrelated account jobs")
    args = parser.parse_args()
    print(json.dumps(launch(args.gpu, args.run_dir, args.method, args.checkpoint,
                           spatial_native_proof=args.spatial_native_proof, gpu_scope=args.gpu_scope,
                           num_videos=args.num_videos, prompt_start=args.prompt_start), indent=2))


if __name__ == "__main__":
    main()
