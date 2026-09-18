"""Isolated complete-GCA experiment for the unpromoted Jeffreys selector.

Parent owns one clean H200 and monitors its only worker. Worker validates
frozen inputs, audits actual native seed scores, evaluates all21-frame L33
outputs, then times two full40-head complete operators against forced SDPA.
No model/video advances; production dispatch and default files are unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from cache_rf0_probe_banks import fingerprint, require, validate_inputs, validate_capture_metadata
from screen_native_seed_trajectory import MODES as EXISTING_MODES, check_capture_environment

MODES = {
    "hellinger_rf0": EXISTING_MODES["rf0"],
    "mass_rf0": EXISTING_MODES["mass_c32"],
    "jeffreys_rf0": EXISTING_MODES["mass_c32"],
    "hellinger_rf10": EXISTING_MODES["rf0"].replace("_rf0_", "_rf10_"),
}
ITERATIONS, REPEATS, WARMUP = 5, 6, 3
BASELINE_SHA = "e7d05b1419e75fb05b8440b653ea6dec0565c9430ab662007630110f3d86170b"
HOLDOUT_IDENTITIES = {
    "call020_holdout": (20, "922ec98b7f8d7f36a974cf126f7565b6cde1c2a48467c1cd80700e6213594bcd"),
    "call100_holdout": (100, "9e261e934f95b4237aef88d58e4c26de5c5964f776f71b2ebb5b13f1207068f2"),
}
ATTENTION_OPTIONS = dict(h=45, w=80, h_reduce=1, w_reduce=1, sm_scale=128 ** -0.5,
    query_outer_chunk=21, head_chunk=8, hard_route_backend="hard_c4",
    exact_intra_frame_attention=False, exact_neighbor_window_columns=0, exact_query_rows=0,
    first_frame_full_attention=False, full_attention_anchor_time_count=0,
    full_attention_anchor_row_count=0, full_attention_anchor_scale=1.0)
SOURCES = (
    "analysis/experiment_conditional_jeffreys_native.py",
    "analysis/verify_conditional_jeffreys_native.py",
    "analysis/conditional_jeffreys_seed.py", "analysis/cache_rf0_probe_banks.py",
    "analysis/screen_native_seed_trajectory.py", "attention/token_merging.py",
    "attention/conditional_jeffreys_seed_triton.py", "attention/entropy_mass_seed_triton.py",
    "attention/adaptive_seed_pytorch.py", "attention/adaptive_seed_triton.py",
    "attention/singleton_c4_wan21_14b_720p_triton.py",
    "attention/hard_c4_wan21_14b_720p_triton.py")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metrics(output, reference):
    import torch
    require(output.shape == reference.shape and output.dtype == reference.dtype == torch.bfloat16
            and output.is_contiguous() and reference.is_contiguous(), "wrong complete output layout")
    require(bool(torch.isfinite(output).all()) and bool(torch.isfinite(reference).all()), "nonfinite output")
    error = (output.float() - reference.float()).square()
    heads = output.shape[2]
    frame_head = error.reshape(1, 21, 3600, heads, 128).sum((0, 2, 4), dtype=torch.float64)
    sse = float(frame_head.sum())
    elements = error.numel()
    return {"numel": elements, "squared_error_sum": sse, "mse": sse / elements,
        "frame_head_squared_error_sum": frame_head.cpu().tolist(),
        "frame_mse": (frame_head.sum(1) / (elements // 21)).cpu().tolist(),
        "head_mse": (frame_head.sum(0) / (elements // heads)).cpu().tolist(),
        "rf0_mse": float(frame_head[0].sum()) / (elements // 21),
        "other20_frames_mse": float(frame_head[1:].sum()) / (elements * 20 // 21)}


def pool_metrics(items):
    numel = sum(item["numel"] for item in items)
    sse = math.fsum(item["squared_error_sum"] for item in items)
    return {"calls": len(items), "numel": numel, "squared_error_sum": sse, "mse": sse / numel,
        "rf0_mse": math.fsum(item["rf0_mse"] * item["numel"] for item in items) / numel,
        "other20_frames_mse": math.fsum(item["other20_frames_mse"] * item["numel"] for item in items) / numel}


def selector_symbol(name):
    return "hellinger_span_seed_rows_triton" if name.startswith("hellinger") else "entropy_mass_seed_rows_triton"


def seed_check(table, heads):
    require(table.shape == (1, 1, 21, 45, 80, heads, 4) and table.dtype == np.uint8, "wrong actual seed table")
    require(np.all(table < 45) and np.all(np.diff(np.sort(table.astype(np.int16), axis=-1), axis=-1) > 0),
            "invalid or repeated native seed rows")
    require(np.all(table[..., 0] == np.arange(45).reshape(1, 1, 1, 45, 1, 1)), "anchor lost")
    return {"sha256": hashlib.sha256(table.tobytes()).hexdigest(), "shape": list(table.shape),
            "distinct_real_rows_and_anchor": True}


def candidate_context(token, name):
    if name != "jeffreys_rf0":
        return nullcontext()
    from attention.conditional_jeffreys_seed_triton import conditional_jeffreys_seed_rows_triton
    return patch.object(token, "entropy_mass_seed_rows_triton", conditional_jeffreys_seed_rows_triton)


def attention(token, q, k, v, name):
    return token._hard_c4_attention(q, k, v, seed_mode=MODES[name], **ATTENTION_OPTIONS).contiguous()


def observed_attention(token, q, k, v, name):
    import torch
    records, tables = [], []
    with candidate_context(token, name):
        real = getattr(token, selector_symbol(name))
        def observe(*args, **kwargs):
            require(not records and len(args) == 3, "unexpected selector invocation")
            router_frame = 10 if name == "hellinger_rf10" else 0
            expected = q.reshape(1, 21, 45, 80, q.shape[2], 128)[:, router_frame:router_frame + 1]
            require(torch.equal(args[0], expected), "wrong router frame passed")
            require(torch.equal(args[1], k.reshape(1, 21, 45, 80, k.shape[2], 128)), "K changed")
            require(args[2] == 128 ** -0.5 and kwargs["candidate_count"] == 32
                    and kwargs["candidate_geometry_count"] == 45 and kwargs["block_j"] == 4
                    and kwargs["gs_num_warps"] == 4 and kwargs["output_dtype"] == torch.uint8,
                    "unexpected selector settings")
            if name in ("mass_rf0", "jeffreys_rf0"):
                require(set(kwargs) == {"candidate_count", "candidate_geometry_count", "block_j",
                    "probe_count", "global_probe_count", "output_dtype", "gs_num_warps"}, "unknown selector override")
                require(kwargs["probe_count"] == 16 and kwargs["global_probe_count"] == 256,
                        "changed local/global probes")
            else:
                require(kwargs["uniform_probes"] == 16 and kwargs["top_probes"] == 0, "not Uniform16")
            result = real(*args, **kwargs)
            records.append({key: str(value) if key == "output_dtype" else value for key, value in kwargs.items()})
            tables.append(result.detach().cpu().numpy().copy())
            return result
        with patch.object(token, selector_symbol(name), observe):
            output = attention(token, q, k, v, name)
    require(len(records) == 1, "expected exactly one observed selector call")
    return output, tables[0], records[0]


def load_holdout(path, device):
    import torch
    result, stored = [], {}
    with np.load(path, allow_pickle=False) as payload:
        require(all(name in payload for name in ("q", "k", "v")), "holdout QKV missing")
        require(str(np.asarray(payload["tag"]).item()) == "wan_self_attn"
                and np.asarray(payload["shape"]).reshape(-1).tolist() == [1, 75600, 40, 128]
                and np.asarray(payload["call_index"]).reshape(-1).tolist() == [HOLDOUT_IDENTITIES[path.parent.name][0]],
                "holdout metadata mismatch")
        for name in ("q", "k", "v"):
            value = payload[name]
            require(value.shape == (1, 40, 75600, 128) and value.dtype == np.float16
                    and np.isfinite(value).all(), "wrong full40 capture")
            stored[name] = str(value.dtype)
            tensor = torch.from_numpy(value.copy()).permute(0, 2, 1, 3).contiguous()
            result.append(tensor.to(device=device, dtype=torch.bfloat16))
    return (*result, stored)


def benchmark_full40(args, report, token):
    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel
    def dense(q, k, v):
        return functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
            v.transpose(1, 2), dropout_p=0, is_causal=False, scale=128 ** -0.5).transpose(1, 2).contiguous()
    for path in args.holdout:
        row = {"input": fingerprint(path), "complete": False, "metrics": {}, "timings": {}, "orders": []}
        require(path.parent.name in HOLDOUT_IDENTITIES and row["input"]["bytes"] == 2322433660
                and row["input"]["sha256"] == HOLDOUT_IDENTITIES[path.parent.name][1], "full40 holdout receipt mismatch")
        row["input_identity_scope"] = "SHA256 freshly measured by independent read-only check for this experiment, not original producer receipt"
        report["full40"].append(row)
        q, k, v, stored = load_holdout(path, "cuda:0")
        row["stored_dtypes"] = stored
        row["conversion"] = "All stored inputs converted to BF16 for every method and SDPA"
        functions = {name: partial(attention, token, q, k, v, name) for name in MODES}
        functions["sdpa"] = partial(dense, q, k, v)
        def context(name):
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION) if name == "sdpa" else candidate_context(token, name)
        with context("sdpa"):
            reference = functions["sdpa"]()
        for name in MODES:
            with context(name):
                output = functions[name]()
            observed, table, received = observed_attention(token, q, k, v, name)
            require(torch.equal(output, observed), f"full40 {name}: observation changed output")
            row["metrics"][name] = metrics(output, reference)
            row.setdefault("selectors", {})[name] = seed_check(table, 40) | {
                "received": received, "observation_output_equal": True}
            seed_path = args.output / f"{path.parent.name}_{name}_seeds.npz"
            np.savez_compressed(seed_path, rows=table)
            row["selectors"][name]["cache"] = fingerprint(seed_path)
            del output, observed, table
            write_json(args.output / "results.json", report)
        del reference
        for name, function in functions.items():
            with context(name):
                for _ in range(WARMUP):
                    output = function()
            del output
        torch.cuda.synchronize()
        samples = {name: [] for name in functions}
        for repeat in range(REPEATS):
            order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
            row["orders"].append(order)
            for name in order:
                with context(name):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(ITERATIONS):
                        output = functions[name]()
                    end.record()
                    end.synchronize()
                del output
                milliseconds = start.elapsed_time(end) / ITERATIONS
                samples[name].append(milliseconds)
                row["timings"][name] = {"mean_attention_ms": statistics.mean(samples[name]),
                    "std_ms": statistics.pstdev(samples[name]), "samples_ms": samples[name],
                    "calls_per_sample": ITERATIONS}
                write_json(args.output / "results.json", report)
                print(path.parent.name, repeat, name, milliseconds, flush=True)
        for name, timing in row["timings"].items():
            timing["speedup_vs_sdpa"] = row["timings"]["sdpa"]["mean_attention_ms"] / timing["mean_attention_ms"]
            for base in ("hellinger_rf0", "hellinger_rf10", "mass_rf0"):
                timing[f"ratio_vs_{base}"] = timing["mean_attention_ms"] / row["timings"][base]["mean_attention_ms"]
        row["complete"] = True
        write_json(args.output / "results.json", report)
        del q, k, v, functions, function
        torch.cuda.empty_cache()


def worker(args):
    check_capture_environment(os.environ)
    require(not (args.output / "results.json").exists(), "refuse to overwrite native results")
    oracle, identities, jobs = validate_inputs(args.capture_root, args.oracle_root)
    baseline_path = args.oracle_root.parent / "rf0_layer33_output_diagnostic_20260904o" / "analysis.json"
    baseline_identity = fingerprint(baseline_path)
    require(baseline_identity["sha256"] == BASELINE_SHA, "baseline receipt mismatch")
    baseline = json.loads(baseline_path.read_text())
    require(baseline["valid"] is True and baseline["complete"] is True, "baseline incomplete")
    baseline_rows = {item["metadata"]["capture_file"]: item for item in baseline["records"]}
    report = {"valid": False, "complete": False, "eligible_candidate": False,
        "scope": "Native complete-GCA screening, not model/video generation",
        "parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": {name: fingerprint(ROOT / name) for name in SOURCES},
        "input_validation": identities, "baseline_identity": baseline_identity, "methods": MODES,
        "candidate_identity": "jeffreys_rf0 substitutes ONLY the selector callable under the Mass RF0 mode",
        "attention_options": ATTENTION_OPTIONS, "iterations": ITERATIONS, "repeats": REPEATS, "warmup": WARMUP,
        "l33": [], "full40": [], "psnr": None, "ssim": None, "lpips": None, "vbench": None,
        "limitations": ["Three fixed step0 L33 captures are not all-layer coverage.",
            "Four saved L33 heads do not provide a full40-head timing headline.",
            "Two historical full40-head holdout captures are not video-quality evidence.",
            "Parent completion and empty purity log are required for eligible timings.",
            "No change to RF0, default selection, training-free status or full compressed route."]}
    write_json(args.output / "results.json", report)
    for name, identity in report["sources"].items():
        destination = args.output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        require(fingerprint(destination)["sha256"] == identity["sha256"], "source snapshot mismatch")
    import torch
    import attention.token_merging as token
    from verify_conditional_jeffreys_native import verify_native
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    report.update(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  device=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"])
    with torch.inference_mode():
        for job in jobs:
            identity = job["capture"]
            require(fingerprint(identity["path"]) == identity, "capture changed after validation")
            saved = torch.load(identity["path"], map_location="cpu", weights_only=True)
            validate_capture_metadata(saved, job["metadata"])
            require(all(tuple(saved[name].shape) == (1, 75600, 4, 128)
                        and saved[name].dtype == torch.bfloat16 and saved[name].is_contiguous()
                        and bool(torch.isfinite(saved[name]).all())
                        for name in ("q", "k", "v", "dense_output")), "wrong original capture tensors")
            q, k, v, reference = [saved[name].to("cuda:0") for name in ("q", "k", "v", "dense_output")]
            del saved
            row = {"input": identity, "metadata": job["metadata"], "metrics": {}, "selectors": {}}
            report["l33"].append(row)
            row["native_parity"], parity_arrays = verify_native(q, k)
            parity_path = args.output / f"prompt{job['metadata']['prompt_index']:03d}_native_parity.npz"
            np.savez_compressed(parity_path, **parity_arrays)
            row["parity_cache"] = fingerprint(parity_path)
            write_json(args.output / "results.json", report)
            require(row["native_parity"]["valid"] is True, "native candidate score parity failed")
            tables = {}
            for name in MODES:
                # First unobserved complete call, then an observation-only replay.
                with candidate_context(token, name):
                    output = attention(token, q, k, v, name)
                observed, tables[name], received = observed_attention(token, q, k, v, name)
                require(torch.equal(output, observed), f"{name}: observation changed output")
                row["metrics"][name] = metrics(output, reference)
                row["selectors"][name] = seed_check(tables[name], 4) | {"received": received, "observation_output_equal": True}
                if name == "jeffreys_rf0":
                    require(np.array_equal(tables[name], parity_arrays["all_native_rows"]),
                            "score auditing changed actual candidate seeds")
                    row["selectors"][name]["score_audit_seeds_equal"] = True
                if name in ("hellinger_rf0", "mass_rf0"):
                    old_name = "rf0" if name == "hellinger_rf0" else "mass_c32"
                    old = baseline_rows[Path(identity["path"]).name]
                    checks = {"seed_sha_equal": row["selectors"][name]["sha256"] == old["seed_diagnostics"][old_name]["sha256_uint8"]}
                    for field in ("mse", "frame_mse", "head_mse"):
                        checks[field + "_replicated"] = bool(np.allclose(row["metrics"][name][field],
                            old["methods"][old_name][field], rtol=2e-6, atol=1e-12))
                    row.setdefault("baseline_replication", {})[name] = checks
                    write_json(args.output / "results.json", report)
                    require(all(checks.values()), f"{name}: unchanged baseline failed replication")
                del output, observed
                write_json(args.output / "results.json", report)
            seeds_path = args.output / f"prompt{job['metadata']['prompt_index']:03d}_native_seeds.npz"
            np.savez_compressed(seeds_path, **tables)
            row["seed_cache"] = fingerprint(seeds_path)
            row["complete"] = True
            report["l33_summary"] = {name: pool_metrics([item["metrics"][name] for item in report["l33"]]) for name in MODES}
            write_json(args.output / "results.json", report)
            print(f"L33 prompt {job['metadata']['prompt_index']} complete", row["metrics"]["jeffreys_rf0"]["mse"], flush=True)
            del q, k, v, reference, tables
            torch.cuda.empty_cache()
        benchmark_full40(args, report, token)
    report.update(valid=True, complete=True)
    write_json(args.output / "results.json", report)


def gpu_state(index):
    line = subprocess.check_output(["nvidia-smi", "-i", str(index),
        "--query-gpu=name,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    fields = [value.strip() for value in line.split(",")]
    require(len(fields) == 4 and "H200" in fields[0], "require exactly one H200")
    return {"name": fields[0], "uuid": fields[1], "memory_mib": int(fields[2]), "utilization": int(fields[3])}


def gpu_processes(uuid):
    text = subprocess.check_output(["nvidia-smi", "-i", uuid,
        "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader,nounits"], text=True).strip()
    return [] if not text else [{"pid": int(row.split(",", 1)[0]), "raw": row} for row in text.splitlines()]


def parent(args):
    require(not args.output.exists(), "refuse an existing experiment directory")
    require(args.gpu is not None, "one physical GPU index is required")
    check_capture_environment(os.environ)
    state = gpu_state(args.gpu)
    require(not gpu_processes(state["uuid"]) and state["memory_mib"] <= 100 and state["utilization"] == 0,
            "GPU is not exclusively clean; do not queue or launch")
    args.output.mkdir(parents=True)
    write_json(args.output / "startup.json", {"physical_gpu": args.gpu, "state": state, "compute_processes": []})
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", WAN_SDPA_BACKEND="flash",
                       TRITON_CACHE_DIR=str(args.output / "triton_cache"))
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
        "--capture-root", str(args.capture_root), "--oracle-root", str(args.oracle_root),
        "--output", str(args.output), "--holdout", *map(str, args.holdout)]
    foreign = []
    monitor_error = None
    process = None
    def interrupted(number, frame):
        raise KeyboardInterrupt(f"received signal {number}")
    def stop_worker():
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()  # Only our own worker.
                process.wait(timeout=15)
    prior_signals = {number: signal.signal(number, interrupted) for number in
                     (signal.SIGTERM, signal.SIGINT)}
    with (args.output / "worker.log").open("x", encoding="utf-8") as log, (args.output / "purity.log").open("x", encoding="utf-8") as purity:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        write_json(args.output / "process.json", {"parent_pid": os.getpid(), "worker_pid": process.pid, "command": command})
        try:
            while process.poll() is None:
                foreign = [item for item in gpu_processes(state["uuid"]) if item["pid"] != process.pid]
                if foreign:
                    purity.write(json.dumps({"foreign": foreign}) + "\n")
                    purity.flush()
                    stop_worker()
                    break
                time.sleep(5)
            status = process.wait(timeout=15)
        except BaseException as error:
            monitor_error = f"{type(error).__name__}: {error}"
            purity.write(json.dumps({"monitor_error": str(error)}) + "\n")
            purity.flush()
        finally:
            stop_worker()
            status = process.returncode
            for number, previous in prior_signals.items():
                signal.signal(number, previous)
    try:
        results = json.loads((args.output / "results.json").read_text())
        complete = results.get("valid") is True and results.get("complete") is True and len(results["l33"]) == 3 and len(results["full40"]) == 2
    except (OSError, ValueError, KeyError):
        complete = False
    valid = status == 0 and complete and not foreign and monitor_error is None and (args.output / "purity.log").stat().st_size == 0
    write_json(args.output / "completion.json", {"valid": valid, "worker_exit_code": status,
        "parent_pid": os.getpid(), "worker_pid": process.pid, "gpu_uuid": state["uuid"], "foreign": foreign,
        "results_complete": complete, "monitor_error": monitor_error})
    require(valid, "native experiment failed; inspect worker.log and completion.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", nargs=2, type=Path, required=True)
    args = parser.parse_args()
    require(len({path.resolve() for path in args.holdout}) == 2, "two distinct holdouts required")
    return worker(args) if args.worker else parent(args)


if __name__ == "__main__":
    main()
