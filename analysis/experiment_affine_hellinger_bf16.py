"""Isolated BF16-only affine Hellinger quality screen, not video or timing.

The parent admits one clean H200 and monitors its single worker. The worker
first audits actual affine selector scores, then evaluates all 21 frames of
three frozen L33 four-head captures. HF/Mass/Jeff controls must reproduce the
frozen BF16 control b. Affine changes only the selector under HF's BF16 mode.
No defaults, production dispatch, merge mathematics or paper files are edited.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from cache_rf0_probe_banks import fingerprint, require, validate_capture_metadata, validate_inputs
from experiment_conditional_jeffreys_native import (
    ATTENTION_OPTIONS, SOURCES as OLD_SOURCES, gpu_processes, gpu_state,
    metrics, pool_metrics, seed_check, write_json,
)
from quality_first_mode_contract import BF16_MODES
from screen_native_seed_trajectory import check_capture_environment

MODES = dict(BF16_MODES) | {"affine_rf0": BF16_MODES["hellinger_rf0"]}
CONTROLS = tuple(BF16_MODES)
BASELINE_SHA = "5a7cb6e15e37bb8f40ffec82d67b16ccb53b5d5ffae357bc98e5d9b8bb557b03"
BASELINE_COMPLETION_SHA = "0542405f1e87ad690b1e619bdcc261f42c71529c54f89eaa6f905e17ed5019d2"
NATIVE_SHA = "2c6fa5c20eb26b935ec4d4c06a86cc337ca3a07c85c07cee53dafc56d4780339"
NATIVE_COMPLETION_SHA = "8be9fd2fa7622d717095684924205fe35640a8e5c216d028d6305ec596e0f228"
MERGE_SYMBOL = "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks"
SOURCES = tuple(dict.fromkeys((*OLD_SOURCES,
    "analysis/experiment_affine_hellinger_bf16.py", "analysis/affine_hellinger_seed.py",
    "analysis/verify_affine_hellinger_native.py", "analysis/quality_first_mode_contract.py",
    "attention/affine_hellinger_seed_triton.py")))


def selector_symbol(method):
    require(method in MODES, "unknown method")
    return ("hellinger_span_seed_rows_triton" if method in ("hellinger_rf0", "affine_rf0")
            else "entropy_mass_seed_rows_triton")


def validate_selector_kwargs(kwargs, method):
    """Validate exact historical parser kwargs; alpha is never used by affine."""
    require(method in MODES, "unknown selector method")
    expected = {"candidate_count": 32, "candidate_geometry_count": 45,
                "block_j": 4, "gs_num_warps": 4}
    if method in ("hellinger_rf0", "affine_rf0"):
        expected.update(alpha=.2, top_probes=0, uniform_probes=16, probe_anchor_count=2,
                        group_tile=16, selection_mode="gs", force_local_minus_one=False)
    else:
        expected.update(probe_count=16, global_probe_count=256)
    require(set(kwargs) == set(expected) | {"output_dtype"}, "unexpected selector kwargs")
    require(all(type(kwargs[key]) is type(value) and kwargs[key] == value for key, value in expected.items()),
            "selector criterion/probes changed")
    require(str(kwargs["output_dtype"]) == "torch.uint8", "seed dtype changed")
    return {key: str(value) if key == "output_dtype" else value for key, value in kwargs.items()}


def validate_merge_kwargs(kwargs):
    """Observe the actual BF16 complete-merge/route substrate, fail closed."""
    expected = {
        "head_start": 0, "head_count": 4, "top_r": 1, "selector": "query_column",
        "compressed_fp8_kv": False, "scaled_fp8_kv": False, "mean_centered_routing": False,
        "gaussian_partition_routing": False, "reuse_partition_exp": False,
        "direct_bulk_normalization": False, "fp8_score_dot": False,
        "fp8_merge_k_dot": False, "fp8_merge_v_dot": False,
        "query_frame_group": 21, "split_merge": False, "materialize_singleton": False,
        "query_column_major": True, "query_column_secondary_frame_stride": 1,
        "query_column_primary_row_mod4_omit2": False, "packed_singleton_kv": False,
        "seed_rows_native_layout": True, "merge_frame_group": 1, "merge_col_group": 1,
        "merge_loop_stages": 2, "merge_disable_licm": False, "split_kv_accum": True,
    }
    require(set(kwargs) == set(expected) | {"output"}, "unexpected merge/route kwargs")
    require(all(type(kwargs[key]) is type(value) and kwargs[key] == value for key, value in expected.items()),
            "merge/route substrate changed or FP8 enabled")
    return expected


def affine_selector_adapter(*args, **kwargs):
    """Accept only exact HF-mode kwargs, forward geometry but NOT alpha."""
    require(len(args) == 3 and args[2] == 128 ** -.5, "unexpected affine selector arguments")
    validate_selector_kwargs(kwargs, "affine_rf0")
    from attention.affine_hellinger_seed_triton import affine_hellinger_seed_rows_triton
    return affine_hellinger_seed_rows_triton(
        *args, candidate_count=kwargs["candidate_count"],
        candidate_geometry_count=kwargs["candidate_geometry_count"],
        block_j=kwargs["block_j"], probe_count=kwargs["uniform_probes"],
        output_dtype=kwargs["output_dtype"], gs_num_warps=kwargs["gs_num_warps"],
    )


def candidate_context(token, method):
    if method == "affine_rf0":
        return patch.object(token, selector_symbol(method), affine_selector_adapter)
    if method == "jeffreys_rf0":
        from attention.conditional_jeffreys_seed_triton import conditional_jeffreys_seed_rows_triton
        return patch.object(token, selector_symbol(method), conditional_jeffreys_seed_rows_triton)
    require(method in CONTROLS, "unknown method")
    return nullcontext()


def attention(token, q, k, v, method):
    require("_fp8" not in MODES[method] and "_rf0_" in MODES[method], "wrong BF16 RF0 mode")
    return token._hard_c4_attention(q, k, v, seed_mode=MODES[method], **ATTENTION_OPTIONS).contiguous()


def observed_attention(token, q, k, v, method):
    """Non-timed replay; observe exact RF0 QK, seeds and BF16 merge options."""
    import torch
    records, tables, merges = [], [], []
    selected_tensor = None
    grids = [tensor.reshape(1, 21, 45, 80, 4, 128) for tensor in (q, k, v)]
    with candidate_context(token, method):
        actual_selector = getattr(token, selector_symbol(method))
        actual_merge = getattr(token, MERGE_SYMBOL)

        def select(*args, **kwargs):
            nonlocal selected_tensor
            require(not records and len(args) == 3 and args[2] == 128 ** -.5, "unexpected selector call")
            require(torch.equal(args[0], grids[0][:, :1]) and torch.equal(args[1], grids[1]),
                    "selector did not receive original RF0 Q and all key frames")
            records.append(validate_selector_kwargs(kwargs, method))
            selected_tensor = actual_selector(*args, **kwargs)
            tables.append(selected_tensor.detach().cpu().numpy().copy())
            return selected_tensor

        def merge(*args, **kwargs):
            require(not merges and len(args) == 5 and args[4] == 128 ** -.5, "unexpected merge call")
            require(selected_tensor is not None and args[3] is selected_tensor,
                    "merge did not consume the observed native seed table")
            require(all(torch.equal(args[i], grids[i]) and args[i].dtype == torch.bfloat16 for i in range(3)),
                    "merge QKV changed")
            merges.append(validate_merge_kwargs(kwargs))
            return actual_merge(*args, **kwargs)

        with patch.object(token, selector_symbol(method), select), patch.object(token, MERGE_SYMBOL, merge):
            output = attention(token, q, k, v, method)
    require(len(records) == len(tables) == len(merges) == 1, "missing selector or complete merge/route call")
    return output, tables[0], {"selector": records[0], "merge_route": merges[0]}


def baseline_replication(measured, received, table, old, native_table):
    """Frozen b complete BF16 output metrics and exact frozen x seed table."""
    checks = {
        "mode_replicated": old["mode"] in BF16_MODES.values(),
        "selector_kwargs_replicated": received == old["received"],
        "exact_seed_table_equal": bool(np.array_equal(table, native_table)),
        "numel_replicated": measured["numel"] == old["metrics"]["numel"],
    }
    for field in ("squared_error_sum", "mse", "frame_head_squared_error_sum", "frame_mse",
                  "head_mse", "rf0_mse", "other20_frames_mse"):
        checks[field + "_replicated"] = bool(np.allclose(measured[field], old["metrics"][field],
                                                         rtol=2e-6, atol=1e-12))
    checks["valid"] = all(checks.values())
    return checks


def validate_worker_parent(args, environment, parent_pid):
    """The internal --worker entry is not a GPU-purity bypass."""
    check_capture_environment(environment)
    require(environment.get("GCA_AFFINE_SCREEN_PARENT_PID") == str(parent_pid), "worker must be spawned by its monitoring parent")
    startup = json.loads((args.output / "startup.json").read_text())
    require(startup["parent_pid"] == parent_pid and startup["compute_processes"] == [], "wrong monitoring-parent receipt")
    require(environment.get("CUDA_VISIBLE_DEVICES") == str(startup["physical_gpu"]), "worker GPU differs from parent")
    require(environment.get("WAN_SDPA_BACKEND") == "flash", "reference backend changed")


def worker(args):
    validate_worker_parent(args, os.environ, os.getppid())
    require(not (args.output / "results.json").exists(), "refuse overwrite/resume")
    baseline_identity = fingerprint(args.baseline_root / "results.json")
    native_identity = fingerprint(args.native_root / "results.json")
    require(baseline_identity["sha256"] == BASELINE_SHA, "not frozen BF16 control b")
    require(fingerprint(args.baseline_root / "completion.json")["sha256"] == BASELINE_COMPLETION_SHA,
            "BF16 control completion receipt mismatch")
    require(native_identity["sha256"] == NATIVE_SHA, "not frozen native seed run x")
    require(fingerprint(args.native_root / "completion.json")["sha256"] == NATIVE_COMPLETION_SHA,
            "native seed completion receipt mismatch")
    for directory in (args.baseline_root, args.native_root):
        require(json.loads((directory / "completion.json").read_text())["valid"] is True
                and (directory / "purity.log").stat().st_size == 0, "baseline did not finish cleanly")
    baseline = json.loads((args.baseline_root / "results.json").read_text())
    native_prior = json.loads((args.native_root / "results.json").read_text())
    require(baseline["valid"] and baseline["complete"] and len(baseline["records"]) == 3, "baseline incomplete")
    require(native_prior["valid"] and native_prior["complete"] and len(native_prior["l33"]) == 3, "native archive incomplete")
    _, identities, jobs = validate_inputs(args.capture_root, args.oracle_root)
    report = {
        "valid": False, "complete": False, "promoted": False,
        "scope": "Three frozen step0 L33 four-head captures; complete 21-frame BF16 GCA outputs, not video",
        "candidate_identity": "Affine unweighted anchor-difference Hellinger substitutes only HF RF0 selector callable",
        "methods": MODES, "attention_options": ATTENTION_OPTIONS,
        "input_validation": identities, "baseline_b": baseline_identity, "native_seed_archive_x": native_identity,
        "sources": {}, "l33": [], "parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "mean_attention_ms": None, "speedup_vs_sdpa": None,
        "psnr": None, "ssim": None, "lpips": None, "vbench": None,
        "limitations": ["Quality screen only: speed is unmeasured.",
            "L33/four-head frozen-input error is not all-layer or 20-video quality evidence.",
            "All methods use original BF16 QKV and no FP8 score/K/V casts; BF16 rounding still remains.",
            "Merge normalization, singleton and full compressed route are unchanged; direct bulk is disabled.",
            "Parent completion and empty purity log are required; no candidate promotion is automatic."],
    }
    for name in SOURCES:
        identity = fingerprint(ROOT / name)
        report["sources"][name] = identity
        destination = args.output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        require(fingerprint(destination)["sha256"] == identity["sha256"], "source snapshot mismatch")
    write_json(args.output / "results.json", report)
    import torch
    import attention.token_merging as token
    from analysis.verify_affine_hellinger_native import verify_native
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    report.update(device=torch.cuda.get_device_name(0), torch_version=str(torch.__version__),
                  cuda_version=torch.version.cuda, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"])
    with torch.inference_mode():
        for job, old_b, old_x in zip(jobs, baseline["records"], native_prior["l33"]):
            identity, metadata = job["capture"], job["metadata"]
            require(metadata == old_b["metadata"] == old_x["metadata"], "baseline input metadata mismatch")
            require(fingerprint(identity["path"]) == identity, "capture changed after validation")
            require(identity["sha256"] == old_b["input"]["sha256"] == old_x["input"]["sha256"], "baseline capture bytes mismatch")
            saved = torch.load(identity["path"], map_location="cpu", weights_only=True)
            validate_capture_metadata(saved, metadata)
            require(all(saved[key].dtype == torch.bfloat16 and tuple(saved[key].shape) == (1, 75600, 4, 128)
                        and saved[key].is_contiguous() and bool(torch.isfinite(saved[key]).all())
                        for key in ("q", "k", "v", "dense_output")), "invalid original capture tensors")
            q, k, v, reference = [saved[key].to("cuda:0") for key in ("q", "k", "v", "dense_output")]
            del saved
            pid = metadata["prompt_index"]
            row = {"input": identity, "metadata": metadata, "metrics": {}, "selectors": {}, "baseline_replication": {}, "complete": False}
            report["l33"].append(row)
            row["native_parity"], parity_arrays = verify_native(q, k)
            parity_path = args.output / f"prompt{pid:03d}_affine_native_parity.npz"
            np.savez_compressed(parity_path, **parity_arrays)
            row["parity_cache"] = fingerprint(parity_path)
            write_json(args.output / "results.json", report)
            require(row["native_parity"]["valid"] is True, "native affine parity failed; no candidate evaluation")
            old_seed_path = args.native_root / f"prompt{pid:03d}_native_seeds.npz"
            old_seed_identity = fingerprint(old_seed_path)
            require(old_seed_identity["sha256"] == old_x["seed_cache"]["sha256"] == old_b["seed_cache"]["sha256"], "frozen seed archive mismatch")
            with np.load(old_seed_path, allow_pickle=False) as cache:
                old_tables = {name: cache[name].copy() for name in CONTROLS}
            tables = {}
            for method in MODES:
                with candidate_context(token, method):
                    output = attention(token, q, k, v, method)
                observed, tables[method], dispatch = observed_attention(token, q, k, v, method)
                require(torch.equal(output, observed), f"{method}: observation changed complete output")
                row["metrics"][method] = metrics(output, reference)
                row["selectors"][method] = seed_check(tables[method], 4) | {
                    "received": dispatch, "observation_output_equal": True,
                    "mean_attention_ms": None, "speedup_vs_sdpa": None,
                }
                if method == "affine_rf0":
                    same = bool(np.array_equal(tables[method], parity_arrays["all_native_rows"]))
                    row["selectors"][method]["score_audit_seeds_equal"] = same
                    require(same, "complete affine call changed audited seeds")
                else:
                    old = old_b["bf16_score_control"][method]
                    require(old["mode"] == MODES[method] and old["exact_seed_table_equal"] is True,
                            "frozen BF16 mode/seed guarantee changed")
                    row["baseline_replication"][method] = baseline_replication(
                        row["metrics"][method], dispatch["selector"], tables[method], old, old_tables[method])
                # Persist actual selected rows and complete metrics even before
                # a replication assertion stops a failed screen.
                seed_path = args.output / f"prompt{pid:03d}_native_seeds.npz"
                np.savez_compressed(seed_path, **tables)
                row["seed_cache"] = fingerprint(seed_path)
                write_json(args.output / "results.json", report)
                if method in CONTROLS:
                    require(row["baseline_replication"][method]["valid"], f"{method}: frozen b BF16 control failed replication")
                del output, observed
                print(f"L33 prompt {pid} {method} BF16 MSE={row['metrics'][method]['mse']:.12g}", flush=True)
            row["complete"] = True
            report["l33_summary"] = {name: pool_metrics([item["metrics"][name] for item in report["l33"]]) for name in MODES}
            write_json(args.output / "results.json", report)
            del q, k, v, reference, tables, old_tables, parity_arrays
            torch.cuda.empty_cache()
    for name, identity in report["sources"].items():
        require(fingerprint(ROOT / name)["sha256"] == identity["sha256"], "source changed during screen")
    report.update(valid=True, complete=True, source_freeze_verified_at_end=True)
    write_json(args.output / "results.json", report)


def results_complete(result):
    if not (result.get("valid") is True and result.get("complete") is True and len(result.get("l33", [])) == 3):
        return False
    for row in result["l33"]:
        if not (row.get("complete") is True and row.get("native_parity", {}).get("valid") is True
                and set(row.get("metrics", {})) == set(MODES)
                and set(row.get("selectors", {})) == set(MODES)
                and all(row.get("baseline_replication", {}).get(name, {}).get("valid") is True for name in CONTROLS)):
            return False
    return result.get("source_freeze_verified_at_end") is True


def parent(args):
    require(type(args.gpu) is int and args.gpu >= 0, "parent requires one explicit physical GPU")
    require(not args.output.exists(), "refuse overwrite/resume")
    check_capture_environment(os.environ)
    state = gpu_state(args.gpu)
    require(not gpu_processes(state["uuid"]) and state["memory_mib"] <= 100 and state["utilization"] == 0,
            "GPU is not exclusively idle; do not run or queue")
    args.output.mkdir(parents=True)
    write_json(args.output / "startup.json", {"physical_gpu": args.gpu, "state": state,
               "parent_pid": os.getpid(), "compute_processes": []})
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", WAN_SDPA_BACKEND="flash",
        TRITON_CACHE_DIR=str(args.output / "triton_cache"), GCA_AFFINE_SCREEN_PARENT_PID=str(os.getpid()))
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"]
    for name in ("capture_root", "oracle_root", "native_root", "baseline_root", "output"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    process, foreign, monitor_error = None, [], None

    def stop():
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()  # Only the worker spawned by this parent.
                process.wait(timeout=15)

    def interrupted(number, frame):
        raise KeyboardInterrupt(f"signal {number}")

    prior = {number: signal.signal(number, interrupted) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (args.output / "worker.log").open("x") as log, (args.output / "purity.log").open("x") as purity:
            # Recheck immediately before spawning, after filesystem preparation.
            latest = gpu_state(args.gpu)
            require(latest["uuid"] == state["uuid"] and latest["memory_mib"] <= 100
                    and latest["utilization"] == 0 and not gpu_processes(state["uuid"]), "GPU ceased to be exclusively idle")
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write_json(args.output / "process.json", {"parent_pid": os.getpid(), "worker_pid": process.pid, "command": command})
            try:
                while process.poll() is None:
                    foreign = [entry for entry in gpu_processes(state["uuid"]) if entry["pid"] != process.pid]
                    if foreign:
                        purity.write(json.dumps({"foreign": foreign}) + "\n")
                        purity.flush()
                        break
                    time.sleep(5)
                if not foreign:
                    foreign = [entry for entry in gpu_processes(state["uuid"]) if entry["pid"] != process.pid]
                    if foreign:
                        purity.write(json.dumps({"foreign": foreign, "terminal_check": True}) + "\n")
                        purity.flush()
            except BaseException as error:
                monitor_error = f"{type(error).__name__}: {error}"
                purity.write(json.dumps({"monitor_error": monitor_error}) + "\n")
                purity.flush()
            finally:
                stop()
    finally:
        stop()
        for number, handler in prior.items():
            signal.signal(number, handler)
    try:
        complete = results_complete(json.loads((args.output / "results.json").read_text()))
    except (OSError, ValueError, KeyError):
        complete = False
    valid = (process is not None and process.returncode == 0 and complete and not foreign
             and monitor_error is None and (args.output / "purity.log").stat().st_size == 0)
    write_json(args.output / "completion.json", {"valid": valid, "worker_exit_code": process.returncode if process else None,
        "parent_pid": os.getpid(), "worker_pid": process.pid if process else None,
        "results_complete": complete, "foreign": foreign, "monitor_error": monitor_error, "gpu_uuid": state["uuid"]})
    require(valid, "affine BF16 screen failed; inspect completion and worker.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int)
    for name in ("capture-root", "oracle-root", "native-root", "baseline-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    return worker(args) if args.worker else parent(args)


if __name__ == "__main__":
    main()
