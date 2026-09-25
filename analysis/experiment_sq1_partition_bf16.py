"""Isolated BF16-only SQ1 partition-mass quality screen, not video or timing.

The parent admits one clean H200 and monitors its single frozen-source worker.
The worker first audits both actual SQ1 selector paths, then evaluates all21
frames of three frozen L33 four-head captures. HF/Mass/Jeff controls must
reproduce frozen BF16 control b. The candidate replaces only Mass's selector,
with stored_seed_logp fixed; the reuse path is a diagnostic, not a tuned mode.
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

MODES = dict(BF16_MODES) | {"sq1_partition_rf0": BF16_MODES["mass_rf0"]}
CONTROLS = tuple(BF16_MODES)
CANDIDATE = "sq1_partition_rf0"
PARITY_PATHS = ("stored_seed_logp", "reuse_owner_logp")
NULL_METRICS = ("mean_attention_ms", "speedup_vs_sdpa", "psnr", "ssim", "lpips", "vbench")
REPLICATION_CHECKS = frozenset(("mode_replicated", "selector_kwargs_replicated", "exact_seed_table_equal",
    "numel_replicated", "squared_error_sum_replicated", "mse_replicated",
    "frame_head_squared_error_sum_replicated", "frame_mse_replicated", "head_mse_replicated",
    "rf0_mse_replicated", "other20_frames_mse_replicated", "valid"))
BASELINE_SHA = "5a7cb6e15e37bb8f40ffec82d67b16ccb53b5d5ffae357bc98e5d9b8bb557b03"
BASELINE_COMPLETION_SHA = "0542405f1e87ad690b1e619bdcc261f42c71529c54f89eaa6f905e17ed5019d2"
NATIVE_SHA = "2c6fa5c20eb26b935ec4d4c06a86cc337ca3a07c85c07cee53dafc56d4780339"
NATIVE_COMPLETION_SHA = "8be9fd2fa7622d717095684924205fe35640a8e5c216d028d6305ec596e0f228"
MERGE_SYMBOL = "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks"
SOURCES = tuple(sorted(set((*OLD_SOURCES,
    "analysis/experiment_sq1_partition_bf16.py", "analysis/affine_hellinger_seed.py",
    "analysis/sq1_partition_seed_reference.py", "analysis/verify_sq1_partition_native.py",
    "analysis/quality_first_mode_contract.py",
    *(path.relative_to(ROOT).as_posix() for path in (ROOT / "attention").rglob("*.py"))))))


def selector_symbol(method):
    require(method in MODES, "unknown method")
    return ("hellinger_span_seed_rows_triton" if method == "hellinger_rf0"
            else "entropy_mass_seed_rows_triton")


def validate_selector_kwargs(kwargs, method):
    """Accept exactly the historical parser kwargs; never strip unknown keys."""
    require(method in MODES, "unknown selector method")
    expected = {"candidate_count": 32, "candidate_geometry_count": 45,
                "block_j": 4, "gs_num_warps": 4}
    if method == "hellinger_rf0":
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


def sq1_selector_adapter(*args, **kwargs):
    """Exact Mass kwargs plus a fixed stored-profile implementation, no tuning."""
    require(len(args) == 3 and args[2] == 128 ** -.5, "unexpected SQ1 selector arguments")
    validate_selector_kwargs(kwargs, "sq1_partition_rf0")
    from attention.sq1_partition_mass_seed_triton import sq1_partition_mass_seed_rows_triton
    return sq1_partition_mass_seed_rows_triton(
        *args, candidate_count=kwargs["candidate_count"],
        candidate_geometry_count=kwargs["candidate_geometry_count"],
        block_j=kwargs["block_j"], probe_count=kwargs["probe_count"],
        global_probe_count=kwargs["global_probe_count"],
        output_dtype=kwargs["output_dtype"], gs_num_warps=kwargs["gs_num_warps"],
        reuse_owner_logp=False,
    )


def candidate_context(token, method):
    if method == "sq1_partition_rf0":
        return patch.object(token, selector_symbol(method), sq1_selector_adapter)
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
    require(environment.get("GCA_SQ1_SCREEN_PARENT_PID") == str(parent_pid), "worker must be spawned by its monitoring parent")
    startup = json.loads((args.output / "startup.json").read_text())
    require(startup["parent_pid"] == parent_pid and startup["compute_processes"] == [], "wrong monitoring-parent receipt")
    require(environment.get("CUDA_VISIBLE_DEVICES") == str(startup["physical_gpu"]), "worker GPU differs from parent")
    require(environment.get("WAN_SDPA_BACKEND") == "flash", "reference backend changed")
    require(Path(__file__).resolve() == (args.output / "sources" / "analysis" / "experiment_sq1_partition_bf16.py").resolve(),
            "worker must execute the frozen source snapshot")
    require(startup["source_provenance_sha256"] == fingerprint(args.output / "source_provenance.json")["sha256"],
            "source provenance differs from monitoring-parent receipt")


def snapshot_sources(output):
    """Freeze every local execution dependency before spawning the worker."""
    manifest = {"root": str(ROOT), "parent_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "files": {}}
    for name in SOURCES:
        original = fingerprint(ROOT / name)
        destination = output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        copied = fingerprint(destination)
        require((copied["bytes"], copied["sha256"]) == (original["bytes"], original["sha256"]), "source snapshot mismatch")
        manifest["files"][name] = original
    write_json(output / "source_provenance.json", manifest)
    return fingerprint(output / "source_provenance.json")


def verify_source_snapshot(output):
    manifest = json.loads((output / "source_provenance.json").read_text())
    require(set(manifest["files"]) == set(SOURCES), "source snapshot inventory mismatch")
    for name, expected in manifest["files"].items():
        actual = fingerprint(output / "sources" / name)
        require((actual["bytes"], actual["sha256"]) == (expected["bytes"], expected["sha256"]), "frozen source changed")
    return manifest


def parity_complete(parity):
    """Do not trust a bare top-level valid flag for the two-path raw gate."""
    if not isinstance(parity, dict) or parity.get("valid") is not True or set(parity.get("paths", {})) != set(PARITY_PATHS):
        return False
    for name in PARITY_PATHS:
        path = parity["paths"][name]
        raw = path.get("raw_score_parity", {})
        if not (path.get("valid") is True and path.get("audit_on_off_all_rows_bitwise_equal") is True
                and raw.get("valid") is True and raw.get("pass") is True
                and raw.get("structure_pass") is True and raw.get("banks") == 600
                and raw.get("score_count") == 57600 and raw.get("rank_count") == 1800
                and raw.get("score_failures") == raw.get("rank_failures") == raw.get("nonfinite_scores") == 0
                and raw.get("score_atol") == 2e-5 and raw.get("score_rtol") == 1e-4):
            return False
        for key in ("all_native_rows", "all_audit_free_rows"):
            structure = path.get(key, {})
            if not (structure.get("valid") is True and structure.get("groups") == 302400
                    and all(structure.get(field) == 0 for field in
                            ("out_of_range_rows", "anchor_failures", "non_distinct_groups", "out_of_bank_rows"))):
                return False
    return True


def worker(args):
    validate_worker_parent(args, os.environ, os.getppid())
    require(not (args.output / "results.json").exists(), "refuse overwrite/resume")
    provenance = verify_source_snapshot(args.output)
    baseline_identity = fingerprint(args.baseline_root / "results.json")
    native_identity = fingerprint(args.native_root / "results.json")
    require(baseline_identity["sha256"] == BASELINE_SHA, "not frozen BF16 control b")
    require(fingerprint(args.baseline_root / "completion.json")["sha256"] == BASELINE_COMPLETION_SHA,
            "BF16 control completion receipt mismatch")
    require(native_identity["sha256"] == NATIVE_SHA, "not frozen native seed run x")
    require(fingerprint(args.native_root / "completion.json")["sha256"] == NATIVE_COMPLETION_SHA,
            "native seed completion receipt mismatch")
    for directory in (args.baseline_root, args.native_root):
        completed = json.loads((directory / "completion.json").read_text())
        require(completed.get("valid") is True and type(completed.get("worker_exit_code")) is int
                and completed["worker_exit_code"] == 0 and completed.get("results_complete") is True
                and completed.get("foreign") == [] and completed.get("monitor_error") is None
                and (directory / "purity.log").stat().st_size == 0, "baseline did not finish cleanly")
    baseline = json.loads((args.baseline_root / "results.json").read_text())
    native_prior = json.loads((args.native_root / "results.json").read_text())
    require(baseline["valid"] and baseline["complete"] and len(baseline["records"]) == 3, "baseline incomplete")
    require(native_prior["valid"] and native_prior["complete"] and len(native_prior["l33"]) == 3, "native archive incomplete")
    _, identities, jobs = validate_inputs(args.capture_root, args.oracle_root)
    report = {
        "valid": False, "complete": False, "promoted": False,
        "scope": "Three frozen step0 L33 four-head captures; complete 21-frame BF16 GCA outputs, not video",
        "candidate_identity": "SQ1-aware current partition-mass deficit substitutes only Mass RF0 selector callable",
        "candidate_implementation": {"reuse_owner_logp": False, "audit_scores": False,
                                     "primary_parity_path": "stored_seed_logp", "reuse_path_is_diagnostic_only": True},
        "methods": MODES, "attention_options": ATTENTION_OPTIONS,
        "input_validation": identities, "baseline_b": baseline_identity, "native_seed_archive_x": native_identity,
        "sources": provenance["files"], "source_provenance": fingerprint(args.output / "source_provenance.json"),
        "l33": [], "parent_commit": provenance["parent_commit"],
        "mean_attention_ms": None, "speedup_vs_sdpa": None,
        "psnr": None, "ssim": None, "lpips": None, "vbench": None,
        "limitations": ["Quality screen only: speed is unmeasured.",
            "L33/four-head frozen-input error is not all-layer or 20-video quality evidence.",
            "All methods use original BF16 QKV and no FP8 score/K/V casts; BF16 rounding still remains.",
            "Merge normalization, singleton and full compressed route are unchanged; direct bulk is disabled.",
            "Parent completion and empty purity log are required; no candidate promotion is automatic."],
    }
    write_json(args.output / "results.json", report)
    import torch
    import attention.token_merging as token
    from analysis.verify_sq1_partition_native import verify_native
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
            parity_path = args.output / f"prompt{pid:03d}_sq1_native_parity.npz"
            np.savez_compressed(parity_path, **parity_arrays)
            row["parity_cache"] = fingerprint(parity_path)
            write_json(args.output / "results.json", report)
            require(parity_complete(row["native_parity"]), "native SQ1 parity failed; no candidate evaluation")
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
                if method == "sq1_partition_rf0":
                    same = bool(np.array_equal(tables[method], parity_arrays["stored_seed_logp__all_native_rows"]))
                    row["selectors"][method]["score_audit_seeds_equal"] = same
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
                else:
                    require(row["selectors"][method]["score_audit_seeds_equal"], "complete SQ1 call changed audited seeds")
                del output, observed
                print(f"L33 prompt {pid} {method} BF16 MSE={row['metrics'][method]['mse']:.12g}", flush=True)
            row["complete"] = True
            report["l33_summary"] = {name: pool_metrics([item["metrics"][name] for item in report["l33"]]) for name in MODES}
            write_json(args.output / "results.json", report)
            del q, k, v, reference, tables, old_tables, parity_arrays
            torch.cuda.empty_cache()
    verify_source_snapshot(args.output)
    require(fingerprint(args.output / "source_provenance.json") == report["source_provenance"], "source provenance changed")
    report.update(valid=True, complete=True, source_freeze_verified_at_end=True)
    write_json(args.output / "results.json", report)


def metric_complete(item):
    """Recompute complete frame/head aggregates, not just a valid flag."""
    try:
        sse = np.asarray(item["frame_head_squared_error_sum"], dtype=np.float64)
        if (sse.shape != (21, 4) or not np.isfinite(sse).all() or np.any(sse < 0)
                or type(item["numel"]) is not int or item["numel"] != 38707200):
            return False
        n = item["numel"]
        expected = {"squared_error_sum": float(sse.sum()), "mse": float(sse.sum()) / n,
                    "frame_mse": sse.sum(1) / (n // 21), "head_mse": sse.sum(0) / (n // 4),
                    "rf0_mse": float(sse[0].sum()) / (n // 21),
                    "other20_frames_mse": float(sse[1:].sum()) / (n * 20 // 21)}
        return all(np.asarray(item[name]).shape == np.asarray(value).shape
                   and np.isfinite(item[name]).all()
                   and np.allclose(item[name], value, rtol=1e-12, atol=1e-12)
                   for name, value in expected.items())
    except (KeyError, TypeError, ValueError):
        return False


def results_complete(result):
    """Strict terminal contract; optional descriptive claims cannot promote it."""
    try:
        if not (result.get("valid") is True and result.get("complete") is True
                and result.get("promoted") is False and len(result.get("l33", [])) == 3
                and result.get("source_freeze_verified_at_end") is True
                and result.get("methods") == MODES and result.get("attention_options") == ATTENTION_OPTIONS
                and set(result.get("sources", {})) == set(SOURCES)
                and all(name in result and result[name] is None for name in NULL_METRICS)
                and result.get("candidate_implementation") == {
                    "reuse_owner_logp": False, "audit_scores": False,
                    "primary_parity_path": "stored_seed_logp", "reuse_path_is_diagnostic_only": True}):
            return False
        for prompt, row in enumerate(result["l33"]):
            metadata = row["metadata"]
            if not (row.get("complete") is True and parity_complete(row.get("native_parity"))
                    and metadata["prompt_index"] == prompt and metadata["layer_index"] == 33
                    and metadata["call_index"] == 33 and metadata["step_index"] == 0
                    and metadata["heads"] == [0, 10, 20, 30] and metadata["shape"] == [1, 75600, 4, 128]
                    and set(row.get("metrics", {})) == set(MODES)
                    and set(row.get("selectors", {})) == set(MODES)
                    and set(row.get("baseline_replication", {})) == set(CONTROLS)):
                return False
            for name in MODES:
                selected = row["selectors"][name]
                if not (metric_complete(row["metrics"][name])
                        and selected.get("distinct_real_rows_and_anchor") is True
                        and selected.get("shape") == [1, 1, 21, 45, 80, 4, 4]
                        and selected.get("observation_output_equal") is True
                        and "mean_attention_ms" in selected and selected["mean_attention_ms"] is None
                        and "speedup_vs_sdpa" in selected and selected["speedup_vs_sdpa"] is None):
                    return False
                validate_selector_kwargs(selected["received"]["selector"], name)
                validate_merge_kwargs(selected["received"]["merge_route"] | {"output": None})
                if name == CANDIDATE:
                    if selected.get("score_audit_seeds_equal") is not True:
                        return False
                else:
                    checks = row["baseline_replication"][name]
                    if set(checks) != REPLICATION_CHECKS or not all(value is True for value in checks.values()):
                        return False
        if set(result.get("l33_summary", {})) != set(MODES):
            return False
        for name in MODES:
            expected = pool_metrics([row["metrics"][name] for row in result["l33"]])
            actual = result["l33_summary"][name]
            if set(actual) != set(expected) or not all(
                    np.isfinite(actual[key]) and np.isclose(actual[key], value, rtol=1e-12, atol=1e-12)
                    for key, value in expected.items()):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def verify_saved_artifacts(result, output):
    """Parent rechecks persisted receipts; paths must remain in this stage."""
    verify_source_snapshot(output)
    expected = [(output / "source_provenance.json", result["source_provenance"])]
    for row in result["l33"]:
        prompt = row["metadata"]["prompt_index"]
        expected.extend(((output / f"prompt{prompt:03d}_sq1_native_parity.npz", row["parity_cache"]),
                         (output / f"prompt{prompt:03d}_native_seeds.npz", row["seed_cache"])))
    receipts = {}
    for path, receipt in expected:
        actual = fingerprint(path)
        require(actual == receipt, "saved primary artifact changed or escaped its stage")
        receipts[path.name] = actual
    receipts["results.json"] = fingerprint(output / "results.json")
    return receipts


def parent(args):
    require(type(args.gpu) is int and args.gpu >= 0, "parent requires one explicit physical GPU")
    require(not args.output.exists(), "refuse overwrite/resume")
    check_capture_environment(os.environ)
    state = gpu_state(args.gpu)
    require(not gpu_processes(state["uuid"]) and state["memory_mib"] <= 100 and state["utilization"] == 0,
            "GPU is not exclusively idle; do not run or queue")
    args.output.mkdir(parents=True)
    provenance = snapshot_sources(args.output)
    write_json(args.output / "startup.json", {"physical_gpu": args.gpu, "state": state,
               "parent_pid": os.getpid(), "compute_processes": [],
               "source_provenance_sha256": provenance["sha256"]})
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", WAN_SDPA_BACKEND="flash",
        TRITON_CACHE_DIR=str(args.output / "triton_cache"), GCA_SQ1_SCREEN_PARENT_PID=str(os.getpid()))
    command = [sys.executable, "-u", str((args.output / "sources" / "analysis" / "experiment_sq1_partition_bf16.py").resolve()), "--worker"]
    for name in ("capture_root", "oracle_root", "native_root", "baseline_root", "output"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name).resolve())]
    process, foreign, monitor_error, monitor_polls = None, [], None, 0

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
        with (args.output / "worker.log").open("x") as log, \
                (args.output / "purity.log").open("x") as purity, \
                (args.output / "gpu_monitor.jsonl").open("x") as monitor:
            # Recheck immediately before spawning, after filesystem preparation.
            verify_source_snapshot(args.output)
            latest = gpu_state(args.gpu)
            require(latest["uuid"] == state["uuid"] and latest["memory_mib"] <= 100
                    and latest["utilization"] == 0 and not gpu_processes(state["uuid"]), "GPU ceased to be exclusively idle")
            process = subprocess.Popen(command, cwd=args.output / "sources", env=environment, stdout=log, stderr=subprocess.STDOUT)
            write_json(args.output / "process.json", {"parent_pid": os.getpid(), "worker_pid": process.pid, "command": command})
            try:
                while process.poll() is None:
                    processes = gpu_processes(state["uuid"])
                    foreign = [entry for entry in processes if entry["pid"] != process.pid]
                    monitor_polls += 1
                    monitor.write(json.dumps({"unix_time": time.time(), "gpu_uuid": state["uuid"],
                        "worker_pid": process.pid, "compute_processes": processes,
                        "foreign": foreign, "terminal_check": False}) + "\n")
                    monitor.flush()
                    if foreign:
                        purity.write(json.dumps({"foreign": foreign}) + "\n")
                        purity.flush()
                        break
                    time.sleep(5)
                if not foreign:
                    processes = gpu_processes(state["uuid"])
                    foreign = [entry for entry in processes if entry["pid"] != process.pid]
                    monitor_polls += 1
                    monitor.write(json.dumps({"unix_time": time.time(), "gpu_uuid": state["uuid"],
                        "worker_pid": process.pid, "compute_processes": processes,
                        "foreign": foreign, "terminal_check": True}) + "\n")
                    monitor.flush()
                    if foreign:
                        purity.write(json.dumps({"foreign": foreign, "terminal_check": True}) + "\n")
                        purity.flush()
            except BaseException as error:
                monitor_error = f"{type(error).__name__}: {error}"
                monitor.write(json.dumps({"unix_time": time.time(), "monitor_error": monitor_error}) + "\n")
                monitor.flush()
                purity.write(json.dumps({"monitor_error": monitor_error}) + "\n")
                purity.flush()
            finally:
                stop()
    except BaseException as error:
        monitor_error = f"{type(error).__name__}: {error}"
    finally:
        stop()
        for number, handler in prior.items():
            signal.signal(number, handler)
    receipts = {}
    try:
        result = json.loads((args.output / "results.json").read_text())
        complete = results_complete(result)
        if complete:
            receipts = verify_saved_artifacts(result, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        complete = False
        monitor_error = monitor_error or f"artifact validation: {type(error).__name__}: {error}"
    valid = (process is not None and process.returncode == 0 and complete and not foreign
             and monitor_polls > 0 and monitor_error is None and (args.output / "purity.log").stat().st_size == 0)
    write_json(args.output / "worker.exitcode", process.returncode if process else None)
    write_json(args.output / "screen.exitcode", 0 if valid else 1)
    write_json(args.output / "completion.json", {"valid": valid, "worker_exit_code": process.returncode if process else None,
        "parent_pid": os.getpid(), "worker_pid": process.pid if process else None,
        "results_complete": complete, "foreign": foreign, "monitor_error": monitor_error, "gpu_uuid": state["uuid"],
        "monitor_polls": monitor_polls, "verified_artifacts": receipts})
    require(valid, "SQ1 BF16 screen failed; inspect completion and worker.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int)
    for name in ("capture-root", "oracle-root", "native-root", "baseline-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("capture_root", "oracle_root", "native_root", "baseline_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return worker(args) if args.worker else parent(args)


if __name__ == "__main__":
    main()
