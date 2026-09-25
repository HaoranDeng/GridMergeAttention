"""Off-path FP64 oracle: fixed native seeds, all key blocks, selected RF0 queries.

Not a candidate implementation or a runtime benchmark. Separates conditional
block error, global reweighting and native-vs-ideal output disagreement.
"""
from __future__ import annotations

import argparse
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
from cache_rf0_probe_banks import (HEADS, QUERY_COLUMNS, fingerprint, require,
                                 validate_inputs, validate_capture_metadata)
from experiment_conditional_jeffreys_native import (MODES, SOURCES, gpu_processes,
    gpu_state, metrics, observed_attention, seed_check, write_json)
from screen_native_seed_trajectory import check_capture_environment

# Keep the entire immutable receipt, not a path-dependent identity.
NATIVE_SHA = "2c6fa5c20eb26b935ec4d4c06a86cc337ca3a07c85c07cee53dafc56d4780339"
COMPLETION_SHA = "8be9fd2fa7622d717095684924205fe35640a8e5c216d028d6305ec596e0f228"


def ideal_merge(seed_q, k, v, singleton_column, scale):
    """B,C,D seed queries; B,K,D original keys. Return B,C+1,D KV.

    Normalize each seed over all keys before ownership, then peel SQ1 and
    normalize within its disjoint owned cell. Empty cells have zero weight
    and -infinity route logits. Every dot product starts from FP64 inputs.
    """
    import torch
    require(all(x.dtype == torch.float64 for x in (seed_q, k, v)), "FP64 inputs required")
    logits = torch.bmm(seed_q, k.transpose(1, 2)) * scale
    logp = logits.log_softmax(-1)
    owner = logp.argmax(1)
    mask = owner[:, None, :] == torch.arange(seed_q.shape[1], device=k.device)[None, :, None]
    mask[:, :, singleton_column] = False
    occupied = mask.any(-1)
    masked = logits.masked_fill(~mask, -torch.inf)
    lognorm = masked.logsumexp(-1)
    logw = masked - torch.where(occupied, lognorm, 0)[:, :, None]
    weights = logw.exp()
    entropy = -(weights * torch.where(mask, logw, 0)).sum(-1)
    mk, mv = torch.bmm(weights, k), torch.bmm(weights, v)
    mk = torch.cat((mk, k[:, singleton_column:singleton_column + 1]), 1)
    mv = torch.cat((mv, v[:, singleton_column:singleton_column + 1]), 1)
    entropy = torch.cat((entropy, torch.zeros_like(entropy[:, :1])), 1)
    valid = torch.cat((occupied, torch.ones_like(occupied[:, :1])), 1)
    return mk, mv, entropy, valid, weights, owner


def block_reference(q, k, v, scale):
    """True dense reference for Q,D queries over ALL B,K,D key blocks."""
    import torch
    logits = torch.einsum("qd,bkd->qbk", q, k) * scale
    logz = logits.logsumexp(-1)
    p = logits.softmax(-1)
    output = torch.einsum("qbk,bkd->qbd", p, v)
    rho = logz.softmax(-1)
    dense = torch.einsum("qb,qbd->qd", rho, output)
    return dense, logz, rho, output


def route_decomposition(q, merged, reference, scale):
    """Exact vector decomposition, with no assumption of orthogonal errors."""
    import torch
    mk, mv, entropy, valid = merged[:4]
    dense, logz, rho, block_dense = reference
    logits = (torch.einsum("qd,bcd->qbc", q, mk) * scale + entropy[None]).masked_fill(~valid[None], -torch.inf)
    approx_logz = logits.logsumexp(-1)
    conditional = torch.einsum("qbc,bcd->qbd", logits.softmax(-1), mv)
    approx_rho = approx_logz.softmax(-1)
    ideal = torch.einsum("qb,qbd->qd", approx_rho, conditional)
    true_mass_output = torch.einsum("qb,qbd->qd", rho, conditional)
    within = true_mass_output - dense
    reweight = ideal - true_mass_output
    # Direct independent definitions must agree, including the cross term.
    direct_within = torch.einsum("qb,qbd->qd", rho, conditional - block_dense)
    direct_reweight = torch.einsum("qb,qbd->qd", approx_rho - rho, conditional)
    require(torch.allclose(within, direct_within, atol=1e-10, rtol=1e-9), "within decomposition failed")
    require(torch.allclose(reweight, direct_reweight, atol=1e-10, rtol=1e-9), "mass decomposition failed")
    gap = logz.logsumexp(-1) - approx_logz.logsumexp(-1)
    eps = logz - approx_logz
    require(float(eps.min()) >= -1e-8 and float(gap.min()) >= -1e-8, "variational lower bound failed")
    alternate_gap = -(rho.log() - eps).logsumexp(-1)
    require(torch.allclose(gap, alternate_gap, atol=1e-9, rtol=1e-8), "global gap identity failed")
    summary = {
        "numel": ideal.numel(), "ideal_mse": float((ideal - dense).square().mean()),
        "within_mse": float(within.square().mean()), "reweight_mse": float(reweight.square().mean()),
        "cross_term": float(2 * (within * reweight).mean()),
        "global_reverse_kl_mean": float(gap.mean()), "global_reverse_kl_max": float(gap.max()),
        "block_mass_tv_mean": float((rho - approx_rho).abs().sum(-1).mean() / 2),
        "weighted_local_conditional_mse": float((rho * (conditional - block_dense).square().mean(-1)).sum(-1).mean()),
        "mean_block_gap": float((rho * eps).sum(-1).mean()),
        "within_diagonal_mse": float((rho.square() * (conditional - block_dense).square().mean(-1)).sum(-1).mean()),
    }
    summary["within_cross_block_term"] = summary["within_mse"] - summary["within_diagonal_mse"]
    require(abs(summary["ideal_mse"] - sum(summary[n] for n in ("within_mse", "reweight_mse", "cross_term"))) < 1e-10,
            "squared-error decomposition failed")
    arrays = {"ideal": ideal, "true_mass_output": true_mass_output, "within": within,
              "reweight": reweight, "global_gap": gap, "block_mass_tv": (rho - approx_rho).abs().sum(-1) / 2}
    return summary, arrays


def worker(args):
    check_capture_environment(os.environ)
    old_identity = fingerprint(args.native_root / "results.json")
    require(old_identity["sha256"] == NATIVE_SHA, "native result identity mismatch")
    require(fingerprint(args.native_root / "completion.json")["sha256"] == COMPLETION_SHA, "native completion mismatch")
    previous = json.loads((args.native_root / "results.json").read_text())
    require(previous["valid"] and previous["complete"], "native result incomplete")
    _, identities, jobs = validate_inputs(args.capture_root, args.oracle_root)
    report = {"valid": False, "complete": False, "candidate": False,
        "scope": "Off-path FP64 oracle; fixed native seed rows; all 945 key blocks; RF0 queries at ten columns",
        "query_columns": list(QUERY_COLUMNS), "heads": list(HEADS), "query_rows": list(range(45)),
        "key_blocks": 945, "key_tokens": 75600, "methods": dict(MODES),
        "bf16_score_modes": {n: m.replace("_fp8score", "") for n, m in MODES.items()},
        "input_validation": identities, "native_results": old_identity,
        "parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": {}, "records": [], "mean_attention_ms": None, "speedup_vs_sdpa": None,
        "psnr": None, "ssim": None, "lpips": None, "vbench": None,
        "limitations": ["Oracle only, not a candidate or timing benchmark.", "Three step0 L33 captures, four heads and RF0 selected columns, not a 20-video evaluation.",
            "Native-versus-ideal disagreement includes FP8, normalization and threshold semantics; it does not identify a unique source.",
            "True-mass output is an off-path counterfactual, never a candidate dense residual."]}
    for name in (*SOURCES, "analysis/diagnose_fixed_seed_global.py"):
        report["sources"][name] = fingerprint(ROOT / name)
        target = args.output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    write_json(args.output / "results.json", report)
    import torch
    import attention.token_merging as token
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    report.update(device=torch.cuda.get_device_name(0), torch_version=str(torch.__version__),
                  cuda_version=torch.version.cuda, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"])
    with torch.inference_mode():
        for job, old in zip(jobs, previous["l33"]):
            identity, metadata = job["capture"], job["metadata"]
            require(fingerprint(identity["path"]) == identity, "capture changed")
            require(metadata == old["metadata"], "native/capture identity mismatch")
            saved = torch.load(identity["path"], map_location="cpu", weights_only=True)
            validate_capture_metadata(saved, metadata)
            require(all(saved[n].dtype == torch.bfloat16 and saved[n].shape == (1, 75600, 4, 128)
                        for n in ("q", "k", "v", "dense_output")), "wrong original tensors")
            q, k, v, flash = [saved[n].to("cuda:0") for n in ("q", "k", "v", "dense_output")]
            del saved
            pid = metadata["prompt_index"]
            seed_path = args.native_root / f"prompt{pid:03d}_native_seeds.npz"
            require(fingerprint(seed_path)["sha256"] == old["seed_cache"]["sha256"], "native seed cache mismatch")
            with np.load(seed_path, allow_pickle=False) as archive:
                tables = {n: archive[n].copy() for n in MODES}
            row = {"input": identity, "metadata": metadata, "seed_cache": fingerprint(seed_path),
                   "native_replication": {}, "groups": [], "complete": False}
            report["records"].append(row)
            selected_indices = torch.tensor([[r * 80 + c for r in range(45)] for c in QUERY_COLUMNS], device=q.device)
            native, bf16 = {}, {}
            for name in MODES:
                output, table, received = observed_attention(token, q, k, v, name)
                measured = metrics(output, flash)
                require(np.array_equal(table, tables[name]), f"{name}: fixed seeds not replicated")
                require(seed_check(table, 4)["sha256"] == old["selectors"][name]["sha256"], "seed receipt mismatch")
                require(np.allclose(measured["frame_head_squared_error_sum"], old["metrics"][name]["frame_head_squared_error_sum"],
                                    rtol=2e-6, atol=1e-10), f"{name}: native complete output changed")
                row["native_replication"][name] = {"exact_seed_table_equal": True, "metrics": measured, "received": received}
                native[name] = output[0, selected_indices].double()
                del output
                # Only remove the FP8 score cast. All algorithms, selectors,
                # original inputs, SQ1 and complete compressed route are fixed.
                bf16_mode = report["bf16_score_modes"][name]
                require("_fp8" not in bf16_mode, "BF16 control retained FP8")
                with patch.dict(MODES, {name: bf16_mode}):
                    bf16_output, bf16_table, bf16_received = observed_attention(token, q, k, v, name)
                require(np.array_equal(bf16_table, table), "disabling FP8 changed seed selector")
                row.setdefault("bf16_score_control", {})[name] = {
                    "exact_seed_table_equal": True, "mode": bf16_mode,
                    "metrics": metrics(bf16_output, flash), "received": bf16_received,
                    "mean_attention_ms": None, "speedup_vs_sdpa": None}
                bf16[name] = bf16_output[0, selected_indices].double()
                del bf16_output
                write_json(args.output / "results.json", report)
            cache = {}
            qgrid = q.reshape(21, 45, 80, 4, 128)
            for slot, head in enumerate(HEADS):
                kh = k[0, :, slot].double().reshape(945, 80, 128)
                vh = v[0, :, slot].double().reshape(945, 80, 128)
                for ci, col in enumerate(QUERY_COLUMNS):
                    query = qgrid[0, :, col, slot].double()
                    reference = block_reference(query, kh, vh, 128 ** -0.5)
                    dense, _, _, _ = reference
                    flash_slice = flash[0, selected_indices[ci], slot].double()
                    key = f"h{head:02d}_c{col:02d}"
                    cache[key + "_dense64"] = dense.cpu().numpy()
                    cache[key + "_flash"] = flash_slice.cpu().numpy()
                    group = {"head": head, "query_column": col, "numel": dense.numel(),
                             "flash_vs_dense64_mse": float((flash_slice - dense).square().mean()), "methods": {}}
                    row["groups"].append(group)
                    for name in MODES:
                        seed_rows = torch.tensor(tables[name][0, 0, :, :, col, slot].reshape(945, 4).astype(np.int64), device=q.device)
                        # RF10 changes the selected row IDs only. Merge always
                        # reloads those real rows from the CURRENT query frame.
                        seed_q = qgrid[0, :, col, slot].double()[seed_rows]
                        merged = ideal_merge(seed_q, kh, vh, col, 128 ** -0.5)
                        result, arrays = route_decomposition(query, merged, reference, 128 ** -0.5)
                        native_slice = native[name][ci, :, slot]
                        native_error = native_slice - dense
                        ideal_error = arrays["ideal"] - dense
                        implementation = native_slice - arrays["ideal"]
                        result.update(native_mse=float(native_error.square().mean()),
                            native_vs_flash_mse=float((native_slice - flash_slice).square().mean()),
                            ideal_vs_flash_mse=float((arrays["ideal"] - flash_slice).square().mean()),
                            native_vs_ideal_mse=float(implementation.square().mean()),
                            native_ideal_cross_term=float(2 * (ideal_error * implementation).mean()),
                            empty_cells=int((~merged[3][:, :4]).sum()))
                        bf16_slice = bf16[name][ci, :, slot]
                        result.update(bf16_mse=float((bf16_slice - dense).square().mean()),
                            bf16_vs_flash_mse=float((bf16_slice - flash_slice).square().mean()),
                            bf16_vs_ideal_mse=float((bf16_slice - arrays["ideal"]).square().mean()))
                        require(abs(result["native_mse"] - result["ideal_mse"] - result["native_vs_ideal_mse"]
                                    - result["native_ideal_cross_term"]) < 1e-10, "native SSE decomposition failed")
                        group["methods"][name] = result
                        for field, value in arrays.items():
                            cache[f"{key}_{name}_{field}"] = value.cpu().numpy()
                        cache[f"{key}_{name}_native"] = native_slice.cpu().numpy()
                        cache[f"{key}_{name}_bf16"] = bf16_slice.cpu().numpy()
                        del merged, arrays, seed_q
                    write_json(args.output / "results.json", report)
                    print(f"prompt {pid} head {head} column {col}", {n: group["methods"][n]["ideal_mse"] for n in MODES}, flush=True)
                    del reference
            cache_path = args.output / f"prompt{pid:03d}_outputs.npz"
            np.savez_compressed(cache_path, **cache)
            row.update(complete=True, output_cache=fingerprint(cache_path))
            write_json(args.output / "results.json", report)
            del q, k, v, flash, qgrid, kh, vh, native, bf16, cache
            torch.cuda.empty_cache()
    report.update(valid=True, complete=True)
    write_json(args.output / "results.json", report)


def parent(args):
    require(not args.output.exists(), "refuse existing output")
    check_capture_environment(os.environ)
    state = gpu_state(args.gpu)
    require(not gpu_processes(state["uuid"]) and state["memory_mib"] <= 100 and state["utilization"] == 0,
            "GPU is not exclusively idle; do not queue")
    args.output.mkdir(parents=True)
    write_json(args.output / "startup.json", {"physical_gpu": args.gpu, "state": state, "compute_processes": []})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4", WAN_SDPA_BACKEND="flash", TRITON_CACHE_DIR=str(args.output / "triton_cache"))
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--capture-root", str(args.capture_root),
               "--oracle-root", str(args.oracle_root), "--native-root", str(args.native_root), "--output", str(args.output)]
    process, foreign, monitor_error = None, [], None
    def stop():
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
    def interrupted(number, frame):
        raise KeyboardInterrupt(f"signal {number}")
    prior = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (args.output / "worker.log").open("x") as log, (args.output / "purity.log").open("x") as purity:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(args.output / "process.json", {"parent_pid": os.getpid(), "worker_pid": process.pid, "command": command})
            try:
                while process.poll() is None:
                    foreign = [p for p in gpu_processes(state["uuid"]) if p["pid"] != process.pid]
                    if foreign:
                        purity.write(json.dumps({"foreign": foreign}) + "\n")
                        purity.flush()
                        break
                    time.sleep(5)
            except BaseException as exc:
                monitor_error = f"{type(exc).__name__}: {exc}"
                purity.write(json.dumps({"monitor_error": monitor_error}) + "\n")
            finally:
                stop()
    finally:
        stop()
        for s, handler in prior.items():
            signal.signal(s, handler)
    try:
        result = json.loads((args.output / "results.json").read_text())
        complete = result["valid"] is True and result["complete"] is True and len(result["records"]) == 3
        complete = complete and all(r["complete"] and len(r["groups"]) == 40 for r in result["records"])
    except (OSError, ValueError, KeyError):
        complete = False
    valid = process is not None and process.returncode == 0 and not foreign and monitor_error is None and complete
    write_json(args.output / "completion.json", {"valid": valid, "worker_exit_code": process.returncode if process else None,
        "results_complete": complete, "foreign": foreign, "monitor_error": monitor_error, "gpu_uuid": state["uuid"]})
    require(valid, "oracle failed; inspect completion and worker.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", type=int)
    for name in ("capture-root", "oracle-root", "native-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    (worker if arguments.worker else parent)(arguments)
