"""Recompute Fisher-GS for each query-frame/key-block on frozen real QKV.

CPU mathematical diagnostic with every key retained in global compressed
attention. No GPU, native kernel, model trajectory or video is executed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from analysis.fisher_query_kmedoids import fisher_features, fisher_gs
from analysis.check_global_route_medoids_cpu import full80_state

HEADS = (0, 10, 20, 30)
COLUMNS = (0, 40, 75)
QUERY_FRAMES = (0, 10, 20)
METHODS = ("fisher_rf0", "fisher_perframe", "uniform_rf0", "uniform_perframe")
SCALE = 128 ** -0.5
CAPTURE_IDENTITIES = (
    (309661725, "98df0871af091aee6a80a256bacb95f4cb5cb4abd5e4429aaae83cbaa133dcce"),
    (309661725, "5639fa83ebf2eacc1d738ee013a66293a90722308b19bbea2becb75a569e415b"),
    (309661789, "da9a0aced1bad63e113742baef928af147882c7dd754ba765662296c2c357481"),
)
CANDIDATE_SHA = "ab288d3a1d70013f912a944f912337633cfc6fe2228b9ee4649febcb94c42e22"


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_from_logits(z, anchors, geometry):
    """Same anchored GS, either geometric Fisher or uniform logit geometry."""
    z = np.asarray(z, dtype=np.float64)
    if geometry == "fisher":
        features = fisher_features(z)
    elif geometry == "uniform":
        features = (z - z.mean(-1, keepdims=True)) / np.sqrt(z.shape[-1])
    else:
        raise ValueError("unknown geometry")
    return fisher_gs(features, anchors)


def candidate_table(torch, source_root):
    path = source_root / "attention" / "adaptive_seed_pytorch.py"
    require(digest(path) == CANDIDATE_SHA, "production candidate table source changed")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "multiscale_candidate_table"]
    require(len(nodes) == 1, "candidate function missing")
    scope = {"torch": torch, "_MULTISCALE_TABLE_CACHE": {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    table, anchors = scope["multiscale_candidate_table"](45, 32, "cpu", geometry_count=45)
    rows, anchors = table.numpy().astype(np.int64), anchors.numpy().astype(np.int64)
    require(np.array_equal(rows[np.arange(45), anchors], np.arange(45)), "anchor not matched")
    return np.tile(rows, (21, 1)), np.tile(anchors, 21)


def aggregate(records):
    result = {}
    for method in METHODS:
        rows = [r["methods"][method] for r in records]
        result[method] = dict(groups=len(rows), target_queries=45 * len(rows),
            mean_global_reverse_kl=float(np.mean([r["mean_global_reverse_kl"] for r in rows])),
            mse_vs_dense_fp64=float(np.mean([r["mse_vs_dense_fp64"] for r in rows])),
            mse_vs_captured_dense_bf16=float(np.mean([r["mse_vs_captured_dense_bf16"] for r in rows])),
            changed_seed_set_fraction=float(np.mean([r["changed_seed_set_fraction"] for r in rows])),
            max_kl_identity_error=max(r["max_kl_identity_error"] for r in rows),
            max_probability_mass_error=max(r["max_probability_mass_error"] for r in rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"), "explicitly disable CUDA")
    require(all(os.environ.get(k) == "4" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")), "four CPU threads required")
    require(not args.output.exists(), "fresh output only")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "CUDA already initialized")
    root = Path(__file__).resolve().parents[1]
    candidates, anchors = candidate_table(torch, root)
    args.output.mkdir(parents=True)
    source_dir = args.output / "sources"
    source_dir.mkdir()
    sources = [Path(__file__), root / "analysis/fisher_query_kmedoids.py",
               root / "analysis/check_global_route_medoids_cpu.py",
               root / "attention/adaptive_seed_pytorch.py"]
    source_hashes = {str(p): digest(p) for p in sources}
    for path in sources:
        shutil.copy2(path, source_dir / path.name)
    report = dict(complete=False, valid=False, device="cpu", cuda_initialized=False,
        scope="3 original BF16 L33 step0 captures; 4 heads; columns0/40/75; query frames0/10/20; full global key support",
        methods=list(METHODS), primary_comparison="fisher_perframe versus fisher_rf0",
        primary_change="current query frame recomputes Fisher metric and all four seed indices for every key frame/row; real matched anchor retained",
        controls="uniform centered-logit geometry uses identical GS, candidates and merge; selection sees no value or dense target",
        selection="C32/g45, U16, anchored GS. BF16 QK operands cast FP32, CPU FP32 dot then FP64 scale/features/GS",
        measurement="original BF16 QKV promoted to FP64; full80 normalized ownership, SQ1, direct cell normalization, all945x5 components in one global softmax",
        query_frames=list(QUERY_FRAMES), query_columns=list(COLUMNS), heads=list(HEADS), key_blocks=945,
        limitations=["Fixed sparse query sample from one layer and step, not a video or full-network result",
                     "Ideal arithmetic output, not native BF16 candidate execution or seed parity",
                     "Every measured query uses all75600 original keys through compressed global attention",
                     "CPU diagnostic wall time is not native attention latency"],
        mean_attention_ms=None, speedup_vs_sdpa=None, psnr=None, ssim=None, lpips=None, vbench=None,
        cpu_threads=4, source_hashes=source_hashes, captures=[], records=[])
    def save():
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    started = time.perf_counter()
    for prompt in range(3):
        path = args.capture_root / f"prompt{prompt:03d}_call033_layer33.pt"
        size, expected_sha = CAPTURE_IDENTITIES[prompt]
        require(path.stat().st_size == size and digest(path) == expected_sha, "capture fingerprint mismatch")
        stat_before = (path.stat().st_size, path.stat().st_mtime_ns)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        require(set(saved) == {"q", "k", "v", "dense_output", "metadata"}, "wrong capture keys")
        for name in ("q", "k", "v", "dense_output"):
            tensor = saved[name]
            require(tuple(tensor.shape) == (1, 75600, 4, 128) and tensor.dtype == torch.bfloat16
                    and tensor.device.type == "cpu" and bool(torch.isfinite(tensor).all()), "invalid captured tensor")
        report["captures"].append(dict(path=str(path), sha256=expected_sha, bytes=size, metadata=saved["metadata"]))
        q, k, v, reference = [saved[n].reshape(21, 45, 80, 4, 128) for n in ("q", "k", "v", "dense_output")]
        arrays = {"candidate_rows": candidates, "anchor_slots": anchors}
        for head_slot, head in enumerate(HEADS):
            keys32 = k[..., head_slot, :].reshape(945, 80, 128).float()
            keys64 = keys32.double().numpy()
            values64 = v[..., head_slot, :].reshape(945, 80, 128).double().numpy()
            probe_keys = keys32[:, ::5]
            for column in COLUMNS:
                shared = None
                for frame in QUERY_FRAMES:
                    query32 = q[frame, :, column, head_slot].float()
                    query64 = query32.double().numpy()
                    # Same computation at RF0 for shared and per-frame paths.
                    z = (torch.bmm(query32[torch.from_numpy(candidates)], probe_keys.transpose(1, 2)).double() * SCALE).numpy()
                    selected = {g: candidates[np.arange(945)[:, None], select_from_logits(z, anchors, g)]
                                for g in ("fisher", "uniform")}
                    if frame == 0:
                        shared = {g: s.copy() for g, s in selected.items()}
                    seeds = dict(fisher_rf0=shared["fisher"], fisher_perframe=selected["fisher"],
                                 uniform_rf0=shared["uniform"], uniform_perframe=selected["uniform"])
                    full = query64 @ keys64.reshape(-1, 128).T * SCALE
                    dense_log_z = np.logaddexp.reduce(full, axis=-1)
                    dense_output = np.exp(full - dense_log_z[:, None]) @ values64.reshape(-1, 128)
                    all_logits = full.reshape(45, 945, 80).transpose(1, 0, 2)
                    captured_output = reference[frame, :, column, head_slot].double().numpy()
                    item = dict(prompt=prompt, head=head, query_column=column, query_frame=frame, methods={})
                    prefix = f"h{head}_c{column}_f{frame}"
                    states = {}
                    for method in METHODS:
                        rows = seeds[method]
                        require(np.array_equal(rows[:, 0], np.tile(np.arange(45), 21))
                                and np.all(np.diff(np.sort(rows, axis=-1), axis=-1) > 0), "anchor/distinct seed failure")
                        state = full80_state(all_logits, values64, rows, column, dense_log_z, dense_output)
                        states[method] = state
                        base = shared["fisher" if method.startswith("fisher") else "uniform"]
                        changed = np.any(np.sort(rows, axis=1) != np.sort(base, axis=1), axis=1)
                        item["methods"][method] = dict(mean_global_reverse_kl=float(state["gap"].mean()),
                            mse_vs_dense_fp64=float(state["per_row_mse"].mean()),
                            mse_vs_captured_dense_bf16=float(np.square(state["output"] - captured_output).mean()),
                            changed_seed_set_fraction=float(changed.mean()),
                            max_kl_identity_error=state["direct_kl_max_error"],
                            max_probability_mass_error=state["distribution_mass_max_error"])
                        for name in ("output", "gap", "per_row_mse"):
                            arrays[f"{prefix}_{method}_{name}"] = state[name]
                        arrays[f"{prefix}_{method}_seeds"] = rows
                    if frame == 0:
                        for geometry in ("fisher", "uniform"):
                            require(np.array_equal(states[geometry + "_rf0"]["output"], states[geometry + "_perframe"]["output"]), "RF0 paths disagree")
                    arrays[f"{prefix}_dense_fp64"] = dense_output
                    report["records"].append(item)
                    save()
                    print(json.dumps(dict(prompt=prompt, head=head, column=column, frame=frame,
                        mse={m: item["methods"][m]["mse_vs_dense_fp64"] for m in METHODS})), flush=True)
                    del states, full, all_logits
        np.savez_compressed(args.output / f"prompt{prompt:03d}_arrays.npz", **arrays)
        require(stat_before == (path.stat().st_size, path.stat().st_mtime_ns), "capture changed during diagnostic")
        del saved, q, k, v, reference, arrays
    require(len(report["records"]) == 108, "incomplete fixed query sample")
    report["summary"] = dict(all=aggregate(report["records"]),
        later_frames=aggregate([r for r in report["records"] if r["query_frame"] != 0]),
        by_query_frame={str(f): aggregate([r for r in report["records"] if r["query_frame"] == f]) for f in QUERY_FRAMES},
        by_prompt={str(p): aggregate([r for r in report["records"] if r["prompt"] == p]) for p in range(3)},
        by_head={str(h): aggregate([r for r in report["records"] if r["head"] == h]) for h in HEADS})
    require(all(digest(Path(p)) == sha for p, sha in source_hashes.items()), "source changed during diagnostic")
    require(not torch.cuda.is_initialized(), "CUDA was initialized")
    report.update(complete=True, valid=True, cpu_diagnostic_seconds=time.perf_counter() - started)
    save()
    print(json.dumps(report["summary"]["all"], indent=2), flush=True)


if __name__ == "__main__":
    main()
