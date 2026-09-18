"""CPU diagnostic of anchor-free Fisher query clustering on audited real QK.

Only frozen inference tensors are read. Full80/SQ1 conditional metrics are
evaluation-only and never select seeds. No GPU, model, video or native timing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from analysis.fisher_query_kmedoids import coverage, fisher_features, fisher_gs, lloyd_medoids
from analysis.screen_conditional_jeffreys_banks import aggregate, evaluate, fingerprint, pool, require

METHODS = ("fisher_gs", "fisher_medoid_fixed_gs", "fisher_medoid_free_gs", "fisher_medoid_free_direct")


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"), "explicitly disable CUDA")
    require(not args.output_dir.exists(), "fresh output required")
    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("valid") is True and manifest.get("complete") is True, "invalid input cache")
    args.output_dir.mkdir(parents=True)
    sources = [Path(__file__).with_name(name) for name in (
        "screen_fisher_query_kmedoids_cpu.py", "fisher_query_kmedoids.py",
        "screen_conditional_jeffreys_banks.py", "conditional_jeffreys_seed.py")]
    source_dir = args.output_dir / "sources"
    source_dir.mkdir()
    for path in sources:
        shutil.copy2(path, source_dir / path.name)
    report = dict(complete=False, valid=False, device="cpu", methods=list(METHODS),
        scope="3 saved L33 step0 captures; 1800 fixed QK banks; 81000 RF0 target rows; conditional FP64 only",
        selection="C32/g45, Uniform16, shared geometric Fisher metric, four distinct real queries",
        primary="fisher_medoid_free_direct: nearest-mean real seed then farthest-first initialization; all four medoids updated to convergence",
        convergence="stop when real seed indices stop changing; preserve numerical ties; abort if 256 rounds insufficient",
        evaluation="full80 normalized key ownership then SQ1; optimal entropy-corrected beta within each sampled block; all45 RF0 queries",
        limitations=["No global routing, current-frame query reuse, BF16 merge, model or video execution",
                     "Local conditional errors are weighted by cached dense block mass for evaluation only",
                     "CPU float64 reference is not native selector parity or speed evidence"],
        mean_attention_ms=None, speedup_vs_sdpa=None, psnr=None, ssim=None, lpips=None, vbench=None,
        cache_manifest=fingerprint(args.cache_dir / "manifest.json"),
        sources={p.name: fingerprint(p) for p in sources}, records=[])
    for prompt in range(3):
        path = args.cache_dir / f"prompt{prompt:03d}_banks.npz"
        actual = fingerprint(path)
        receipt = next(r for r in manifest["records"] if r["prompt_index"] == prompt)["cache"]
        require(actual["sha256"] == receipt["sha256"] and actual["bytes"] == receipt["bytes"], "cache hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        logits, values, rows, anchor = (arrays[k] for k in ("full_logits", "local_values", "candidate_rows", "anchor"))
        require(logits.shape == (600, 45, 80) and rows.shape == (600, 32), "unexpected geometry")
        require(np.array_equal(rows[np.arange(600), anchor], arrays["key_row"]), "candidate anchor identity mismatch")
        require(np.array_equal(arrays["uniform16_columns"], np.arange(0, 80, 5)), "probe policy mismatch")
        z = logits[np.arange(600)[:, None], rows][..., ::5]
        x = fisher_features(z)
        gs = fisher_gs(x, anchor)
        variants = {"fisher_gs": dict(seeds=gs, iterations=np.zeros(600, dtype=np.int64),
                                    objective_history=coverage(x, gs)[None])}
        variants["fisher_medoid_fixed_gs"] = lloyd_medoids(x, gs, fixed_first=True)
        variants["fisher_medoid_free_gs"] = lloyd_medoids(x, gs)
        variants["fisher_medoid_free_direct"] = lloyd_medoids(x)
        record = dict(prompt_index=prompt, cache=actual, methods={})
        raw = {}
        for method, result in variants.items():
            slots = result["seeds"]
            chosen = rows[np.arange(600)[:, None], slots]
            require(np.all(np.diff(np.sort(chosen, axis=-1), axis=-1) > 0), "repeated real seed")
            measured = evaluate(logits, values, chosen, arrays["query_column"], arrays["global_log_Z"])
            record["methods"][method] = dict(
                all=aggregate(measured),
                by_head={str(h): aggregate(measured, arrays["head"] == h) for h in (0, 10, 20, 30)},
                mean_fisher_nearest_squared=float(coverage(x, slots).mean() / 32),
                mean_iterations=float(result["iterations"].mean()), max_iterations=int(result["iterations"].max()),
                anchor_in_set_fraction=float(np.any(chosen == arrays["key_row"][:, None], axis=1).mean()),
                first_seed_moved_fraction=float((slots[:, 0] != gs[:, 0]).mean()))
            for key in ("seeds", "iterations", "objective_history"):
                raw[f"{method}_{key}"] = result[key]
            raw[f"{method}_real_rows"] = chosen
            for key in ("reverse_gap", "forward_kl", "conditional_output_sse", "rho"):
                raw[f"{method}_{key}"] = measured[key]
            print(json.dumps(dict(prompt=prompt, method=method, **record["methods"][method]["all"],
                                  fisher=record["methods"][method]["mean_fisher_nearest_squared"])), flush=True)
        np.savez_compressed(args.output_dir / f"prompt{prompt:03d}_metrics.npz", **raw)
        report["records"].append(record)
        write(args.output_dir / "diagnostic.json", report)
    report["summary"] = {}
    for method in METHODS:
        rows = [r["methods"][method] for r in report["records"]]
        summary = dict(all=pool([r["all"] for r in rows]),
            by_head={str(h): pool([r["by_head"][str(h)] for r in rows]) for h in (0, 10, 20, 30)})
        for key in ("mean_fisher_nearest_squared", "mean_iterations", "anchor_in_set_fraction", "first_seed_moved_fraction"):
            summary[key] = float(np.mean([r[key] for r in rows]))
        summary["max_iterations"] = max(r["max_iterations"] for r in rows)
        report["summary"][method] = summary
    for path in sources:
        require(fingerprint(path)["sha256"] == report["sources"][path.name]["sha256"], "source changed during run")
    report.update(complete=True, valid=True)
    write(args.output_dir / "diagnostic.json", report)


if __name__ == "__main__":
    main()
