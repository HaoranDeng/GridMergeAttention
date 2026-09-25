"""CPU-only objective/probe cross-audit on frozen RF0 L33 probe banks.

This is not a native attention implementation, a video experiment, or a timing
benchmark. P80 and all-key-Z settings are diagnostic oracles, not production
candidates. All methods use the same production C32/g45 bank and real row seeds.
Final diagnostics reconstruct exact-arithmetic full-80 ownership, then SQ1,
on all 45 RF0 target rows. They do not emulate FP8 merge or global routing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform

import numpy as np

try:
    from .conditional_jeffreys_seed import partition_metrics, select_jeffreys_seeds
except ImportError:
    from conditional_jeffreys_seed import partition_metrics, select_jeffreys_seeds


METHODS = (
    "hellinger_p16", "mass_p16_g256", "jeffreys_p16_g256",
    "jeffreys_p16_g256_no_sq_score", "mass_p16_exact_z", "jeffreys_p16_exact_z",
    "hellinger_p80", "mass_p80_exact_z", "jeffreys_p80_exact_z",
)
PROBES = np.arange(0, 80, 5)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def lse(values):
    return np.logaddexp.reduce(values, axis=-1)


def mass_weights(local, global_norm):
    log_weight = lse(local) - global_norm
    return np.exp(log_weight - log_weight.max(axis=-1, keepdims=True))


def response_weights(local):
    # Production standardization uses base-2 local log partitions and eps=1e-6.
    log2_norm = lse(local) / math.log(2)
    centered = log2_norm - log2_norm.mean(axis=-1, keepdims=True)
    standardized = centered / np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    return np.exp(0.4 * standardized)


def select_span(logits, anchor, weights):
    """Float64 reference for the existing anchored, centered Hellinger GS."""
    require(logits.ndim == 3 and weights.shape == logits.shape[:2], "bad span shape")
    require(np.isfinite(logits).all() and np.isfinite(weights).all()
            and np.min(weights) >= 0, "nonfinite logits or invalid weights")
    count, rows, _ = logits.shape
    require(count > 0 and rows >= 4 and logits.shape[-1] > 0, "span requires four candidates")
    anchor = np.asarray(anchor, dtype=np.int64)
    require(anchor.shape == (count,) and np.all((anchor >= 0) & (anchor < rows)), "bad anchor")
    probability = np.exp(logits - lse(logits)[..., None])
    residual = np.sqrt(probability)
    residual -= residual.mean(axis=-1, keepdims=True)
    used = np.zeros((count, rows), dtype=bool)
    chosen = np.empty((count, 4), dtype=np.int64)
    chosen[:, 0] = anchor
    batch = np.arange(count)
    basis_source = residual[batch, anchor].copy()
    used[batch, anchor] = True
    for step in range(1, 4):
        basis = basis_source / np.sqrt((basis_source ** 2).sum(axis=-1) + 1e-12)[:, None]
        residual -= np.einsum("brp,bp->br", residual, basis)[..., None] * basis[:, None, :]
        score = (residual ** 2).sum(axis=-1) * weights
        score[used] = -np.inf
        pick = score.argmax(axis=-1)
        chosen[:, step] = pick
        used[batch, pick] = True
        basis_source = residual[batch, pick].copy()
    return chosen


def choose(method, full_logits, rows, anchor, global_z, global_g, query_column):
    require(method in METHODS, "unknown prespecified method")
    batch = np.arange(len(full_logits))[:, None]
    candidate = full_logits[batch, rows]
    local = candidate if "p80" in method else candidate[..., PROBES]
    if method.startswith("hellinger"):
        slots = select_span(local, anchor, response_weights(local))
    else:
        denominator = (global_z if "exact_z" in method else global_g)[batch, rows]
        weights = mass_weights(local, denominator)
        if method.startswith("mass"):
            slots = select_span(local, anchor, weights)
        else:
            singleton = query_column if "p80" in method else np.where(query_column % 5 == 0, query_column // 5, -1)
            if method.endswith("no_sq_score"):
                singleton = None
            slots = select_jeffreys_seeds(local, anchor, weights=weights,
                                          num_seeds=4, singleton_index=singleton)
    return rows[batch, slots]


def evaluate(full_logits, values, chosen_rows, query_column, global_z):
    """Full80/SQ1 metrics on all target rows, never the probe partition."""
    batch = np.arange(len(full_logits))[:, None]
    metrics = partition_metrics(full_logits, full_logits[batch, chosen_rows],
                                singleton_index=query_column, return_distributions=True)
    for name in ("reverse_gap", "forward_kl", "jeffreys"):
        require(np.isfinite(metrics[name]).all() and metrics[name].min() >= -1e-10,
                f"invalid {name} in full80 audit")
    require(np.allclose(metrics["reverse_gap"] + metrics["forward_kl"],
                        metrics["jeffreys"], rtol=1e-9, atol=1e-10), "Jeffreys identity failed")
    probability = np.exp(full_logits - lse(full_logits)[..., None])
    log_mass = lse(full_logits) - global_z
    require(log_mass.max() < 2e-4, "local partition exceeds cached global partition")
    # Only FP32 CPU-vs-GPU summation overshoots of an oracle mass are clamped.
    rho = np.exp(np.minimum(log_mass, 0))
    output_error = (metrics["p_hat"] - probability) @ values.astype(np.float64)
    output_sse = (output_error ** 2).sum(axis=-1)
    return {
        "reverse_gap": metrics["reverse_gap"], "forward_kl": metrics["forward_kl"],
        "jeffreys": metrics["jeffreys"], "rho": rho,
        "deficit": -np.expm1(-metrics["reverse_gap"]),
        "conditional_output_sse": output_sse,
        "oracle_logmass_positive_roundoff_count": int(np.count_nonzero(log_mass > 0)),
        "oracle_logmass_max_positive_roundoff": float(max(0, log_mass.max())),
    }


def aggregate(metrics, mask=None):
    mask = np.ones(len(metrics["rho"]), dtype=bool) if mask is None else mask
    rho = metrics["rho"][mask]
    total = float(rho.sum())
    require(total > 0, "zero diagnostic target mass")
    result = {"banks": int(mask.sum()), "target_rows": int(rho.size), "rho_sum": total}
    for name in ("reverse_gap", "forward_kl", "jeffreys", "deficit"):
        result[f"weighted_{name}_sum"] = float(np.sum(rho * metrics[name][mask]))
        result[f"weighted_mean_{name}"] = result[f"weighted_{name}_sum"] / total
    sse = float(np.sum(rho * metrics["conditional_output_sse"][mask]))
    result.update(weighted_conditional_output_sse=sse,
                  weighted_conditional_output_mse=sse / (total * 128),
                  mean_partition_deficit_contribution=result["weighted_deficit_sum"] / rho.size)
    return result


def pool(rows):
    result = {key: sum(row[key] for row in rows) for key in (
        "banks", "target_rows", "rho_sum", "weighted_reverse_gap_sum",
        "weighted_forward_kl_sum", "weighted_jeffreys_sum", "weighted_deficit_sum",
        "weighted_conditional_output_sse")}
    for name in ("reverse_gap", "forward_kl", "jeffreys", "deficit"):
        result[f"weighted_mean_{name}"] = result[f"weighted_{name}_sum"] / result["rho_sum"]
    result["weighted_conditional_output_mse"] = result["weighted_conditional_output_sse"] / (128 * result["rho_sum"])
    result["mean_partition_deficit_contribution"] = result["weighted_deficit_sum"] / result["target_rows"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing to overwrite an earlier audit")
    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("valid") is True and manifest.get("complete") is True,
            "portable cache is not validated and complete")
    args.output_dir.mkdir(parents=True)
    report = {
        "valid": False, "complete": False, "eligible_candidate": False,
        "scope": "CPU FP64 reference; 3 RF0 L33 step0 captures, four heads, 150 fixed banks/head; no global attention or model advanced",
        "selection": "Same production C32/g45 candidate rows, fixed first anchor; 4 real seeds",
        "no_sq_score_semantics": "Only disables singleton in seed-selection scoring; final evaluation retains full80 SQ1 for every method",
        "evaluation": "All45 RF0 query rows, full80 normalized ownership before singleton removal; mathematical merge, not native FP8",
        "mean_attention_ms": None, "speedup_vs_sdpa": None,
        "psnr": None, "ssim": None, "lpips": None, "vbench": None,
        "methods": list(METHODS), "cache_manifest": fingerprint(args.cache_dir / "manifest.json"),
        "execution": {"python": platform.python_version(), "numpy": np.__version__,
                      "platform": platform.platform(), "device": "cpu",
                      "thread_environment": {key: os.environ.get(key) for key in
                          ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        "source": {name: fingerprint(Path(__file__).with_name(name)) for name in (
            "screen_conditional_jeffreys_banks.py", "conditional_jeffreys_seed.py")},
        "records": [],
        "limitations": [
            "An error concentration diagnostic, not a representative model/video benchmark.",
            "P80 and exact global normalizers are off-path oracles, not speed-eligible methods.",
            "Conditional-output MSE excludes global block reweighting, temporal reuse, FP8 merge, and later network effects.",
            "Sampled banks are a fixed post-hoc diagnostic grid; no significance or PSNR inference.",
            "Jeffreys upper-bounds the reverse gap but its decrease need not reduce that gap or output error.",
            "Candidate selection is CPU float64, not a bitwise reproduction of original Triton seed tables.",
        ],
    }
    for prompt in range(3):
        path = args.cache_dir / f"prompt{prompt:03d}_banks.npz"
        receipt = next(item for item in manifest["records"] if item["prompt_index"] == prompt)["cache"]
        actual = fingerprint(path)
        require(actual["sha256"] == receipt["sha256"] and actual["bytes"] == receipt["bytes"], "cache bytes mismatch")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        logits, values = arrays["full_logits"], arrays["local_values"]
        rows, anchor = arrays["candidate_rows"], arrays["anchor"]
        require(logits.shape == (600, 45, 80) and values.shape == (600, 80, 128), "wrong cache tensor shape")
        require(rows.shape == (600, 32) and anchor.shape == (600,), "wrong C32 table")
        require(np.array_equal(rows[np.arange(600), anchor], arrays["key_row"]), "anchor row mismatch")
        record = {"prompt_index": prompt, "cache": actual, "methods": {}}
        save_arrays = {}
        for method in METHODS:
            chosen = choose(method, logits, rows, anchor, arrays["global_log_Z"],
                            arrays["global_log_Z_G256"], arrays["query_column"])
            require(np.array_equal(chosen[:, 0], arrays["key_row"])
                    and np.all(np.diff(np.sort(chosen, axis=-1), axis=-1) > 0), "invalid selected real rows")
            metrics = evaluate(logits, values, chosen, arrays["query_column"], arrays["global_log_Z"])
            record["methods"][method] = {"all": aggregate(metrics),
                "by_head": {str(head): aggregate(metrics, arrays["head"] == head) for head in (0, 10, 20, 30)},
                "oracle_logmass_positive_roundoff_count": metrics["oracle_logmass_positive_roundoff_count"],
                "oracle_logmass_max_positive_roundoff": metrics["oracle_logmass_max_positive_roundoff"]}
            save_arrays[f"{method}_seeds"] = chosen
            for key in ("reverse_gap", "forward_kl", "jeffreys", "conditional_output_sse", "rho"):
                save_arrays[f"{method}_{key}"] = metrics[key]
            print(f"prompt {prompt}: {method} complete", flush=True)
        result_path = args.output_dir / f"prompt{prompt:03d}_metrics.npz"
        np.savez_compressed(result_path, **save_arrays)
        record["metrics_cache"] = fingerprint(result_path)
        report["records"].append(record)
        (args.output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report["summary"] = {method: {
        "all": pool([r["methods"][method]["all"] for r in report["records"]]),
        "by_head": {str(head): pool([r["methods"][method]["by_head"][str(head)] for r in report["records"]])
                    for head in (0, 10, 20, 30)},
    } for method in METHODS}
    report.update(valid=True, complete=True)
    (args.output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
