"""Diagnose the existing P16/G256 mass sketch on three audited RF0 L33 calls.

This measures fixed sampling error, not a new selector or inference method.
Only step 0, four saved heads, and the declared deterministic grid are used.
GPU execution requires a separate approved launcher with exclusive-GPU checks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUERY_COLUMNS = (0, 1, 20, 21, 40, 41, 60, 61, 75, 76)
KEY_ROWS = (0, 11, 22, 33, 44)
KEY_FRAMES = (0, 10, 20)
HEADS = (0, 10, 20, 30)
ROWS, COLS, FRAMES, D = 45, 80, 21, 128
N, P, G = FRAMES * ROWS * COLS, 16, 256
PHASE_SIZE = N // 5
BLOCKS = tuple((frame, row) for frame in KEY_FRAMES for row in KEY_ROWS)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def historical_indices():
    """The actual historical arithmetic indices, not a changed probe policy."""
    phase = np.flatnonzero(np.arange(N) % COLS % 5 == 0)
    global_probe = np.arange(G, dtype=np.int64) * (N // G)
    require(len(phase) == PHASE_SIZE and np.isin(global_probe, phase).all(),
            "P16/G256 phase identity changed")
    return phase, global_probe


def softmax64(logits):
    values = np.asarray(logits, dtype=np.float64)
    require(values.ndim == 1 and values.size > 0 and np.isfinite(values).all(),
            "Require a finite nonempty candidate vector")
    exponential = np.exp(values - values.max())
    return exponential / exponential.sum()


def centered_rms(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean((values - values.mean()) ** 2)))


def diagnose_bank(log_global, log_phase, log_global_probe, log_local,
                  log_local_probe, candidate_rows):
    """Exact finite phase decomposition; all inputs use natural logarithms.

    All partitions must come from the SAME per-query QK values. The global
    sketch estimates Z by (N/G) Z_G and the local sketch estimates Z_b by
    (80/16) Z_b0. Constants cancel across candidate rows, but must be restored
    before testing whether an estimated block mass exceeds one.
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in
              (log_global, log_phase, log_global_probe, log_local, log_local_probe)]
    rows = np.asarray(candidate_rows)
    require(rows.ndim == 1 and len(rows) > 0 and len(np.unique(rows)) == len(rows),
            "Candidate rows must be nonempty and distinct")
    require(all(v.shape == rows.shape and np.isfinite(v).all() for v in arrays),
            "All log-partitions must be finite vectors of candidate-bank size")
    z, z0, zg, zb, zb0 = arrays
    require(np.all(z0 <= z + 1e-10) and np.all(zg <= z0 + 1e-10)
            and np.all(zb <= z + 1e-10) and np.all(zb0 <= zb + 1e-10)
            and np.all(zb0 <= z0 + 1e-10), "Invalid subset partition ordering")
    log_rho = zb - z
    log_rho_hat = math.log(G / PHASE_SIZE) + zb0 - zg
    phase_mismatch = (zb0 - zb) - (z0 - z)
    within_phase_error = math.log(PHASE_SIZE / G) + zg - z0
    log_error = log_rho_hat - log_rho
    identity_error = float(np.max(np.abs(log_error - phase_mismatch + within_phase_error)))
    require(identity_error < 1e-9, "Phase decomposition does not reconstruct mass error")
    pi, pi_hat = softmax64(log_rho), softmax64(log_rho_hat)
    # Components isolate candidate-relative effects, not error in a shared scale.
    pi_phase = softmax64(log_rho + phase_mismatch)
    pi_subsample = softmax64(log_rho - within_phase_error)
    best, estimated = int(pi.argmax()), int(pi_hat.argmax())
    return {
        "candidate_rows": rows.astype(int).tolist(),
        "pi_tv": float(0.5 * np.abs(pi_hat - pi).sum()),
        "phase_only_pi_tv": float(0.5 * np.abs(pi_phase - pi).sum()),
        "within_phase_only_pi_tv": float(0.5 * np.abs(pi_subsample - pi).sum()),
        "exact_top_row": int(rows[best]), "estimated_top_row": int(rows[estimated]),
        "top_row_flip": bool(estimated != best),
        "estimated_top_outside_exact_max_set": bool(pi[estimated] < pi[best] - 1e-12),
        "estimated_top_exact_rank": int(1 + np.count_nonzero(pi > pi[estimated] + 1e-12)),
        "candidate_centered_log_error_rms": centered_rms(log_error),
        "candidate_centered_phase_mismatch_rms": centered_rms(phase_mismatch),
        "candidate_centered_within_phase_error_rms": centered_rms(within_phase_error),
        "mean_log_error": float(log_error.mean()),
        "mean_phase_mismatch": float(phase_mismatch.mean()),
        "mean_within_phase_error": float(within_phase_error.mean()),
        "estimated_mass_above_one_count": int(np.count_nonzero(log_rho_hat > 0)),
        "candidate_count": len(rows),
        "max_log_estimated_mass": float(log_rho_hat.max()),
        "mean_exact_global_phase_mass": float(np.exp(z0 - z).mean()),
        "mean_exact_local_phase_mass": float(np.exp(zb0 - zb).mean()),
        "decomposition_max_abs_error": identity_error,
    }


def key_chunk_plan(start, stop):
    """Disjoint streaming chunks, with subset offsets reused for the same QK."""
    require(0 <= start < stop <= N, "Invalid key chunk")
    indices = np.arange(start, stop, dtype=np.int64)
    subsets = [np.arange(stop - start), np.flatnonzero(indices % COLS % 5 == 0),
               np.flatnonzero((indices % (N // G) == 0) & (indices // (N // G) < G))]
    for frame, row in BLOCKS:
        block = (frame * ROWS + row) * COLS
        inside = (indices >= block) & (indices < block + COLS)
        subsets.extend((np.flatnonzero(inside), np.flatnonzero(inside & (indices % COLS % 5 == 0))))
    return subsets


def stream_partitions(q, k, *, device, query_chunk, key_chunk):
    """FP32 QK; FP64 logsumexp/logaddexp, retaining only scalar partitions.

    Output columns are Z, Z_phase, Z_G, followed by (Z_b, Z_b0) per BLOCKS.
    No full N-key score matrix survives a streaming chunk.
    """
    import torch
    require(tuple(k.shape) == (N, D) and q.ndim == 2 and q.shape[1] == D,
            "Unexpected query/key shape")
    require(query_chunk > 0 and key_chunk > 0, "Chunk sizes must be positive")
    result = np.empty((len(q), 3 + 2 * len(BLOCKS)), dtype=np.float64)
    keys = k.to(device=device, dtype=torch.float32).contiguous()
    plans = [(start, min(start + key_chunk, N), key_chunk_plan(start, min(start + key_chunk, N)))
             for start in range(0, N, key_chunk)]
    with torch.inference_mode():
        for query_start in range(0, len(q), query_chunk):
            query = q[query_start:query_start + query_chunk].to(device=device, dtype=torch.float32)
            cache = torch.full((len(query), result.shape[1]), -torch.inf,
                               dtype=torch.float64, device=device)
            for key_start, key_stop, subsets in plans:
                scores = (query @ keys[key_start:key_stop].T) * (D ** -0.5)
                require(bool(torch.isfinite(scores).all()), "Nonfinite FP32 QK dot")
                scores = scores.to(torch.float64)
                for column, subset in enumerate(subsets):
                    if len(subset):
                        index = torch.as_tensor(subset, dtype=torch.long, device=device)
                        value = torch.logsumexp(scores.index_select(1, index), dim=1)
                        cache[:, column] = torch.logaddexp(cache[:, column], value)
            result[query_start:query_start + len(query)] = cache.cpu().numpy()
    require(np.isfinite(result).all(), "Incomplete/nonfinite partition cache")
    return result


def aggregate(records):
    require(bool(records), "Cannot summarize an empty sample")
    means = ("pi_tv", "phase_only_pi_tv", "within_phase_only_pi_tv",
             "candidate_centered_log_error_rms", "candidate_centered_phase_mismatch_rms",
             "candidate_centered_within_phase_error_rms", "mean_log_error",
             "mean_phase_mismatch", "mean_within_phase_error",
             "mean_exact_global_phase_mass", "mean_exact_local_phase_mass")
    count = sum(r["candidate_count"] for r in records)
    result = {"sampled_banks": len(records), "candidate_block_observations": count,
              **{key if key.startswith("mean_") else f"mean_{key}":
                 math.fsum(r[key] for r in records) / len(records) for key in means},
              "top_row_flip_fraction": sum(r["top_row_flip"] for r in records) / len(records),
              "estimated_top_outside_exact_max_set_fraction":
                  sum(r["estimated_top_outside_exact_max_set"] for r in records) / len(records),
              "estimated_mass_above_one_fraction":
                  sum(r["estimated_mass_above_one_count"] for r in records) / count,
              "max_decomposition_abs_error": max(r["decomposition_max_abs_error"] for r in records)}
    result["pi_tv_quantiles"] = dict(zip(("min", "median", "p90", "max"),
                                        np.quantile([r["pi_tv"] for r in records], [0, .5, .9, 1]).tolist()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True,
                        help="The wrapper run's captures subdirectory, not the wrapper root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--query-chunk", type=int, default=32)
    parser.add_argument("--key-chunk", type=int, default=4096)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite an existing diagnostic")
    require(args.query_chunk > 0 and args.key_chunk > 0, "Chunk sizes must be positive")
    for directory in (ROOT, ROOT / "analysis"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from analyze_rf0_layer33_captures import validate_bundle, load_capture, fingerprint
    manifest, audit = validate_bundle(args.capture_root)
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.device == "cuda":
        require(torch.cuda.is_available(), "CUDA is unavailable")
    table, anchors = multiscale_candidate_table(ROWS, 16, "cpu", geometry_count=16)
    table, anchors = table.numpy(), anchors.numpy()
    selected_rows = sorted(set(table[list(KEY_ROWS)].reshape(-1).tolist()))
    query_identity = [(row, col) for row in selected_rows for col in QUERY_COLUMNS]
    query_lookup = {identity: i for i, identity in enumerate(query_identity)}
    query_tokens = torch.tensor([row * COLS + col for row, col in query_identity], dtype=torch.long)
    phase, probes = historical_indices()
    report = {
        "complete": False, "scope": "L33 step0 only; fixed RF0 inference-time QK mass-sketch diagnosis",
        "limitations": ["No candidate output, new selector, fitting, PSNR or runtime measurement.",
                        "Deterministic spatial/block sample, not an unbiased population estimate.",
                        "All partitions are exact sums over the indicated subsets of the same FP32 QK dots, not real-arithmetic exact values.",
                        "Mass weights exclude V, the hard-partition objective and SQ1; this isolates the existing unpeeled local-mass sketch.",
                        "Normalized candidate mass is not seed quality; rank changes do not prove output error causality.",
                        "The shared producer's bundle gate fingerprints all 12 files, but only three step0 tensor payloads are loaded here.",
                        "External launcher must enforce and monitor one exclusively clean GPU; this script does not do GPU purity monitoring."],
        "capture_audit": audit, "checkpoint_weights": manifest["checkpoint_weights"],
        "precision": {"source": "unchanged BF16", "dot": "FP32 matmul, TF32 disabled, highest precision",
                      "log_partitions": "FP64 logsumexp and streaming logaddexp", "numpy_summary": "float64",
                      "device": args.device, "torch_version": torch.__version__,
                      "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "float32_matmul_precision": torch.get_float32_matmul_precision()},
        "sampling": {"query_frame": 0, "query_columns": QUERY_COLUMNS, "key_rows": KEY_ROWS,
                     "key_frames": KEY_FRAMES, "heads": HEADS, "candidate_count": 16, "candidate_geometry_count": 16,
                     "candidate_bank": {str(row): table[row].tolist() for row in KEY_ROWS},
                     "anchor_positions": {str(row): int(anchors[row]) for row in KEY_ROWS},
                     "distinct_query_rows": selected_rows, "distinct_queries_per_head": len(query_identity),
                     "global_key_count": N, "phase_key_count": len(phase), "global_probe_count": G,
                     "global_probe_stride": N // G, "global_probe_indices": probes.tolist(),
                     "local_probe_columns": list(range(0, COLS, 5)),
                     "query_chunk": args.query_chunk, "key_chunk": args.key_chunk},
        "aggregation": "Unweighted means over declared candidate banks; repeated query/block observations are not independent replicates.",
        "sources": {name: fingerprint(ROOT / name) for name in
                    ("analysis/diagnose_rf0_mass_probe_alias.py", "analysis/analyze_rf0_layer33_captures.py",
                     "attention/adaptive_seed_pytorch.py", "attention/entropy_mass_seed_triton.py")},
        "partition_caches": [], "records": [],
    }
    captures = [item for item in manifest["captures"] if item["step_index"] == 0]
    require([item["prompt_index"] for item in captures] == [0, 1, 2]
            and all(item["call_index"] == 33 for item in captures), "Require exactly the three step0 captures")
    for item in captures:
        name = item["capture_file"]
        payload = load_capture(args.capture_root / name, item, audit["capture_files"][name])
        for head_slot, head in enumerate(HEADS):
            q = payload["q"][0, :, head_slot].index_select(0, query_tokens)
            k = payload["k"][0, :, head_slot]
            cache = stream_partitions(q, k, device=args.device,
                                      query_chunk=args.query_chunk, key_chunk=args.key_chunk)
            report["partition_caches"].append({"capture_file": name, "head": head,
                "query_row_columns": query_identity, "columns": ["log_Z", "log_Z_phase", "log_Z_G256"]
                    + [f"log_Z_{suffix}_t{frame}_r{row}" for frame, row in BLOCKS for suffix in ("80", "P16")],
                "natural_log_partitions": cache.tolist()})
            for col in QUERY_COLUMNS:
                for block_index, (frame, row) in enumerate(BLOCKS):
                    bank = table[row]
                    bank_cache = cache[[query_lookup[(int(r), col)] for r in bank]]
                    result = diagnose_bank(bank_cache[:, 0], bank_cache[:, 1], bank_cache[:, 2],
                                           bank_cache[:, 3 + 2 * block_index], bank_cache[:, 4 + 2 * block_index], bank)
                    report["records"].append({"capture_file": name, "prompt_index": item["prompt_index"],
                        "step_index": 0, "call_index": 33, "layer_index": 33, "head": head,
                        "query_frame": 0, "query_column": col, "key_frame": frame, "key_row": row, **result})
            print(f"prompt={item['prompt_index']} head={head}: fixed-grid mass partitions complete", flush=True)
        del payload
        audit["tensor_payloads_validated"] += 1
    records = report["records"]
    require(len(records) == 3 * len(HEADS) * len(QUERY_COLUMNS) * len(BLOCKS), "Incomplete sampled-bank coverage")
    report["summary"] = {"all_sampled_heads": aggregate(records),
        "head30": aggregate([r for r in records if r["head"] == 30]),
        "other_heads": aggregate([r for r in records if r["head"] != 30]),
        "by_head": {str(head): aggregate([r for r in records if r["head"] == head]) for head in HEADS},
        "by_prompt": {str(prompt): aggregate([r for r in records if r["prompt_index"] == prompt]) for prompt in range(3)}}
    report["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
