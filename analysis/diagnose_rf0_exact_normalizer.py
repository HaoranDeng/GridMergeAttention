"""OFF-PATH exact-normalizer oracle on three RF0 L33 step0 captures.

Compare original partition-G256 against all-key and phase0-only partition
normalizers. Neither dense-normalizer oracle is an eligible GCA candidate,
video experiment, hybrid inference method, or attention speed claim.
No production file, candidate table, local P16 rule, merge or route is changed.
GPU execution belongs to a separately approved exclusive-GPU launcher.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "analysis"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import analyze_rf0_layer33_captures as base

require = base.require
HEADS, ROWS, COLS, FRAMES, DIM = (0, 10, 20, 30), 45, 80, 21, 128
N, G, PHASE_N, SCALE = 75600, 256, 15120, 128 ** -0.5
MODE = base.MODES["partition_c16"]
METHODS = ("partition_g256", "partition_oracle_allkeys", "partition_oracle_phase0")
NORM_SHAPE = (1, 1, 45, 80, 4)
QUERY_CHUNK, KEY_CHUNK = 64, 4096


def phase_offsets(start, stop, *, phase_stride=5):
    """Offsets for original Uniform16 phase: flat key index modulo five=0."""
    require(0 <= start < stop and phase_stride > 0, "invalid key chunk")
    return range((-start) % phase_stride, stop - start, phase_stride)


def log2_g256_convention(log_partition, key_count, *, probe_count=G):
    """log2[(G/key_count) Z]; the shared scale cancels across candidates."""
    require(key_count >= probe_count > 0, "invalid normalizer cardinalities")
    return (log_partition - math.log(key_count / probe_count)) / math.log(2)


def stream_normalizers(query, keys, *, query_chunk=QUERY_CHUNK, key_chunk=KEY_CHUNK, scale=SCALE):
    """Return CPU FP64 [H,Q,2] natural LSEs and true global-peak diagnostics.

    Inputs [H,Q,D]/[H,N,D] are the original RF0 queries and all-frame keys.
    Only bounded QK tiles exist: FP32 non-TF32 dots, FP64 streaming LSE.
    There are no local-block partitions, attention outputs, or fitted values.
    """
    import torch
    require(query.ndim == keys.ndim == 3 and query.shape[0] == keys.shape[0]
            and query.shape[2] == keys.shape[2] and query.device == keys.device,
            "normalizer query/key layout mismatch")
    require(query_chunk > 0 and key_chunk > 0 and math.isfinite(scale), "invalid streaming settings")
    heads, query_count, _ = query.shape
    require(query_count > 0 and keys.shape[1] > 0, "empty query/key population")
    result = torch.empty((heads, query_count, 2), dtype=torch.float64, device="cpu")
    peak_value = torch.empty((heads, query_count), dtype=torch.float32, device="cpu")
    peak_index = torch.empty((heads, query_count), dtype=torch.long, device="cpu")
    keys = keys.float().contiguous()
    prior_precision, prior_tf32 = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode():
            for qs in range(0, query_count, query_chunk):
                q = query[:, qs:qs + query_chunk].float().contiguous()
                total = torch.full((heads, q.shape[1], 2), -torch.inf, device=q.device, dtype=torch.float64)
                best = torch.full((heads, q.shape[1]), -torch.inf, device=q.device)
                best_index = torch.zeros((heads, q.shape[1]), device=q.device, dtype=torch.long)
                for ks in range(0, keys.shape[1], key_chunk):
                    ke = min(ks + key_chunk, keys.shape[1])
                    logits = torch.bmm(q, keys[:, ks:ke].transpose(1, 2)) * scale
                    require(bool(torch.isfinite(logits).all()), "nonfinite FP32 oracle logits")
                    value, local_index = logits.max(-1)
                    better = value > best  # Keep the earliest flat key on exact ties.
                    best_index = torch.where(better, local_index + ks, best_index)
                    best = torch.maximum(best, value)
                    logits = logits.double()
                    total[..., 0] = torch.logaddexp(total[..., 0], torch.logsumexp(logits, -1))
                    offsets = phase_offsets(ks, ke)
                    if len(offsets):
                        index = torch.tensor(list(offsets), device=q.device, dtype=torch.long)
                        total[..., 1] = torch.logaddexp(total[..., 1], torch.logsumexp(logits.index_select(-1, index), -1))
                require(bool(torch.isfinite(total).all()), "incomplete oracle normalizer")
                result[:, qs:qs + q.shape[1]] = total.cpu()
                peak_value[:, qs:qs + q.shape[1]], peak_index[:, qs:qs + q.shape[1]] = best.cpu(), best_index.cpu()
    finally:
        torch.set_float32_matmul_precision(prior_precision)
        torch.backends.cuda.matmul.allow_tf32 = prior_tf32
    peak = {"max_logit_fp32": peak_value, "argmax_key_index": peak_index,
            "peak_mass_fp64": (peak_value.double() - result[..., 0]).exp(),
            "argmax_in_phase0": peak_index.remainder(5) == 0,
            "argmax_in_g256": (peak_index.remainder(N // G) == 0) & (peak_index // (N // G) < G)}
    require(bool(torch.isfinite(peak["peak_mass_fp64"]).all())
            and bool((peak["peak_mass_fp64"] <= 1 + 1e-12).all()), "invalid exact peak mass")
    return result, peak


def native_layout(raw, key_count, device):
    """Map [four heads, 3600 RF0 queries] to the unchanged native norm layout."""
    import torch
    require(tuple(raw.shape) == (4, ROWS * COLS) and bool(torch.isfinite(raw).all()), "wrong oracle array")
    return log2_g256_convention(raw, key_count).transpose(0, 1).reshape(NORM_SHAPE).to(
        device=device, dtype=torch.float32).contiguous()


def normalizer_adapter(real, expected_q, expected_k, calls, *, replacement=None):
    """Observe G256 unchanged, or bypass it with a precomputed oracle norm."""
    import torch
    def dispatch(query, keys, scale=None, **options):
        require(not calls, "expected exactly one global-normalizer invocation")
        require(scale == SCALE and options == {"global_probe_count": G}, "normalizer arguments changed")
        require(query.dtype == expected_q.dtype and keys.dtype == expected_k.dtype
                and torch.equal(query, expected_q) and torch.equal(keys, expected_k),
                "normalizer must receive original BF16 RF0 query and unchanged all-frame K")
        # Deliberate branch: an oracle never calls the replaced G256 routine.
        norm = real(query, keys, scale=scale, **options) if replacement is None else replacement
        require(tuple(norm.shape) == NORM_SHAPE and norm.dtype == torch.float32
                and norm.device == query.device and norm.is_contiguous() and bool(torch.isfinite(norm).all()),
                "normalizer must be finite contiguous FP32 native log2 layout")
        calls.append({"query_shape": list(query.shape), "key_shape": list(keys.shape), "scale": scale,
                      "kwargs": options, "original_rf0_qk_verified": True,
                      "original_g256_called": replacement is None, "value": norm.detach().cpu().clone()})
        return norm
    return dispatch


def unpatched_attention(token, q, k, v):
    """The baseline executes before any observation/normalizer monkeypatch."""
    return token._hard_c4_attention(q, k, v, h=ROWS, w=COLS, h_reduce=1, w_reduce=1, sm_scale=SCALE,
        query_outer_chunk=21, head_chunk=8, hard_route_backend="hard_c4", seed_mode=MODE,
        exact_intra_frame_attention=False, exact_neighbor_window_columns=0, exact_query_rows=0,
        first_frame_full_attention=False, full_attention_anchor_time_count=0,
        full_attention_anchor_row_count=0, full_attention_anchor_scale=1.0).contiguous()


def seed_changes(table, baseline):
    import numpy as np
    result = base.seed_comparison(table)  # Validate anchors, real distinct rows and uint8 layout.
    require(table.shape == baseline.shape and table.dtype == baseline.dtype, "seed comparison shape mismatch")
    shared = (table[..., :, None] == baseline[..., None, :]).any(-1).sum(-1)[0, 0]
    changed = (table != baseline).any(-1)[0, 0]
    result.update(reference="partition_g256", mean_shared_rows=float(shared.mean()),
        ordered_tuple_changed_fraction=float(changed.mean()),
        set_changed_fraction=float((np.sort(table, axis=-1) != np.sort(baseline, axis=-1)).any(-1).mean()),
        mean_shared_rows_by_head={str(h): float(shared[..., i].mean()) for i, h in enumerate(HEADS)},
        ordered_tuple_changed_fraction_by_head={str(h): float(changed[..., i].mean()) for i, h in enumerate(HEADS)})
    return result


def summarize(records):
    result = {name: base.aggregate_metrics([row["methods"][name] for row in records]) for name in METHODS}
    original = result["partition_g256"]["mse"]
    for value in result.values():
        value["mse_relative_change_vs_partition_g256"] = value["mse"] / original - 1 if original else None
    return result


def run(args, manifest, audit, report):
    import torch
    import attention.token_merging as token
    import attention.entropy_mass_seed_triton as norm_module
    from attention.partition_mass_seed_triton import partition_mass_seed_rows_triton
    require(torch.cuda.is_available(), "native oracle output evaluation requires CUDA")
    base.benchmark.check_capture_environment(os.environ)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    require(token.partition_mass_seed_rows_triton is partition_mass_seed_rows_triton, "production selector already wrapped")
    real_norm = norm_module.global_probe_log_norm_triton
    require(real_norm.__module__ == norm_module.__name__ and real_norm.__name__ == "global_probe_log_norm_triton",
            "global normalizer was already wrapped before the diagnostic")
    report["execution"] = {"device": torch.cuda.get_device_name(device), "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "query_chunk": QUERY_CHUNK, "key_chunk": KEY_CHUNK,
        "precision": "FP32 non-TF32 oracle dots; FP64 LSE; native injected norms FP32 base2"}
    captures = [item for item in manifest["captures"] if item["step_index"] == 0]
    require([v["prompt_index"] for v in captures] == [0, 1, 2]
            and all(v["call_index"] == 33 for v in captures), "need exactly three step0 L33 captures")
    with torch.inference_mode():
        for item in captures:
            filename = item["capture_file"]
            payload = base.load_capture(args.capture_root / filename, item, audit["capture_files"][filename])
            audit["tensor_payloads_validated"] += 1
            q, k, v, reference = [payload[name].to(device) for name in base.TENSORS]
            del payload
            expected_q = q.reshape(1, FRAMES, ROWS, COLS, 4, DIM)[:, :1].contiguous()
            expected_k = k.reshape(1, FRAMES, ROWS, COLS, 4, DIM).contiguous()
            baseline_output = unpatched_attention(token, q, k, v)
            row = {"metadata": item, "input_file": audit["capture_files"][filename], "methods": {
                "partition_g256": base.output_metrics(baseline_output, reference)},
                "normalizer_calls": {}, "selector_kwargs": {}, "seed_changes": {}}
            calls = []
            with patch.object(norm_module, "global_probe_log_norm_triton", normalizer_adapter(
                    real_norm, expected_q, expected_k, calls)):
                observed, baseline_table, kwargs = base.observed_attention(token, q, k, v, "partition_c16")
            require(len(calls) == 1 and torch.equal(baseline_output.view(torch.uint8), observed.view(torch.uint8)),
                    "observation-only baseline is not bitwise equal to unpatched production")
            row["unpatched_baseline_observation_bitwise_equal"] = True
            del observed, baseline_output
            sampled_norm = calls[0].pop("value")
            row["normalizer_calls"]["partition_g256"] = calls[0]
            row["selector_kwargs"]["partition_g256"] = kwargs
            tables = {"partition_g256": baseline_table}
            raw, peaks = stream_normalizers(expected_q.reshape(3600, 4, DIM).permute(1, 0, 2),
                                            expected_k.reshape(N, 4, DIM).permute(1, 0, 2))
            row["true_global_peak_by_head"] = {}
            for slot, head in enumerate(HEADS):
                ids, counts = torch.unique(peaks["argmax_key_index"][slot], return_counts=True)
                order = counts.argsort(descending=True, stable=True)[:10]
                row["true_global_peak_by_head"][str(head)] = {
                    "top_argmax_keys": [{"key_index": int(ids[i]), "query_count": int(counts[i])} for i in order],
                    "distinct_argmax_keys": len(ids), "mean_peak_mass": float(peaks["peak_mass_fp64"][slot].mean()),
                    "argmax_in_g256_fraction": float(peaks["argmax_in_g256"][slot].double().mean()),
                    "argmax_in_phase0_fraction": float(peaks["argmax_in_phase0"][slot].double().mean())}
            norms = {"partition_g256": sampled_norm,
                "partition_oracle_allkeys": native_layout(raw[..., 0], N, device),
                "partition_oracle_phase0": native_layout(raw[..., 1], PHASE_N, device)}
            for name in METHODS[1:]:
                calls = []
                with patch.object(norm_module, "global_probe_log_norm_triton", normalizer_adapter(
                        real_norm, expected_q, expected_k, calls, replacement=norms[name])):
                    output, tables[name], options = base.observed_attention(token, q, k, v, "partition_c16")
                require(len(calls) == 1 and not calls[0]["original_g256_called"], "oracle failed to bypass G256")
                require(options == kwargs, "oracle changed selector launch/algorithm arguments")
                calls[0].pop("value")
                row["normalizer_calls"][name], row["selector_kwargs"][name] = calls[0], options
                row["methods"][name] = base.output_metrics(output, reference)
                del output
            for name in METHODS:
                row["seed_changes"][name] = seed_changes(tables[name], baseline_table)
            cache_path = args.output.parent / f"prompt{item['prompt_index']:03d}_exact_normalizers.pt"
            cache = {"metadata": {"capture_file": filename, "capture_sha256": audit["capture_files"][filename]["sha256"],
                    "prompt_index": item["prompt_index"], "step_index": 0, "router_frame": 0, "heads": list(HEADS),
                    "normalizer_layout": list(NORM_SHAPE), "raw_natural_layout": [4, 3600, 2],
                    "raw_natural_columns": ["allkeys", "phase0"], "normalizer_convention": report["normalizer_convention"],
                    "evaluation_source_sha256": report["evaluation_provenance"]["source_sha256"], "eligible_candidate": False},
                "raw_log_partitions_natural_fp64": raw,
                "true_global_peaks": peaks,
                "normalizers_log2_g256_scale": {name: value.detach().cpu().contiguous() for name, value in norms.items()},
                "seed_tables": {name: torch.from_numpy(table.copy()) for name, table in tables.items()}}
            with cache_path.open("xb") as handle:
                torch.save(cache, handle)
            row["normalizer_and_seed_cache"] = base.fingerprint(cache_path)
            report["records"].append(row)
            report["summary"] = summarize(report["records"])
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"completed prompt {item['prompt_index']}: three fixed normalizer diagnostics", flush=True)
            del q, k, v, reference, expected_q, expected_k, raw, norms, cache, tables
    require(len(report["records"]) == 3 and audit["tensor_payloads_validated"] == 3, "incomplete oracle run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True, help="Validated capture run's captures child directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite an existing diagnostic")
    require(not any((args.output.parent / f"prompt{p:03d}_exact_normalizers.pt").exists() for p in range(3)),
            "refusing to overwrite existing portable normalizers")
    report = {"complete": False, "valid": False, "records": [], "methods": list(METHODS), "mode": MODE,
        "eligible_candidate": False, "scope": "OFF-PATH L33 step0 normalizer-only oracle; three prompts, original four heads",
        "reference": "Saved same-call original full40 Flash SDPA output slice; no reference recomputation",
        "normalizer_convention": {"partition_g256": "Original native log2 Z_G256",
            "partition_oracle_allkeys": "log2 Z_all75600 - log2(75600/256)",
            "partition_oracle_phase0": "log2 Z_phase15120 - log2(15120/256)"},
        "remaining_mass_semantics": {"partition_oracle_allkeys": "Candidate weight proportional to Z_localP16 / Z_all, NOT exact full80 block mass",
            "partition_oracle_phase0": "Candidate weight proportional to Z_localP16 / Z_phase0; within-phase G256 subsampling removed only"},
        "unchanged": "RF0, C16/g16, P16 columns0,5,...75, anchor+3 partition-deficit steps, SQ1, merge, full route, gs4/reuse",
        "mean_attention_ms": None, "speedup_vs_sdpa": None,
        "limitations": ["Dense normalizers are external oracles, not deployable GCA candidates or speed evidence.",
            "Exact means all indicated keys summed with specified floating-point arithmetic, not real-arithmetic exact.",
            "All local P16 sampling error remains; normalizer improvement alone need not improve output MSE.",
            "Four saved heads at L33/step0 only; no video PSNR or pure full-model result.",
            "External launcher alone is responsible for clean single-GPU evaluation purity."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle)
    try:
        manifest, audit = base.validate_bundle(args.capture_root)
        report["capture_validation"] = audit
        report["evaluation_provenance"] = base.benchmark.provenance()
        for name in ("analysis/diagnose_rf0_exact_normalizer.py", "analysis/analyze_rf0_layer33_captures.py",
                     "analysis/verify_partition_rf0_production.py"):
            report["evaluation_provenance"]["source_sha256"][name] = base.fingerprint(ROOT / name)["sha256"]
        run(args, manifest, audit, report)
        report.update(valid=True, complete=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("valid", "complete", "error")}))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
