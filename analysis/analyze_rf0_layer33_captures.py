"""Evaluate twelve audited L33 captures with the four existing RF0 methods.

No model is advanced, no method is fitted, and no medoid selector is used.
The reference is the saved same-call full-40-head dense output slice, never
a recomputed four-head dense result. Preparation/tests are CPU-only; actual
operator evaluation requires an externally approved, exclusively clean GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "analysis"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import benchmark_rf0_seed_attention as benchmark
import capture_rf0_layer33_diagnostic as producer
from utils.prompt_contract import build_prompt_manifest, load_nonempty_prompts, validate_canonical_prompt_contract
from verify_partition_rf0_production import require_production_options

MODES = dict(benchmark.MODES)
CONFIG = "configs/dense_wan_14b_720p_fewstep.yaml"
TENSORS = ("q", "k", "v", "dense_output")
SEED_SHAPE = (1, 1, 21, 45, 80, 4, 4)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_metadata(item, row, manifest):
    for key, value in row.items():
        require(item.get(key) == value, f"capture metadata {key} differs from ordered record")
    expected = {"heads": list(producer.HEADS), "shape": list(producer.SAVED_SHAPE),
        "dtype": "torch.bfloat16", "sample_seed": row["prompt_index"], "checkpoint_weights": "raw_state_dict",
        "prompt": manifest["prompt_manifest"]["prompts"][row["prompt_index"]],
        "checkpoint_identity": manifest["checkpoint_identity"], "config_sha256": manifest["config_sha256"],
        "capture_source_sha256": manifest["source_sha256"]["analysis/capture_rf0_layer33_diagnostic.py"],
        "frozen_stream_sha256": producer.FROZEN_STREAM_SHA256, "scale": 128 ** -0.5,
        "qk_position": "after Q/K RMSNorm and RoPE; V after projection",
        "reference": "Same call's original 40-head forced Flash SDPA output, sliced to saved heads"}
    for key, value in expected.items():
        require(item.get(key) == value, f"capture metadata {key} mismatch")


def validate_bundle(capture_root, *, source_root=ROOT):
    """Strict small-manifest checks and streaming fingerprints; no Torch/GPU."""
    root, source_root = Path(capture_root).resolve(), Path(source_root)
    require((root / "exit_code").read_text().strip() == "0", "capture exit_code is not zero")
    require(root.name == "captures", "capture-root must be the wrapper's captures child directory")
    wrapper_root = root.parent
    for filename in ("exit_code", "worker_exit_code"):
        require((wrapper_root / filename).read_text().strip() == "0", f"wrapper {filename} is not zero")
    require((wrapper_root / "purity.log").stat().st_size == 0, "capture purity log is not empty")
    manifest = read_json(root / "capture_manifest.json")
    require(manifest.get("complete") is True, "capture manifest is incomplete")
    for key, value in {"total_self_attention_calls": 480, "expected_total_calls": 480,
                       "total_captures": 12, "expected_captures": 12,
                       "planned_raw_tensor_bytes": producer.RAW_CAPTURE_BYTES}.items():
        require(type(manifest.get(key)) is int and manifest[key] == value, f"manifest {key} mismatch")
    require(manifest.get("heads") == list(producer.HEADS), "wrong saved head order")
    require(manifest.get("capture_calls_per_prompt") == list(producer.TARGET_CALLS), "wrong capture calls")
    require(manifest.get("actual_warped_timesteps") == list(producer.ACTUAL_TIMESTEPS), "wrong actual steps")
    require(manifest.get("checkpoint_weights") == "raw_state_dict", "wrong loaded checkpoint branch")
    require(manifest.get("actual_forced_sdpa_backend") == "flash", "capture did not force Flash SDPA")
    require(manifest.get("actual_use_torch_rmsnorm") is False, "capture did not use native RMSNorm")
    identity = manifest["checkpoint_identity"]
    require(isinstance(identity.get("path"), str) and bool(identity["path"]), "checkpoint path missing")
    require(type(identity.get("bytes")) is int and identity["bytes"] > 0, "checkpoint size missing")
    require(type(identity.get("mtime_ns")) is int and identity["mtime_ns"] > 0, "checkpoint mtime missing")
    names = producer.validate_complete(manifest["records"])
    require(sorted(p.name for p in root.glob("*.pt")) == sorted(names), "capture filename set is not exactly twelve")
    prompts_path = source_root / "prompts/MovieGenVideoBench_extended.txt"
    prompts = load_nonempty_prompts(prompts_path)
    validate_canonical_prompt_contract(prompts_path, prompts)
    canonical = build_prompt_manifest(prompts_path, prompts, selected_count=3, seed=0, num_samples=1,
                                      vbench_filenames=False, allow_noncanonical=False)
    recorded_prompt = manifest["prompt_manifest"]
    # Portable copied artifacts can have a different absolute prompt path.
    require({k: v for k, v in recorded_prompt.items() if k != "prompt_path"}
            == {k: v for k, v in canonical.items() if k != "prompt_path"}, "noncanonical prompt identities")
    for filename in ("prompt_manifest.json", "prompt_preflight_manifest.json"):
        require(read_json(root / filename) == recorded_prompt, f"{filename} differs from capture manifest")
    sources = manifest["source_sha256"]
    require(set(sources) == set(producer.SOURCES), "capture source inventory differs from the reviewed producer")
    for name in producer.SOURCES:
        require(fingerprint(source_root / name)["sha256"] == sources[name], f"source hash mismatch: {name}")
    require(sources["analysis/screen_native_seed_trajectory.py"] == producer.FROZEN_STREAM_SHA256,
            "frozen stream source mismatch")
    require(PurePosixPath(str(manifest["settings"]["config"]).replace("\\", "/")).name == Path(CONFIG).name,
            "unexpected capture configuration")
    require(fingerprint(source_root / CONFIG)["sha256"] == manifest["config_sha256"], "config hash mismatch")
    captures = manifest["captures"]
    require(len(captures) == 12 and [v["capture_file"] for v in captures] == names, "capture metadata list mismatch")
    selected_rows = [row for row in manifest["records"] if row["capture_file"]]
    files = {}
    for item, row in zip(captures, selected_rows):
        validate_metadata(item, row, manifest)
        path = root / row["capture_file"]
        require(path.is_file() and path.resolve().is_relative_to(root), "capture path escapes its input root")
        require(type(item.get("bytes")) is int and path.stat().st_size == item["bytes"] > 0, "capture size mismatch")
        files[row["capture_file"]] = fingerprint(path)
        if "sha256" in item:
            require(files[row["capture_file"]]["sha256"] == item["sha256"], "producer capture SHA256 mismatch")
    inputs = {name: fingerprint(root / name) for name in
              ("capture_manifest.json", "prompt_manifest.json", "prompt_preflight_manifest.json", "exit_code")}
    inputs.update({f"wrapper_{name}": fingerprint(wrapper_root / name)
                   for name in ("exit_code", "worker_exit_code", "purity.log")})
    return manifest, {"valid": True, "input_root": str(root), "inputs": inputs, "capture_files": files,
        "tensor_payloads_validated": 0, "source_matches_current_reviewed_files": True,
        "hash_scope": "Computed fingerprints of the bytes read now; original producer has no per-file SHA256 receipt",
        "checkpoint_scope": "Recorded path/size/mtime and actual raw branch; not a checkpoint-byte hash proof"}


def load_capture(path, item, file_identity):
    import torch
    require(fingerprint(path) == file_identity, "capture bytes changed after bundle validation")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(set(payload) == set(TENSORS) | {"metadata"}, "unexpected capture payload keys")
    expected = {key: value for key, value in item.items() if key not in ("bytes", "sha256")}
    require(payload["metadata"] == expected, "PT metadata differs from capture_manifest")
    for name in TENSORS:
        value = payload[name]
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == producer.SAVED_SHAPE,
                f"{name}: wrong tensor shape/type")
        require(value.dtype == torch.bfloat16 and value.device.type == "cpu" and value.is_contiguous(),
                f"{name}: require unchanged contiguous CPU BF16")
        require(bool(torch.isfinite(value).all()), f"{name}: nonfinite tensor")
    return payload


def output_metrics(output, reference):
    import torch
    require(all(tuple(value.shape) == producer.SAVED_SHAPE and value.dtype == torch.bfloat16
                and value.is_contiguous() for value in (output, reference)), "wrong complete output/reference shape/dtype/layout")
    require(bool(torch.isfinite(output).all()) and bool(torch.isfinite(reference).all()), "nonfinite output/reference")
    error, energy = (output.float() - reference.float()).square(), reference.float().square()
    total, reference_total = float(error.sum(dtype=torch.float64)), float(energy.sum(dtype=torch.float64))
    frame = error.reshape(1, 21, 3600, 4, 128).sum((0, 2, 3, 4), dtype=torch.float64).tolist()
    head = error.sum((0, 1, 3), dtype=torch.float64).tolist()
    return {"numel": error.numel(), "squared_error_sum": total, "reference_squared_sum": reference_total,
            "mse": total / error.numel(), "relative_l2": math.sqrt(total / reference_total) if reference_total else None,
            "frame_squared_error_sum": frame, "head_squared_error_sum": head,
            "frame_mse": [value / (error.numel() // 21) for value in frame],
            "head_mse": [value / (error.numel() // 4) for value in head], "finite": True}


def aggregate_metrics(items):
    require(bool(items), "cannot aggregate empty metrics")
    require(all(type(item["numel"]) is int and item["numel"] > 0
                and all(math.isfinite(item[key]) and item[key] >= 0
                        for key in ("squared_error_sum", "reference_squared_sum")) for item in items),
            "invalid metric count or energy")
    count = sum(item["numel"] for item in items)
    sse = math.fsum(item["squared_error_sum"] for item in items)
    energy = math.fsum(item["reference_squared_sum"] for item in items)
    result = {"calls": len(items), "numel": count, "squared_error_sum": sse, "reference_squared_sum": energy,
              "mse": sse / count, "relative_l2": math.sqrt(sse / energy) if energy else None}
    for axis in ("frame", "head"):
        key = f"{axis}_squared_error_sum"
        size = len(items[0][key])
        require(size > 0 and all(len(item[key]) == size and item["numel"] % size == 0 for item in items),
                f"incompatible {axis} marginals")
        require(all(all(math.isfinite(value) and value >= 0 for value in item[key])
                    and math.isclose(math.fsum(item[key]), item["squared_error_sum"], rel_tol=1e-10, abs_tol=1e-12)
                    for item in items), f"{axis} marginal SSE does not match total")
        sums = [math.fsum(item[key][i] for item in items) for i in range(size)]
        result[key] = sums
        result[f"{axis}_numel"] = count // size
        result[f"{axis}_mse"] = [value / (count // size) for value in sums]
    return result


def summarize(records):
    groups = {"overall": records}
    groups.update({f"step_{s}": [row for row in records if row["metadata"]["step_index"] == s] for s in range(4)})
    groups.update({f"prompt_{p}": [row for row in records if row["metadata"]["prompt_index"] == p] for p in range(3)})
    result = {}
    for group, rows in groups.items():
        if not rows:
            continue
        values = {name: aggregate_metrics([row["methods"][name] for row in rows]) for name in MODES}
        base = values["rf0"]["mse"]
        for item in values.values():
            item["mse_relative_change_vs_rf0"] = item["mse"] / base - 1 if base else None
        result[group] = values
    return result


def seed_comparison(table, reference=None):
    import numpy as np
    require(table.shape == SEED_SHAPE and table.dtype == np.uint8, "wrong native seed table layout/dtype")
    require(bool((table < 45).all()) and bool((np.diff(np.sort(table.astype(np.int16), axis=-1), axis=-1) > 0).all()),
            "seeds must be four distinct real rows")
    anchors = np.arange(45).reshape(1, 1, 1, 45, 1, 1)
    require(bool((table[..., 0] == anchors).all()), "existing anchored method lost its first seed")
    result = {"valid": True, "sha256_uint8": hashlib.sha256(table.tobytes()).hexdigest(),
              "groups": math.prod(table.shape[:-1]), "anchor_required": True}
    if reference is not None:
        require(reference.shape == table.shape and reference.dtype == table.dtype, "incompatible reference seed table")
        shared = (table[..., :, None] == reference[..., None, :]).any(-1).sum(-1)[0, 0]
        result.update(mean_shared_rows_vs_hellinger=float(shared.mean()),
            mean_shared_rows_by_original_head={str(h): float(shared[..., i].mean()) for i, h in enumerate(producer.HEADS)},
            mean_shared_rows_by_key_frame=shared.mean((1, 2, 3)).tolist(),
            exact_ordered_tuple_fraction=float((table == reference).all(-1).mean()),
            same_set_fraction=float((np.sort(table, axis=-1) == np.sort(reference, axis=-1)).all(-1).mean()))
    return result


def observed_attention(token, q, k, v, name):
    import torch
    symbol = {"rf0": "hellinger_span_seed_rows_triton", "mass_c32": "entropy_mass_seed_rows_triton",
              "mass_c16": "entropy_mass_seed_rows_triton", "partition_c16": "partition_mass_seed_rows_triton"}[name]
    real, tables, received = getattr(token, symbol), [], []
    expected_q = q.reshape(1, 21, 45, 80, 4, 128)[:, :1]
    expected_k = k.reshape(1, 21, 45, 80, 4, 128)
    def observe(*inputs, **options):
        require(not tables and len(inputs) == 3 and inputs[2] == 128 ** -0.5, "unexpected selector call count/scale")
        require(options.get("output_dtype") == torch.uint8, "selector must return original uint8 rows")
        require(torch.equal(inputs[0], expected_q) and torch.equal(inputs[1], expected_k), "selector must receive RF0 and unchanged all-frame K")
        if name == "partition_c16":
            require_production_options(options)  # Observe production; never repair kwargs.
        table = real(*inputs, **options)
        tables.append(table.detach().cpu().numpy().copy())
        received.append({key: str(value) if key == "output_dtype" else value for key, value in options.items()})
        return table
    with patch.object(token, symbol, observe):
        output = token._hard_c4_attention(q, k, v, h=45, w=80, h_reduce=1, w_reduce=1, sm_scale=128 ** -0.5,
            query_outer_chunk=21, head_chunk=8, hard_route_backend="hard_c4", seed_mode=MODES[name],
            exact_intra_frame_attention=False, exact_neighbor_window_columns=0, exact_query_rows=0,
            first_frame_full_attention=False, full_attention_anchor_time_count=0,
            full_attention_anchor_row_count=0, full_attention_anchor_scale=1.0).contiguous()
    require(len(tables) == 1, "expected exactly one real selector invocation")
    return output, tables[0], received[0]


def probe_statistics(logits, log_mass, positions):
    """Independent finite NumPy statistics, not native selector register values.

    Mass is normalized only across the existing candidate bank. Selected mass
    is candidate-weight share, NOT attention coverage or a quality objective.
    """
    import numpy as np
    logits, log_mass = np.asarray(logits, dtype=np.float64), np.asarray(log_mass, dtype=np.float64)
    require(logits.ndim == 3 and log_mass.shape == logits.shape[:2]
            and np.isfinite(logits).all() and np.isfinite(log_mass).all(), "invalid finite probe arrays")
    centered = logits - logits.max(-1, keepdims=True)
    logp = centered - np.log(np.exp(centered).sum(-1, keepdims=True))
    log_weight = log_mass - log_mass.max(-1, keepdims=True)
    log_weight -= np.log(np.exp(log_weight).sum(-1, keepdims=True))
    weight = np.exp(log_weight)
    entropy = -(np.exp(logp) * logp).sum(-1) / math.log(logits.shape[-1])
    values = {"max_normalized_candidate_weight": weight.max(-1),
              "candidate_weight_effective_count": 1 / np.square(weight).sum(-1),
              "normalized_candidate_weight_entropy": -(weight * log_weight).sum(-1) / math.log(logits.shape[-2]),
              "mean_normalized_probe_entropy": entropy.mean(-1),
              "weight_averaged_normalized_probe_entropy": (weight * entropy).sum(-1)}
    for name, index in positions.items():
        index = np.asarray(index)
        require(index.shape == (len(logits), 4) and np.issubdtype(index.dtype, np.integer)
                and (index >= 0).all() and (index < logits.shape[-2]).all(), "invalid selected candidate positions")
        values[f"{name}_selected_candidate_weight_share"] = np.take_along_axis(weight, index, -1).sum(-1)
    return values


def head30_probe_diagnostics(q, k, tables):
    """Bounded step0-only C16/G16/P16/G256 distributions, all 75,600 blocks."""
    import numpy as np
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    from attention.entropy_mass_seed_triton import global_probe_log_norm_triton
    query = q.reshape(1, 21, 45, 80, 4, 128)[:, :1, :, :, 3:4].contiguous()
    keys = k.reshape(1, 21, 45, 80, 4, 128)[..., 3:4, :].contiguous()
    norm = global_probe_log_norm_triton(query, keys, scale=128 ** -0.5, global_probe_count=256)
    bank, _ = multiscale_candidate_table(45, 16, q.device, geometry_count=16)
    bank, samples = bank.long(), {}
    selected = {name: tables[name][0, 0, ..., 3, :].reshape(-1, 4) for name in ("mass_c16", "partition_c16")}
    prior_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        probes = torch.arange(16, device=q.device) * 5
        for start in range(0, 75600, 256):
            index = torch.arange(start, min(start + 256, 75600), device=q.device)
            column, row, frame = index % 80, (index // 80) % 45, index // 3600
            rows = bank[row]
            q_part = query[0, 0, rows, column[:, None], 0].float()
            k_part = keys[0, frame[:, None], row[:, None], probes[None, :], 0].float()
            logits = torch.bmm(q_part, k_part.transpose(1, 2)) * (128 ** -0.5)
            log_mass = torch.logsumexp(logits, -1) - norm[0, 0, rows, column[:, None], 0] * math.log(2)
            row_bank, positions = rows.cpu().numpy(), {}
            for name, table in selected.items():
                matches = row_bank[:, :, None] == table[start:start + len(index), None, :]
                require(bool((matches.sum(1) == 1).all()), "actual C16 seed is not in the existing candidate bank")
                positions[name] = matches.argmax(1)
            values = probe_statistics(logits.cpu().numpy(), log_mass.cpu().numpy(), positions)
            for name, values_part in values.items():
                samples.setdefault(name, []).append(values_part)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_tf32
    distributions = {}
    for name, parts in samples.items():
        values = np.concatenate(parts)
        require(len(values) == 75600 and bool(np.isfinite(values).all()), "incomplete probe diagnostic")
        distributions[name] = {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
            "p05_p50_p95": np.quantile(values, [0.05, 0.5, 0.95]).tolist(),
            "mean_by_key_frame": values.reshape(21, 45, 80).mean((1, 2)).tolist()}
    return {"original_head": 30, "router_frame": 0, "key_frames": list(range(21)), "blocks": 75600,
        "candidate_count": 16, "candidate_geometry_count": 16, "probe_count": 16, "global_probe_count": 256,
        "scope": "Independent FP32 non-TF32 local logits from actual BF16 QK; original native G256 normalizer; NumPy FP64 statistics",
        "limitations": "Not native register dumps, exact global masses, full-key objectives, or attention coverage. Actual tables remain untouched.",
        "distributions": distributions}


def run(args, manifest, audit, report):
    import torch
    import attention.token_merging as token
    require(torch.cuda.is_available(), "native evaluation requires CUDA; use --validate-only for CPU checks")
    benchmark.check_capture_environment(os.environ)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    report.update(device=torch.cuda.get_device_name(device), torch_version=torch.__version__, cuda_version=torch.version.cuda)
    with torch.inference_mode():
        for item in manifest["captures"]:
            filename = item["capture_file"]
            saved = load_capture(args.capture_root / filename, item, audit["capture_files"][filename])
            audit["tensor_payloads_validated"] += 1
            q, k, v, reference = [saved[name].to(device) for name in TENSORS]
            row = {"metadata": saved["metadata"], "input_file": audit["capture_files"][filename],
                   "methods": {}, "seed_diagnostics": {}, "selector_kwargs": {}}
            del saved
            tables = {}
            for name in MODES:
                output, tables[name], row["selector_kwargs"][name] = observed_attention(token, q, k, v, name)
                row["methods"][name] = output_metrics(output, reference)
                row["seed_diagnostics"][name] = seed_comparison(tables[name], tables.get("rf0"))
                del output
            if args.probe_statistics and item["step_index"] == 0:
                row["head30_probe_diagnostics"] = head30_probe_diagnostics(q, k, tables)
            report["records"].append(row)
            report["summary"] = summarize(report["records"])
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"completed {filename}: {len(report['records'])}/12", flush=True)
            del q, k, v, reference, tables
    require(len(report["records"]) == 12 and audit["tensor_payloads_validated"] == 12, "incomplete L33 evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true", help="CPU-only bundle and all tensor-payload checks; no native evaluation")
    parser.add_argument("--probe-statistics", action="store_true", help="Optional extra step0/head30 C16 probe distributions; default off")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing analysis report")
    report = {"complete": False, "valid": False, "records": [], "methods": MODES, "heads": list(producer.HEADS),
        "router_frame": 0, "reference": "Stored same-call original full-40-head Flash SDPA output sliced to four heads",
        "aggregation": "sum SSE / sum numel; FP32 squared differences, FP64 reductions; no mean of relative changes",
        "mean_attention_ms": None, "speedup_vs_sdpa": None,
        "timing_scope": "Not measured: four sampled heads and diagnostic observation, no speed headline",
        "scope": "Twelve L33 captures only; identical pure RF0 operator per method, all saved heads/frames; no video PSNR",
        "limitations": ["Candidates do not advance a model; four heads are not all 40 heads.",
                        "Seed agreement is not seed optimality; no unvalidated selector is included.",
                        "Capture purity is checked; evaluation GPU exclusivity requires the external launcher."],
        "validation_only": args.validate_only, "extra_probe_statistics_requested": args.probe_statistics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle)
    try:
        manifest, audit = validate_bundle(args.capture_root)
        report["capture_validation"] = audit
        report["evaluation_provenance"] = benchmark.provenance()
        for name in ("analysis/analyze_rf0_layer33_captures.py", "analysis/verify_partition_rf0_production.py"):
            report["evaluation_provenance"]["source_sha256"][name] = fingerprint(ROOT / name)["sha256"]
        if args.validate_only:
            for item in manifest["captures"]:
                payload = load_capture(args.capture_root / item["capture_file"], item, audit["capture_files"][item["capture_file"]])
                audit["tensor_payloads_validated"] += 1
                del payload
        else:
            run(args, manifest, audit, report)
        report.update(valid=True, complete=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"valid": report["valid"], "complete": report["complete"], "error": report.get("error")}))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
