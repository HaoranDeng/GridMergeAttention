"""Matched full40 attention timing on 24 original BF16 dense captures.

Diagnostic sample: 3 prompts x 4 steps x layers 0/33, NOT all480 call inputs,
an all-layer trajectory, or video quality. Each candidate is the same pure
operator eligible for every layer; no sampled-layer exemption is introduced.
No launch until the specific bb capture has a valid completed parent receipt.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
CAPTURE_NAME = "full40_bf16_dense_capture_20260905bb"
AUDIT_SHA = "5d75c179120fa9e2f25b8f4a9b860578da3f0489eb56aea412a0925d7f8714a5"
HF10_MODE = "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2"
METHODS = ("sdpa_flash", "historical_hf_rf10_fp8score", "hellinger_rf0",
           "fisher_anchor_rf0", "mean_fisher_rf0", "barycentric_medoid_rf0")
HISTORICAL_HF = "historical_hf_rf10_fp8score"
STABLE_METHODS = ("hellinger_stable_sq1_rf0", "fisher_anchor_stable_sq1_rf0")
STABLE_SQ1_SUITE = ("sdpa_flash", HISTORICAL_HF, "hellinger_rf0", "fisher_anchor_rf0", *STABLE_METHODS)
WARMUP, REPEATS, CALLS_PER_REPEAT = 2, 4, 5
SHAPE = (1, 75600, 40, 128)
MIN_CAPTURE_BYTES = 4 * math.prod(SHAPE) * 2
GRID = (1, 21, 45, 80, 40, 128)
SEED_SHAPE = (1, 1, 21, 45, 80, 40, 4)
TARGET_CALLS = (0, 33, 40, 73, 80, 113, 120, 153)
TIMESTEPS = (1000.0, 937.5, 833.3333129882812, 625.0)
MERGE = "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks"
SCOPE = "24 full40 original-BF16 call inputs: 3 prompts x 4 steps x layers0/33; not all480 inputs, video quality, or end-to-end runtime"


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def expected_records():
    return [{"prompt_index": p, "call_index": c, "layer_index": c % 40,
             "step_index": c // 40, "actual_timestep": TIMESTEPS[c // 40],
             "capture_file": f"prompt{p:03d}_call{c:03d}_layer{c%40}.pt" if c in TARGET_CALLS else None}
            for p in range(3) for c in range(160)]


def check_capture_stat(root, receipt):
    path = (Path(root) / receipt["capture_file"]).resolve(strict=True)
    require(path.parent == Path(root).resolve(), "capture path escapes input directory")
    info = path.stat(); original = receipt["file"]
    require(path == Path(original["path"]).resolve(), "capture identity path differs")
    require((info.st_size, info.st_mtime_ns) == (original["bytes"], original["mtime_ns"]),
            "capture changed after producer/parent SHA256 verification")
    return path


def validate_capture_run(run):
    """Reuse completed producer SHA receipts, checking current size/mtime.

The capture parent already rehashed every saved file after completion. This
avoids rehashing 74GB at each repeat; metadata checks run before/after load.
It cannot detect an adversary restoring both original size and mtime.
"""
    run = Path(run).resolve()
    require(run.name == CAPTURE_NAME, "require the original full40 bb capture, not substituted/repeated heads")
    completion_path, root = run / "completion.json", run / "captures"
    completion = json.loads(completion_path.read_text())
    require(completion.get("valid") is True and completion.get("failure") is None
            and completion.get("foreign") == [] and completion.get("worker_exit_code") == 0,
            "bb parent has not completed validly; do not queue a benchmark")
    manifest_path = root / "capture_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verified = completion.get("verified_artifacts") or {}
    require(verified.get("manifest_sha256") == sha(manifest_path), "capture manifest SHA changed")
    require(manifest.get("complete") is True and manifest.get("total_self_attention_calls") == 480
            and manifest.get("total_captures") == 24 and manifest.get("records") == expected_records()
            and (root / "exit_code").read_text().strip() == "0", "require all480 dense calls and24 captures")
    require(manifest.get("checkpoint_weights") == "raw_state_dict"
            and manifest.get("actual_forced_sdpa_backend") == "flash"
            and manifest.get("actual_use_torch_rmsnorm") is False
            and manifest.get("heads") == list(range(40)), "wrong dense capture provenance")
    captures = manifest.get("captures", [])
    expected = [r for r in expected_records() if r["capture_file"]]
    require(len(captures) == 24, "incomplete capture list")
    require([c["capture_file"] for c in captures] == [c["capture_file"] for c in expected], "capture order differs")
    require(sorted(p.name for p in root.glob("*.pt")) == sorted(c["capture_file"] for c in expected), "extra/missing input files")
    receipts = [{k: row[k] for k in ("capture_file", "bytes", "sha256")} for row in captures]
    require(verified.get("captures") == receipts, "capture parent SHA receipts differ")
    for row, identity in zip(captures, expected):
        require(all(row.get(k) == v for k, v in identity.items()), "capture call identity differs")
        require(row.get("shape") == list(SHAPE) and row.get("dtype") == "torch.bfloat16"
                and row.get("heads") == list(range(40)) and row.get("all_tensors_finite") is True,
                "require original full40 BF16 metadata, not head replication/conversion")
        require(row["bytes"] == row["file"]["bytes"] and row["sha256"] == row["file"]["sha256"]
                and len(row["sha256"]) == 64 and row["bytes"] >= MIN_CAPTURE_BYTES,
                "conflicting or undersized input SHA receipts")
        check_capture_stat(root, row)
    return {"capture_run": str(run), "capture_root": str(root), "captures": captures,
            "completion_sha256": sha(completion_path), "manifest_sha256": sha(manifest_path),
            "checksum_policy": "Reuse producer+completed-parent SHA256 receipts; verify exact size/mtime before and after every load"}


def historical_mode(audit_path):
    require(sha(audit_path) == AUDIT_SHA, "historical HF reference audit is not the verified run-f audit")
    section = json.loads(Path(audit_path).read_text())["hellinger_rf10"]
    require(section["policy"]["router_frame"] == 10 and section["policy"]["candidate_count"] == 32
            and section["policy"]["pure_all_layer_config"] is True, "historical HF policy changed")
    config = section["config_text"]
    require(hashlib.sha256(config.encode()).hexdigest() == section["config"]["sha256"], "historical config text differs")
    modes = [line.split(":", 1)[1].strip() for line in config.splitlines() if line.strip().startswith("seed_mode:")]
    require(modes == [HF10_MODE], "historical mode differs")
    return modes[0]


def source_hashes(stable_sq1_suite=False):
    paths = list((ROOT / "attention").rglob("*.py"))
    paths += [ROOT / "analysis" / name for name in (
        "benchmark_barycentric_full40.py", "quality_first_mode_contract.py", "mean_anchor_video_adapter.py",
        "zero_anchor_video_adapter.py", "launch_quality_first_videos.py", "run_quality_first_videos.py")]
    if stable_sq1_suite:
        paths += [ROOT / "analysis" / name for name in ("stable_sq1_bulk_kernel.py", "stable_sq1_video_adapter.py")]
    return {str(p.resolve()): sha(p) for p in sorted(set(paths))}


def suite_methods(stable_sq1_suite=False):
    require(type(stable_sq1_suite) is bool, "suite switch must be Boolean")
    return STABLE_SQ1_SUITE if stable_sq1_suite else METHODS


def expected_stable_receipt(output):
    """CPU-only source prediction; no Triton import or generated-file write."""
    from analysis.stable_sq1_bulk_kernel import generate_stable_sq1_source
    path = (Path(output) / "stable_kernel.py").resolve()
    require(not path.is_relative_to(ROOT), "generated kernel must be outside immutable source")
    source = generate_stable_sq1_source((ROOT / "attention/hard_c4_wan21_14b_720p_triton.py").read_text())
    encoded = source.encode("utf-8")
    return {"path": str(path), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def verify_stable_receipt(receipt, expected):
    require(isinstance(receipt, dict) and all(receipt.get(key) == value for key, value in expected.items()),
            "generated stable receipt differs from CPU source prediction")
    path = Path(expected["path"]).resolve(strict=True)
    require(path.is_file() and path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"],
            "generated stable source changed")
    return receipt


def require_parent():
    require(os.getppid() > 1 and os.environ.get("BARYCENTRIC_BENCH_PARENT") == str(os.getppid()),
            "worker requires its live clean-GPU monitoring parent")


def validate_payload(payload, receipt):
    import torch
    expected_metadata = {k: v for k, v in receipt.items() if k not in ("file", "bytes", "sha256")}
    require(payload.get("metadata") == expected_metadata, "loaded metadata differs from verified capture")
    require(set(payload) == {"q", "k", "v", "dense_output", "metadata"}, "unexpected capture payload")
    for name in ("q", "k", "v", "dense_output"):
        tensor = payload[name]
        require(tuple(tensor.shape) == SHAPE and tensor.dtype == torch.bfloat16
                and tensor.device.type == "cpu" and tensor.is_contiguous(), "not original contiguous CPU full40 BF16 tensors")


def signature(tensor):
    return (tuple(tensor.shape), str(tensor.dtype), str(tensor.device), tensor.data_ptr(),
            tensor.storage_offset(), tuple(tensor.stride()))


def identity(function):
    file = Path(inspect.getsourcefile(function)).resolve()
    require(file.is_relative_to(ROOT), "operator imported outside pinned source root")
    return {"callable": function.__module__ + "." + function.__name__, "source": str(file), "sha256": sha(file)}


def operator_options(mode):
    return dict(sm_scale=128 ** -0.5, query_outer_chunk=21, head_chunk=8, layout="hard_route",
                num_merge_tokens=4, seed_mode=mode, hard_route_backend="hard_c4", q_init=None,
                exact_intra_frame_attention=False, exact_neighbor_window_columns=0, exact_query_rows=0,
                first_frame_full_attention=False, full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0, full_attention_anchor_width_count=0,
                full_attention_anchor_scale=1.0, dense_heads=(), seed_query=None,
                seed_table_cache=None, cache_timestep=None)


class Binding:
    """Install method bindings outside timing; mean boundaries remain inside."""
    def __init__(self, name, qkv, *, stable_kernel=None, stable_receipt=None):
        import attention.token_merging as token
        import attention.core as core
        from analysis.quality_first_mode_contract import VIDEO_MODES, method_contract
        self.name, self.qkv, self.token, self.core = name, qkv, token, core
        self.stable_kernel, self.stable_receipt = stable_kernel, stable_receipt
        self.original_merge = getattr(token, MERGE)
        self.originals = {"complete_operator": identity(core.flash_attention if name == "sdpa_flash" else token.token_merging_attn)}
        self.selector = self.symbol = self.mode = None
        if name != "sdpa_flash":
            historical = name == HISTORICAL_HF
            declaration = None if historical else method_contract(name)
            self.mode = HF10_MODE if historical else VIDEO_MODES[name]
            self.symbol = "hellinger_span_seed_rows_triton" if historical else declaration["selector_symbol"]
            override = None if historical else declaration["selector_override"]
            self.selector = getattr(importlib.import_module(override.rsplit(".", 1)[0]), override.rsplit(".", 1)[1]) if override else getattr(token, self.symbol)
            self.originals.update(selector=identity(self.selector), merge_route=identity(self.original_merge))
            expected = override or "attention.adaptive_seed_triton.hellinger_span_seed_rows_triton"
            require(self.originals["selector"]["callable"] == expected, "loaded selector identity differs from declared method")
        if name in STABLE_METHODS:
            require(stable_kernel is not None and isinstance(stable_receipt, dict), "stable suite requires one archived generated kernel")
            self.originals["stable_merge_kernel"] = stable_receipt

    @contextmanager
    def context(self, *, observe=False):
        mean_adapter = stable_adapter = None
        if self.name == "mean_fisher_rf0":
            from analysis.mean_anchor_video_adapter import MeanSeedMergeAdapter
            mean_adapter = MeanSeedMergeAdapter(self.original_merge, importlib.import_module("attention.hard_c4_wan21_14b_720p_triton"))
            self.originals["mean_adapter"] = identity(MeanSeedMergeAdapter.__call__)
        elif self.name in STABLE_METHODS:
            from analysis.stable_sq1_video_adapter import StableSQ1MergeAdapter
            stable_adapter = StableSQ1MergeAdapter(self.original_merge, importlib.import_module("attention.hard_c4_wan21_14b_720p_triton"), self.stable_kernel)
            self.originals["stable_merge_adapter"] = identity(StableSQ1MergeAdapter.__call__)
        merge_callable = stable_adapter or mean_adapter or self.original_merge
        state = {"calls": 0, "selector_calls": 0, "merge_calls": 0, "metadata": {}, "seed": None}
        grid = tuple(x.reshape(GRID) for x in self.qkv)
        frame = 10 if self.name == HISTORICAL_HF else 0

        def select(*args, **kwargs):
            require(len(args) == 3 and args[2] == 128 ** -0.5, "selector scale changed")
            require(signature(args[0]) == signature(grid[0][:, frame:frame + 1])
                    and signature(args[1]) == signature(grid[1]), "selector not original expected RF query view/full K")
            require(kwargs.get("candidate_count") == 32 and kwargs.get("candidate_geometry_count") == 45
                    and kwargs.get("block_j") == 4, "selector candidate geometry changed")
            if self.symbol == "hellinger_span_seed_rows_triton":
                require(kwargs.get("top_probes") == 0 and kwargs.get("uniform_probes") == 16
                        and kwargs.get("alpha") == .2, "historical/RF0 Hellinger probe rule changed")
            else:
                require(kwargs.get("probe_count") == 16 and kwargs.get("global_probe_count") == 256, "native selector probe contract changed")
            result = self.selector(*args, **kwargs)
            require(tuple(result.shape) == SEED_SHAPE and str(result.dtype) == "torch.uint8", "wrong native seed table")
            state["selector_calls"] += 1; state["seed"] = signature(result)
            state["metadata"].setdefault("selector", {"q": signature(args[0]), "k": signature(args[1]), "seed": signature(result)})
            return result

        def merge(*args, **kwargs):
            require(len(args) == 5 and args[4] == 128 ** -0.5, "merge scale/arity changed")
            require(all(signature(args[i]) == signature(grid[i]) for i in range(3))
                    and signature(args[3]) == state["seed"], "merge changed actual QKV/selected rows")
            require(kwargs.get("head_start") == (state["merge_calls"] % 5) * 8 and kwargs.get("head_count") == 8,
                    "wrong complete40 head chunks")
            required = {"top_r": 1, "selector": "query_column", "seed_rows_native_layout": True,
                        "merge_frame_group": 1, "merge_col_group": 1, "query_frame_group": 21,
                        "query_column_major": True, "split_kv_accum": True,
                        "fp8_score_dot": self.name == HISTORICAL_HF}
            require(all(kwargs.get(k) == v for k, v in required.items()), "SQ1/fullroute/precision substrate differs")
            for key in ("direct_bulk_normalization", "compressed_fp8_kv", "scaled_fp8_kv", "fp8_merge_k_dot",
                        "fp8_merge_v_dot", "split_merge", "materialize_singleton", "mean_centered_routing",
                        "gaussian_partition_routing", "reuse_partition_exp", "packed_singleton_kv"):
                require(kwargs.get(key) is False, f"unexpected merge override {key}")
            state["merge_calls"] += 1
            return merge_callable(*args, **kwargs)

        def invoke():
            if mean_adapter: mean_adapter.begin_attention()
            if self.name == "sdpa_flash":
                require(self.core._FORCED_SDPA_BACKEND == "flash", "SDPA denominator is not forced Flash")
                output = self.core.flash_attention(*self.qkv).contiguous()
            else:
                output = self.token.token_merging_attn(*self.qkv, 1, 1, 1, 45, 80,
                                                     **operator_options(self.mode)).contiguous()
            if mean_adapter: mean_adapter.end_attention()
            state["calls"] += 1
            return output

        with ExitStack() as stack:
            if self.selector:
                stack.enter_context(patch.object(self.token, self.symbol, select if observe else self.selector))
                stack.enter_context(patch.object(self.token, MERGE, merge if observe else merge_callable))
            yield invoke, state
        if observe and self.name != "sdpa_flash":
            require(state["selector_calls"] == state["calls"] and state["merge_calls"] == 5 * state["calls"], "missing/extra selection or head chunks")
        if mean_adapter:
            require(mean_adapter.mean_computations == mean_adapter.completed_attention_calls == state["calls"]
                    and mean_adapter.total_calls == mean_adapter.kernel_calls == 5 * state["calls"], "mean must rebuild exactly once per complete attention")
            state["mean_counts"] = {"means": mean_adapter.mean_computations, "complete_calls": mean_adapter.completed_attention_calls,
                                    "merge_kernel_calls": mean_adapter.kernel_calls}
        if stable_adapter:
            require(stable_adapter.calls == stable_adapter.kernel_calls == 5 * state["calls"], "stable requires five actual kernel launches per complete attention")
            state["stable_counts"] = {"complete_calls": state["calls"], "merge_calls": stable_adapter.calls,
                                      "stable_merge_kernel_calls": stable_adapter.kernel_calls}


def output_error(output, reference):
    import torch
    require(tuple(output.shape) == tuple(reference.shape) == SHAPE and output.dtype == reference.dtype == torch.bfloat16
            and output.is_contiguous(), "wrong complete output or reference")
    values = []
    for head in range(40):
        actual, target = output[0, :, head].float(), reference[0, :, head].float()
        require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(target).all()), "nonfinite output/reference")
        values.append(float((actual - target).square().sum(dtype=torch.float64)))
    count = math.prod(SHAPE)
    return {"numel": count, "squared_error_sum": math.fsum(values), "mse": math.fsum(values) / count,
            "per_head_squared_error_sum": values, "per_head_mse": [x / (count // 40) for x in values]}


def repeat_order(repeat, methods=METHODS):
    require(repeat in range(REPEATS), "unexpected repeat")
    return methods if repeat % 2 == 0 else tuple(reversed(methods))


def summarize(records, methods=METHODS):
    require(len(records) == 24 and len({r["capture_file"] for r in records}) == 24, "need all24 matched captures")
    summary = {}
    for name in methods:
        rows = [record["methods"][name] for record in records]
        timings = [block for row in rows for block in row["repeat_ms"]]
        require(len(timings) == 24 * REPEATS and all(math.isfinite(x) and x > 0 for x in timings), "incomplete/nonfinite timings")
        count = sum(row["error"]["numel"] for row in rows)
        sse = math.fsum(row["error"]["squared_error_sum"] for row in rows)
        heads = [math.fsum(row["error"]["per_head_squared_error_sum"][h] for row in rows) for h in range(40)]
        summary[name] = {"mean_attention_ms": math.fsum(timings) / len(timings), "timed_calls": len(timings) * CALLS_PER_REPEAT,
                         "capture_inputs": 24, "numel": count, "mse": sse / count,
                         "per_head_mse": [value / (count // 40) for value in heads]}
        if name in STABLE_METHODS:
            for row in rows:
                warmup = row["warmup_observation"]["stable_counts"]
                require(warmup == {"complete_calls": WARMUP, "merge_calls": 5 * WARMUP,
                                   "stable_merge_kernel_calls": 5 * WARMUP}, "incomplete stable warmup kernels")
                counts = row["timed_stable_counts"]
                expected = {"complete_calls": CALLS_PER_REPEAT, "merge_calls": 5 * CALLS_PER_REPEAT,
                            "stable_merge_kernel_calls": 5 * CALLS_PER_REPEAT}
                require(len(counts) == REPEATS and all(value == expected for value in counts), "incomplete stable timed kernels")
            summary[name]["stable_merge_kernel_calls"] = 24 * 5 * REPEATS * CALLS_PER_REPEAT
            summary[name]["warmup_stable_merge_kernel_calls"] = 24 * 5 * WARMUP
    for values in summary.values():
        values["speedup_vs_sdpa"] = summary["sdpa_flash"]["mean_attention_ms"] / values["mean_attention_ms"]
        values["speedup_vs_historical_hf_rf10"] = summary[HISTORICAL_HF]["mean_attention_ms"] / values["mean_attention_ms"]
        values["speedup_vs_hf_rf0_bf16"] = summary["hellinger_rf0"]["mean_attention_ms"] / values["mean_attention_ms"]
    return summary


def worker(args):
    require_parent()  # Before importing Torch or initializing GPU.
    methods = suite_methods(args.stable_sq1_suite)
    preflight = json.loads((args.output / "preflight.json").read_text())
    require(source_hashes(args.stable_sq1_suite) == preflight["source_hashes"] and preflight["methods"] == list(methods),
            "benchmark source or suite changed before worker")
    inputs = validate_capture_run(args.capture_run)
    require(inputs == preflight["inputs"], "input receipt changed before worker")
    require(historical_mode(args.historical_audit) == HF10_MODE, "historical mode changed")
    import torch
    import triton
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    stable_kernel = stable_receipt = None
    if args.stable_sq1_suite:
        from analysis.stable_sq1_video_adapter import load_generated_kernel
        hard = importlib.import_module("attention.hard_c4_wan21_14b_720p_triton")
        expected = expected_stable_receipt(args.output)
        require(expected == preflight["expected_generated_stable_kernel"], "stable source prediction changed")
        stable_kernel, receipt = load_generated_kernel(hard._hard_c4_720p_temporal_shared_merge_kernel, Path(expected["path"]))
        stable_receipt = verify_stable_receipt(receipt | {"bytes": Path(receipt["path"]).stat().st_size}, expected)
    report = {"complete": False, "scope": SCOPE, "methods": list(methods), "records": [],
              "timing": {"warmup": WARMUP, "repeats": REPEATS, "calls_per_repeat": CALLS_PER_REPEAT,
                         "order": [repeat_order(i, methods) for i in range(REPEATS)], "clock": "CUDA events, synchronized per block",
                         "included": "complete selection + optional per-call current-frame means + SQ1 merge + global compressed route + contiguous output",
                         "excluded": "disk load, host/device transfer, compilation/warmup, context installation, external observation/error checks",
                         "adapter_overhead": "stable adapter per-call metadata validation, kernel-symbol binding and counters remain inside timing; no inference work moved outside"},
              "versions": {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__},
              "historical_mode": HF10_MODE,
              "historical_control_scope": "Verified historical method mode on the common pinned current kernels, not a replay of the old checkout/runtime",
              "video_metrics": None}
    if args.stable_sq1_suite:
        from analysis.quality_first_mode_contract import method_contract
        report["generated_stable_kernel"] = stable_receipt
        report["stable_declarations"] = {name: method_contract(name) for name in STABLE_METHODS}
        report["stable_kernel_loader"] = identity(load_generated_kernel)
    with torch.inference_mode():
        for receipt in inputs["captures"]:
            require_parent()
            path = check_capture_stat(inputs["capture_root"], receipt)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            validate_payload(payload, receipt); check_capture_stat(inputs["capture_root"], receipt)
            qkv = tuple(payload[name].to(device="cuda") for name in ("q", "k", "v"))
            reference = payload["dense_output"].to(device="cuda")
            del payload
            storage_before = [signature(tensor) for tensor in qkv]
            bindings = {name: Binding(name, qkv, stable_kernel=stable_kernel, stable_receipt=stable_receipt) for name in methods}
            record = {"capture_file": receipt["capture_file"], "input_sha256": receipt["sha256"],
                      "identity": {key: receipt[key] for key in ("prompt_index", "call_index", "step_index", "layer_index")}, "methods": {}}
            for name in methods:
                with bindings[name].context(observe=True) as (invoke, state):
                    for _ in range(WARMUP): output = invoke()
                torch.cuda.synchronize()
                record["methods"][name] = {"warmup_observation": state, "callable_identities": bindings[name].originals,
                                           "error": output_error(output, reference), "repeat_ms": [], "timed_mean_counts": []}
                if name in STABLE_METHODS:
                    record["methods"][name]["timed_stable_counts"] = []
                del output
            for repeat in range(REPEATS):
                for name in repeat_order(repeat, methods):
                    with bindings[name].context() as (invoke, state):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize(); start.record()
                        for _ in range(CALLS_PER_REPEAT): output = invoke()
                        end.record(); end.synchronize()
                        elapsed = start.elapsed_time(end) / CALLS_PER_REPEAT
                    require(state["calls"] == CALLS_PER_REPEAT and math.isfinite(elapsed) and elapsed > 0, "timed invocation incomplete")
                    record["methods"][name]["repeat_ms"].append(elapsed)
                    if "mean_counts" in state: record["methods"][name]["timed_mean_counts"].append(state["mean_counts"])
                    if "stable_counts" in state: record["methods"][name]["timed_stable_counts"].append(state["stable_counts"])
                    del output
            # Inference tensors have no version counters. This checks storage
            # and layout, not a checksum of device values or arbitrary writes.
            require([signature(tensor) for tensor in qkv] == storage_before, "operator changed Q/K/V storage/layout")
            report["records"].append(record); write(args.output / "results.json", report)
            print(json.dumps({"capture": receipt["capture_file"], "complete_inputs": len(report["records"]),
                              "means_ms": {name: math.fsum(record["methods"][name]["repeat_ms"]) / REPEATS for name in methods}}), flush=True)
            del bindings, qkv, reference
        report["summary"] = summarize(report["records"], methods)
        require(source_hashes(args.stable_sq1_suite) == preflight["source_hashes"] and validate_capture_run(args.capture_run) == inputs, "source/input changed during benchmark")
        if args.stable_sq1_suite:
            verify_stable_receipt(report["generated_stable_kernel"], preflight["expected_generated_stable_kernel"])
        report["complete"] = True; write(args.output / "results.json", report)


def launch(args):
    from analysis import launch_quality_first_videos as guard
    methods = suite_methods(args.stable_sq1_suite)
    inputs = validate_capture_run(args.capture_run)  # Refuse missing bb completion before clean-GPU admission.
    historical_mode(args.historical_audit)
    require(not args.output.exists(), "fresh benchmark output required; no resume/queue")
    initial = guard.require_clean(args.gpu)
    limit = guard.account_limit()
    last_account_devices = guard.account_devices()
    require(len(last_account_devices) < limit, "account GPU cap has no remaining slot; refusing to queue")
    hashes = source_hashes(args.stable_sq1_suite)
    expected = expected_stable_receipt(args.output) if args.stable_sq1_suite else None
    args.output.mkdir(parents=True)
    write(args.output / "preflight.json", {"initial_gpu": initial, "inputs": inputs, "source_hashes": hashes,
           "historical_audit_sha256": sha(args.historical_audit), "scope": SCOPE, "methods": list(methods),
           "expected_generated_stable_kernel": expected, "account_gpu_limit": limit,
           "initial_account_devices": last_account_devices})
    process = None; foreign = []; failure = None
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(*unused): raise KeyboardInterrupt("benchmark parent interrupted")
    for number in previous: signal.signal(number, interrupted)
    try:
        immediate = guard.require_clean(args.gpu, expected_uuid=initial["gpu"]["uuid"])
        last_account_devices = guard.account_devices()
        require(len(last_account_devices) < limit, "account GPU cap reached before spawn")
        environment = guard.child_environment(args.output, initial["gpu"]["uuid"])
        environment["BARYCENTRIC_BENCH_PARENT"] = str(os.getpid())
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--gpu", str(args.gpu),
                   "--capture-run", str(args.capture_run), "--historical-audit", str(args.historical_audit), "--output", str(args.output)]
        if args.stable_sq1_suite: command.append("--stable-sq1-suite")
        with (args.output / "worker.log").open("x") as log, (args.output / "gpu_monitor.jsonl").open("x") as monitor:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write(args.output / "process.json", {"parent_pid": os.getpid(), "worker_pid": process.pid,
                  "command": command, "immediate_gpu": immediate, "spawned_utc": guard.now(),
                  "account_gpu_limit": limit, "immediate_account_devices": last_account_devices})
            while True:
                active = guard.gpu_processes(initial["gpu"]["uuid"])
                last_account_devices = guard.account_devices()
                foreign = [row for row in active if row["pid"] != process.pid]
                monitor.write(json.dumps({"utc": guard.now(), "processes": active, "foreign": foreign,
                                          "account_devices": last_account_devices}) + "\n"); monitor.flush()
                require(not foreign, "foreign GPU process; timing invalid")
                require(len(last_account_devices) <= limit, "account GPU cap exceeded; stop only own child")
                if process.poll() is not None: break
                time.sleep(5)
        require(process.returncode == 0, f"benchmark worker exited {process.returncode}")
        require(source_hashes(args.stable_sq1_suite) == hashes, "source changed during benchmark")
        result = json.loads((args.output / "results.json").read_text())
        require(result.get("complete") is True and result.get("methods") == list(methods)
                and result.get("summary") == summarize(result["records"], methods), "incomplete benchmark result")
        if args.stable_sq1_suite: verify_stable_receipt(result["generated_stable_kernel"], expected)
    except BaseException as error: failure = f"{type(error).__name__}: {error}"
    finally:
        for number in previous: signal.signal(number, signal.SIG_IGN)
        try: guard.stop_own_child(process)
        finally:
            for number, handler in previous.items(): signal.signal(number, handler)
    receipt = {"valid": failure is None, "timing_valid": failure is None, "failure": failure,
               "foreign": foreign, "worker_exit_code": process.returncode if process else None,
               "gpu_uuid": initial["gpu"]["uuid"], "completed_utc": guard.now(), "scope": SCOPE,
               "monitor_interval_seconds": 5, "account_gpu_limit": limit, "last_account_devices": last_account_devices,
               "purity_caveat": "sampled compute-process exclusivity, not host-wide contention isolation"}
    write(args.output / "completion.json", receipt)
    require(failure is None, failure)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--stable-sq1-suite", action="store_true", help="explicit nondefault paired stable SQ1 suite; leaves historical default suite unchanged")
    parser.add_argument("--gpu", type=int, required=True)
    for name in ("capture-run", "historical-audit", "output"):
        parser.add_argument("--" + name, type=lambda x: Path(x).resolve(), required=True)
    args = parser.parse_args()
    return worker(args) if args.worker else print(json.dumps(launch(args), indent=2))


if __name__ == "__main__": main()
