"""Guarded, off-path fixed-seed SQ1 witness diagnosis, never a candidate.

One original 309 MB capture; prompt1, head30 (stored slot3), RF0, column40,
all45 query rows and all945 key blocks. The unchanged native entry computes
the original21-frame/80-column merge and route geometry for this one head;
only the witness is retained. No video, timing, new seeds, precision default,
or production-kernel edits. The old BF16 output MUST reproduce bitwise first.

A separately generated observer copies the current merge kernel and adds
stores only. Its witness KV/bias and output MUST equal the untouched kernel
bitwise before its intermediate arrays can be interpreted. Counterfactuals
retain observed owners and separately control native versus structural cell
validity; each native-route replay uses the original layout and precision.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OLD_ROOT = Path("/local2/denghaoran/GridMergeAttention/experiments/quality_first_bf16_20260905b")
OLD_RESULT_SHA = "5a7cb6e15e37bb8f40ffec82d67b16ccb53b5d5ffae357bc98e5d9b8bb557b03"
CAPTURE_SHA = "5639fa83ebf2eacc1d738ee013a66293a90722308b19bbea2becb75a569e415b"
SEEDS_SHA = "5775f4d59884828d3b419254646f959bbccb75599b9809716217b57cecb57374"
CACHE_SHA = "0bd17b85e1af64f2d323a7818707be35640864286fad13420797ca4dda495296"
GRID = (1, 21, 45, 80, 4, 128)
METHOD, HEAD_SLOT, COLUMN, SCALE = "mass_rf0", 3, 40, 128 ** -.5
KERNEL = "_hard_c4_720p_temporal_shared_merge_kernel"
TRACE_NAMES = ("logits2", "logp2", "owner", "pre_weights", "pre_bias", "prob_sum", "alpha", "residual", "bulk_valid")
COUNTERFACTUALS = (
    "native_KV_stable_bias_native_valid", "stable_KV_native_bias_native_valid",
    "stable_KV_stable_bias_native_valid", "stable_KV_stable_bias_structural_valid",
)
MERGE_OPTIONS = dict(head_start=3, head_count=1, top_r=1, selector="query_column",
    query_column_major=True, split_kv_accum=True, seed_rows_native_layout=True,
    merge_loop_stages=2, merge_disable_licm=False, block_j=16,
    query_frame_group=21, merge_frame_group=1, merge_col_group=1,
    direct_bulk_normalization=False, fp8_score_dot=False, fp8_merge_k_dot=False,
    fp8_merge_v_dot=False, compressed_fp8_kv=False, scaled_fp8_kv=False,
    mean_centered_routing=False, gaussian_partition_routing=False, reuse_partition_exp=False,
    split_merge=False, materialize_singleton=False, packed_singleton_kv=False)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def input_contract(old_root):
    """Small receipt reads only; exactly one named capture, never a glob."""
    old_root = Path(old_root)
    require(digest(old_root / "results.json") == OLD_RESULT_SHA, "wrong completed BF16 diagnostic")
    old = json.loads((old_root / "results.json").read_text())
    completion = json.loads((old_root / "completion.json").read_text())
    require(old["valid"] is True and old["complete"] is True
            and completion["valid"] is True and completion["worker_exit_code"] == 0
            and completion["foreign"] == [], "old BF16 diagnostic did not complete cleanly")
    records = [row for row in old["records"] if row["metadata"]["prompt_index"] == 1]
    require(len(records) == 1, "exactly one prompt1 witness required")
    row = records[0]
    meta = row["metadata"]
    require(meta["heads"] == [0, 10, 20, 30] and meta["call_index"] == 33
            and meta["step_index"] == 0 and meta["layer_index"] == 33
            and meta["shape"] == [1, 75600, 4, 128] and meta["dtype"] == "torch.bfloat16",
            "wrong witness coordinates or original dtype")
    require(row["input"]["sha256"] == CAPTURE_SHA and row["input"]["bytes"] == 309661725
            and row["seed_cache"]["sha256"] == SEEDS_SHA and row["output_cache"]["sha256"] == CACHE_SHA,
            "witness identity changed")
    groups = [group for group in row["groups"] if group["head"] == 30 and group["query_column"] == COLUMN]
    require(len(groups) == 1 and row["bf16_score_control"][METHOD]["exact_seed_table_equal"] is True,
            "missing fixed-seed witness")
    require(row["bf16_score_control"][METHOD]["mode"] ==
            "entropy_mass_span_c32_p16_g256_j4_t1_rf0_u16_cg45_sq1_qm_skv_nl_ml2", "old BF16 mode changed")
    return old, row, groups[0]


def check_inputs(row):
    for name in ("input", "seed_cache", "output_cache"):
        receipt = row[name]
        path = Path(receipt["path"])
        require(path.stat().st_size == receipt["bytes"] and digest(path) == receipt["sha256"],
                f"changed {name}")


def source_hashes():
    names = {"analysis/diagnose_sq1_fixed_seed_witness.py", "analysis/launch_quality_first_videos.py",
             "analysis/run_quality_first_videos.py", "analysis/quality_first_mode_contract.py"}
    names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "attention").rglob("*.py"))
    if (ROOT / "analysis/__init__.py").exists():
        names.add("analysis/__init__.py")
    return {name: digest(ROOT / name) for name in sorted(names)}


def verify_sources(output, hashes):
    require(source_hashes() == hashes, "live sources changed")
    require(all(digest(output / "sources" / name) == value for name, value in hashes.items()),
            "frozen sources changed")


def make_observer_source(source):
    """Copy arithmetic AST unchanged; insert bounded stores at four anchors.

    New code is an experiment artifact, never a shared-kernel source edit.
    Unknown source structure fails instead of applying an approximate patch.
    """
    tree = ast.parse(textwrap.dedent(source))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == KERNEL)
    function.name = "_sq1_fixed_seed_observer_kernel"
    function.decorator_list = [ast.parse("triton.jit", mode="eval").body]
    for name in TRACE_NAMES:
        # This Triton version binds only node.args.args in its code generator,
        # so the appended pointers must be ordinary arguments, not keyword-
        # only arguments. Explicit defaults also keep its binder signature
        # legal after the original optional arguments. CaptureKernel supplies
        # every pointer by name; none of the original positions is changed.
        function.args.args.append(ast.arg(arg="witness_" + name + "_ptr"))
        function.args.defaults.append(ast.Constant(value=None))
    prefix = """
witness_block = key_outer * ROWS + key_row
witness_blocks = witness_block + tl.zeros((BLOCK_J,), tl.int32)
witness_columns = query_col_start + tl.arange(0, BLOCK_J)
witness_match = (witness_columns == 40) & (query_outer == 0)
witness_cells = tl.arange(0, C)
"""

    def vector_store(pointer, main, tail):
        return f"""
tl.store(witness_{pointer}_ptr + ((witness_blocks[:, None, None] * C + witness_cells[None, :, None]) * COLS + main_cols[None, None, :]), {main}, mask=witness_match[:, None, None])
tl.store(witness_{pointer}_ptr + ((witness_blocks[:, None, None] * C + witness_cells[None, :, None]) * COLS + tail_cols[None, None, :]), {tail}, mask=witness_match[:, None, None])
"""

    additions = {
        "log_probs_tail": prefix + vector_store("logits2", "tl.reshape(scores_main, (BLOCK_J, C, 64))", "tl.reshape(scores_tail, (BLOCK_J, C, 16))")
            + vector_store("logp2", "log_probs_main", "log_probs_tail"),
        "winner_tail, route_tail": """
tl.store(witness_owner_ptr + witness_blocks[:, None] * COLS + main_cols[None, :], route_main, mask=witness_match[:, None])
tl.store(witness_owner_ptr + witness_blocks[:, None] * COLS + tail_cols[None, :], route_tail, mask=witness_match[:, None])
""",
        "pre_bias": vector_store("pre_weights", "weights_main", "weights_tail") + """
tl.store(witness_pre_bias_ptr + witness_blocks[:, None] * C + witness_cells[None, :], route_bias, mask=witness_match[:, None])
tl.store(witness_prob_sum_ptr + witness_blocks[:, None] * C + witness_cells[None, :], prob_sum, mask=witness_match[:, None])
""",
        "safe_residual_mass": """
tl.store(witness_alpha_ptr + witness_blocks[:, None] * C + witness_cells[None, :], selected_cluster_weight, mask=witness_match[:, None])
tl.store(witness_residual_ptr + witness_blocks[:, None] * C + witness_cells[None, :], residual_mass, mask=witness_match[:, None])
tl.store(witness_bulk_valid_ptr + witness_blocks[:, None] * C + witness_cells[None, :], bulk_has_key.to(tl.int32), mask=witness_match[:, None])
""",
    }
    counts = dict.fromkeys(additions, 0)

    class InsertStores(ast.NodeTransformer):
        def visit_Assign(self, node):
            name = ast.unparse(node.targets[0])
            if name == "(winner_tail, route_tail)":
                name = "winner_tail, route_tail"
            if name == "route_bias" and ast.unparse(node.value) == "tl.where(has_key, route_bias, float('inf'))":
                name = "pre_bias"
            if name not in additions:
                return node
            counts[name] += 1
            return [node, *ast.parse(additions[name]).body]

    InsertStores().visit(function)
    require(all(count == 1 for count in counts.values()), f"observer anchors changed: {counts}")
    tree.body = [function]
    ast.fix_missing_locations(tree)
    return "import triton\nimport triton.language as tl\n\n" + ast.unparse(tree) + "\n"


def witness_view(tensor, vector=True):
    """Actual merge layout [B,Qframe,column,activehead,Kframe,keyrow,C,D]."""
    shape = (1, 21, 80, 1, 21, 45, 4) + ((128,) if vector else ())
    return tensor.view(shape)[0, 0, COLUMN, 0].reshape((945, 4) + ((128,) if vector else ()))


class CaptureKernel:
    """Forward unchanged native launch; optionally substitute witness buffers."""
    def __init__(self, kernel, torch, traces=None, replacement=None):
        self.kernel, self.torch, self.traces, self.replacement = kernel, torch, traces, replacement
        self.calls, self.observed, self.flags = 0, None, None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls += 1
            require(self.calls == 1, "unexpected repeated merge launch")
            require(tuple(args[0].shape) == GRID and tuple(args[1].shape) == (1, 1, 21, 45, 80, 4, 4)
                    and tuple(args[12:24]) == (21, 1, 21, 4, 1, 3, 128, 45, 80, 4, 4, 16),
                    "native geometry changed")
            require(tuple(args[24:30]) == (2, 1, 1, False, "ieee", True),
                    "old ml2 LICM/loop/precision/entropy contract changed")
            for name in ("FP8_SCORE_DOT", "FP8_MERGE_K_DOT", "FP8_MERGE_V_DOT", "DIRECT_BULK_NORMALIZATION",
                         "MEAN_CENTERED_ROUTING", "GAUSSIAN_PARTITION_ROUTING", "REUSE_PARTITION_EXP",
                         "STORE_PACKED_SINGLETON_KV", "MATERIALIZE_GLOBAL_MASS", "GLOBAL_MASS_DEFER_CORRECTION"):
                require(kwargs.get(name) is False, f"unexpected merge option {name}")
            require(kwargs["GLOBAL_MASS_TOP_R"] == 1 and kwargs["GLOBAL_QUERY_COLUMN_SELECTOR"] is True
                    and kwargs["SPLIT_KV_ACCUM"] is True and kwargs["SEED_ROWS_NATIVE_LAYOUT"] is True
                    and kwargs.get("MEAN_SEED_ZERO", False) is False, "not original SQ1 substrate")
            self.flags = {name: value for name, value in kwargs.items() if isinstance(value, (bool, int, float, str))}
            self.flags.update(block_j=args[23], merge_loop_stages=args[24], merge_frame_group=args[25],
                              merge_col_group=args[26], merge_disable_licm=args[27],
                              input_precision=args[28], use_entropy_bias=args[29])
            extra = {} if self.traces is None else {"witness_" + name + "_ptr": value for name, value in self.traces.items()}
            result = self.kernel[grid](*args, **kwargs, **extra)
            views = (witness_view(args[5]), witness_view(args[6]), witness_view(args[9], False))
            self.observed = tuple(value.clone() for value in views)
            if self.replacement is not None:
                for value, desired in zip(views, self.replacement):
                    converted = desired.to(value.dtype)
                    value.copy_(converted)
                    require(bitwise_equal(self.torch, value, converted), "counterfactual substitution not bitwise")
            return result
        return launch


def native_call(torch, q, k, v, table, hard, singleton, *, kernel=None, traces=None, replacement=None):
    actual = getattr(hard, KERNEL)
    observer = CaptureKernel(actual if kernel is None else kernel, torch, traces, replacement)
    output = torch.full_like(q, float("nan"))
    with patch.object(hard, KERNEL, observer):
        singleton.hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks(
            q, k, v, table, SCALE, output=output, **MERGE_OPTIONS)
    require(observer.calls == 1 and observer.observed is not None, "missing native merge")
    selected = output[0, 0, :, COLUMN, HEAD_SLOT].clone()
    require(bool(torch.isfinite(selected).all()), "nonfinite native witness output")
    return selected, observer.observed, observer.flags


def stable_merge(torch, logits, keys, values, *, owners=None):
    """FP64 direct cell softmax AFTER full-support ownership and SQ1 peeling."""
    require(logits.dtype == keys.dtype == values.dtype == torch.float64, "FP64 reference required")
    logp = logits.log_softmax(-1)
    if owners is None:
        owners = logp.argmax(1)
    mask = owners[:, None, :] == torch.arange(4, device=keys.device)[None, :, None]
    pre_mask = mask.clone()
    occupied = pre_mask.any(-1)
    pre_logz = logits.masked_fill(~pre_mask, -torch.inf).logsumexp(-1)
    pre_logw = logits - torch.where(occupied, pre_logz, 0)[:, :, None]
    pre_weight = torch.where(pre_mask, pre_logw.exp(), 0)
    pre_beta = (pre_weight * torch.where(pre_mask, pre_logw, 0)).sum(-1)
    alpha = pre_weight[..., COLUMN]
    mask[..., COLUMN] = False
    count = mask.sum(-1)
    valid = count > 0
    # Direct bulk maximum: no 1-alpha, no S-p_s, no E-E_s, and no
    # cancellation between a large natural logit and its cell log-normalizer.
    maximum = logits.masked_fill(~mask, -torch.inf).amax(-1)
    centered = logits - torch.where(valid, maximum, 0)[:, :, None]
    exponent = torch.where(mask, centered.exp(), 0)
    normalizer = exponent.sum(-1)
    safe_normalizer = torch.where(valid, normalizer, 1)
    weights = exponent / safe_normalizer[:, :, None]
    beta = (weights * torch.where(mask, centered, 0)).sum(-1) - safe_normalizer.log()
    cell_logz = maximum + safe_normalizer.log()
    beta = beta.masked_fill(~valid, torch.inf)
    merged_k, merged_v = torch.bmm(weights, keys), torch.bmm(weights, values)
    return {"logits": logits, "logp": logp, "owners": owners, "pre_weights": pre_weight,
            "pre_beta": pre_beta, "alpha": alpha, "residual_subtracted": 1 - alpha,
            "residual_direct": (pre_weight * mask).sum(-1), "bulk_count": count,
            "weights": weights, "valid": valid, "bias": beta, "k": merged_k, "v": merged_v,
            "cell_logz": cell_logz, "mask": mask}


def route64(torch, query, keys, values, bias, singleton_k, singleton_v):
    """All945 blocks, C4 plus SQ1, no truncated top-k or dense rescue."""
    keys = torch.cat((keys.double(), singleton_k[:, None]), 1)
    values = torch.cat((values.double(), singleton_v[:, None]), 1)
    bias = torch.cat((bias.double(), torch.zeros_like(bias[:, :1], dtype=torch.float64)), 1)
    logits = torch.einsum("qd,bcd->qbc", query, keys) * SCALE - bias[None]
    weights = logits.flatten(1).softmax(-1).reshape(logits.shape)
    output = torch.einsum("qbc,bcd->qd", weights, values)
    return {"output": output, "weights": weights, "logits": logits, "values": values}


def finite_stats(torch, actual, expected):
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    error = (actual.double() - expected.double()).abs()[finite]
    return {"values": actual.numel(), "finite_pairs": int(finite.sum()),
            "finite_mask_mismatches": int((torch.isfinite(actual) != torch.isfinite(expected)).sum()),
            "max_abs_error": float(error.max()) if error.numel() else None,
            "mean_abs_error": float(error.mean()) if error.numel() else None}


def hull_diagnostics(torch, merged, original, mask, valid, tau):
    """Raw coordinate violations; storage/dot rounding is not silently zeroed."""
    ordinary, scaled = [], []
    for cell in range(4):
        lower = original.masked_fill(~mask[:, cell, :, None], torch.inf).amin(1)
        upper = original.masked_fill(~mask[:, cell, :, None], -torch.inf).amax(1)
        value = merged[:, cell].double()
        active = valid[:, cell, None]
        raw = torch.maximum(lower - value, value - upper).clamp_min(0).masked_fill(~active, 0)
        factor = tau[:, cell, None]
        adjusted = torch.maximum(factor * lower - value, value - factor * upper).clamp_min(0).masked_fill(~active, 0)
        ordinary.append(raw.amax(-1))
        scaled.append(adjusted.amax(-1))
    ordinary, scaled = torch.stack(ordinary, -1), torch.stack(scaled, -1)
    summary = dict(raw_convex_hull_max_violation=float(ordinary.max()),
        rounded_weight_mass_scaled_hull_max_violation=float(scaled.max()),
        note="Raw errors include BF16 storage/FP32 dot rounding; scaled hull uses sum(BF16 pre-peel bulk weights)/native residual.")
    return summary, ordinary, scaled


def mse(torch, output, reference):
    return float((output.double() - reference.double()).square().mean())


def top_witness_tokens(torch, row, native_route, ideal_route, native_buffers, stable, trace, seed_rows):
    """Attribution is vector contribution, not an unsupported causal label."""
    native_k, native_v, native_bias = native_buffers
    difference = (native_route["weights"][row, :, :, None] * native_route["values"]
                  - ideal_route["weights"][row, :, :, None] * ideal_route["values"])
    contribution = difference.square().sum(-1).sqrt()
    top_indices = set(native_route["weights"][row].flatten().topk(16).indices.cpu().tolist())
    top_indices.update(contribution.flatten().topk(16).indices.cpu().tolist())
    result = []
    for index in sorted(top_indices, key=lambda index: -float(contribution.flatten()[index])):
        block, cell = divmod(index, 5)
        frame, key_row = divmod(block, 45)
        actual_logit = native_route["logits"][row, block, cell]
        ideal_logit = ideal_route["logits"][row, block, cell]
        item = {"query_row": row, "block": block, "key_frame": frame, "key_row": key_row, "cell": cell,
            "singleton": cell == 4, "native_weight": float(native_route["weights"][row, block, cell]),
            "ideal_weight": float(ideal_route["weights"][row, block, cell]),
            "contribution_delta_l2": float(contribution[block, cell]),
            "native_logit": float(actual_logit) if bool(torch.isfinite(actual_logit)) else None,
            "ideal_logit": float(ideal_logit) if bool(torch.isfinite(ideal_logit)) else None,
            "logit_delta": float(actual_logit - ideal_logit)
                if bool(torch.isfinite(actual_logit) & torch.isfinite(ideal_logit)) else None}
        if cell < 4:
            item.update(seed_row=int(seed_rows[block, cell]), bulk_count=int(stable["bulk_count"][block, cell]),
                native_bias=float(native_bias[block, cell]) if bool(torch.isfinite(native_bias[block, cell])) else None,
                stable_bias=float(stable["bias"][block, cell]) if bool(stable["valid"][block, cell]) else None,
                native_alpha=float(trace["alpha"][block, cell]), native_residual=float(trace["residual"][block, cell]),
                direct_residual=float(stable["residual_direct"][block, cell]),
                native_bulk_valid=bool(trace["bulk_valid"][block, cell]),
                key_delta_l2=float((native_k[block, cell].double() - stable["k"][block, cell]).norm()),
                value_delta_l2=float((native_v[block, cell].double() - stable["v"][block, cell]).norm()))
        result.append(item)
    return result


def bitwise_equal(torch, left, right):
    require(left.dtype == right.dtype and left.shape == right.shape, "bit comparison contract")
    integer = torch.int16 if left.dtype == torch.bfloat16 else torch.int32
    require(left.dtype in (torch.bfloat16, torch.float32), "unsupported bit comparison dtype")
    return bool(torch.equal(left.contiguous().view(integer), right.contiguous().view(integer)))


def require_old_replay(torch, native, old):
    require(old.dtype == torch.float64 and old.shape == native.shape, "wrong old native slice")
    converted = old.to(torch.bfloat16)
    require(bool(torch.equal(converted.double(), old)), "old native cache is not exact BF16 values")
    require(bitwise_equal(torch, native, converted), "old BF16 output did not reproduce bitwise; stop diagnosis")


def validate_completed_result(report):
    require(report.get("valid") is True and report.get("complete") is True, "worker incomplete")
    require(report.get("candidate") is False and report.get("timing_benchmark") is False, "diagnostic scope changed")
    require(report.get("key_blocks") == 945 and report.get("query_rows") == 45, "incomplete witness scope")
    required = ("old_native_bitwise", "observer_output_bitwise", "observer_kv_bias_bitwise",
                "old_ideal_fp64", "old_dense_fp64", "original_inputs_bitwise", "sources_unchanged", "inputs_unchanged")
    require(all(report.get("gates", {}).get(name) is True for name in required), "missing witness hard gate")
    require(set(report.get("counterfactuals", {})) == set(COUNTERFACTUALS), "counterfactual coverage changed")
    require(all(record.get("original_merge_bitwise") is True and record.get("substitution_bitwise") is True
                and record.get("native_route_geometry_unchanged") is True
                for record in report["counterfactuals"].values()), "counterfactual integrity gate missing")


def worker(args):
    # No CUDA imports or allocation before proving this is the owned child.
    require(os.environ.get("SQ1_WITNESS_PARENT") == str(os.getppid()), "worker requires its guarded live parent")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
            and "," not in os.environ["CUDA_VISIBLE_DEVICES"], "one UUID-visible GPU required")
    startup = json.loads((args.output / "startup.json").read_text())
    hashes = startup["source_sha256"]
    verify_sources(args.output, hashes)
    old, row, group = input_contract(args.old_root)
    check_inputs(row)
    report = dict(valid=False, complete=False, candidate=False, timing_benchmark=False,
        scope="prompt1 call33/L33 step0; original saved head30(slot3); RF0 column40; fixed Mass RF0 seeds",
        key_blocks=945, key_tokens=75600, query_rows=45, metadata=row["metadata"],
        inputs={name: row[name] for name in ("input", "seed_cache", "output_cache")},
        old_result_sha256=OLD_RESULT_SHA, old_group=group, source_sha256=hashes,
        merge_options=MERGE_OPTIONS, gates={}, counterfactuals={},
        fp64_replay_tolerance=dict(atol=1e-10, rtol=1e-9),
        psnr=None, ssim=None, lpips=None, mean_attention_ms=None, speedup_vs_sdpa=None,
        limitations=["One fixed-seed witness, not a selector, benchmark, or video result.",
            "Observer trace is usable only if its output and witness KV/bias match the unmodified kernel bitwise.",
            "Counterfactuals alter witness intermediates only and retain the normal full-head route geometry.",
            "The unchanged merge/route still computes all21 frames and80 columns for one head; only RF0/col40 is interpreted.",
            "Five-second sampled GPU exclusivity is not continuous proof of absence of short foreign processes."])
    write_json(args.output / "results.json", report)
    import numpy as np
    import torch
    from attention import hard_c4_wan21_14b_720p_triton as hard
    from attention import singleton_c4_wan21_14b_720p_triton as singleton
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    report.update(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  device=torch.cuda.get_device_name(0))
    saved = torch.load(row["input"]["path"], map_location="cpu", weights_only=True)
    expected_meta = {key: value for key, value in row["metadata"].items() if key not in ("bytes", "sha256")}
    require(set(saved) == {"q", "k", "v", "dense_output", "metadata"}
            and saved["metadata"] == expected_meta, "capture metadata changed")
    require(all(value.dtype == torch.bfloat16 and tuple(value.shape) == (1, 75600, 4, 128)
                and bool(torch.isfinite(value).all()) for name, value in saved.items() if name != "metadata"),
            "not finite original BF16 four-head storage")
    with np.load(row["seed_cache"]["path"], allow_pickle=False) as archive:
        seed_array = archive[METHOD].copy()
    require(seed_array.dtype == np.uint8 and seed_array.shape == (1, 1, 21, 45, 80, 4, 4), "seed table shape/dtype")
    require(bool((seed_array < 45).all()) and bool((np.diff(np.sort(seed_array, axis=-1), axis=-1) != 0).all()),
            "four distinct real rows required")
    with np.load(row["output_cache"]["path"], allow_pickle=False) as archive:
        cached = {name: torch.from_numpy(archive["h30_c40_" + suffix].copy()).to("cuda:0") for name, suffix in
                  (("native", METHOD + "_bf16"), ("ideal", METHOD + "_ideal"), ("dense", "dense64"))}
    with torch.inference_mode():
        q, k, v = [saved[name].to("cuda:0").reshape(GRID) for name in ("q", "k", "v")]
        originals = tuple(value.clone() for value in (q, k, v))
        table = torch.from_numpy(seed_array).to("cuda:0")
        seeds_original = table.clone()
        native, buffers, flags = native_call(torch, q, k, v, table, hard, singleton)
        require_old_replay(torch, native, cached["native"])
        report["gates"]["old_native_bitwise"] = True
        report["actual_merge_launch"] = flags
        write_json(args.output / "results.json", report)
        print("Old native BF16 witness reproduced bitwise: 45 x 128.", flush=True)

        original_kernel = getattr(hard, KERNEL)
        source = getattr(original_kernel, "src", None)
        if source is None:
            source = inspect.getsource(original_kernel.fn)
        observer_path = args.output / "observer_kernel.py"
        observer_path.write_text(make_observer_source(source), encoding="utf-8")
        report["observer_source_sha256"] = digest(observer_path)
        spec = importlib.util.spec_from_file_location("_sq1_witness_observer", observer_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        trace = {name: torch.full((945, 80) if name == "owner" else
                    ((945, 4, 80) if name in ("logits2", "logp2", "pre_weights") else (945, 4)),
                    -99 if name in ("owner", "bulk_valid") else float("nan"), device=q.device,
                    dtype=torch.int32 if name in ("owner", "bulk_valid") else torch.float32)
                 for name in TRACE_NAMES}
        observed, observed_buffers, observed_flags = native_call(torch, q, k, v, table, hard, singleton,
            kernel=module._sq1_fixed_seed_observer_kernel, traces=trace)
        observer_output = bitwise_equal(torch, observed, native)
        observer_buffers = all(bitwise_equal(torch, a, b) for a, b in zip(buffers, observed_buffers))
        report["gates"].update(observer_output_bitwise=observer_output, observer_kv_bias_bitwise=observer_buffers)
        report["observer_parity"] = dict(output=finite_stats(torch, observed, native),
            buffers=[finite_stats(torch, a, b) for a, b in zip(observed_buffers, buffers)])
        write_json(args.output / "results.json", report)
        require(observer_output and observer_buffers and flags == observed_flags,
                "observer changed native arithmetic outputs; no trace attribution permitted")
        require(bool(((trace["owner"] >= 0) & (trace["owner"] < 4)).all())
                and bool(((trace["bulk_valid"] == 0) | (trace["bulk_valid"] == 1)).all()), "incomplete integer trace")
        require(all(bool(torch.isfinite(trace[name]).all()) for name in TRACE_NAMES
                    if name not in ("owner", "bulk_valid", "pre_bias")), "nonfinite/incomplete trace")
        require(bool((torch.isfinite(trace["pre_bias"]) | torch.isposinf(trace["pre_bias"])).all()), "incomplete bias trace")
        print("Observer output and KV/bias are bitwise identical; all 945 banks captured.", flush=True)

        query = q[0, 0, :, COLUMN, HEAD_SLOT].double()
        keys = k[0, :, :, :, HEAD_SLOT].reshape(945, 80, 128).double()
        values = v[0, :, :, :, HEAD_SLOT].reshape(945, 80, 128).double()
        seed_rows = table[0, 0, :, :, COLUMN, HEAD_SLOT].reshape(945, 4).long()
        seed_query = query[seed_rows]
        true_logits = torch.bmm(seed_query, keys.transpose(1, 2)) * SCALE
        ideal = stable_merge(torch, true_logits, keys, values)
        fixed_owner_true = stable_merge(torch, true_logits, keys, values, owners=trace["owner"])
        # Freeze the actual native normalized log profile AND ownership.
        # logp2 includes the native FP32 subtraction, unlike fresh true logits.
        stable = stable_merge(torch, trace["logp2"].double() * 0.6931471805599453,
                              keys, values, owners=trace["owner"])
        sqk, sqv = keys[:, COLUMN], values[:, COLUMN]
        ideal_route = route64(torch, query, ideal["k"], ideal["v"], ideal["bias"], sqk, sqv)
        # Use the old block decomposition, independently of compressed routing.
        dense_logits = torch.einsum("qd,bkd->qbk", query, keys) * SCALE
        block_dense = torch.einsum("qbk,bkd->qbd", dense_logits.softmax(-1), values)
        dense = torch.einsum("qb,qbd->qd", dense_logits.logsumexp(-1).softmax(-1), block_dense)
        report["gates"].update(old_ideal_fp64=bool(torch.allclose(ideal_route["output"], cached["ideal"], atol=1e-10, rtol=1e-9)),
            old_dense_fp64=bool(torch.allclose(dense, cached["dense"], atol=1e-10, rtol=1e-9)))
        report["old_fp64_replay"] = dict(ideal=finite_stats(torch, ideal_route["output"], cached["ideal"]),
                                         dense=finite_stats(torch, dense, cached["dense"]))
        write_json(args.output / "results.json", report)
        require(report["gates"]["old_ideal_fp64"] and report["gates"]["old_dense_fp64"], "old FP64 replay failed")
        native_k, native_v, native_bias = buffers
        native_valid = torch.isfinite(native_bias)
        require(bool(torch.equal(native_valid, trace["bulk_valid"].bool())), "trace/final validity mismatch")
        native_route = route64(torch, query, *buffers, sqk, sqv)
        stable_route = route64(torch, query, stable["k"], stable["v"], stable["bias"], sqk, sqv)
        fixed_true_route = route64(torch, query, fixed_owner_true["k"], fixed_owner_true["v"], fixed_owner_true["bias"], sqk, sqv)
        masked_weight = trace["pre_weights"].double().clone()
        masked_weight[..., COLUMN] = 0
        actual_direct_residual = masked_weight.sum(-1)
        safe_r = torch.where(native_valid, trace["residual"], 1).double()
        rounded_tau = masked_weight.to(torch.bfloat16).double().sum(-1) / safe_r
        alpha = trace["alpha"].double()
        alpha_log_alpha = alpha * alpha.clamp_min(torch.finfo(torch.float64).tiny).log()
        peeled_beta_from_native_prestate = ((trace["pre_bias"].double() - alpha_log_alpha) / safe_r
                                            - safe_r.log()).masked_fill(~native_valid, torch.inf)
        report["stage_comparison"] = dict(
            owners_native_vs_true_mismatches=int((trace["owner"] != ideal["owners"]).sum()),
            logits_native_base2_to_true=finite_stats(torch, trace["logits2"].double() * 0.6931471805599453, true_logits),
            pre_weights_native_vs_stable=finite_stats(torch, trace["pre_weights"], stable["pre_weights"]),
            pre_bias_native_vs_stable=finite_stats(torch, trace["pre_bias"], stable["pre_beta"]),
            residual_subtracted_vs_observed_direct=finite_stats(torch, trace["residual"], actual_direct_residual),
            bias_native_vs_stable=finite_stats(torch, native_bias, stable["bias"]),
            bias_native_vs_fp64_peel_of_observed_prestate=finite_stats(torch, native_bias, peeled_beta_from_native_prestate),
            observed_prestate_fp64_peel_vs_stable=finite_stats(torch, peeled_beta_from_native_prestate, stable["bias"]),
            key_native_vs_stable=finite_stats(torch, native_k, stable["k"]),
            value_native_vs_stable=finite_stats(torch, native_v, stable["v"]),
            native_valid_cells=int(native_valid.sum()), structural_valid_cells=int(stable["valid"].sum()),
            structural_nonempty_dropped=int((stable["valid"] & ~native_valid).sum()),
            native_valid_structurally_empty=int((native_valid & ~stable["valid"]).sum()),
            native_valid_tau_max_abs_from_one=float((rounded_tau[native_valid] - 1).abs().max()))
        write_json(args.output / "results.json", report)
        require(report["stage_comparison"]["native_valid_structurally_empty"] == 0,
                "native-valid structurally empty cell: fixed-mask counterfactual not defined; stop without changing mask")
        entropy_upper_violation = native_bias.double().clamp_min(0).masked_fill(~native_valid, 0)
        entropy_lower_violation = (-stable["bulk_count"].clamp_min(1).double().log() - native_bias.double()).clamp_min(0).masked_fill(~native_valid, 0)
        own_native_logit = (seed_query * native_k.double()).sum(-1) * SCALE - native_bias.double()
        own_stable_logit = (seed_query * fixed_owner_true["k"]).sum(-1) * SCALE - fixed_owner_true["bias"]
        report["physical_diagnostics"] = dict(
            note="Raw violations, not hard gates: BF16 KV storage/dot and FP32 bias have rounding error.",
            beta_above_zero_max=float(entropy_upper_violation.max()),
            beta_below_negative_log_bulk_count_max=float(entropy_lower_violation.max()),
            seed_cell_lse_native=finite_stats(torch, own_native_logit, fixed_owner_true["cell_logz"]),
            seed_cell_lse_fp64=finite_stats(torch, own_stable_logit, fixed_owner_true["cell_logz"]))
        key_hull, key_hull_raw, key_hull_scaled = hull_diagnostics(torch, native_k, keys, stable["mask"], native_valid, rounded_tau)
        value_hull, value_hull_raw, value_hull_scaled = hull_diagnostics(torch, native_v, values, stable["mask"], native_valid, rounded_tau)
        report["physical_diagnostics"].update(key_hull=key_hull, value_hull=value_hull)
        outputs = dict(native=native.double(), dense=dense, ideal=ideal_route["output"],
            native_KV_bias_fp64_route=native_route["output"],
            stable_native_profile_structural_fp64_route=stable_route["output"],
            true_logits_native_owner_structural_fp64_route=fixed_true_route["output"])
        counter_buffers = (
            (native_k, native_v, stable["bias"].masked_fill(~native_valid, torch.inf)),
            (stable["k"], stable["v"], native_bias),
            (stable["k"], stable["v"], stable["bias"].masked_fill(~native_valid, torch.inf)),
            (stable["k"], stable["v"], stable["bias"]),
        )
        for name, replacement in zip(COUNTERFACTUALS, counter_buffers):
            replacement = (replacement[0].to(torch.bfloat16), replacement[1].to(torch.bfloat16), replacement[2].float())
            result, actual_buffers, actual_flags = native_call(torch, q, k, v, table, hard, singleton, replacement=replacement)
            require(actual_flags == flags and all(bitwise_equal(torch, a, b) for a, b in zip(actual_buffers, buffers)),
                    "counterfactual changed original merge before substitution")
            outputs[name] = result.double()
            report["counterfactuals"][name] = dict(mse_vs_dense=mse(torch, result, dense),
                mse_vs_ideal=mse(torch, result, ideal_route["output"]),
                row9_mse_vs_dense=mse(torch, result[9], dense[9]), row8_mse_vs_dense=mse(torch, result[8], dense[8]),
                original_merge_bitwise=True, native_route_geometry_unchanged=True)
            report["counterfactuals"][name]["substitution_bitwise"] = True
            print(name, report["counterfactuals"][name], flush=True)
        report["output_comparison"] = {name: dict(mse_vs_dense=mse(torch, value, dense),
            mse_vs_ideal=mse(torch, value, ideal_route["output"]),
            per_row_mse_vs_dense=(value - dense).square().mean(-1).cpu().tolist(),
            per_row_mse_vs_ideal=(value - ideal_route["output"]).square().mean(-1).cpu().tolist())
            for name, value in outputs.items()}
        report["top_witness_tokens"] = {str(index): top_witness_tokens(torch, index, native_route,
            ideal_route, buffers, stable, trace, seed_rows) for index in (9, 8)}
        for index in (9, 8):
            for item in report["top_witness_tokens"][str(index)]:
                block, cell = item["block"], item["cell"]
                if cell == 4:
                    continue
                cell_lse = dense_logits[index, block].masked_fill(~stable["mask"][block, cell], -torch.inf).logsumexp(-1)
                actual_logit = native_route["logits"][index, block, cell]
                item.update(native_owner_cell_lse=float(cell_lse) if bool(torch.isfinite(cell_lse)) else None,
                    native_cell_lse_upper_violation=float(actual_logit - cell_lse)
                        if bool(torch.isfinite(actual_logit) & torch.isfinite(cell_lse)) else None,
                    ideal_true_bulk_count=int(ideal["bulk_count"][block, cell]),
                    ideal_true_bias=float(ideal["bias"][block, cell]) if bool(ideal["valid"][block, cell]) else None,
                    direct_residual_observed_weights=float(actual_direct_residual[block, cell]),
                    rounded_weight_mass_tau=float(rounded_tau[block, cell]),
                    key_convex_hull_violation=float(key_hull_raw[block, cell]),
                    value_convex_hull_violation=float(value_hull_raw[block, cell]))
        report["row_summary"] = {str(index): dict(dense_l2=float(dense[index].norm()),
            native_l2=float(native[index].double().norm()),
            native_singleton_mass=float(native_route["weights"][index, :, 4].sum()),
            ideal_singleton_mass=float(ideal_route["weights"][index, :, 4].sum())) for index in (9, 8)}
        arrays = {"trace_" + name: value.cpu().numpy() for name, value in trace.items()}
        arrays.update({"output_" + name: value.cpu().numpy() for name, value in outputs.items()})
        arrays.update(seed_rows=seed_rows.cpu().numpy(), native_k=native_k.float().cpu().numpy(),
            native_v=native_v.float().cpu().numpy(), native_bias=native_bias.cpu().numpy(),
            actual_direct_residual=actual_direct_residual.cpu().numpy(), rounded_weight_mass_tau=rounded_tau.cpu().numpy(),
            peeled_beta_from_native_prestate_fp64=peeled_beta_from_native_prestate.cpu().numpy(),
            key_hull_raw=key_hull_raw.cpu().numpy(), key_hull_scaled=key_hull_scaled.cpu().numpy(),
            value_hull_raw=value_hull_raw.cpu().numpy(), value_hull_scaled=value_hull_scaled.cpu().numpy(),
            native_route_logits=native_route["logits"].cpu().numpy(), native_route_weights=native_route["weights"].cpu().numpy(),
            ideal_route_logits=ideal_route["logits"].cpu().numpy(), ideal_route_weights=ideal_route["weights"].cpu().numpy(),
            seed_cell_lse_native=own_native_logit.cpu().numpy(), seed_cell_lse_true=fixed_owner_true["cell_logz"].cpu().numpy())
        for name in ("owners", "pre_weights", "pre_beta", "alpha", "residual_direct", "bulk_count", "bias", "k", "v", "weights"):
            arrays["stable_observed_profile_" + name] = stable[name].cpu().numpy()
        for name in ("owners", "bias", "k", "v"):
            arrays["ideal_true_" + name] = ideal[name].cpu().numpy()
        arrays_path = args.output / "witness_arrays.npz"
        np.savez_compressed(arrays_path, **arrays)
        report["arrays"] = dict(path=str(arrays_path), bytes=arrays_path.stat().st_size, sha256=digest(arrays_path))
        report["gates"]["original_inputs_bitwise"] = all(bitwise_equal(torch, a, b) for a, b in zip((q, k, v), originals)) and bool(torch.equal(table, seeds_original))
    check_inputs(row)
    report["gates"]["inputs_unchanged"] = True
    verify_sources(args.output, hashes)
    report["gates"]["sources_unchanged"] = True
    old_sources = old.get("sources", {})
    report["source_changes_from_old_b"] = {name: dict(old_sha256=receipt["sha256"], current_sha256=hashes[name])
        for name, receipt in old_sources.items() if name in hashes and receipt["sha256"] != hashes[name]}
    report.update(valid=True, complete=True)
    validate_completed_result(report)
    write_json(args.output / "results.json", report)
    print("Complete fixed-seed witness diagnosis; no video or performance conclusion.", flush=True)


def parent(args):
    from analysis import launch_quality_first_videos as guard
    require(not args.output.exists(), "fresh output directory required; no resume/overwrite")
    initial = guard.require_clean(args.gpu)
    old, row, _ = input_contract(args.old_root)
    check_inputs(row)
    hashes = source_hashes()
    args.output.mkdir(parents=True)
    for name, sha in hashes.items():
        target = args.output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
        require(digest(target) == sha, "source copy mismatch")
    verify_sources(args.output, hashes)
    process, foreign, failure, status = None, [], None, None
    polls, artifacts, previous = 0, None, {}
    started = guard.now()

    def interrupted(number, _frame):
        raise KeyboardInterrupt(f"received signal {number}")

    try:
        previous = {number: signal.signal(number, interrupted) for number in (signal.SIGTERM, signal.SIGINT)}
        immediate = guard.require_clean(args.gpu, expected_uuid=initial["gpu"]["uuid"])
        write_json(args.output / "startup.json", dict(parent_pid=os.getpid(), initial=initial,
            immediately_before_spawn=immediate, source_sha256=hashes, monitor_interval_seconds=5,
            source_root=str(ROOT), old_result_sha256=OLD_RESULT_SHA,
            parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
        environment = guard.child_environment(args.output, initial["gpu"]["uuid"])
        environment["SQ1_WITNESS_PARENT"] = str(os.getpid())
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--gpu", str(args.gpu),
                   "--output", str(args.output), "--old-root", str(args.old_root)]
        with (args.output / "worker.log").open("x", encoding="utf-8") as log, \
             (args.output / "gpu_monitor.jsonl").open("x", encoding="utf-8") as monitor:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write_json(args.output / "process.json", dict(parent_pid=os.getpid(), worker_pid=process.pid,
                command=command, cwd=str(ROOT), spawned_utc=guard.now(), cuda_visible_devices=environment["CUDA_VISIBLE_DEVICES"]))
            while True:
                active = guard.gpu_processes(initial["gpu"]["uuid"])
                polls += 1
                monitor.write(json.dumps(dict(utc=guard.now(), compute_processes=active)) + "\n")
                monitor.flush()
                foreign = [record for record in active if record["pid"] != process.pid]
                require(not foreign, "foreign process detected; stop only own child")
                if process.poll() is not None:
                    break
                time.sleep(5)
            status = process.wait(timeout=15)
            require(status == 0, f"worker exit {status}")
            report = json.loads((args.output / "results.json").read_text())
            validate_completed_result(report)
            verify_sources(args.output, hashes)
            require(digest(args.output / "witness_arrays.npz") == report["arrays"]["sha256"], "output arrays changed")
            artifacts = {name: digest(args.output / name) for name in ("results.json", "witness_arrays.npz", "observer_kernel.py")}
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        try:
            guard.stop_own_child(process)
            if process is not None:
                status = process.returncode
        except BaseException as error:
            failure = (failure + "; " if failure else "") + f"cleanup: {error}"
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
        completion = dict(valid=failure is None and status == 0 and artifacts is not None and not foreign,
            started_utc=started, completed_utc=guard.now(), worker_exit_code=status,
            parent_pid=os.getpid(), worker_pid=process.pid if process else None,
            requested_gpu=args.gpu, gpu_uuid=initial["gpu"]["uuid"], foreign=foreign,
            failure=failure, monitor_interval_seconds=5, monitor_polls=polls, verified_artifacts=artifacts,
            timing_benchmark=False, video_quality_result=False)
        write_json(args.output / "completion.json", completion)
    require(completion["valid"], f"witness diagnostic invalid: {failure}; inspect worker.log")
    print(json.dumps(completion, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    (worker if args.worker else parent)(args)


if __name__ == "__main__":
    main()
