"""Guard-owned, OFF-PATH stable SQ1 witness control; never launch a GPU here.

Accept only a completed primary observer diagnosis with all its hard gates.
Replay its saved real seeds and old native output before a generated stable
merge. The stable merge's own three-array observer must preserve its output
and KV/bias bitwise. FP64 references use its ACTUAL raw scores and owners;
old-prefix differences are reported, not assumed absent or rejected.
Structural nonempty validity replaces the old residual cutoff explicitly.
All 945 blocks/45 witness rows retain the original native route geometry.
No selector, video integration, quality result, or speed claim is provided.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np

from analysis import diagnose_sq1_fixed_seed_witness as base
from analysis import stable_sq1_bulk_kernel as generator


TRACE_NAMES = ("logits2", "logp2", "owner")
OBSERVER_KERNEL = "_stable_sq1_observer_kernel"
ARTIFACTS = ("results.json", "stable_kernel.py", "stable_observer_kernel.py", "control_arrays.npz")
REQUIRED_GATES = (
    "primary_complete", "old_native_bitwise", "primary_buffers_bitwise",
    "stable_observer_output_bitwise", "stable_observer_buffers_bitwise",
    "stable_trace_complete", "stable_structural_mask_exact", "native_route_geometry_unchanged",
    "reference_substitution_bitwise", "old_mask_substitution_bitwise",
    "original_inputs_bitwise", "sources_unchanged", "inputs_unchanged", "primary_unchanged",
)


def _receipt(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=base.digest(path))


def read_completed_primary(primary, old_root=base.OLD_ROOT):
    """Read/hash receipts only, before CUDA imports; no prospective run pin."""
    primary = Path(primary).resolve(strict=True)
    report = json.loads((primary / "results.json").read_text())
    completion = json.loads((primary / "completion.json").read_text())
    base.validate_completed_result(report)
    base.require(completion.get("valid") is True and completion.get("worker_exit_code") == 0
                 and completion.get("foreign") == [] and completion.get("failure") is None,
                 "primary parent did not complete cleanly")
    verified = completion.get("verified_artifacts", {})
    names = ("results.json", "witness_arrays.npz", "observer_kernel.py")
    base.require(set(verified) == set(names), "primary artifact coverage changed")
    receipts = {name: _receipt(primary / name) for name in names}
    base.require(all(receipts[name]["sha256"] == verified[name] for name in names),
                 "primary checked artifact changed")
    base.require(report["arrays"]["sha256"] == receipts["witness_arrays.npz"]["sha256"]
                 and report["arrays"]["bytes"] == receipts["witness_arrays.npz"]["bytes"]
                 and report["observer_source_sha256"] == receipts["observer_kernel.py"]["sha256"],
                 "primary worker artifact receipt mismatch")
    _, row, _ = base.input_contract(old_root)
    base.require(report["old_result_sha256"] == base.OLD_RESULT_SHA
                 and report["metadata"] == row["metadata"]
                 and report["inputs"] == {key: row[key] for key in ("input", "seed_cache", "output_cache")}
                 and report["merge_options"] == base.MERGE_OPTIONS, "primary witness identity changed")
    base.check_inputs(row)
    current = base.source_hashes()
    base.require(current == report["source_sha256"], "primary merge/route/diagnostic sources changed")
    for name, digest in current.items():
        base.require(base.digest(primary / "sources" / name) == digest, "primary archived source changed")
    receipts["completion.json"] = _receipt(primary / "completion.json")
    return dict(directory=str(primary), report=report, row=row, receipts=receipts)


def make_stable_observer_source(source):
    """Add stores only: ordinary trailing pointer args and explicit J-shaped offsets."""
    tree = ast.parse(source)
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == generator.GENERATED_KERNEL]
    base.require(len(matches) == 1, "expected one generated stable kernel")
    function = matches[0]
    base.require(not any(isinstance(n, ast.Name) and n.id.startswith("witness_")
                         for n in ast.walk(function)), "reserved observer names already present")
    function.name = OBSERVER_KERNEL
    for name in TRACE_NAMES:
        function.args.args.append(ast.arg(arg="witness_" + name + "_ptr"))
        function.args.defaults.append(ast.Constant(value=None))
    prefix = """
witness_block = key_outer * ROWS + key_row
witness_blocks = witness_block + tl.zeros((BLOCK_J,), tl.int32)
witness_columns = query_col_start + tl.arange(0, BLOCK_J)
witness_match = (witness_columns == 40) & (query_outer == 0)
witness_cells = tl.arange(0, C)
"""
    def vector(pointer, main, tail):
        return f"""
tl.store(witness_{pointer}_ptr + ((witness_blocks[:, None, None] * C + witness_cells[None, :, None]) * COLS + main_cols[None, None, :]), {main}, mask=witness_match[:, None, None])
tl.store(witness_{pointer}_ptr + ((witness_blocks[:, None, None] * C + witness_cells[None, :, None]) * COLS + tail_cols[None, None, :]), {tail}, mask=witness_match[:, None, None])
"""
    additions = {
        "log_probs_tail": prefix + vector("logits2", "tl.reshape(scores_main, (BLOCK_J, C, 64))", "tl.reshape(scores_tail, (BLOCK_J, C, 16))")
            + vector("logp2", "log_probs_main", "log_probs_tail"),
        "winner_tail, route_tail": """
tl.store(witness_owner_ptr + witness_blocks[:, None] * COLS + main_cols[None, :], route_main, mask=witness_match[:, None])
tl.store(witness_owner_ptr + witness_blocks[:, None] * COLS + tail_cols[None, :], route_tail, mask=witness_match[:, None])
""",
    }
    counts = dict.fromkeys(additions, 0)
    class Insert(ast.NodeTransformer):
        def visit_Assign(self, node):
            name = ast.unparse(node.targets[0]).strip("()")
            if name not in additions:
                return node
            counts[name] += 1
            return [node, *ast.parse(additions[name]).body]
    Insert().visit(function)
    base.require(all(n == 1 for n in counts.values()), "stable observer insertion anchors changed")
    tree.body = [function]
    ast.fix_missing_locations(tree)
    return "import triton\nimport triton.language as tl\n\n" + ast.unparse(tree) + "\n"


class ReviewedStableKernel:
    """Validate every actual constexpr, then forward the exact launch unchanged."""
    def __init__(self, kernel, original_function):
        self.kernel, self.signature = kernel, inspect.signature(original_function)
        self.calls = 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls += 1
            base.require(self.calls == 1, "stable kernel repeated launch")
            bound = self.signature.bind_partial(*args, **{k: v for k, v in kwargs.items()
                                                         if k in self.signature.parameters})
            bound.apply_defaults()
            generator.validate_launch_contract(bound.arguments)
            return self.kernel[grid](*args, **kwargs)
        return launch


def direct_bulk_reference_numpy(logits2, owners, keys, values, singleton=40):
    """Independent FP64 direct bulk conditioning on observed base-2 scores."""
    raw = np.asarray(logits2)
    owner = np.asarray(owners)
    k, v = np.asarray(keys, dtype=np.float64), np.asarray(values, dtype=np.float64)
    base.require(raw.ndim == 3 and raw.shape[1:] == (4, 80), "raw scores must be [banks,4,80]")
    base.require(owner.dtype.kind in "iu" and owner.shape == (len(raw), 80)
                 and np.all((owner >= 0) & (owner < 4)), "invalid observed owners")
    base.require(k.ndim == 3 and k.shape[:2] == (len(raw), 80) and v.shape == k.shape,
                 "reference K/V geometry changed")
    base.require(isinstance(singleton, int) and not isinstance(singleton, bool) and 0 <= singleton < 80,
                 "invalid singleton")
    base.require(np.isfinite(raw).all() and np.isfinite(k).all() and np.isfinite(v).all(), "nonfinite reference input")
    scores = raw.astype(np.float64)
    weights = np.zeros(scores.shape, dtype=np.float64)
    bias = np.full(scores.shape[:2], np.inf)
    count = np.zeros(scores.shape[:2], dtype=np.int64)
    for b in range(len(scores)):
        for c in range(4):
            selected = np.flatnonzero((owner[b] == c) & (np.arange(80) != singleton))
            count[b, c] = len(selected)
            if not len(selected):
                continue
            centered = scores[b, c, selected] - scores[b, c, selected].max()
            exponential = np.exp2(centered)
            w = exponential / exponential.sum()
            weights[b, c, selected] = w
            logw = centered * np.log(2.0) - np.log(exponential.sum())
            bias[b, c] = np.sum(w * logw)
    return dict(weights=weights, bias=bias, valid=count > 0, bulk_count=count,
                k=weights @ k, v=weights @ v)


def route_reference_numpy(query, keys, values, bias, singleton_k, singleton_v, scale=base.SCALE):
    """One global FP64 route over every supplied C4+SQ1 block, not top-k."""
    q = np.asarray(query, dtype=np.float64)
    k = np.concatenate((np.asarray(keys, dtype=np.float64), np.asarray(singleton_k)[:, None]), axis=1)
    v = np.concatenate((np.asarray(values, dtype=np.float64), np.asarray(singleton_v)[:, None]), axis=1)
    b = np.concatenate((np.asarray(bias, dtype=np.float64), np.zeros((len(k), 1))), axis=1)
    logits = q @ k.reshape(-1, k.shape[-1]).T * scale - b.reshape(-1)
    centered = logits - logits.max(axis=-1, keepdims=True)
    probability = np.exp(centered)
    probability /= probability.sum(axis=-1, keepdims=True)
    return probability @ v.reshape(-1, v.shape[-1])


def compare_numpy(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    base.require(a.shape == b.shape, "comparison shape mismatch")
    finite = np.isfinite(a) & np.isfinite(b)
    error = np.abs(a.astype(np.float64)[finite] - b.astype(np.float64)[finite])
    return dict(values=a.size, finite_pairs=int(finite.sum()),
                finite_mask_mismatches=int(np.count_nonzero(np.isfinite(a) != np.isfinite(b))),
                max_abs_error=float(error.max()) if error.size else None,
                mean_abs_error=float(error.mean()) if error.size else None)


def witness_errors_numpy(output, dense, ideal):
    """Errors against the SAME cached 45-row dense/ideal witness; no recomputation."""
    arrays = [np.asarray(value, dtype=np.float64) for value in (output, dense, ideal)]
    base.require(all(value.shape == (45, 128) and np.isfinite(value).all() for value in arrays),
                 "cached/output witness must be finite [45,128]")
    answer = {}
    for name, reference in zip(("dense", "ideal"), arrays[1:]):
        per_row = np.mean(np.square(arrays[0] - reference), axis=-1)
        answer["mse_vs_" + name] = float(per_row.mean())
        answer["per_row_mse_vs_" + name] = per_row.tolist()
        for row in (8, 9):
            answer[f"row{row}_mse_vs_{name}"] = float(per_row[row])
    return answer


def _load_kernel(path, source, name):
    Path(path).write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, name)


def validate_completed_result(report):
    base.require(report.get("valid") is True and report.get("complete") is True, "control incomplete")
    base.require(report.get("candidate") is False and report.get("timing_benchmark") is False
                 and report.get("video_quality_result") is False, "off-path scope changed")
    base.require(report.get("key_blocks") == 945 and report.get("query_rows") == 45, "witness truncated")
    base.require(all(report.get("gates", {}).get(key) is True for key in REQUIRED_GATES), "missing stable control gate")
    base.require(report.get("native_calls") == 5, "unexpected native replay coverage")
    base.require(set(report.get("generated_sources", {})) == {"stable_kernel.py", "stable_observer_kernel.py"},
                 "missing generated source receipts")
    base.require(report.get("prefix_comparison", {}).get("owner_mismatches") is not None
                 and report.get("structural_mask", {}).get("revived_vs_old") is not None, "missing semantic differences")


def worker(primary: Path, output: Path, old_root: Path = base.OLD_ROOT):
    """Owned child only. The separate parent handles GPU eligibility/monitoring."""
    base.require(os.environ.get("STABLE_SQ1_WITNESS_PARENT") == str(os.getppid()), "guarded live parent required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    base.require(visible.startswith("GPU-") and "," not in visible, "one UUID-visible GPU required")
    primary, output = Path(primary), Path(output)
    checked = read_completed_primary(primary, old_root)
    base.require(output.is_dir() and not any((output / name).exists() for name in ARTIFACTS), "fresh parent output required")
    hashes = base.source_hashes()
    hashes.update({name: base.digest(base.ROOT / name) for name in (
        "analysis/stable_sq1_bulk_kernel.py", "analysis/validate_stable_sq1_witness.py")})
    report = dict(valid=False, complete=False, candidate=False, timing_benchmark=False, video_quality_result=False,
                  key_blocks=945, query_rows=45, source_sha256=hashes,
                  primary=dict(directory=checked["directory"], receipts=checked["receipts"]),
                  inputs=checked["report"]["inputs"], gates={"primary_complete": True}, native_calls=0,
                  limitations=["One off-path fixed-real-seed witness; no video or latency conclusion.",
                    "Structural nonempty mask differs from the old residual cutoff.",
                    "KV uses BF16 weight/store rounding; numerical errors are reported, not waived by fitted tolerances.",
                    "New score/owner values need not be bitwise identical to old compilation; both are retained."])
    base.write_json(output / "results.json", report)
    import torch
    from attention import hard_c4_wan21_14b_720p_triton as hard
    from attention import singleton_c4_wan21_14b_720p_triton as singleton_module
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    row = checked["row"]
    saved = torch.load(row["input"]["path"], map_location="cpu", weights_only=True)
    expected_meta = {k: v for k, v in row["metadata"].items() if k not in ("bytes", "sha256")}
    base.require(set(saved) == {"q", "k", "v", "dense_output", "metadata"}
                 and saved["metadata"] == expected_meta, "saved original metadata changed")
    base.require(all(t.dtype == torch.bfloat16 and tuple(t.shape) == (1, 75600, 4, 128)
                     and bool(torch.isfinite(t).all()) for n, t in saved.items() if n != "metadata"), "original BF16 inputs required")
    with np.load(row["seed_cache"]["path"], allow_pickle=False) as archive:
        seed_array = archive[base.METHOD].copy()
    base.require(seed_array.shape == (1, 1, 21, 45, 80, 4, 4) and seed_array.dtype == np.uint8
                 and np.all(seed_array < 45) and np.all(np.diff(np.sort(seed_array, axis=-1), axis=-1) != 0), "saved real seed table changed")
    with np.load(row["output_cache"]["path"], allow_pickle=False) as archive:
        old_output = archive["h30_c40_" + base.METHOD + "_bf16"].copy()
        cached_dense = archive["h30_c40_dense64"].copy()
        cached_ideal = archive["h30_c40_" + base.METHOD + "_ideal"].copy()
    base.require(old_output.dtype == cached_dense.dtype == cached_ideal.dtype == np.float64,
                 "old output cache dtype changed")
    witness_errors_numpy(old_output, cached_dense, cached_ideal)
    with np.load(primary / "witness_arrays.npz", allow_pickle=False) as archive:
        old_arrays = {name: archive[name].copy() for name in (
            "trace_logits2", "trace_logp2", "trace_owner", "native_k", "native_v", "native_bias", "seed_rows")}
    def cpu(t):
        return t.detach().double().cpu().numpy()
    def native(*args, **kwargs):
        report["native_calls"] += 1
        return base.native_call(torch, *args, **kwargs)
    with torch.inference_mode():
        q, k, v = [saved[name].to("cuda:0").reshape(base.GRID) for name in ("q", "k", "v")]
        originals = [t.clone() for t in (q, k, v)]
        table = torch.from_numpy(seed_array).to("cuda:0")
        table_original = table.clone()
        args = (q, k, v, table, hard, singleton_module)
        old, old_buffers, flags = native(*args)
        base.require_old_replay(torch, old, torch.from_numpy(old_output).to(q.device))
        report["gates"]["old_native_bitwise"] = True
        for tensor, name in zip(old_buffers, ("native_k", "native_v", "native_bias")):
            stored = torch.from_numpy(old_arrays[name]).to(q.device, dtype=tensor.dtype)
            base.require(np.array_equal(cpu(stored), old_arrays[name].astype(np.float64))
                         and base.bitwise_equal(torch, tensor, stored), "primary native buffer replay changed")
        base.require(flags == checked["report"]["actual_merge_launch"], "old launch differs from primary")
        report["gates"]["primary_buffers_bitwise"] = True
        seed_rows = table[0, 0, :, :, base.COLUMN, base.HEAD_SLOT].reshape(945, 4)
        base.require(np.array_equal(seed_rows.cpu().numpy(), old_arrays["seed_rows"]), "witness rows changed")
        original_kernel = getattr(hard, base.KERNEL)
        # Triton's .src may omit @triton.jit. The pinned AST includes it.
        source = inspect.getsource(original_kernel.fn)
        stable_source = generator.generate_stable_sq1_source(source)
        stable_kernel = _load_kernel(output / "stable_kernel.py", stable_source, generator.GENERATED_KERNEL)
        stable, stable_buffers, stable_flags = native(*args,
            kernel=ReviewedStableKernel(stable_kernel, original_kernel.fn))
        observer_source = make_stable_observer_source(stable_source)
        observer_kernel = _load_kernel(output / "stable_observer_kernel.py", observer_source, OBSERVER_KERNEL)
        trace = {name: torch.full((945, 80) if name == "owner" else (945, 4, 80),
                                 -99 if name == "owner" else float("nan"), device=q.device,
                                 dtype=torch.int32 if name == "owner" else torch.float32) for name in TRACE_NAMES}
        observed, observed_buffers, observed_flags = native(*args,
            kernel=ReviewedStableKernel(observer_kernel, original_kernel.fn), traces=trace)
        report["gates"].update(stable_observer_output_bitwise=base.bitwise_equal(torch, observed, stable),
            stable_observer_buffers_bitwise=all(base.bitwise_equal(torch, a, b) for a, b in zip(observed_buffers, stable_buffers)))
        base.require(report["gates"]["stable_observer_output_bitwise"] and report["gates"]["stable_observer_buffers_bitwise"],
                     "stable observer changed arithmetic; do not interpret trace")
        base.require(bool(((trace["owner"] >= 0) & (trace["owner"] < 4)).all())
                     and all(bool(torch.isfinite(trace[name]).all()) for name in ("logits2", "logp2")), "incomplete stable trace")
        report["gates"]["stable_trace_complete"] = True
        report["prefix_comparison"] = {name: compare_numpy(cpu(trace[name]), old_arrays["trace_" + name]) for name in ("logits2", "logp2")}
        report["prefix_comparison"].update(owner_mismatches=int(np.count_nonzero(trace["owner"].cpu().numpy() != old_arrays["trace_owner"])),
            raw_scores_bitwise=bool(np.array_equal(trace["logits2"].cpu().numpy().view(np.uint32), old_arrays["trace_logits2"].view(np.uint32))),
            logp_bitwise=bool(np.array_equal(trace["logp2"].cpu().numpy().view(np.uint32), old_arrays["trace_logp2"].view(np.uint32))))
        keys, values = [cpu(t[0, :, :, :, base.HEAD_SLOT].reshape(945, 80, 128)) for t in (k, v)]
        query = cpu(q[0, 0, :, base.COLUMN, base.HEAD_SLOT])
        actual_reference = direct_bulk_reference_numpy(cpu(trace["logits2"]), trace["owner"].cpu().numpy(), keys, values)
        old_reference = direct_bulk_reference_numpy(old_arrays["trace_logits2"], old_arrays["trace_owner"], keys, values)
        actual = [cpu(t) for t in stable_buffers]
        old_valid, stable_valid = np.isfinite(cpu(old_buffers[2])), np.isfinite(actual[2])
        base.require(np.array_equal(stable_valid, actual_reference["valid"]), "stable structural mask mismatch")
        report["gates"]["stable_structural_mask_exact"] = True
        report["structural_mask"] = dict(old_valid=int(old_valid.sum()), stable_valid=int(stable_valid.sum()),
            revived_vs_old=int(np.count_nonzero(stable_valid & ~old_valid)),
            dropped_vs_old=int(np.count_nonzero(old_valid & ~stable_valid)),
            old_mask_control_uses_intersection=True,
            old_mask_control_equals_old_mask=bool(np.all(~old_valid | stable_valid)),
            old_vs_new_structural_differences=int(np.count_nonzero(old_reference["valid"] != actual_reference["valid"])))
        report["buffer_comparison"] = {name: compare_numpy(t, actual_reference[name]) for name, t in zip(("k", "v", "bias"), actual)}
        def substitute(buffers):
            result, before_buffers, launch_flags = native(*args, replacement=buffers)
            base.require(launch_flags == flags and all(base.bitwise_equal(torch, a, b)
                         for a, b in zip(before_buffers, old_buffers)), "substitution changed original merge")
            return result
        reference_buffers = tuple(torch.from_numpy(actual_reference[name]).to(q.device, dtype=dtype)
                                  for name, dtype in (("k", torch.bfloat16), ("v", torch.bfloat16), ("bias", torch.float32)))
        reference_native = substitute(reference_buffers)
        report["gates"]["reference_substitution_bitwise"] = True  # CaptureKernel verifies copies.
        old_mask_buffers = (stable_buffers[0], stable_buffers[1], stable_buffers[2].masked_fill(
            ~torch.from_numpy(old_valid).to(q.device), torch.inf))
        # If compiler-level prefix differences moved an old-valid cell to
        # structural emptiness, no stable representative exists there. This
        # control uses the INTERSECTION, never fabricates an old-mask token.
        old_mask_native = substitute(old_mask_buffers)
        report["gates"]["old_mask_substitution_bitwise"] = True
        report["gates"]["native_route_geometry_unchanged"] = flags == stable_flags == observed_flags
        sqk, sqv = keys[:, base.COLUMN], values[:, base.COLUMN]
        reference_route = route_reference_numpy(query, actual_reference["k"], actual_reference["v"], actual_reference["bias"], sqk, sqv)
        old_reference_route = route_reference_numpy(query, old_reference["k"], old_reference["v"], old_reference["bias"], sqk, sqv)
        stable_buffer_route = route_reference_numpy(query, *actual, sqk, sqv)
        outputs = dict(old_native=cpu(old), stable_native=cpu(stable), reference_native_route=cpu(reference_native),
                       stable_old_mask_native_route=cpu(old_mask_native), actual_prefix_reference_fp64_route=reference_route,
                       old_prefix_reference_fp64_route=old_reference_route, stable_buffers_fp64_route=stable_buffer_route)
        report["output_comparison"] = {name: compare_numpy(value, reference_route) for name, value in outputs.items()}
        report["cached_witness_errors"] = {name: witness_errors_numpy(value, cached_dense, cached_ideal)
                                           for name, value in outputs.items()}
        report["stable_vs_reference_native_route"] = compare_numpy(cpu(stable), cpu(reference_native))
        arrays = {"output_" + name: value for name, value in outputs.items()}
        arrays.update(cached_dense=cached_dense, cached_ideal=cached_ideal)
        arrays.update({"stable_trace_" + name: value.cpu().numpy() for name, value in trace.items()})
        for name, value in zip(("k", "v", "bias"), actual): arrays["stable_" + name] = value
        for label, reference in (("actual_prefix_reference", actual_reference), ("old_prefix_reference", old_reference)):
            arrays.update({label + "_" + name: reference[name] for name in ("weights", "k", "v", "bias", "bulk_count", "valid")})
        arrays.update(seed_rows=seed_rows.cpu().numpy(), old_valid=old_valid)
        np.savez_compressed(output / "control_arrays.npz", **arrays)
        report["arrays"] = _receipt(output / "control_arrays.npz")
        report["gates"]["original_inputs_bitwise"] = all(base.bitwise_equal(torch, a, b) for a, b in zip((q, k, v), originals)) and bool(torch.equal(table, table_original))
    base.check_inputs(row)
    report["gates"]["inputs_unchanged"] = True
    report["gates"]["primary_unchanged"] = read_completed_primary(primary, old_root)["receipts"] == checked["receipts"]
    report["gates"]["sources_unchanged"] = all(base.digest(base.ROOT / name) == digest for name, digest in hashes.items())
    report["generated_sources"] = {name: _receipt(output / name) for name in ("stable_kernel.py", "stable_observer_kernel.py")}
    report.update(valid=True, complete=True)
    validate_completed_result(report)
    base.write_json(output / "results.json", report)
    return report
