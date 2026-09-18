"""Frozen RF10/C16 partition-deficit screen on the two existing Wan captures.

Only six timed methods: RF10, mass C32/G45, mass C16/G16, partition
C16/G16 J2 or J4 (four warps, owner-logp reuse), and forced Flash SDPA.
The partition settings are BENCHMARK-ONLY overrides: production currently
passes eight warps and does not request owner-logp reuse. Every quality and
timed call uses the same override and recomputes the complete selector.
No cached tables enter timing; the original SQ1 merge/route is unchanged.

Run only after an external exclusive-H200 check. No GPU work is performed by
--cpu-selfcheck. Capture-local MSE is not video PSNR or a promotion decision.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "analysis"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

CAPTURE_ROOT = "/local2/denghaoran/GridMergeAttention/experiments/learned_seed_capture_20260822"
DEFAULT_CAPTURES = [f"{CAPTURE_ROOT}/call{call}_holdout/qkv.pt.npz" for call in ("020", "100")]
MODES = {
    "rf10": "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2",
    "mass_c32": "entropy_mass_span_c32_p16_g256_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2",
    "mass_c16": "entropy_mass_span_c16_p16_g256_j4_t1_u16_cg16_sq1_qm_fp8score_skv_nl_ml2",
    "partition_c16_j2": "entropy_partition_mass_c16_p16_g256_j2_t1_u16_cg16_sq1_qm_fp8score_skv_nl_ml2",
    "partition_c16_j4": "entropy_partition_mass_c16_p16_g256_j4_t1_u16_cg16_sq1_qm_fp8score_skv_nl_ml2",
}
PARITY_HEADS = [0, 10, 20, 30]
SEED_GATE = 3.98
SCALE = 128 ** -0.5


def _selector_options(name):
    if name not in MODES:
        raise ValueError(name)
    count = 32 if name in ("rf10", "mass_c32") else 16
    options = dict(candidate_count=count, candidate_geometry_count=45 if count == 32 else 16,
                   block_j=2 if name == "partition_c16_j2" else 4, gs_num_warps=4)
    if name != "rf10":
        options.update(probe_count=16, global_probe_count=256)
    if name.startswith("partition_"):
        options.update(reuse_owner_logp=True, num_stages=1)
    elif name.startswith("mass_"):
        options.update(weight_mode="mass", native_low_precision_dot=False,
                       tf32_low_precision_dot=False, singleton_complement=False)
    else:
        options.update(alpha=0.2, top_probes=0, uniform_probes=16,
                       probe_anchor_count=2, group_tile=16)
    return options


def _cpu_selfcheck():
    """Actual wrapper/bank dispatch plus C16 natural/base2 algebra, no GPU."""
    import ast
    import math
    from types import ModuleType, SimpleNamespace
    import numpy as np

    def function_ast(path, name):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        return ast.Module(body=[node], type_ignores=[])

    candidate_scope = {"torch": SimpleNamespace(device=str, int32=np.int32,
                       tensor=lambda data, **kw: np.array(data, dtype=kw["dtype"])),
                       "_MULTISCALE_TABLE_CACHE": {}}
    source = REPO_ROOT / "attention" / "adaptive_seed_pytorch.py"
    exec(compile(function_ast(source, "multiscale_candidate_table"), str(source), "exec"), candidate_scope)
    bank, anchors = candidate_scope["multiscale_candidate_table"](45, 16, "cpu", geometry_count=16)
    assert bank.shape == (45, 16) and (np.diff(bank, axis=1) > 0).all()
    assert np.array_equal(bank[np.arange(45), anchors], np.arange(45))

    class Tensor:
        def __init__(self, shape, dtype="bf16"):
            self.shape, self.dtype = tuple(shape), dtype
            self.ndim, self.device, self.is_cuda = len(shape), "metadata", True
        def size(self, dim):
            return self.shape[dim]
        def contiguous(self):
            return self
        def is_contiguous(self):
            return True

    launches, norm_calls = [], []
    class Kernel:
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                launches.append((grid, args, kwargs))
            return call

    package = ModuleType("_partition_c16_cpu")
    package.__path__ = []
    candidates = ModuleType(package.__name__ + ".adaptive_seed_pytorch")
    candidates.multiscale_candidate_table = candidate_scope["multiscale_candidate_table"]
    normalizer = ModuleType(package.__name__ + ".entropy_mass_seed_triton")
    def norm(q, k, **kwargs):
        norm_calls.append(kwargs)
        return Tensor(q.shape[:-1], "fp32")
    normalizer.global_probe_log_norm_triton = norm
    scope = {"__package__": package.__name__, "math": math,
             "torch": SimpleNamespace(Tensor=Tensor, bfloat16="bf16", float16="fp16", float32="fp32",
                                      uint8="u8", int32="i32", empty=lambda shape, **kw: Tensor(shape, kw["dtype"])),
             "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b,
                                       next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
             "_partition_mass_seed_kernel": Kernel()}
    source = REPO_ROOT / "attention" / "partition_mass_seed_triton.py"
    exec(compile(function_ast(source, "partition_mass_seed_rows_triton"), str(source), "exec"), scope)
    with patch.dict(sys.modules, {module.__name__: module for module in (package, candidates, normalizer)}):
        for name in ("partition_c16_j2", "partition_c16_j4"):
            output = scope["partition_mass_seed_rows_triton"](
                Tensor((1, 1, 45, 80, 40, 128)), Tensor((1, 21, 45, 80, 40, 128)),
                SCALE, **_selector_options(name))
            assert output.shape == (1, 1, 21, 45, 80, 40, 4)
            grid, _, options = launches[-1]
            j = _selector_options(name)["block_j"]
            assert grid == (45 * (80 // j), 40)
            assert (options["CANDIDATES"], options["BLOCK_ROWS"], options["BLOCK_J"], options["num_warps"]) == (16, 16, j, 4)
            assert options["REUSE_OWNER_LOGP"] is True
            assert norm_calls[-1] == {"scale": SCALE, "global_probe_count": 256}

    def select(logits, log_mass, base2):
        shifted = logits - logits.max(-1, keepdims=True)
        if base2:
            shifted /= np.log(2)
            logp = shifted - np.log2(np.exp2(shifted).sum(-1, keepdims=True))
        else:
            logp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
        batch = np.arange(logits.shape[0])
        first = np.array([0, 3, 8, 15])
        chosen = first[:, None]
        mass = np.exp(log_mass - log_mass.max(-1, keepdims=True))
        empty = 0
        for _ in range(3):
            profiles = logp[batch[:, None], chosen]
            owner = profiles.argmax(1)
            represented = np.zeros(logits.shape[:-1])
            bounds = []
            for cell_id in range(chosen.shape[1]):
                cell = owner == cell_id
                active = cell.any(-1)
                empty += int((~active).sum())
                values = profiles[:, cell_id]
                maximum = np.where(cell, values, -np.inf).max(-1)
                maximum = np.where(active, maximum, 0)
                exponential = np.where(cell, (np.exp2 if base2 else np.exp)(values - maximum[:, None]), 0)
                den = np.where(active, exponential.sum(-1), 1)
                weight = exponential / den[:, None]
                logw = np.where(cell, values - maximum[:, None]
                                - (np.log2 if base2 else np.log)(den)[:, None], 0)
                bound = (logp * weight[:, None]).sum(-1) - (weight * logw).sum(-1)[:, None]
                represented += np.where(active[:, None], (np.exp2 if base2 else np.exp)(bound), 0)
                bounds.append(np.where(active[:, None], bound, -np.inf))
            if base2:
                deficit = np.maximum(1 - represented, 0)
            else:
                # Independent stable logsumexp/-expm1 form used by Torch.
                bound = np.stack(bounds, -1)
                maximum = bound.max(-1)
                gap = np.maximum(-(maximum + np.log(np.exp(bound - maximum[..., None]).sum(-1))), 0)
                deficit = -np.expm1(-gap)
            assert np.isfinite(deficit).all()
            score = mass * deficit
            score[batch[:, None], chosen] = -1
            chosen = np.concatenate((chosen, score.argmax(-1)[:, None]), -1)
        assert (np.diff(np.sort(chosen, axis=-1), axis=-1) > 0).all()
        return chosen, empty

    index = np.arange(4 * 16 * 16).reshape(4, 16, 16)
    ordinary = 3 * np.sin(index / 17) + np.cos(index / 31)
    duplicate = ordinary.copy()
    duplicate[:, 1::2] = duplicate[:, ::2]
    fixtures = {"uniform": np.zeros_like(ordinary), "ordinary": ordinary, "duplicate": duplicate}
    empty = 0
    for logits in fixtures.values():
        mass = np.sin(np.arange(64).reshape(4, 16) / 9)
        natural, _ = select(logits, mass, False)
        native, count = select(logits, mass, True)
        assert np.array_equal(natural, native)
        empty += count
    assert empty > 0
    return {"passed": True, "candidate_banks_checked": 45, "actual_wrapper_launches": len(launches),
            "fixtures": list(fixtures), "empty_cells": empty,
            "scope": "CPU metadata and NumPy C16 algebra only; no Torch/Triton compilation or GPU claim"}


def _load_runtime():
    global torch, helper, mass_selector, partition_selector, rf10_selector, custom_tables
    import torch
    import benchmark_joint_frame_mass_seeds as helper
    from attention.entropy_mass_seed_triton import entropy_mass_seed_rows_triton as mass_selector
    from attention.partition_mass_seed_triton import partition_mass_seed_rows_triton as partition_selector
    from attention.adaptive_seed_triton import hellinger_span_seed_rows_triton as rf10_selector
    from screen_partition_population_seeds import custom_tables


def _configured_partition(name, *inputs, **kwargs):
    # Override explicit dispatcher values; functools.partial defaults would not.
    kwargs.update(_selector_options(name))
    return partition_selector(*inputs, **kwargs)


def _context(name):
    if name.startswith("partition_"):
        return patch("attention.token_merging.partition_mass_seed_rows_triton",
                     new=partial(_configured_partition, name))
    return helper._context(name)


def _native_table(name, query_grid, key_grid):
    selector = (partition_selector if name.startswith("partition_") else
                mass_selector if name.startswith("mass_") else rf10_selector)
    return selector(query_grid[:, 10:11].contiguous(), key_grid, SCALE,
                    output_dtype=torch.uint8, **_selector_options(name))


def _partition_reference(q_grid, k_grid, chunk):
    """Same existing natural-log deficit reference, only the bank becomes C16/G16."""
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    from partition_variational_selector import select_partition_variational_seeds
    heads, frames, rows, columns, dim = q_grid.shape
    candidates, anchors = multiscale_candidate_table(rows, 16, q_grid.device, geometry_count=16)
    block_rows = torch.arange(frames * rows, device=q_grid.device) % rows
    candidate_rows, anchor_rows = candidates[block_rows].long(), anchors[block_rows].long()
    probe_cols = torch.arange(16, device=q_grid.device) * (columns // 16)
    table = torch.empty((1, 1, frames, rows, columns, heads, 4), device=q_grid.device, dtype=torch.uint8)
    for head in range(heads):
        query = q_grid[head, 10].permute(1, 0, 2).float()
        key = k_grid[head].reshape(frames * rows, columns, dim).float()
        flat_key = key.flatten(0, 1)
        positions = torch.arange(256, device=q_grid.device) * (flat_key.size(0) // 256)
        normalizer = (torch.einsum("jrd,pd->jrp", query, flat_key[positions]) * SCALE).logsumexp(-1)
        flat_out = table[0, 0, :, :, :, head].reshape(frames * rows, columns, 4)
        for start in range(0, frames * rows, chunk):
            end = min(start + chunk, frames * rows)
            bank = candidate_rows[start:end]
            logits = torch.einsum("jbcd,bpd->jbcp", query[:, bank], key[start:end, probe_cols]) * SCALE
            mass = logits.logsumexp(-1) - normalizer[:, bank]
            selected = select_partition_variational_seeds(
                logits, anchor_rows[start:end][None].expand(columns, -1),
                variant="gap_deficit_weighted", weights=mass.softmax(-1))
            chosen_rows = bank[None].expand(columns, -1, -1).gather(-1, selected)
            flat_out[start:end].copy_(chosen_rows.permute(1, 0, 2).to(torch.uint8))
        print(f"partition C16 reference head {head + 1}/{heads}", flush=True)
    return table


def _table_output(q, k, v, table):
    with patch("attention.token_merging.hellinger_span_seed_rows_triton", return_value=table):
        return helper._gca(q, k, v, MODES["rf10"])


def _dispatch_output(name, q, k, v, query_grid):
    symbol = ("partition_mass_seed_rows_triton" if name.startswith("partition_") else
              "entropy_mass_seed_rows_triton" if name.startswith("mass_") else "hellinger_span_seed_rows_triton")
    selector = (partial(_configured_partition, name) if name.startswith("partition_") else
                mass_selector if name.startswith("mass_") else rf10_selector)
    with patch(f"attention.token_merging.{symbol}", wraps=selector) as spy:
        output = helper._gca(q, k, v, MODES[name])
    if spy.call_count != 1:
        raise AssertionError(f"{name}: expected one full-head selector, got {spy.call_count}")
    positional, keyword = spy.call_args
    if not torch.equal(positional[0], query_grid[:, 10:11]):
        raise AssertionError(f"{name}: router is not original frame 10")
    for key in ("candidate_count", "candidate_geometry_count", "block_j"):
        if keyword.get(key) != _selector_options(name)[key]:
            raise AssertionError(f"{name}: unexpected production {key}")
    return output, {"selector_calls": spy.call_count, "router_frame": 10,
                    "query_shape": list(positional[0].shape), "key_shape": list(positional[1].shape),
                    "incoming_production_warps": keyword["gs_num_warps"],
                    "effective_options": _selector_options(name)}


def _parity(host_grids, device, record, save, chunk):
    grids = tuple(grid[PARITY_HEADS].to(device=device, dtype=torch.bfloat16) for grid in host_grids)
    q, k, v = tuple(helper._tokens(grid) for grid in grids)
    qg, kg = q.reshape(1, 21, 45, 80, 4, 128), k.reshape(1, 21, 45, 80, 4, 128)
    # GS controls reuse their existing screening implementations unchanged.
    existing = custom_tables(grids[0], grids[1], ["gs_z_raw", "gs_256", "gs_c16_256"],
                             chunk=chunk, probes=16, router="rf10")
    partition = _partition_reference(grids[0], grids[1], chunk)
    refs = {"rf10": existing["gs_z_raw"], "mass_c32": existing["gs_256"],
            "mass_c16": existing["gs_c16_256"], "partition_c16_j2": partition,
            "partition_c16_j4": partition}
    j2_table = j2_output = None
    for name in MODES:
        native = _native_table(name, qg, kg)
        invariants = helper._seed_invariants(native, 4)
        agreement = helper._seed_agreement(native, refs[name])
        item = {"native_invariants": invariants, "seed_vs_torch": agreement,
                "seed_gate_passed": agreement["mean_shared_seeds"] >= SEED_GATE}
        record["parity_4heads"][name] = item
        save()
        if not item["seed_gate_passed"] or not all(invariants[key] for key in
                    ("rows_in_range", "matched_anchor_first", "distinct_rows")):
            raise AssertionError(f"{name}: invalid rows or seed agreement below {SEED_GATE}/4")
        output, item["dispatch"] = _dispatch_output(name, q, k, v, qg)
        injected = _table_output(q, k, v, native)
        item["dispatch_vs_native_table"] = helper._difference(output, injected)
        del injected
        ref_output = _table_output(q, k, v, refs[name])
        item["output_vs_torch_table"] = helper._difference(output, ref_output)
        del ref_output
        if name == "partition_c16_j2":
            j2_table, j2_output = native, output
        elif name == "partition_c16_j4":
            item["seed_vs_j2"] = helper._seed_agreement(native, j2_table)
            item["output_vs_j2"] = helper._difference(output, j2_output)
            if item["seed_vs_j2"]["mean_shared_seeds"] < SEED_GATE:
                raise AssertionError("J2/J4 disagree beyond the numerical launch gate")
        save()
        print("parity_4heads", name, json.dumps(item), flush=True)
        if not item["dispatch_vs_native_table"]["bitwise_equal"]:
            raise AssertionError(f"{name}: dispatch differs from explicit native table")
        if not item["output_vs_torch_table"]["finite"]:
            raise AssertionError(f"{name}: nonfinite output")


def _timing(functions, args, record, save, device):
    import statistics
    names = list(functions)
    for name in names:
        with _context(name):
            for _ in range(args.warmup):
                output = functions[name]()
        del output
    torch.cuda.synchronize(device)
    samples = {name: [] for name in names}
    record["timing_orders"] = []
    for repeat in range(args.repeats):
        order = names if repeat % 2 == 0 else names[::-1]
        record["timing_orders"].append(order)
        for name in order:
            # Overrides/backend setup and event construction precede recording.
            with _context(name):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    output = functions[name]()
                end.record()
                end.synchronize()
            del output
            latency = start.elapsed_time(end) / args.iterations
            samples[name].append(latency)
            record["attention_latency"][name] = {
                "mean_attention_ms": statistics.mean(samples[name]),
                "std_ms": statistics.pstdev(samples[name]), "samples_ms": list(samples[name]),
                "calls_per_sample": args.iterations}
            save()
            print("attention_latency", repeat, name, latency, flush=True)
    timing = record["attention_latency"]
    for item in timing.values():
        item["speedup_vs_sdpa"] = timing["sdpa"]["mean_attention_ms"] / item["mean_attention_ms"]
        item["ratio_vs_rf10"] = item["mean_attention_ms"] / timing["rf10"]["mean_attention_ms"]
    best = min(("partition_c16_j2", "partition_c16_j4"), key=lambda name: timing[name]["mean_attention_ms"])
    record["faster_partition_launch"] = best
    record["faster_partition_not_slower_than_rf10"] = timing[best]["ratio_vs_rf10"] <= 1
    save()


def _run(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    result = {
        "settings": {**vars(args), "qkv": [str(path) for path in args.qkv], "output": str(args.output)},
        "device": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "parity_heads": PARITY_HEADS, "timing_heads": list(range(40)),
        "dtype": "torch.bfloat16", "modes": MODES,
        "effective_selector_options": {name: _selector_options(name) for name in MODES},
        "benchmark_only_partition_override": {"incoming_production_warps": 8,
             "effective_warps": 4, "reuse_owner_logp": True,
             "note": "Quality and timing both use this override; NOT unmodified production latency"},
        "frozen_algorithm": {"router_frame": 10, "seeds": 4, "local_probes": 16, "global_probes": 256,
             "head_chunk": 8, "query_outer_chunk": 21, "sq1": True, "bank_c16_geometry": 16,
             "normalizer": "original FP32 TF32x3; native and TF32-single optimizations OFF",
             "partition_score": "rho_i * (1 - exp(-exact_probe_partition_gap_i))",
             "scope_note": "Exact only on Uniform16 probes, not exact full-key or all-frame objective"},
        "attention_scope": "complete selector + original merge + original route + contiguous output; no QKV projection/RoPE",
        "timing": "compile/warmup/context setup excluded; interleaved CUDA events; full selector recomputed each call",
        "sdpa": "external reference only; PyTorch SDPA forced FLASH_ATTENTION",
        "minimum_mean_shared_seeds": SEED_GATE, "records": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    save()
    for path in args.qkv:
        info = path.stat()
        record = {"capture": {"path": str(path.resolve()), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns},
                  "parity_4heads": {}, "output_vs_sdpa_40heads": {}, "attention_latency": {}}
        result["records"][str(path)] = record
        save()
        try:
            host = helper._load_grid(str(path), list(range(40)))
            if any(tuple(grid.shape) != (40, 21, 45, 80, 128) for grid in host):
                raise ValueError("captures must contain all 40 heads of 21x45x80 Wan BF16 QKV")
            _parity(host, device, record, save, args.reference_block_chunk)
            q, k, v = tuple(helper._tokens(grid).to(device=device, dtype=torch.bfloat16) for grid in host)
            del host
            functions = {name: partial(helper._gca, q, k, v, mode) for name, mode in MODES.items()}
            functions["sdpa"] = partial(helper._dense, q, k, v)
            # Full-40 seed checks also compare J2/J4 on exactly the same input.
            qg, kg = q.reshape(1, 21, 45, 80, 40, 128), k.reshape(1, 21, 45, 80, 40, 128)
            j2 = _native_table("partition_c16_j2", qg, kg)
            j4 = _native_table("partition_c16_j4", qg, kg)
            record["partition_launch_seed_agreement_40heads"] = helper._seed_agreement(j4, j2)
            record["partition_seed_invariants_40heads"] = {name: helper._seed_invariants(table, 40)
                                                        for name, table in (("j2", j2), ("j4", j4))}
            del j2, j4
            if not all(item[key] for item in record["partition_seed_invariants_40heads"].values()
                       for key in ("rows_in_range", "matched_anchor_first", "distinct_rows")):
                raise AssertionError("full-40 partition C4 row invariants failed")
            if record["partition_launch_seed_agreement_40heads"]["mean_shared_seeds"] < SEED_GATE:
                raise AssertionError("full-40 J2/J4 seed agreement failed")
            with _context("sdpa"):
                dense = functions["sdpa"]()
            j2_output = None
            for name in MODES:
                with _context(name):
                    output = functions[name]()
                if tuple(output.shape) != tuple(q.shape) or not output.is_contiguous():
                    raise AssertionError(f"{name}: output layout mismatch")
                record["output_vs_sdpa_40heads"][name] = helper._difference(output, dense)
                if name == "partition_c16_j2":
                    j2_output = output
                elif name == "partition_c16_j4":
                    record["partition_launch_output_agreement_40heads"] = helper._difference(output, j2_output)
                if not record["output_vs_sdpa_40heads"][name]["finite"]:
                    raise AssertionError(f"{name}: nonfinite full-head output")
                del output
                save()
            del dense, j2_output
            quality = record["output_vs_sdpa_40heads"]
            for item in quality.values():
                for baseline in ("rf10", "mass_c32", "mass_c16"):
                    item[f"mse_change_vs_{baseline}"] = item["mse"] / quality[baseline]["mse"] - 1
            save()
            _timing(functions, args, record, save, device)
            record["complete"] = True
            save()
            del q, k, v, qg, kg, functions
            torch.cuda.empty_cache()
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv", nargs=2, type=Path, default=[Path(path) for path in DEFAULT_CAPTURES])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reference-block-chunk", type=int, default=21)
    parser.add_argument("--cpu-selfcheck", action="store_true")
    args = parser.parse_args()
    if args.cpu_selfcheck:
        print(json.dumps(_cpu_selfcheck(), indent=2))
        return
    if args.output is None:
        parser.error("--output is required unless --cpu-selfcheck is used")
    if min(args.iterations, args.repeats, args.warmup, args.reference_block_chunk) < 1:
        parser.error("iterations/repeats/warmup/reference-block-chunk must be positive")
    if len({str(path.resolve()) for path in args.qkv}) != 2:
        parser.error("two distinct captures are required")
    _load_runtime()
    with torch.inference_mode():
        _run(args)


if __name__ == "__main__":
    main()
