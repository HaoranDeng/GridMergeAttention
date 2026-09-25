"""Isolated H/W-swapped Anchor-Fisher RF0 GCA; never installed on import.

Original THW is permuted to TWH: a key block is one original H=45 column,
and a query group is one original W=80 row.  Four real RF0 seed columns are
chosen per intersection, then evaluated in every actual query frame.  C32,
the g45 *candidate geometry budget*, Fisher curvature/GS, stable C4+SQ1,
entropy bias, and the complete compressed route are retained.  g45 is NOT
the extent of the candidate axis: all 80 anchors and the entire W axis are
covered.  No FP8, learned component, dense exception, or local route exists.

The original generic full-route Triton kernel is reused with ROWS=80,
COLS=45 and BLOCK_I=128.  The merge kernel below masks all padded positions
before partition normalization/ownership.  Uniform16 is floor(i*45/16).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None

SEED_MODE = "fisher_anchor_rf0_c32_g45_u16_hw_swap"
ROWS, COLS, FRAMES, HEADS, DIM = 80, 45, 21, 40, 128


if triton is not None:

    @triton.jit
    def _transpose_fisher_seed_kernel(
        q_ptr, k_ptr, candidate_ptr, anchor_ptr, seed_ptr, audit_ptr,
        K_OUTER: tl.constexpr, HEADS: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_J: tl.constexpr = 4, AUDIT: tl.constexpr = False,
    ):
        key_column = tl.program_id(0) // 12
        group = tl.program_id(0) % 12
        head = tl.program_id(1) % HEADS
        batch = tl.program_id(1) // HEADS
        dims = tl.arange(0, 128)
        flat = tl.arange(0, BLOCK_J * 32)
        query_rows = group * BLOCK_J + flat // 32
        slots = flat % 32
        columns = tl.load(candidate_ptr + key_column * 32 + slots)
        valid = query_rows < 45
        query_token = (batch * 80 + columns) * 45 + query_rows
        query = tl.load(q_ptr + (query_token[:, None] * HEADS + head) * 128
                        + dims[None, :], mask=valid[:, None], other=0.0)
        candidate_columns = tl.reshape(columns, (BLOCK_J, 32))
        valid_2d = tl.reshape(valid, (BLOCK_J, 32))
        candidate_ids = tl.arange(0, 32)[None, :]
        anchor = tl.load(anchor_ptr + key_column)
        output_rows = group * BLOCK_J + tl.arange(0, BLOCK_J)
        seed_slots = tl.arange(0, 4)[None, :]
        # Fractional stride BEFORE integer division, not i*(45//16).
        probes = (tl.arange(0, 16) * 45) // 16
        for key_frame in tl.range(0, K_OUTER):
            key_token = ((batch * K_OUTER + key_frame) * 80 + key_column) * 45
            keys = tl.load(k_ptr + ((key_token + probes[:, None]) * HEADS + head)
                           * 128 + dims[None, :])
            dot_logits = tl.dot(query, tl.trans(keys), input_precision="ieee")
            z = tl.reshape(dot_logits.to(tl.float64) * SCALE, (BLOCK_J, 32, 16))
            used = (candidate_ids == anchor) & valid_2d
            anchor_z = tl.sum(tl.where(used[:, :, None], z, 0.0), axis=1)
            rho = libdevice.exp(anchor_z - tl.max(anchor_z, axis=1)[:, None])
            rho /= tl.sum(rho, axis=1)[:, None]
            delta = z - anchor_z[:, None, :]
            centered = delta - tl.sum(delta * rho[:, None, :], axis=2)[:, :, None]
            residual = centered * tl.sqrt(rho[:, None, :])
            seed0 = tl.sum(tl.where(used, candidate_columns, 0), axis=1)
            seeds = seed0[:, None] * (seed_slots == 0).to(tl.int32)
            for step in tl.static_range(1, 4):
                score = tl.sum(residual * residual, axis=2)
                if AUDIT:
                    offset = (((((batch * K_OUTER + key_frame) * 80 + key_column)
                                * 45 + output_rows[:, None]) * HEADS + head)
                              * 3 + step - 1) * 32 + candidate_ids
                    tl.store(audit_ptr + offset, score, mask=valid_2d)
                _, picked = tl.max(tl.where(valid_2d & ~used, score, -float("inf")),
                    axis=1, return_indices=True, return_indices_tie_break_left=True)
                hit = (candidate_ids == picked[:, None]) & valid_2d
                used |= hit
                column = tl.sum(tl.where(hit, candidate_columns, 0), axis=1)
                seeds += column[:, None] * (seed_slots == step).to(tl.int32)
                if step < 3:
                    direction = tl.sum(tl.where(hit[:, :, None], residual, 0.0), axis=1)
                    basis = direction / tl.sqrt(tl.sum(direction * direction, axis=1)
                                                + 1.0e-12)[:, None]
                    projection = tl.sum(residual * basis[:, None, :], axis=2)
                    residual -= projection[:, :, None] * basis[:, None, :]
            offset = (((((batch * K_OUTER + key_frame) * 80 + key_column) * 45
                        + output_rows[:, None]) * HEADS + head) * 4 + seed_slots)
            tl.store(seed_ptr + offset, seeds, mask=output_rows[:, None] < 45)

    @triton.jit
    def _transpose_stable_merge_kernel(
        q_ptr, k_ptr, v_ptr, seed_ptr, mk_ptr, mv_ptr, bias_ptr,
        Q_OUTER: tl.constexpr, K_OUTER: tl.constexpr, HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr, HEAD_START: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_J: tl.constexpr = 16,
    ):
        """Full-valid-45 ownership, then direct stable bulk conditioning."""
        key_frame = tl.program_id(0) % K_OUTER
        key_column = tl.program_id(0) // K_OUTER
        head = tl.program_id(1) % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = tl.program_id(1) // ACTIVE_HEADS
        batch64 = batch.to(tl.int64)
        dims = tl.arange(0, 128)
        key_rows = tl.arange(0, 64)
        valid_key = key_rows < 45
        key_token = ((batch * K_OUTER + key_frame) * 80 + key_column) * 45 + key_rows
        address = (key_token[:, None] * HEADS + global_head) * 128 + dims[None, :]
        keys = tl.load(k_ptr + address, mask=valid_key[:, None], other=0.0,
                       eviction_policy="evict_last")
        values = tl.load(v_ptr + address, mask=valid_key[:, None], other=0.0,
                         eviction_policy="evict_last")
        flat = tl.arange(0, BLOCK_J * 4)
        seed_ids = flat % 4
        cluster_ids = tl.arange(0, 4)[None, :, None]
        for row_start in tl.range(0, 45, BLOCK_J):
            query_rows = row_start + flat // 4
            valid_query = query_rows < 45
            seed_offset = (((((batch64 * K_OUTER + key_frame) * 80 + key_column)
                             * 45 + query_rows) * HEADS + global_head) * 4 + seed_ids)
            seed_columns = tl.load(seed_ptr + seed_offset, mask=valid_query, other=0).to(tl.int32)
            for query_frame in tl.range(0, Q_OUTER, num_stages=2, disable_licm=False):
                query_token = ((batch * Q_OUTER + query_frame) * 80 + seed_columns) * 45 + query_rows
                queries = tl.load(q_ptr + (query_token[:, None] * HEADS + global_head)
                                  * 128 + dims[None, :], mask=valid_query[:, None], other=0.0,
                                  eviction_policy="evict_last")
                scores = tl.dot(queries, tl.trans(keys), input_precision="ieee")
                scores *= SCALE * 1.4426950408889634
                masked_scores = tl.where(valid_key[None, :], scores, -float("inf"))
                maximum = tl.max(masked_scores, axis=1)
                partition = tl.exp2(masked_scores - maximum[:, None])
                log_norm = maximum + tl.log2(tl.sum(partition, axis=1))
                log_probs = tl.reshape(masked_scores - log_norm[:, None], (BLOCK_J, 4, 64))
                _, owner = tl.max(log_probs, axis=1, return_indices=True,
                                  return_indices_tie_break_left=True)
                sq_rows = row_start + tl.arange(0, BLOCK_J)
                bulk = ((owner[:, None, :] == cluster_ids) & valid_key[None, None, :]
                        & (key_rows[None, None, :] != sq_rows[:, None, None]))
                has_key = tl.sum(bulk.to(tl.int32), axis=2) > 0
                cells = tl.reshape(scores, (BLOCK_J, 4, 64))
                bulk_max = tl.max(tl.where(bulk, cells, -float("inf")), axis=2)
                safe_max = tl.where(has_key, bulk_max, 0.0)
                centered = tl.where(bulk, cells - safe_max[:, :, None], 0.0)
                exponential = tl.where(bulk, tl.exp2(centered), 0.0)
                bulk_sum = tl.sum(exponential, axis=2)
                safe_sum = tl.where(has_key, bulk_sum, 1.0)
                weights = exponential / safe_sum[:, :, None]
                expected = tl.sum(weights * centered, axis=2)
                bias = (expected - tl.log2(safe_sum)) * 0.6931471805599453
                bias = tl.where(has_key, bias, float("inf"))
                weights2d = tl.reshape(weights, (BLOCK_J * 4, 64)).to(keys.dtype)
                merged_k = tl.dot(weights2d, keys, input_precision="ieee")
                merged_v = tl.dot(weights2d, values, input_precision="ieee")
                out_base = ((((((batch64 * Q_OUTER + query_frame) * 45 + query_rows)
                              * ACTIVE_HEADS + head) * K_OUTER + key_frame) * 80
                            + key_column) * 4 + seed_ids)
                out_address = out_base[:, None] * 128 + dims[None, :]
                tl.store(mk_ptr + out_address, merged_k, mask=valid_query[:, None])
                tl.store(mv_ptr + out_address, merged_v, mask=valid_query[:, None])
                tl.store(bias_ptr + out_base, tl.reshape(bias, (BLOCK_J * 4,)), mask=valid_query)


def fisher_seed_columns(q, k, *, score_audit=None):
    """Native selector on contiguous TWH QRF0/K; flexible frame/head count for QA."""
    if triton is None:
        raise RuntimeError("Triton is required")
    if (q.ndim != 6 or k.ndim != 6 or q.shape[1:4] != (1, 80, 45)
            or k.shape[0] != q.shape[0] or k.shape[2:] != q.shape[2:]
            or k.shape[1] < 1 or q.shape[-1] != 128):
        raise ValueError("expected Q [B,1,80,45,heads,128] and K [B,T,80,45,heads,128]")
    if any(not x.is_cuda or x.dtype != torch.bfloat16 or x.device != q.device
           or not x.is_contiguous() for x in (q, k)):
        raise ValueError("selector requires contiguous original CUDA BF16 tensors")
    from .adaptive_seed_pytorch import multiscale_candidate_table
    candidates, anchors = multiscale_candidate_table(80, 32, q.device, geometry_count=45)
    shape = (q.shape[0], 1, k.shape[1], 80, 45, q.shape[4], 4)
    seeds = torch.empty(shape, device=q.device, dtype=torch.uint8)
    if score_audit is not None and (tuple(score_audit.shape) != shape[:-1] + (3, 32)
            or score_audit.dtype != torch.float32 or score_audit.device != q.device
            or not score_audit.is_contiguous()):
        raise ValueError("score_audit shape/dtype/device mismatch")
    _transpose_fisher_seed_kernel[(80 * 12, q.shape[0] * q.shape[4])](
        q, k, candidates, anchors, seeds, seeds if score_audit is None else score_audit,
        K_OUTER=k.shape[1], HEADS=q.shape[4], SCALE=128 ** -.5,
        BLOCK_J=4, AUDIT=score_audit is not None, num_warps=4, num_stages=1)
    return seeds


def merge_and_full_route(q, k, v, seeds, *, head_start=0, head_count=8, output=None,
                         return_merged=False):
    """One active-head chunk, generic time count only for native parity tests."""
    from .hard_c4_wan21_14b_720p_triton import _hard_c4_720p_route_kernel
    if (q.ndim != 6 or tuple(q.shape[2:4]) != (80, 45) or q.shape[-1] != 128
            or k.shape != v.shape or k.shape[0] != q.shape[0] or k.shape[2:] != q.shape[2:]):
        raise ValueError("expected matching contiguous TWH QKV blocks")
    if any(not x.is_cuda or x.dtype != torch.bfloat16 or x.device != q.device
           or not x.is_contiguous() for x in (q, k, v)):
        raise ValueError("merge requires original contiguous CUDA BF16 tensors")
    batch, q_frames, _, _, heads, _ = q.shape
    k_frames = k.shape[1]
    if not (0 <= head_start < heads and 1 <= head_count <= heads - head_start):
        raise ValueError("invalid active head chunk")
    if (tuple(seeds.shape) != (batch, 1, k_frames, 80, 45, heads, 4)
            or seeds.dtype != torch.uint8 or seeds.device != q.device or not seeds.is_contiguous()):
        raise ValueError("invalid native RF0 seed table")
    shape = (batch, q_frames, 45, head_count, k_frames, 80, 4)
    merged_k = torch.empty((*shape, 128), device=q.device, dtype=q.dtype)
    merged_v = torch.empty_like(merged_k)
    bias = torch.empty(shape, device=q.device, dtype=torch.float32)
    if output is None:
        output = torch.empty_like(q)
    if output.shape != q.shape or output.dtype != q.dtype or output.device != q.device or not output.is_contiguous():
        raise ValueError("invalid output tensor")
    _transpose_stable_merge_kernel[(80 * k_frames, batch * head_count)](
        q, k, v, seeds, merged_k, merged_v, bias, Q_OUTER=q_frames, K_OUTER=k_frames,
        HEADS=heads, ACTIVE_HEADS=head_count, HEAD_START=head_start, SCALE=128 ** -.5,
        BLOCK_J=16, num_warps=4, num_stages=1)
    # Production full compressed route, no surgery, no sparse/dense exceptions.
    _hard_c4_720p_route_kernel[(1, batch * head_count, q_frames * 45)](
        q, k, v, merged_k, merged_v, bias, bias, bias, seeds, bias, output,
        Q_OUTER=q_frames, K_OUTER=k_frames, TOTAL_HEADS=heads, ACTIVE_HEADS=head_count,
        HEAD_START=head_start, D_HEAD=128, HEAD_MAJOR=False, OUTPUT_HEAD_MAJOR=False,
        ROWS=80, COLS=45, C=4, BLOCK_I=128, BLOCK_TOKENS=64, LOOP_STAGES=3,
        INPUT_PRECISION="ieee", EXACT_INTRA_FRAME=False, EXACT_NEIGHBOR_WINDOW_COLS=0,
        RECORD_DISTORTION=False, QUERY_OUTER_OFFSET=0, SINGLETON_TOP_R=1,
        SINGLETON_QUERY_COLUMN=True, SINGLETON_SECONDARY_FRAME_STRIDE=1,
        SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False, QUERY_COLUMN_MAJOR=True,
        PACKED_SINGLETON_KV=False, SCALED_FP8_KV=False, COMPRESSED_FP8=False,
        SCALE=128 ** -.5, num_warps=4, num_stages=1)
    return (output, merged_k, merged_v, bias) if return_merged else output


class SpatialTransposeGCA:
    """Fixed original-Wan geometry and five ordered 8-head chunks per call."""
    seed_mode = SEED_MODE

    def __init__(self):
        from . import hard_c4_wan21_14b_720p_triton as route_module
        from . import adaptive_seed_pytorch as candidates_module
        from . import spatial_transpose_gca_reference as reference_module
        paths = [Path(__file__).resolve(), Path(route_module.__file__).resolve(),
                 Path(candidates_module.__file__).resolve(), Path(reference_module.__file__).resolve()]
        self.source_receipt = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.core_calls = self.selector_calls = self.merge_calls = self.kernel_calls = self.route_calls = 0
        self.active = False

    def verify_kernel(self):
        for path, expected in self.source_receipt.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise RuntimeError("spatial transpose source changed: " + path)

    def counts(self):
        return {"core_calls": self.core_calls, "selector_calls": self.selector_calls,
                "merge_wrapper_calls": self.merge_calls, "stable_kernel_calls": self.kernel_calls}

    def metadata(self):
        return {"seed_mode": self.seed_mode, "original_thw": [21, 45, 80],
                "operator_layout": "TWH", "key_axis": "H45", "query_axis": "W80",
                "layout": {"key_axis": "H", "query_axis": "W", "time_axis": "T",
                           "key_group_size": 45, "query_group_size": 80,
                           "key_groups_per_frame": 80, "query_groups_per_frame": 45,
                           "compressed_slots_per_group": 5, "full_route": True},
                "candidate_count": 32, "candidate_geometry_count": 45,
                "candidate_geometry_meaning": "fixed local/global budget, not axis extent; covers all W80 anchors",
                "probe_indices": [(i * 45) // 16 for i in range(16)],
                "rf": 0, "head_chunk": 8, "merge_block_j": 16, "route_block_i": 128,
                "route_tokens_per_query": 21 * 80 * 5, "compressed_bulk_slots": 4,
                "singleton": "K[t_key,h_query,w_key], removed from bulk before normalization",
                "entropy_bias": True, "dense_fallback": False, "fp8": False,
                "full_compressed_route": True, "source_receipt": self.source_receipt,
                "dispatch_counts": self.counts(), "route_kernel_calls": self.route_calls}

    def __call__(self, query, key, value):
        if self.active:
            raise RuntimeError("nested spatial transpose core")
        for tensor in (query, key, value):
            if (tuple(tensor.shape) != (1, 40, 75600, 128) or tensor.dtype != torch.bfloat16
                    or not tensor.is_cuda or tensor.device != query.device):
                raise ValueError("requires original 720p/81-frame CUDA BF16 head-major QKV")
        self.active = True
        try:
            # Include both actual transposes inside complete-attention timing.
            q, k, v = [x.reshape(1, 40, 21, 45, 80, 128).permute(0, 2, 4, 3, 1, 5)
                       .contiguous() for x in (query, key, value)]
            seeds = fisher_seed_columns(q[:, :1].contiguous(), k)
            self.selector_calls += 1
            output = torch.empty_like(q)
            for head_start in range(0, 40, 8):
                merge_and_full_route(q, k, v, seeds, head_start=head_start, head_count=8, output=output)
                self.merge_calls += 1
                self.kernel_calls += 1
                self.route_calls += 1
            self.core_calls += 1
            return output.permute(0, 4, 1, 3, 2, 5).contiguous().reshape(1, 40, 75600, 128)
        finally:
            self.active = False


def create_spatial_transpose_core(output_dir=None):
    raise RuntimeError("retired C4+SQ1 path: C4 must contain exactly four merged K/V and no singleton")
    """Factory compatible with the isolated benchmark. Does not write artifacts."""
    core = SpatialTransposeGCA()
    core.verify_kernel()
    return core


def validate_native(output_dir):
    """Fail-closed CUDA primitive validation, invoked explicitly before a run.

    This is an external diagnostic, never part of candidate inference or timed
    attention. It uses two synthetic TWH frames, all80 query columns/all45 key rows,
    including query columns64..79, last non-divisible merge group, multiple
    key frames, exact ties and empty bulks. The route oracle consumes the same
    BF16 compressed K/V as native to isolate route correctness. Independent
    FP64 references validate seed selection and stable bulk construction.
    """
    import json
    import numpy as np
    from .spatial_transpose_gca_reference import candidate_table, fisher_seeds, stable_bulk, softmax

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    report = {"valid": False, "passed": False, "kind": "external native primitive correctness diagnostic; not timing/quality",
              "frames": 2, "query_columns": 80, "valid_key_rows": 45,
              "cases": {}, "tolerances": {"seed_mismatches": 0, "score_relative": 2e-5,
                                            "merge_max_abs": .01, "bias_max_abs": 2e-5,
                                            "route_max_abs": .02, "route_rmse": .0025}}

    def require(condition, message):
        if not condition:
            raise RuntimeError("spatial transpose native validation: " + message)

    def array(tensor):
        return tensor.detach().float().cpu().numpy().astype(np.float64)

    def rounded_bf16(x):
        return torch.from_numpy(np.asarray(x)).to(torch.bfloat16).float().numpy().astype(np.float64)

    try:
        require(torch.cuda.is_available(), "CUDA unavailable")
        core = create_spatial_transpose_core()
        report["source_receipt"] = core.source_receipt
        report["device"] = torch.cuda.get_device_name()
        candidates, anchors = candidate_table()
        from .adaptive_seed_pytorch import multiscale_candidate_table
        actual_candidates, actual_anchors = multiscale_candidate_table(80, 32, "cpu", geometry_count=45)
        require(np.array_equal(candidates, actual_candidates.numpy())
                and np.array_equal(anchors, actual_anchors.numpy()), "candidate table reference drift")
        generator = torch.Generator(device="cpu").manual_seed(1729)
        shape = (1, 2, 80, 45, 1, 128)
        cpu_q, cpu_k, cpu_v = [(torch.randn(shape, generator=generator) * .8).to(torch.bfloat16)
                              for _ in range(3)]
        for case in ("random_two_frames", "zero_query_ties_empty_bulks"):
            q = (cpu_q if case == "random_two_frames" else torch.zeros_like(cpu_q)).cuda()
            k, v = cpu_k.cuda(), cpu_v.cuda()
            audit = torch.empty((1, 1, 2, 80, 45, 1, 3, 32), dtype=torch.float32, device=q.device)
            seeds = fisher_seed_columns(q[:, :1].contiguous(), k, score_audit=audit)
            output, mk, mv, bias = merge_and_full_route(q, k, v, seeds, head_count=1, return_merged=True)
            torch.cuda.synchronize()
            for name, tensor in (("output", output), ("merged_k", mk), ("merged_v", mv), ("selector_scores", audit)):
                require(bool(torch.isfinite(tensor).all()), case + " nonfinite " + name)
            require(not bool(torch.isnan(bias).any()) and not bool(torch.isneginf(bias).any()), case + " invalid bias")
            s = seeds.cpu().numpy()[0, 0, :, :, :, 0]
            require(bool(((s >= 0) & (s < 80)).all()), case + " invalid seed index")
            require(bool((np.sort(s, axis=-1)[..., 1:] != np.sort(s, axis=-1)[..., :-1]).all()), case + " duplicate seeds")
            require(bool((s[..., 0] == np.arange(80)[None, :, None]).all()), case + " lost matched anchor")
            qn, kn, vn = array(q)[0, ..., 0, :], array(k)[0, ..., 0, :], array(v)[0, ..., 0, :]
            mkn, mvn, bn = array(mk)[0, :, :, 0], array(mv)[0, :, :, 0], array(bias)[0, :, :, 0]
            scores_native = array(audit)[0, 0, ..., 0, :, :]
            outn = array(output)[0, ..., 0, :]
            probe = np.arange(16) * 45 // 16
            stats = {"seed_cells": 0, "seed_mismatches": 0, "score_max_abs": 0.,
                     "merge_cells": 0, "merge_max_abs": 0., "bias_max_abs": 0.,
                     "route_query_vectors": 0, "route_max_abs": 0., "route_squared_error": 0.,
                     "route_elements": 0, "tail_query_columns_checked": list(range(64, 80)),
                     "query_rows_checked": [0, 16, 32, 44], "all_outputs_finite": True,
                     "empty_bulk_count": int(np.isposinf(bn).sum())}
            for key_frame in range(2):
                for key_column in (0, 31, 79):
                    for query_row in (0, 16, 32, 44):
                        logits = (qn[0, candidates[key_column], query_row]
                                  @ kn[key_frame, key_column, probe].T) * (128 ** -.5)
                        expected_seeds, expected_scores = fisher_seeds(logits, candidates[key_column], anchors[key_column])
                        mismatch = int(not np.array_equal(expected_seeds, s[key_frame, key_column, query_row]))
                        stats["seed_cells"] += 1
                        stats["seed_mismatches"] += mismatch
                        score_error = float(np.max(np.abs(expected_scores - scores_native[key_frame, key_column, query_row])))
                        stats["score_max_abs"] = max(stats["score_max_abs"], score_error)
                        require(score_error <= 2e-5 * (1 + float(np.max(np.abs(expected_scores)))), case + " Fisher score mismatch")
                        for query_frame in range(2):
                            actual_seeds = s[key_frame, key_column, query_row]
                            full_logits = (qn[query_frame, actual_seeds, query_row]
                                           @ kn[key_frame, key_column].T) * (128 ** -.5)
                            weights, expected_bias, owner = stable_bulk(full_logits, query_row)
                            native_bias = bn[query_frame, query_row, key_frame, key_column]
                            require(np.array_equal(np.isposinf(expected_bias), np.isposinf(native_bias)), case + " empty bulk mismatch")
                            finite = np.isfinite(expected_bias)
                            bias_error = float(np.max(np.abs(expected_bias[finite] - native_bias[finite])))
                            stats["bias_max_abs"] = max(stats["bias_max_abs"], bias_error)
                            # Production/native merge rounds normalized weights before the BF16 dot.
                            bf16_weights = rounded_bf16(weights)
                            expected_k = rounded_bf16(bf16_weights @ kn[key_frame, key_column])
                            expected_v = rounded_bf16(bf16_weights @ vn[key_frame, key_column])
                            merge_error = max(float(np.max(np.abs(expected_k - mkn[query_frame, query_row, key_frame, key_column]))),
                                              float(np.max(np.abs(expected_v - mvn[query_frame, query_row, key_frame, key_column]))))
                            stats["merge_max_abs"] = max(stats["merge_max_abs"], merge_error)
                            stats["merge_cells"] += 1
            for query_frame in range(2):
                for query_row in (0, 16, 32, 44):
                    # Full2*80*4 bulk plus2*80 SQ1 stream, including ALL key frames/columns.
                    key_bank = np.concatenate([mkn[query_frame, query_row].reshape(-1, 128),
                                               kn[:, :, query_row].reshape(-1, 128)])
                    value_bank = np.concatenate([mvn[query_frame, query_row].reshape(-1, 128),
                                                 vn[:, :, query_row].reshape(-1, 128)])
                    bank_bias = np.concatenate([bn[query_frame, query_row].reshape(-1), np.zeros(2 * 80)])
                    logits = qn[query_frame, :, query_row] @ key_bank.T * (128 ** -.5) - bank_bias
                    expected = softmax(logits) @ value_bank
                    error = expected - outn[query_frame, :, query_row]
                    stats["route_query_vectors"] += 80
                    stats["route_elements"] += int(error.size)
                    stats["route_squared_error"] += float(np.square(error).sum())
                    stats["route_max_abs"] = max(stats["route_max_abs"], float(np.max(np.abs(error))))
            stats["route_rmse"] = math.sqrt(stats.pop("route_squared_error") / stats["route_elements"])
            report["cases"][case] = stats
            require(stats["seed_mismatches"] == 0, case + " seed disagreement")
            require(stats["merge_max_abs"] <= .01, case + " merge vector disagreement")
            require(stats["bias_max_abs"] <= 2e-5, case + " entropy disagreement")
            require(stats["route_max_abs"] <= .02 and stats["route_rmse"] <= .0025, case + " route disagreement")
            if case == "zero_query_ties_empty_bulks":
                require(stats["empty_bulk_count"] == 2 * 45 * 2 * 80 * 3, "expected exactly three empty C4 bulk slots per cell")
            del q, k, v, seeds, audit, output, mk, mv, bias
        # Compile and exercise the actual21-frame/8-head dispatch, not merely
        # extrapolate a tiny test. Never counted as model quality or timing.
        torch.cuda.empty_cache()
        full_generator = torch.Generator(device="cuda").manual_seed(1730)
        full_shape = (1, 40, 75600, 128)
        full_qkv = [torch.randn(full_shape, dtype=torch.bfloat16, device="cuda", generator=full_generator)
                    for _ in range(3)]
        full_output = core(*full_qkv)
        torch.cuda.synchronize()
        require(tuple(full_output.shape) == full_shape and full_output.dtype == torch.bfloat16
                and full_output.is_contiguous() and bool(torch.isfinite(full_output).all()),
                "full production-shape callable returned invalid output")
        require(core.counts() == {"core_calls": 1, "selector_calls": 1,
                                  "merge_wrapper_calls": 5, "stable_kernel_calls": 5}
                and core.route_calls == 5, "full-shape actual dispatch counts changed")
        report["full_shape_smoke"] = {"qkv_shape": list(full_shape), "dtype": "bfloat16",
                                      "finite_output": True, "counts": core.counts(),
                                      "route_kernel_calls": core.route_calls,
                                      "synthetic_diagnostic_only": True}
        del full_qkv, full_output
        core.verify_kernel()
        report["valid"] = report["passed"] = True
        torch.cuda.empty_cache()
        return report
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        with (directory / "native_validation.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
