"""Isolated query-H/key-T Anchor-Fisher RF0 GCA; never installed on import.

Physical THW becomes WHT: query groups are Q[tq,:,wq], key groups are
K[:,hk,wk]. Physical RF0 always means t=0, including wq!=0. The compact seed
table [B,1,Wk,Hk,Wq,heads,4] contains real height indices, independent of tq.
C32/g45 uses the unchanged candidate policy over all45 height anchors, with
temporal Uniform16 probes floor(i*21/16). Current-time C4 ownership uses the
full21 keys before SQ1 at tq is removed. Empty bulks store zero KV/+inf bias.

The original full-route kernel is reused unchanged with ROWS45/COLS21 and
Q_OUTER=K_OUTER=W80, covering all18000 nominal representatives per query.
All KV/weights are BF16; stable merge reductions FP32 and Fisher/GS metrics
FP64. The twenty2-head chunks bound merged KV+bias scratch to24,966,144,000
bytes. No learned component, FP8, dense exception or sparse route is used.
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

SEED_MODE = "fisher_anchor_rf0_c32_g45_u16_key_t_query_h"
ROWS, COLS, OUTER, HEADS, DIM = 45, 21, 80, 40, 128


if triton is not None:

    @triton.jit
    def _temporal_height_fisher_seed_kernel(
        q_ptr, k_ptr, candidate_ptr, anchor_ptr, seed_ptr, audit_ptr,
        Q_OUTER: tl.constexpr, K_OUTER: tl.constexpr, HEADS: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_J: tl.constexpr = 4, AUDIT: tl.constexpr = False,
    ):
        groups: tl.constexpr = triton.cdiv(Q_OUTER, BLOCK_J)
        key_height = tl.program_id(0) // groups
        group = tl.program_id(0) % groups
        head = tl.program_id(1) % HEADS
        batch = tl.program_id(1) // HEADS
        dims = tl.arange(0, 128)
        flat = tl.arange(0, BLOCK_J * 32)
        query_widths = group * BLOCK_J + flat // 32
        slots = flat % 32
        heights = tl.load(candidate_ptr + key_height * 32 + slots)
        valid = query_widths < Q_OUTER
        # QRF0 is [B,1,H45,Wq,heads,D], obtained from physical t=0.
        query_token = (batch * 45 + heights) * Q_OUTER + query_widths
        query = tl.load(q_ptr + (query_token[:, None] * HEADS + head) * 128
                        + dims[None, :], mask=valid[:, None], other=0.0)
        candidate_heights = tl.reshape(heights, (BLOCK_J, 32))
        valid_2d = tl.reshape(valid, (BLOCK_J, 32))
        candidate_ids = tl.arange(0, 32)[None, :]
        anchor = tl.load(anchor_ptr + key_height)
        output_widths = group * BLOCK_J + tl.arange(0, BLOCK_J)
        seed_slots = tl.arange(0, 4)[None, :]
        probes = (tl.arange(0, 16) * 21) // 16
        for key_width in tl.range(0, K_OUTER):
            key_token = ((batch * K_OUTER + key_width) * 45 + key_height) * 21
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
            seed0 = tl.sum(tl.where(used, candidate_heights, 0), axis=1)
            seeds = seed0[:, None] * (seed_slots == 0).to(tl.int32)
            for step in tl.static_range(1, 4):
                score = tl.sum(residual * residual, axis=2)
                if AUDIT:
                    offset = (((((batch * K_OUTER + key_width) * 45 + key_height)
                                * Q_OUTER + output_widths[:, None]) * HEADS + head)
                              * 3 + step - 1) * 32 + candidate_ids
                    tl.store(audit_ptr + offset, score, mask=valid_2d)
                _, picked = tl.max(tl.where(valid_2d & ~used, score, -float("inf")),
                    axis=1, return_indices=True, return_indices_tie_break_left=True)
                hit = (candidate_ids == picked[:, None]) & valid_2d
                used |= hit
                selected_height = tl.sum(tl.where(hit, candidate_heights, 0), axis=1)
                seeds += selected_height[:, None] * (seed_slots == step).to(tl.int32)
                if step < 3:
                    direction = tl.sum(tl.where(hit[:, :, None], residual, 0.0), axis=1)
                    basis = direction / tl.sqrt(tl.sum(direction * direction, axis=1)
                                                + 1.0e-12)[:, None]
                    projection = tl.sum(residual * basis[:, None, :], axis=2)
                    residual -= projection[:, :, None] * basis[:, None, :]
            offset = (((((batch * K_OUTER + key_width) * 45 + key_height) * Q_OUTER
                        + output_widths[:, None]) * HEADS + head) * 4 + seed_slots)
            tl.store(seed_ptr + offset, seeds, mask=output_widths[:, None] < Q_OUTER)

    @triton.jit
    def _temporal_height_stable_merge_kernel(
        q_ptr, k_ptr, v_ptr, seed_ptr, mk_ptr, mv_ptr, bias_ptr,
        Q_OUTER: tl.constexpr, K_OUTER: tl.constexpr, HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr, HEAD_START: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_J: tl.constexpr = 16,
    ):
        """Full-valid-T21 ownership followed by stable bulk conditioning."""
        key_width = tl.program_id(0) % K_OUTER
        key_height = tl.program_id(0) // K_OUTER
        head = tl.program_id(1) % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = tl.program_id(1) // ACTIVE_HEADS
        batch64 = batch.to(tl.int64)
        dims = tl.arange(0, 128)
        key_times = tl.arange(0, 32)
        valid_key = key_times < 21
        key_token = ((batch * K_OUTER + key_width) * 45 + key_height) * 21 + key_times
        address = (key_token[:, None] * HEADS + global_head) * 128 + dims[None, :]
        keys = tl.load(k_ptr + address, mask=valid_key[:, None], other=0.0,
                       eviction_policy="evict_last")
        values = tl.load(v_ptr + address, mask=valid_key[:, None], other=0.0,
                         eviction_policy="evict_last")
        flat = tl.arange(0, BLOCK_J * 4)
        seed_ids = flat % 4
        cluster_ids = tl.arange(0, 4)[None, :, None]
        for query_width in tl.range(0, Q_OUTER, num_stages=2, disable_licm=False):
            # Compact RF0 seeds are indexed by wq, NEVER by current query time.
            seed_offset = (((((batch64 * K_OUTER + key_width) * 45 + key_height)
                             * Q_OUTER + query_width) * HEADS + global_head) * 4 + seed_ids)
            seed_heights = tl.load(seed_ptr + seed_offset).to(tl.int32)
            for time_start in tl.range(0, 21, BLOCK_J):
                query_times = time_start + flat // 4
                valid_query = query_times < 21
                query_token = ((batch * Q_OUTER + query_width) * 45 + seed_heights) * 21 + query_times
                queries = tl.load(q_ptr + (query_token[:, None] * HEADS + global_head)
                                  * 128 + dims[None, :], mask=valid_query[:, None], other=0.0,
                                  eviction_policy="evict_last")
                scores = tl.dot(queries, tl.trans(keys), input_precision="ieee")
                scores *= SCALE * 1.4426950408889634
                masked_scores = tl.where(valid_key[None, :], scores, -float("inf"))
                maximum = tl.max(masked_scores, axis=1)
                partition = tl.exp2(masked_scores - maximum[:, None])
                log_norm = maximum + tl.log2(tl.sum(partition, axis=1))
                log_probs = tl.reshape(masked_scores - log_norm[:, None], (BLOCK_J, 4, 32))
                _, owner = tl.max(log_probs, axis=1, return_indices=True,
                                  return_indices_tie_break_left=True)
                sq_times = time_start + tl.arange(0, BLOCK_J)
                bulk = ((owner[:, None, :] == cluster_ids) & valid_key[None, None, :]
                        & (key_times[None, None, :] != sq_times[:, None, None]))
                has_key = tl.sum(bulk.to(tl.int32), axis=2) > 0
                cells = tl.reshape(scores, (BLOCK_J, 4, 32))
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
                weights2d = tl.reshape(weights, (BLOCK_J * 4, 32)).to(keys.dtype)
                merged_k = tl.dot(weights2d, keys, input_precision="ieee")
                merged_v = tl.dot(weights2d, values, input_precision="ieee")
                out_base = ((((((batch64 * Q_OUTER + query_width) * 21 + query_times)
                              * ACTIVE_HEADS + head) * K_OUTER + key_width) * 45
                            + key_height) * 4 + seed_ids)
                out_address = out_base[:, None] * 128 + dims[None, :]
                tl.store(mk_ptr + out_address, merged_k, mask=valid_query[:, None])
                tl.store(mv_ptr + out_address, merged_v, mask=valid_query[:, None])
                tl.store(bias_ptr + out_base, tl.reshape(bias, (BLOCK_J * 4,)), mask=valid_query)


def fisher_seed_rows(q_rf0, k, *, score_audit=None):
    """QRF0 [B,1,H45,Wq,heads,D], K [B,Wk,H45,T21,heads,D]."""
    if triton is None:
        raise RuntimeError("Triton is required")
    if (q_rf0.ndim != 6 or k.ndim != 6 or q_rf0.shape[1:3] != (1, 45)
            or k.shape[2:4] != (45, 21) or min(q_rf0.shape[3], k.shape[1]) < 1
            or k.shape[0] != q_rf0.shape[0] or k.shape[4:] != q_rf0.shape[4:]
            or q_rf0.shape[-1] != 128):
        raise ValueError("expected QRF0 [B,1,45,Wq,heads,128] and K [B,Wk,45,21,heads,128]")
    if any(not x.is_cuda or x.dtype != torch.bfloat16 or x.device != q_rf0.device
           or not x.is_contiguous() for x in (q_rf0, k)):
        raise ValueError("selector requires contiguous CUDA BF16 tensors")
    from .adaptive_seed_pytorch import multiscale_candidate_table
    candidates, anchors = multiscale_candidate_table(45, 32, q_rf0.device, geometry_count=45)
    shape = (q_rf0.shape[0], 1, k.shape[1], 45, q_rf0.shape[3], q_rf0.shape[4], 4)
    seeds = torch.empty(shape, device=q_rf0.device, dtype=torch.uint8)
    if score_audit is not None and (tuple(score_audit.shape) != shape[:-1] + (3, 32)
            or score_audit.dtype != torch.float32 or score_audit.device != q_rf0.device
            or not score_audit.is_contiguous()):
        raise ValueError("score_audit shape/dtype/device mismatch")
    _temporal_height_fisher_seed_kernel[(45 * triton.cdiv(q_rf0.shape[3], 4), q_rf0.shape[0] * q_rf0.shape[4])](
        q_rf0, k, candidates, anchors, seeds, seeds if score_audit is None else score_audit,
        Q_OUTER=q_rf0.shape[3], K_OUTER=k.shape[1], HEADS=q_rf0.shape[4], SCALE=128 ** -.5,
        BLOCK_J=4, AUDIT=score_audit is not None, num_warps=4, num_stages=1)
    return seeds


def merge_and_full_route(q, k, v, seeds, *, head_start=0, head_count=2, output=None,
                         return_merged=False):
    """One active-head chunk; outer W extent is flexible only for native QA."""
    if triton is None:
        raise RuntimeError("Triton is required")
    from .hard_c4_wan21_14b_720p_triton import _hard_c4_720p_route_kernel
    if (q.ndim != 6 or tuple(q.shape[2:4]) != (45, 21) or q.shape[-1] != 128
            or k.shape != v.shape or k.shape[0] != q.shape[0] or k.shape[2:] != q.shape[2:]
            or min(q.shape[1], k.shape[1]) < 1):
        raise ValueError("expected matching contiguous WHT QKV blocks")
    if any(not x.is_cuda or x.dtype != torch.bfloat16 or x.device != q.device
           or not x.is_contiguous() for x in (q, k, v)):
        raise ValueError("merge requires contiguous CUDA BF16 tensors")
    batch, q_widths, _, _, heads, _ = q.shape
    k_widths = k.shape[1]
    if not (0 <= head_start < heads and 1 <= head_count <= heads - head_start):
        raise ValueError("invalid active head chunk")
    if (tuple(seeds.shape) != (batch, 1, k_widths, 45, q_widths, heads, 4)
            or seeds.dtype != torch.uint8 or seeds.device != q.device or not seeds.is_contiguous()):
        raise ValueError("invalid compact physical RF0 seed table")
    shape = (batch, q_widths, 21, head_count, k_widths, 45, 4)
    merged_k = torch.empty((*shape, 128), device=q.device, dtype=q.dtype)
    merged_v = torch.empty_like(merged_k)
    bias = torch.empty(shape, device=q.device, dtype=torch.float32)
    if output is None:
        output = torch.empty_like(q)
    if output.shape != q.shape or output.dtype != q.dtype or output.device != q.device or not output.is_contiguous():
        raise ValueError("invalid output tensor")
    _temporal_height_stable_merge_kernel[(45 * k_widths, batch * head_count)](
        q, k, v, seeds, merged_k, merged_v, bias, Q_OUTER=q_widths, K_OUTER=k_widths,
        HEADS=heads, ACTIVE_HEADS=head_count, HEAD_START=head_start, SCALE=128 ** -.5,
        BLOCK_J=16, num_warps=4, num_stages=1)
    # The pointer in the prototype-distance position is deliberately a dummy:
    # SINGLETON_QUERY_COLUMN=True compiles away every prototype/seed lookup.
    _hard_c4_720p_route_kernel[(1, batch * head_count, q_widths * 21)](
        q, k, v, merged_k, merged_v, bias, bias, bias, bias, bias, output,
        Q_OUTER=q_widths, K_OUTER=k_widths, TOTAL_HEADS=heads, ACTIVE_HEADS=head_count,
        HEAD_START=head_start, D_HEAD=128, HEAD_MAJOR=False, OUTPUT_HEAD_MAJOR=False,
        ROWS=45, COLS=21, C=4, BLOCK_I=64, BLOCK_TOKENS=64, LOOP_STAGES=3,
        INPUT_PRECISION="ieee", EXACT_INTRA_FRAME=False, EXACT_NEIGHBOR_WINDOW_COLS=0,
        RECORD_DISTORTION=False, QUERY_OUTER_OFFSET=0, SINGLETON_TOP_R=1,
        SINGLETON_QUERY_COLUMN=True, SINGLETON_SECONDARY_FRAME_STRIDE=1,
        SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False, QUERY_COLUMN_MAJOR=True,
        PACKED_SINGLETON_KV=False, SCALED_FP8_KV=False, COMPRESSED_FP8=False,
        SCALE=128 ** -.5, num_warps=4, num_stages=1)
    return (output, merged_k, merged_v, bias) if return_merged else output


class TemporalKeyHeightGCA:
    """Fixed original Wan geometry with twenty ordered 2-head chunks per call."""
    seed_mode = SEED_MODE

    def __init__(self):
        from . import hard_c4_wan21_14b_720p_triton as route_module
        from . import adaptive_seed_pytorch as candidates_module
        from . import temporal_key_height_gca_reference as reference_module
        paths = [Path(__file__).resolve(), Path(route_module.__file__).resolve(),
                 Path(candidates_module.__file__).resolve(), Path(reference_module.__file__).resolve()]
        self.source_receipt = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.core_calls = self.selector_calls = self.merge_calls = self.kernel_calls = self.route_calls = 0
        self.active = False

    def verify_kernel(self):
        for path, expected in self.source_receipt.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise RuntimeError("temporal key height source changed: " + path)

    def counts(self):
        return {"core_calls": self.core_calls, "selector_calls": self.selector_calls,
                "merge_wrapper_calls": self.merge_calls, "stable_kernel_calls": self.kernel_calls}

    def metadata(self):
        return {"seed_mode": self.seed_mode, "original_thw": [21, 45, 80],
                "operator_layout": "WHT", "key_axis": "T21", "query_axis": "H45",
                "remaining_axis": "W80",
                "layout": {"key_axis": "T", "query_axis": "H", "remaining_axis": "W",
                           "key_group_size": 21, "query_group_size": 45, "outer_size": 80,
                           "key_groups_per_outer": 45, "query_groups_per_outer": 21,
                           "compressed_slots_per_group": 5, "full_route": True},
                "candidate_count": 32, "candidate_geometry_count": 45,
                "candidate_geometry_meaning": "fixed local/global budget on H45; covers all45 height anchors",
                "probe_indices": [(i * 21) // 16 for i in range(16)],
                "rf": 0, "physical_reference_frame": 0, "rf0_layout": "B,1,H45,Wq80,heads,D",
                "seed_table_shape": [1, 1, 80, 45, 80, 40, 4], "seed_table_bytes": 46080000,
                "seed_table_independent_of_query_time": True, "head_chunk": 2,
                "merge_block_j": 16, "merge_key_padding": 32, "route_block_i": 64,
                "route_tokens_per_query": 80 * 45 * 5, "compressed_bulk_slots": 4,
                "singleton": "K[t_query,h_key,w_key], removed after full21 ownership before bulk normalization",
                "entropy_bias": True, "dense_fallback": False, "fp8": False,
                "full_compressed_route": True, "source_receipt": self.source_receipt,
                "dispatch_counts": self.counts(), "route_kernel_calls": self.route_calls}

    def __call__(self, query, key, value):
        if self.active:
            raise RuntimeError("nested temporal key height core")
        for tensor in (query, key, value):
            if (tuple(tensor.shape) != (1, 40, 75600, 128) or tensor.dtype != torch.bfloat16
                    or not tensor.is_cuda or tensor.device != query.device):
                raise ValueError("requires original 720p/81-frame CUDA BF16 head-major QKV")
        self.active = True
        try:
            # Physical t=0 is selected explicitly from original THW, not WHT[:, :1].
            q_rf0 = query.reshape(1, 40, 21, 45, 80, 128)[:, :, 0].permute(0, 2, 3, 1, 4).unsqueeze(1).contiguous()
            q, k, v = [x.reshape(1, 40, 21, 45, 80, 128).permute(0, 4, 3, 2, 1, 5)
                       .contiguous() for x in (query, key, value)]
            seeds = fisher_seed_rows(q_rf0, k)
            self.selector_calls += 1
            output = torch.empty_like(q)
            for head_start in range(0, 40, 2):
                merge_and_full_route(q, k, v, seeds, head_start=head_start, head_count=2, output=output)
                self.merge_calls += 1
                self.kernel_calls += 1
                self.route_calls += 1
            self.core_calls += 1
            return output.permute(0, 4, 3, 2, 1, 5).contiguous().reshape(1, 40, 75600, 128)
        finally:
            self.active = False


def create_temporal_key_height_core(output_dir=None):
    raise RuntimeError("retired C4+SQ1 path: C4 must contain exactly four merged K/V and no singleton")
    """Factory compatible with the isolated benchmark; does not write artifacts."""
    core = TemporalKeyHeightGCA()
    core.verify_kernel()
    return core


def validate_native(output_dir):
    """Explicit external CUDA correctness diagnostic, never a model path.

    Two synthetic W groups retain all T21/H45 positions. Independent CPU FP64
    references check RF0 Fisher selection, full21 ownership and stable bulks.
    The route oracle uses native BF16 representatives over all Wk/Hk groups.
    It checks current times0/1/16/20, H32..44, wq!=wk, temporal padding, ties,
    empty bulks and RF0 invariance to perturbing every nonzero query frame.
    One additional full40-head smoke exercises the actual twenty-chunk callable.
    """
    import json
    import numpy as np
    from .temporal_key_height_gca_reference import candidate_table, fisher_seeds, stable_bulk, softmax

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    report = {"valid": False, "passed": False,
              "kind": "external native primitive correctness diagnostic; not timing/quality",
              "outer_widths": 2, "query_heights": 45, "valid_key_times": 21,
              "physical_reference_frame": 0, "cases": {},
              "tolerances": {"seed_mismatches": 0, "score_relative": 2e-5,
                             "merge_max_abs": .01, "bias_max_abs": 2e-5,
                             "route_max_abs": .02, "route_rmse": .0025}}

    def require(condition, message):
        if not condition:
            raise RuntimeError("temporal key height native validation: " + message)

    def array(tensor):
        return tensor.detach().float().cpu().numpy().astype(np.float64)

    def rounded_bf16(x):
        return torch.from_numpy(np.asarray(x)).to(torch.bfloat16).float().numpy().astype(np.float64)

    def rf0(q):
        return q[:, :, :, 0].permute(0, 2, 1, 3, 4).unsqueeze(1).contiguous()

    try:
        require(torch.cuda.is_available(), "CUDA unavailable")
        core = create_temporal_key_height_core()
        report["source_receipt"] = core.source_receipt
        report["device"] = torch.cuda.get_device_name()
        candidates, anchors = candidate_table()
        from .adaptive_seed_pytorch import multiscale_candidate_table
        actual_candidates, actual_anchors = multiscale_candidate_table(45, 32, "cpu", geometry_count=45)
        require(np.array_equal(candidates, actual_candidates.numpy())
                and np.array_equal(anchors, actual_anchors.numpy()), "candidate table reference drift")
        generator = torch.Generator(device="cpu").manual_seed(1729)
        shape = (1, 2, 45, 21, 1, 128)
        cpu_q, cpu_k, cpu_v = [(torch.randn(shape, generator=generator) * .8).to(torch.bfloat16)
                              for _ in range(3)]
        for case in ("random_two_widths", "zero_query_ties_empty_bulks"):
            q = (cpu_q if case == "random_two_widths" else torch.zeros_like(cpu_q)).cuda()
            k, v = cpu_k.cuda(), cpu_v.cuda()
            audit = torch.empty((1, 1, 2, 45, 2, 1, 3, 32), dtype=torch.float32, device=q.device)
            seeds = fisher_seed_rows(rf0(q), k, score_audit=audit)
            perturbed_q = q.clone()
            perturbed_q[:, :, :, 1:] = (perturbed_q[:, :, :, 1:].float() * -3.0 + 7.0).to(q.dtype)
            perturbed_seeds = fisher_seed_rows(rf0(perturbed_q), k)
            require(torch.equal(seeds, perturbed_seeds), case + " RF0 changed with nonzero-frame queries")
            del perturbed_q, perturbed_seeds
            output, mk, mv, bias = merge_and_full_route(q, k, v, seeds, head_count=1, return_merged=True)
            torch.cuda.synchronize()
            for name, tensor in (("output", output), ("merged_k", mk), ("merged_v", mv), ("selector_scores", audit)):
                require(bool(torch.isfinite(tensor).all()), case + " nonfinite " + name)
            require(not bool(torch.isnan(bias).any()) and not bool(torch.isneginf(bias).any()), case + " invalid bias")
            s = seeds.cpu().numpy()[0, 0, :, :, :, 0]
            require(bool(((s >= 0) & (s < 45)).all()), case + " invalid seed index")
            require(bool((np.sort(s, axis=-1)[..., 1:] != np.sort(s, axis=-1)[..., :-1]).all()), case + " duplicate seeds")
            require(bool((s[..., 0] == np.arange(45)[None, :, None]).all()), case + " lost matched anchor")
            qn, kn, vn = array(q)[0, ..., 0, :], array(k)[0, ..., 0, :], array(v)[0, ..., 0, :]
            mkn, mvn, bn = array(mk)[0, :, :, 0], array(mv)[0, :, :, 0], array(bias)[0, :, :, 0]
            scores_native = array(audit)[0, 0, ..., 0, :, :]
            outn = array(output)[0, ..., 0, :]
            probe = np.arange(16) * 21 // 16
            stats = {"seed_cells": 0, "seed_mismatches": 0, "score_max_abs": 0., "score_max_rel": 0.,
                     "merge_cells": 0, "merge_max_abs": 0., "bias_max_abs": 0.,
                     "route_query_vectors": 0, "route_max_abs": 0., "route_squared_error": 0.,
                     "route_elements": 0, "tail_query_heights_checked": list(range(32, 45)),
                     "query_times_checked": [0, 1, 16, 20], "query_widths_checked": [0, 1],
                     "query_heights_checked": list(range(45)),
                     "key_widths_checked": [0, 1], "checked_key_heights": [0, 22, 44],
                     "selector_key_heights": list(range(45)), "cross_width_route": True,
                     "rf0_nonzero_frame_perturbation_invariant": True,
                     "all_outputs_finite": True, "empty_bulk_count": int(np.isposinf(bn).sum())}
            for key_width in range(2):
                for key_height in range(45):
                    for query_width in range(2):
                        logits = (qn[query_width, candidates[key_height], 0]
                                  @ kn[key_width, key_height, probe].T) * (128 ** -.5)
                        expected_seeds, expected_scores = fisher_seeds(logits, candidates[key_height], anchors[key_height])
                        stats["seed_cells"] += 1
                        stats["seed_mismatches"] += int(not np.array_equal(expected_seeds, s[key_width, key_height, query_width]))
                        score_error = float(np.max(np.abs(expected_scores - scores_native[key_width, key_height, query_width])))
                        stats["score_max_abs"] = max(stats["score_max_abs"], score_error)
                        score_relative = score_error / (1 + float(np.max(np.abs(expected_scores))))
                        stats["score_max_rel"] = max(stats["score_max_rel"], score_relative)
                        require(score_relative <= 2e-5, case + " Fisher score mismatch")
                        if key_height not in (0, 22, 44):
                            continue
                        for query_time in (0, 1, 16, 20):
                            actual_seeds = s[key_width, key_height, query_width]
                            full_logits = (qn[query_width, actual_seeds, query_time]
                                           @ kn[key_width, key_height].T) * (128 ** -.5)
                            weights, expected_bias, _ = stable_bulk(full_logits, query_time)
                            native_bias = bn[query_width, query_time, key_width, key_height]
                            require(np.array_equal(np.isposinf(expected_bias), np.isposinf(native_bias)), case + " empty bulk mismatch")
                            finite = np.isfinite(expected_bias)
                            stats["bias_max_abs"] = max(stats["bias_max_abs"], float(np.max(np.abs(expected_bias[finite] - native_bias[finite]))))
                            bf16_weights = rounded_bf16(weights)
                            expected_k = rounded_bf16(bf16_weights @ kn[key_width, key_height])
                            expected_v = rounded_bf16(bf16_weights @ vn[key_width, key_height])
                            merge_error = max(float(np.max(np.abs(expected_k - mkn[query_width, query_time, key_width, key_height]))),
                                              float(np.max(np.abs(expected_v - mvn[query_width, query_time, key_width, key_height]))))
                            stats["merge_max_abs"] = max(stats["merge_max_abs"], merge_error)
                            stats["merge_cells"] += 1
            for query_width in range(2):
                for query_time in (0, 1, 16, 20):
                    # Every Wk/Hk block participates, including wq!=wk.
                    key_bank = np.concatenate([mkn[query_width, query_time].reshape(-1, 128),
                                               kn[:, :, query_time].reshape(-1, 128)])
                    value_bank = np.concatenate([mvn[query_width, query_time].reshape(-1, 128),
                                                 vn[:, :, query_time].reshape(-1, 128)])
                    bank_bias = np.concatenate([bn[query_width, query_time].reshape(-1), np.zeros(2 * 45)])
                    logits = qn[query_width, :, query_time] @ key_bank.T * (128 ** -.5) - bank_bias
                    expected = softmax(logits) @ value_bank
                    error = expected - outn[query_width, :, query_time]
                    stats["route_query_vectors"] += 45
                    stats["route_elements"] += int(error.size)
                    stats["route_squared_error"] += float(np.square(error).sum())
                    stats["route_max_abs"] = max(stats["route_max_abs"], float(np.max(np.abs(error))))
            stats["route_rmse"] = math.sqrt(stats.pop("route_squared_error") / stats["route_elements"])
            report["cases"][case] = stats
            require(stats["seed_cells"] == 180 and stats["merge_cells"] == 48
                    and stats["route_query_vectors"] == 360 and stats["route_elements"] == 46080,
                    case + " diagnostic coverage count mismatch")
            require(stats["seed_mismatches"] == 0, case + " seed disagreement")
            require(stats["merge_max_abs"] <= .01, case + " merge vector disagreement")
            require(stats["bias_max_abs"] <= 2e-5, case + " entropy disagreement")
            require(stats["route_max_abs"] <= .02 and stats["route_rmse"] <= .0025, case + " route disagreement")
            if case == "zero_query_ties_empty_bulks":
                require(stats["empty_bulk_count"] == 2 * 21 * 2 * 45 * 3, "expected exactly three empty C4 bulk slots per cell")
            del q, k, v, seeds, audit, output, mk, mv, bias
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
                                  "merge_wrapper_calls": 20, "stable_kernel_calls": 20}
                and core.route_calls == 20, "full-shape actual dispatch counts changed")
        report["full_shape_smoke"] = {"qkv_shape": list(full_shape), "dtype": "bfloat16",
                                      "finite_output": True, "counts": core.counts(),
                                      "route_kernel_calls": core.route_calls,
                                      "synthetic_diagnostic_only": True}
        del full_qkv, full_output
        core.verify_kernel()
        report["rf0_nonzero_frame_perturbation_invariant"] = True
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
