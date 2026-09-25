"""Independent phase-average/native row check; no quality or timing claim.

The explicit rotation reference below does not import the production phase
transform. It sums each virtual time separately in NumPy FP64, then the
worker rounds that result once to BF16. Every nonzero result must match
bitwise; +0/-0 differences alone are recorded and allowed, with no numerical
tolerance. Original Q/K and spatial channels retain strict bitwise gates.
This checks fixed fixtures, not an assertion that different FP64 reduction
orders must round identically for every possible query.
"""
from __future__ import annotations

import hashlib
import math

TEMPORAL_WIDTH = 44
SPATIAL_WIDTH = 84
DOTS_PER_BANK = 32 * 16
DOT_DIAGNOSTIC_LIMIT = 8
FP32_UNIT_ROUNDOFF = 2.0 ** -24
DOT_TERMS = 128
DOT_GAMMA = (DOT_TERMS - 1) * FP32_UNIT_ROUNDOFF / (1 - (DOT_TERMS - 1) * FP32_UNIT_ROUNDOFF)


def dot_forward_error(dot32, dot64, absolute_product_sum, *, array):
    """Preset gamma127 forward bound; NumPy/Torch use identical arithmetic.

BF16 operands times these small integer keys have exact FP32 products.
Only their 127 additions can round: |fl(q.k)-q.k| <= gamma127 sum |q_d k_d|.
Inputs are promoted FP64 arrays; no TF32, overflow or underflow is admitted.
This audits Torch accumulation, not hidden native dot intermediates. Native
rows still face the separate exact independent-FP64 row gate.
"""
    error = array.abs(dot32 - dot64)
    bound = DOT_GAMMA * absolute_product_sum
    finite = (array.isfinite(dot32) & array.isfinite(dot64)
              & array.isfinite(absolute_product_sum) & array.isfinite(bound)
              & (absolute_product_sum >= 0))
    zero_violation = (bound == 0) & (error != 0)
    denominator = array.where(bound > 0, bound, array.ones_like(bound))
    ratio = array.where(zero_violation, array.full_like(error, float("inf")), error / denominator)
    violations = (~finite) | (error > bound) | zero_violation
    return error, bound, ratio, finite, violations, zero_violation


def explicit_phase_average(query, temporal_frames):
    """Explicit positive Wan time rotations, independent FP64 reference."""
    import numpy as np

    if (not isinstance(temporal_frames, int) or isinstance(temporal_frames, bool)
            or temporal_frames < 1):
        raise ValueError("temporal_frames must be a positive integer")
    query = np.asarray(query, dtype=np.float64)
    if query.ndim < 1 or query.shape[-1] != 128 or not np.isfinite(query).all():
        raise ValueError("finite original queries with head dimension 128 required")
    result = query.copy()
    result[..., :TEMPORAL_WIDTH] = 0.0
    even, odd = query[..., :TEMPORAL_WIDTH:2], query[..., 1:TEMPORAL_WIDTH:2]
    frequency = 10000.0 ** (-np.arange(22, dtype=np.float64) / 22.0)
    for frame in range(temporal_frames):
        angle = frame * frequency
        cosine, sine = np.cos(angle), np.sin(angle)
        result[..., :TEMPORAL_WIDTH:2] += even * cosine - odd * sine
        result[..., 1:TEMPORAL_WIDTH:2] += even * sine + odd * cosine
    result[..., :TEMPORAL_WIDTH] /= temporal_frames
    return result


def validate_phase_record(record, banks, frames):
    """Independent parent/worker fail-closed gates; no score-value tolerance."""
    zero_fields = (
        "anchor_mismatches", "nonunique_banks", "invalid_candidate_banks",
        "score_audit_mismatched_banks", "transform_nonzero_bit_mismatches",
        "spatial_bit_mismatches", "q_input_bit_mismatches", "k_input_bit_mismatches",
        "fp32_dot_bound_violations", "fp32_dot_zero_bound_violations",
        "nonfinite_dot_values", "nonfinite_score_values",
    )
    true_fields = (
        "score_audit_on_off_equal", "transform_equal_except_zero_sign", "spatial_bitwise_equal",
        "q_input_unchanged", "k_input_unchanged", "fp32_dot_bound_comparison_passed",
        "transformed_q_contiguous",
    )
    raw_bits = record.get("transform_bit_mismatches")
    signed_zeros = record.get("signed_zero_bit_differences")
    dot_mismatches = record.get("fp32_dot_mismatches")
    max_dot_error = record.get("max_fp32_dot_abs_error")
    max_dot_ratio = record.get("max_fp32_dot_bound_ratio")
    if (not isinstance(raw_bits, int) or isinstance(raw_bits, bool)
            or not isinstance(signed_zeros, int) or isinstance(signed_zeros, bool)
            or not 0 <= signed_zeros <= 45 * 80 * 2 * 128
            or raw_bits != signed_zeros
            or record.get("transform_bitwise_equal") is not (raw_bits == 0)
            or record.get("transform_zero_sign_policy") != "nonzero_bits_exact_zero_sign_ignored"
            or any(record.get(name) != 0 for name in zero_fields)
            or any(record.get(name) is not True for name in true_fields)
            or record.get("temporal_frames") != frames
            or record.get("temporal_width") != TEMPORAL_WIDTH
            or record.get("spatial_width") != SPATIAL_WIDTH
            or record.get("frame_banks") != [7200] * frames
            or record.get("frame_mismatched_banks") != [0] * frames
            or record.get("q_dtype") != "torch.bfloat16"
            or record.get("k_dtype") != "torch.bfloat16"
            or record.get("transformed_q_dtype") != "torch.bfloat16"
            or record.get("q_shape") != [1, 1, 45, 80, 2, 128]
            or record.get("k_shape") != [1, frames, 45, 80, 2, 128]
            or record.get("transformed_q_shape") != [1, 1, 45, 80, 2, 128]
            or record.get("fp32_dot_values_checked") != banks * DOTS_PER_BANK
            or not isinstance(dot_mismatches, int) or isinstance(dot_mismatches, bool)
            or not 0 <= dot_mismatches <= banks * DOTS_PER_BANK
            or record.get("fp32_reference_dots_equal_fp64") is not (dot_mismatches == 0)
            or not isinstance(max_dot_error, (int, float)) or not math.isfinite(max_dot_error)
            or max_dot_error < 0 or ((max_dot_error == 0) != (dot_mismatches == 0))
            or not isinstance(max_dot_ratio, (int, float)) or not math.isfinite(max_dot_ratio)
            or not 0 <= max_dot_ratio <= 1
            or record.get("fp32_unit_roundoff") != FP32_UNIT_ROUNDOFF
            or record.get("fp32_dot_terms") != DOT_TERMS
            or record.get("fp32_dot_gamma") != DOT_GAMMA
            or record.get("score_audit_dtype") != "torch.float32"
            or record.get("score_audit_values_observed") != banks * 96
            or record.get("score_values_compared") != 0
            or record.get("score_comparison_performed") is not False
            or record.get("selector_block_j") != 4
            or record.get("oracle_implementation") != "analysis.anchor_fullgrid_seed.anchor_indices"
            or record.get("fixture") != "independent_dense_integer_qk_-4_to_4"):
        raise RuntimeError("phase-average transform, input, dot, structure or audit gate failed")


def check_phase_case(torch, seed, frames, scale):
    from analysis.anchor_fullgrid_seed import anchor_indices
    from analysis.check_barycentric_native import Q_SHAPE, decode_bank_ids
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    from attention.phase_averaged_seed import (
        phase_averaged_query, phase_averaged_fisher_seed_rows,
    )

    torch.manual_seed(seed)
    # Every original element/head/key frame is independently generated. Dense
    # integer K is retained. Legal accumulated FP32 roundoff is observed and
    # checked against the preset gamma127 bound, not rejected as non-exact.
    q = torch.randint(-4, 5, Q_SHAPE, device="cuda").to(torch.bfloat16)
    k_shape = (1, frames, 45, 80, 2, 128)
    k = torch.randint(-4, 5, k_shape, device="cuda").to(torch.bfloat16)
    q_before, k_before = q.clone(), k.clone()
    transformed = phase_averaged_query(q, frames)
    reference_array = explicit_phase_average(q_before.double().cpu().numpy(), frames)
    reference_q = torch.from_numpy(reference_array).to(device=q.device).to(torch.bfloat16)
    if transformed.shape != q.shape or transformed.dtype != torch.bfloat16:
        raise AssertionError("wrong transformed RF0 query shape or dtype")
    transform_difference = transformed.view(torch.int16) != reference_q.view(torch.int16)
    signed_zero_difference = transform_difference & (transformed == 0) & (reference_q == 0)
    transform_errors = int(transform_difference.sum())
    signed_zero_errors = int(signed_zero_difference.sum())
    nonzero_transform_errors = int((transform_difference & ~signed_zero_difference).sum())
    spatial_errors = int((transformed[..., 44:].view(torch.int16)
                          != q_before[..., 44:].view(torch.int16)).sum())
    kwargs = dict(candidate_count=32, candidate_geometry_count=45, block_j=4,
                  probe_count=16, global_probe_count=256, gs_num_warps=4)
    # Only original Q/K enter the native wrapper, never independent reference Q.
    native = phase_averaged_fisher_seed_rows(q, k, scale, **kwargs)
    scores = torch.full((1, 1, frames, 45, 80, 2, 3, 32), float("nan"),
                        dtype=torch.float32, device=q.device)
    audited = phase_averaged_fisher_seed_rows(q, k, scale, score_audit=scores, **kwargs)
    expected_shape = (1, 1, frames, 45, 80, 2, 4)
    for value in (native, audited):
        if tuple(value.shape) != expected_shape or value.dtype != torch.uint8:
            raise AssertionError("wrong native real-row table shape or dtype")
    native, audited = native.reshape(-1, 4), audited.reshape(-1, 4)
    banks = frames * 7200
    expected = torch.empty_like(native)
    table, anchors = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    # Confirm that the reference's slot-zero anchor is the actual key row.
    if not torch.equal(table.gather(1, anchors.long()[:, None]).flatten().long(),
                       torch.arange(45, device=q.device)):
        raise AssertionError("candidate anchor is not the matched actual row")
    invalid = anchor_errors = nonunique = dot_errors = 0
    bound_errors = zero_bound_errors = nonfinite_dots = 0
    max_dot_error = 0.0
    max_dot_ratio = 0.0
    dot_diagnostics = []
    for start in range(0, banks, 256):
        ids = torch.arange(start, min(start + 256, banks), device=q.device)
        frame, row, col, head = decode_bank_ids(ids)
        real = table[row].long()
        qq = reference_q[0, 0, real, col[:, None], head[:, None]].float()
        kk = k_before[0, frame[:, None], row[:, None],
                      torch.arange(0, 80, 5, device=q.device), head[:, None]].float()
        dot32 = torch.bmm(qq, kk.transpose(1, 2))
        dot64 = torch.bmm(qq.double(), kk.double().transpose(1, 2))
        absolute_sum = torch.bmm(qq.double().abs(), kk.double().abs().transpose(1, 2))
        difference, bound, ratio, finite, violations, zero_violations = dot_forward_error(
            dot32.double(), dot64, absolute_sum, array=torch)
        dot_errors += int((difference != 0).sum())
        bound_errors += int(violations.sum())
        zero_bound_errors += int(zero_violations.sum())
        nonfinite_dots += int((~finite).sum())
        max_dot_error = (max(max_dot_error, float(difference.max()))
                         if max_dot_error is not None and bool(torch.isfinite(difference).all()) else None)
        max_dot_ratio = (max(max_dot_ratio, float(ratio.max()))
                         if max_dot_ratio is not None and bool(torch.isfinite(ratio).all()) else None)
        remaining = DOT_DIAGNOSTIC_LIMIT - len(dot_diagnostics)
        if remaining > 0:
            for position, candidate, probe in (difference != 0).nonzero()[:remaining].cpu().tolist():
                dot_diagnostics.append({"bank_id": int(ids[position]), "frame": int(frame[position]),
                    "row": int(row[position]), "column": int(col[position]), "head": int(head[position]),
                    "candidate_slot": candidate, "probe_slot": probe,
                    "dot_fp32": float(dot32[position, candidate, probe]),
                    "dot_fp64": float(dot64[position, candidate, probe]),
                    "absolute_product_sum": float(absolute_sum[position, candidate, probe]),
                    "forward_error_bound": float(bound[position, candidate, probe]),
                    "forward_error_ratio": (float(ratio[position, candidate, probe])
                        if bool(torch.isfinite(ratio[position, candidate, probe])) else None),
                    "bound_violation": bool(violations[position, candidate, probe])})
        # Independent FP64 dots still define every reference row. The bounded
        # FP32 arithmetic observation does not relax exact native row equality.
        z = dot64 * scale
        slots = anchor_indices(z, anchors[row])
        expected[ids] = real.gather(1, slots).to(torch.uint8)
        actual = native[ids].long()
        invalid += int(((real[:, :, None] == actual[:, None, :]).sum(1) != 1).any(-1).sum())
        anchor_errors += int((actual[:, 0] != row).sum())
        nonunique += int((actual.sort(-1).values.diff(dim=-1) == 0).any(-1).sum())
    equal, audit_equal = (native == expected).all(-1), (native == audited).all(-1)
    q_errors = int((q.view(torch.int16) != q_before.view(torch.int16)).sum())
    k_errors = int((k.view(torch.int16) != k_before.view(torch.int16)).sum())
    return {"input_seed": seed, "key_frames": frames, "temporal_frames": frames, "scale": scale,
        "method": "phase_averaged", "banks": banks, "q_shape": list(q.shape), "k_shape": list(k.shape),
        "q_dtype": str(q.dtype), "k_dtype": str(k.dtype),
        "fixture": "independent_dense_integer_qk_-4_to_4",
        "transformed_q_shape": list(transformed.shape), "transformed_q_dtype": str(transformed.dtype),
        "transformed_q_contiguous": transformed.is_contiguous(),
        "temporal_width": TEMPORAL_WIDTH, "spatial_width": SPATIAL_WIDTH,
        "transform_bit_mismatches": transform_errors, "transform_bitwise_equal": transform_errors == 0,
        "signed_zero_bit_differences": signed_zero_errors,
        "transform_nonzero_bit_mismatches": nonzero_transform_errors,
        "transform_equal_except_zero_sign": nonzero_transform_errors == 0,
        "transform_zero_sign_policy": "nonzero_bits_exact_zero_sign_ignored",
        "spatial_bit_mismatches": spatial_errors, "spatial_bitwise_equal": spatial_errors == 0,
        "q_input_bit_mismatches": q_errors, "q_input_unchanged": q_errors == 0,
        "k_input_bit_mismatches": k_errors, "k_input_unchanged": k_errors == 0,
        "fp32_dot_mismatches": dot_errors, "fp32_reference_dots_equal_fp64": dot_errors == 0,
        "fp32_dot_values_checked": banks * DOTS_PER_BANK, "max_fp32_dot_abs_error": max_dot_error,
        "fp32_unit_roundoff": FP32_UNIT_ROUNDOFF, "fp32_dot_terms": DOT_TERMS,
        "fp32_dot_gamma": DOT_GAMMA, "fp32_dot_bound_violations": bound_errors,
        "fp32_dot_zero_bound_violations": zero_bound_errors, "nonfinite_dot_values": nonfinite_dots,
        "max_fp32_dot_bound_ratio": max_dot_ratio,
        "fp32_dot_bound_comparison_passed": bound_errors == 0 and nonfinite_dots == 0,
        "dot_diagnostics": dot_diagnostics,
        "dot_check_scope": "Torch FP32 versus FP64; native internal accumulation not directly observed",
        "group_agreement": float(equal.float().mean()), "mismatched_banks": int((~equal).sum()),
        "anchor_mismatches": anchor_errors, "nonunique_banks": nonunique,
        "invalid_candidate_banks": invalid, "selector_block_j": 4,
        "oracle_implementation": "analysis.anchor_fullgrid_seed.anchor_indices",
        "frame_banks": [7200] * frames,
        "frame_mismatched_banks": [int((~part).sum()) for part in equal.reshape(frames, -1)],
        "score_audit_on_off_equal": bool(audit_equal.all()),
        "score_audit_mismatched_banks": int((~audit_equal).sum()),
        "score_audit_dtype": str(scores.dtype), "score_audit_values_observed": scores.numel(),
        "nonfinite_score_values": int((~torch.isfinite(scores)).sum()),
        "score_values_compared": 0, "score_comparison_performed": False,
        "score_scope": "FP32 audit completeness/on-off rows only; no FP64 score-value claim",
        "native_sha256": hashlib.sha256(native.cpu().numpy().tobytes()).hexdigest()}
