"""Audit native conditional-Jeffreys scores at their actual selected prefixes.

The array-level auditor imports only NumPy and is CPU-testable. ``verify_native``
is an explicitly invoked GPU diagnostic, not an inference path or a timing
benchmark. It shares one G256 normalizer with the selector, checks every native
seed row, and compares 600 fixed banks against independent float64 partition
arithmetic. Near-tie row disagreement is measured by rank defect, not hidden by
requiring a free-running reference selector to follow the same discrete path.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from .conditional_jeffreys_seed import partition_metrics
except ImportError:
    from conditional_jeffreys_seed import partition_metrics


ABS_TOL, REL_TOL = 2e-5, 1e-4
QUERY_COLUMNS = (0, 1, 20, 21, 40, 41, 60, 61, 75, 76)
KEY_FRAMES, KEY_ROWS = (0, 10, 20), (0, 11, 22, 33, 44)
HEAD_LABELS = (0, 10, 20, 30)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def bank_coordinates():
    """Return 600 rows of [head_slot, query_column, key_frame, key_row]."""
    return np.asarray([(h, c, f, r) for h in range(4) for c in QUERY_COLUMNS
                       for f in KEY_FRAMES for r in KEY_ROWS], dtype=np.int64)


def reference_logits_from_vectors(queries, keys, scale):
    """Accumulate sampled original-value QK products in FP64, then scale.

    BF16 source values cast to FP32 are exact. Casting those values to FP64
    BEFORE the product avoids treating another FP32 accumulation order as a
    ground truth for the native FP32 kernel. No tolerance or score is changed.
    """
    queries, keys = np.asarray(queries), np.asarray(keys)
    _require(queries.ndim == keys.ndim == 3 and queries.shape[0] == keys.shape[0]
             and queries.shape[-1] == keys.shape[-1] and min(queries.shape) > 0
             and min(keys.shape) > 0 and queries.dtype.kind in "fiu" and keys.dtype.kind in "fiu",
             "require matched nonempty [banks,rows,dim] and [banks,keys,dim] vectors")
    _require(np.isfinite(queries).all() and np.isfinite(keys).all()
             and np.isfinite(scale) and scale > 0, "nonfinite vectors or invalid scale")
    return (queries.astype(np.float64) @ keys.astype(np.float64).transpose(0, 2, 1)) * float(scale)


def _comparison(actual, expected, atol, rtol):
    absolute = np.abs(actual - expected)
    tolerance = atol + rtol * np.abs(expected)
    failed = absolute > tolerance
    return {
        "valid": bool(np.isfinite(actual).all() and np.isfinite(expected).all()
                      and not failed.any()),
        "values": int(actual.size), "failed_values": int(failed.sum()),
        "max_absolute_error": float(absolute.max()),
        "max_relative_error_with_atol_floor": float(
            (absolute / np.maximum(np.abs(expected), atol)).max()),
        "max_error_over_tolerance": float((absolute / tolerance).max()),
        "first_failures": np.argwhere(failed)[:12].tolist(),
    }


def audit_score_arrays(local_logits, global_log2_normalizer, candidate_rows,
                       native_rows, native_scores, query_columns, anchors,
                       *, atol=ABS_TOL, rtol=REL_TOL):
    """Replay actual prefixes and compare all candidate ranking scores.

    ``local_logits`` is [banks, candidates, 16] in natural-log units;
    ``global_log2_normalizer`` is [banks,candidates]. ``native_scores`` is
    [banks,3,candidates] after nonnegative-Gamma clamp but before used masking.
    The four selected rows are actual query-row numbers, not candidate slots.
    Candidate rows are increasing, hence low-slot and low-row ties coincide.
    Returns JSON-serializable report plus raw reference scores/Gamma/rank arrays.
    Floating tolerances are fixed a priori for FP32 accumulation/transcendentals;
    they are not a license to discard a failed bank or change the algorithm.
    """
    logits = np.asarray(local_logits, dtype=np.float64)
    norm = np.asarray(global_log2_normalizer, dtype=np.float64)
    rows, selected = np.asarray(candidate_rows), np.asarray(native_rows)
    scores = np.asarray(native_scores, dtype=np.float64)
    columns, anchors = np.asarray(query_columns), np.asarray(anchors)
    _require(logits.ndim == 3 and logits.shape[0] > 0 and logits.shape[1] >= 4
             and logits.shape[2] == 16, "require nonempty [banks,R>=4,16] logits")
    banks, count, _ = logits.shape
    _require(norm.shape == rows.shape == (banks, count)
             and selected.shape == (banks, 4) and scores.shape == (banks, 3, count)
             and columns.shape == anchors.shape == (banks,), "bad audit shapes")
    _require(all(np.isfinite(x).all() for x in (logits, norm, scores)),
             "nonfinite input scores/logits/normalizer")
    _require(all(x.dtype.kind in "iu" for x in (rows, selected, columns, anchors)),
             "rows, columns and anchors must be integer arrays")
    _require(np.all((rows >= 0) & (rows < 45)) and np.all(np.diff(rows, axis=1) > 0)
             and np.all((columns >= 0) & (columns < 80))
             and np.all((anchors >= 0) & (anchors < count)), "invalid candidate geometry")
    _require(np.isfinite(atol) and np.isfinite(rtol) and atol > 0 and rtol >= 0,
             "invalid fixed tolerance")
    membership = selected[:, :, None] == rows[:, None, :]
    seed_checks = {
        "anchor_matches": bool(np.array_equal(selected[:, 0], rows[np.arange(banks), anchors])),
        "four_distinct": bool(np.all(np.diff(np.sort(selected, axis=1), axis=1) > 0)),
        "all_in_bank": bool(membership.any(axis=-1).all()),
    }
    report = {
        "valid": False, "banks": banks, "checked_candidate_scores": banks * 3 * count,
        "seed_checks": seed_checks, "atol": atol, "rtol": rtol,
        "score_semantics": "common-max-normalized mass * max(conditional Jeffreys in bits,0); before used mask",
        "reference": "float64 partition_metrics evaluated at each actual native prefix",
        "tolerance_reason": "Prespecified FP32 dot, reduction and exp/log roundoff allowance; failed values remain failures",
    }
    if not all(seed_checks.values()):
        report["failure"] = "Invalid native rows prevent actual-prefix score replay"
        return report, {}
    slots = membership.argmax(axis=-1)
    local_log2_z = np.logaddexp.reduce(logits, axis=-1) / math.log(2)
    log2_mass = local_log2_z - norm
    weights = np.exp2(log2_mass - log2_mass.max(axis=-1, keepdims=True))
    singleton = np.where(columns % 5 == 0, columns // 5, -1)
    reference, gammas = np.empty_like(scores), np.empty_like(scores)
    rank_defect = np.empty((banks, 3), dtype=np.float64)
    rank_tolerance = np.empty_like(rank_defect)
    best_slots = np.empty((banks, 3), dtype=np.int64)
    used = np.zeros((banks, count), dtype=bool)
    batch = np.arange(banks)
    for step in range(3):
        used[batch, slots[:, step]] = True
        seed_logits = logits[batch[:, None], slots[:, :step + 1]]
        gamma = partition_metrics(logits, seed_logits, singleton)["jeffreys"] / math.log(2)
        gammas[:, step] = gamma
        reference[:, step] = weights * np.maximum(gamma, 0.0)
        eligible = np.where(used, -np.inf, reference[:, step])
        best = eligible.argmax(axis=-1)
        best_slots[:, step] = best
        chosen = slots[:, step + 1]
        rank_defect[:, step] = eligible[batch, best] - eligible[batch, chosen]
        rank_tolerance[:, step] = atol + rtol * np.abs(eligible[batch, best])
    score_check = _comparison(scores, reference, atol, rtol)
    rank_fail = rank_defect > rank_tolerance
    report.update(
        score_comparison=score_check,
        rank_comparison={
            "valid": bool(not rank_fail.any()), "choices": int(rank_defect.size),
            "failed_choices": int(rank_fail.sum()),
            "nonidentical_reference_argmax_choices": int(np.count_nonzero(best_slots != slots[:, 1:])),
            "max_absolute_rank_defect": float(rank_defect.max()),
            "max_rank_defect_over_tolerance": float((rank_defect / rank_tolerance).max()),
            "first_failures": np.argwhere(rank_fail)[:12].tolist(),
        },
        min_unclamped_reference_gamma_bits=float(gammas.min()),
        negative_reference_gamma_count=int(np.count_nonzero(gammas < 0)),
        valid=bool(score_check["valid"] and not rank_fail.any()),
    )
    return report, {"reference_scores": reference, "reference_gamma_bits": gammas,
                    "reference_mass_weights": weights, "native_candidate_slots": slots,
                    "reference_best_slots_at_native_prefix": best_slots,
                    "reference_rank_defect": rank_defect,
                    "reference_rank_tolerance": rank_tolerance}


def audit_all_rows(native_rows, candidate_table):
    """Check every [B,1,F,45,80,H,4] group, not just sampled banks."""
    rows, table = np.asarray(native_rows), np.asarray(candidate_table)
    _require(rows.ndim == 7 and rows.shape[0:2] == (1, 1)
             and rows.shape[2:5] == (21, 45, 80) and rows.shape[-2:] == (4, 4)
             and rows.dtype.kind in "iu", "wrong full native row layout")
    _require(table.shape == (45, 32) and table.dtype.kind in "iu"
             and np.all((table >= 0) & (table < 45))
             and np.all(np.diff(table, axis=1) > 0), "wrong full candidate table")
    bounds = (rows >= 0) & (rows < 45)
    anchor_expected = np.broadcast_to(np.arange(45)[None, None, None, :, None, None], rows.shape[:-1])
    anchor_fail = rows[..., 0] != anchor_expected
    distinct_fail = np.any(np.diff(np.sort(rows.astype(np.int64), axis=-1), axis=-1) <= 0, axis=-1)
    lookup = np.zeros((45, 45), dtype=bool)
    lookup[np.arange(45)[:, None], table] = True
    row_axis = np.arange(45)[None, None, None, :, None, None, None]
    membership = lookup[row_axis, np.clip(rows, 0, 44)] & bounds
    report = {"groups": int(rows[..., 0].size), "out_of_range_rows": int((~bounds).sum()),
              "anchor_failures": int(anchor_fail.sum()), "non_distinct_groups": int(distinct_fail.sum()),
              "out_of_bank_rows": int((~membership).sum())}
    report["valid"] = not any(report[key] for key in report if key != "groups")
    return report


def audit_recorded_intermediates(intermediate, candidate_rows, native_rows,
                                 global_log2_normalizer, query_column,
                                 independent_log2_logits):
    """Localize one failed bank without replacing the end-to-end parity gate.

    The 19 native planes contain dot/logits/centered logits/logp/p/mass/local Z,
    then three planes each of ownership, owner logp, unclamped Gamma, priority.
    Full FP64 replay from actual native logits separates dot/front-end changes
    from entropy arithmetic. A second replay fixes the recorded native owner,
    logp, p and mass, isolating the remaining conditional-score arithmetic.
    """
    values = np.asarray(intermediate, dtype=np.float64)
    _require(values.shape == (19, 32, 16) and np.isfinite(values).all(),
             "expected finite native intermediate [19,32,16]")
    rows, selected = np.asarray(candidate_rows), np.asarray(native_rows)
    _require(rows.shape == (32,) and selected.shape == (4,), "bad intermediate row shapes")
    membership = selected[:, None] == rows[None, :]
    _require(membership.any(axis=-1).all(), "intermediate native row is not a candidate")
    slots = membership.argmax(axis=-1)
    native_owners = values[7:10, 0].astype(np.int64)
    _require(np.array_equal(values[7:10], np.broadcast_to(native_owners[:, None, :], (3, 32, 16))),
             "recorded owner planes are not repeated exact integer slots")
    _require(all(np.all((native_owners[step] >= 0) & (native_owners[step] <= step))
                 for step in range(3)), "recorded owner slot out of range")
    for start, stop in ((5, 7), (13, 19)):
        _require(np.array_equal(values[start:stop], np.broadcast_to(values[start:stop, :, :1], values[start:stop].shape)),
                 "recorded row-valued planes do not repeat across keys")
    _require(np.array_equal(values[10:13], np.broadcast_to(values[10:13, :1, :], (3, 32, 16))),
             "recorded owner-logp planes do not repeat across candidates")
    anchors = np.asarray([slots[0]], dtype=np.int64)
    native_scores = values[16:19, :, 0]
    native_logits_report, native_logits_arrays = audit_score_arrays(
        values[1][None] * math.log(2), np.asarray(global_log2_normalizer)[None],
        rows[None], selected[None], native_scores[None], np.asarray([query_column]), anchors)
    profile_gamma = np.zeros((3, 32), dtype=np.float64)
    profile_owner_mismatches, independent_owner_mismatches = [], []
    independent = np.asarray(independent_log2_logits, dtype=np.float64)
    independent_logp = independent - np.logaddexp.reduce(independent * math.log(2), axis=-1)[:, None] / math.log(2)
    profile_probability, profile_logp = values[4], values[3]
    keep = np.arange(0, 80, 5) != query_column
    for step in range(3):
        expected_owner = np.argmax(profile_logp[slots[:step + 1]], axis=0)
        independent_owner = np.argmax(independent_logp[slots[:step + 1]], axis=0)
        profile_owner_mismatches.append(int(np.count_nonzero(expected_owner != native_owners[step])))
        independent_owner_mismatches.append(int(np.count_nonzero(independent_owner != native_owners[step])))
        for cell in range(step + 1):
            mask = (native_owners[step] == cell) & keep
            if not mask.any():
                continue
            owner_logp = values[10 + step, 0, mask]
            centered = owner_logp - owner_logp.max()
            logw = centered - np.log2(np.exp2(centered).sum())
            weight = np.exp2(logw)
            residual = profile_logp[:, mask] - logw[None, :]
            center = np.sum(residual * weight[None, :], axis=-1)
            profile_gamma[step] += np.sum(profile_probability[:, mask] * (residual - center[:, None]), axis=-1)
    profile_scores = values[5, :, 0][None] * np.maximum(profile_gamma, 0)
    report = {
        "scope": "Additional localization only; does not replace or relax original independent-reference gate",
        "actual_native_logits_replay": native_logits_report,
        "recorded_profile_fixed_owner_score_comparison": _comparison(native_scores, profile_scores, ABS_TOL, REL_TOL),
        "recorded_profile_fixed_owner_gamma_comparison": _comparison(values[13:16, :, 0], profile_gamma, ABS_TOL, REL_TOL),
        "native_vs_independent_logits": _comparison(values[1], independent, ABS_TOL, REL_TOL),
        "native_owner_vs_recorded_logp_mismatches_by_prefix": profile_owner_mismatches,
        "native_owner_vs_independent_logp_mismatches_by_prefix": independent_owner_mismatches,
        "max_native_probability_sum_error": float(np.abs(profile_probability.sum(axis=-1) - 1).max()),
        "native_probability_vs_exp2_recorded_logp": _comparison(profile_probability, np.exp2(profile_logp), ABS_TOL, REL_TOL),
    }
    arrays = {"native_intermediate_planes": np.asarray(intermediate),
              "native_profile_fp64_gamma_bits": profile_gamma,
              "native_profile_fp64_scores": profile_scores,
              **{f"actual_native_logits_{key}": value for key, value in native_logits_arrays.items()}}
    return report, arrays


def verify_native(q, k, *, reuse_owner_logp=True, collect_intermediate=True):
    """Run the explicit parity diagnostic on one saved BF16 four-head capture.

    q/k: CUDA BF16 [1,75600,4,128]. Returns (JSON report, NumPy arrays).
    The root runner owns input provenance, exclusivity, timing and output files.
    This helper makes exactly one native G256 normalizer call, then supplies it
    to the selector. Independent FP32 non-TF32 products are diagnostic only.
    """
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    from attention.entropy_mass_seed_triton import global_probe_log_norm_triton
    from attention.conditional_jeffreys_seed_triton import conditional_jeffreys_seed_rows_triton

    _require(tuple(q.shape) == tuple(k.shape) == (1, 75600, 4, 128)
             and q.dtype == k.dtype == torch.bfloat16 and q.is_cuda and k.is_cuda
             and q.device == k.device, "require original CUDA BF16 [1,75600,4,128] QK")
    scale, log2e = 128 ** -0.5, 1.0 / math.log(2)
    q_rf0 = q[:, :3600].reshape(1, 1, 45, 80, 4, 128).contiguous()
    keys = k.reshape(1, 21, 45, 80, 4, 128).contiguous()
    normalizer = global_probe_log_norm_triton(q_rf0, keys, scale, global_probe_count=256)
    score_tensor = torch.empty((1, 1, 21, 45, 80, 4, 3, 32), device=q.device, dtype=torch.float32)
    native = conditional_jeffreys_seed_rows_triton(
        q_rf0, keys, scale, candidate_count=32, candidate_geometry_count=45,
        block_j=4, probe_count=16, global_probe_count=256, output_dtype=torch.uint8,
        gs_num_warps=4, global_normalizer=normalizer, reuse_owner_logp=reuse_owner_logp,
        score_audit=score_tensor)
    intermediate = None
    trace_invariant = None
    if collect_intermediate:
        intermediate = torch.empty((19, 32, 16), device=q.device, dtype=torch.float32)
        traced_scores = torch.empty_like(score_tensor)
        traced_native = conditional_jeffreys_seed_rows_triton(
            q_rf0, keys, scale, candidate_count=32, candidate_geometry_count=45,
            block_j=4, probe_count=16, global_probe_count=256, output_dtype=torch.uint8,
            gs_num_warps=4, global_normalizer=normalizer, reuse_owner_logp=reuse_owner_logp,
            score_audit=traced_scores, intermediate_audit=intermediate,
            intermediate_audit_bank=(0, 10, 22, 20, 3))
        trace_invariant = {"native_rows_bitwise_equal": bool(torch.equal(native, traced_native)),
                           "all_scores_bitwise_equal": bool(torch.equal(score_tensor, traced_scores))}
        trace_invariant["valid"] = all(trace_invariant.values())
        del traced_scores, traced_native
    table_tensor, anchor_tensor = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    table = table_tensor.cpu().numpy().astype(np.int64)
    anchor_table = anchor_tensor.cpu().numpy().astype(np.int64)
    full_rows = native.cpu().numpy()
    all_rows = audit_all_rows(full_rows, table)
    coordinates = bank_coordinates()
    head, column, frame, row = coordinates.T
    gpu_head, gpu_column, gpu_frame, gpu_row = [torch.as_tensor(a, device=q.device) for a in coordinates.T]
    candidate_rows = table[row]
    query_indices = candidate_rows * 80 + column[:, None]
    key_indices = frame[:, None] * 3600 + row[:, None] * 80 + np.arange(0, 80, 5)[None, :]
    query_index_gpu = torch.as_tensor(query_indices, device=q.device)
    key_index_gpu = torch.as_tensor(key_indices, device=q.device)
    # Set and restore the global precision option around diagnostic products.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        sampled_q = q[0, query_index_gpu, gpu_head[:, None]].float()
        sampled_k = k[0, key_index_gpu, gpu_head[:, None]].float()
        local_log2 = torch.bmm(sampled_q, sampled_k.transpose(-1, -2)) * (scale * log2e)
        # Independent global probe products, all 45 rows of the sampled columns.
        all_query_indices = (torch.arange(45, device=q.device)[:, None] * 80
                             + torch.as_tensor(QUERY_COLUMNS, device=q.device)[None, :]).reshape(-1)
        global_key_indices = torch.arange(256, device=q.device) * (75600 // 256)
        global_q = q[0, all_query_indices].permute(1, 0, 2).float()
        global_k = k[0, global_key_indices].permute(1, 0, 2).float()
        global_logits2 = torch.bmm(global_q, global_k.transpose(-1, -2)) * (scale * log2e)
        global_logits2_numpy = global_logits2.cpu().numpy().astype(np.float64)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    local_log2_numpy = local_log2.cpu().numpy()
    fp32_reference_logits = local_log2_numpy.astype(np.float64) * math.log(2)
    sampled_query_numpy = sampled_q.cpu().numpy()
    sampled_key_numpy = sampled_k.cpu().numpy()
    logits = reference_logits_from_vectors(sampled_query_numpy, sampled_key_numpy, scale)
    sampled_normalizer = normalizer[0, 0].reshape(3600, 4)[query_index_gpu, gpu_head[:, None]].cpu().numpy()
    sampled_rows = full_rows[0, 0, frame, row, column, head].astype(np.int64)
    sampled_scores = score_tensor[0, 0, gpu_frame, gpu_row, gpu_column, gpu_head].cpu().numpy()
    report, reference_arrays = audit_score_arrays(logits, sampled_normalizer, candidate_rows,
                                                  sampled_rows, sampled_scores, column, anchor_table[row])
    fp32_reference_report, fp32_reference_arrays = audit_score_arrays(
        fp32_reference_logits, sampled_normalizer, candidate_rows,
        sampled_rows, sampled_scores, column, anchor_table[row])
    global_reference = np.logaddexp.reduce(global_logits2_numpy * math.log(2), axis=-1) / math.log(2)
    global_native = normalizer[0, 0].reshape(3600, 4)[all_query_indices].T.cpu().numpy().astype(np.float64)
    global_check = _comparison(global_native, global_reference, ABS_TOL, REL_TOL)
    report.update(all_native_rows=all_rows, global_normalizer_comparison=global_check,
                  native_global_normalizer_calls=1, reuse_owner_logp=bool(reuse_owner_logp),
                  head_labels=list(HEAD_LABELS), input_shape=list(q.shape),
                  dot_reference="Original BF16 values exactly cast to FP32 and then FP64 BEFORE all sampled QK products; FP64 scaling and entropy arithmetic",
                  former_fp32_reference=fp32_reference_report,
                  reference_precision_reason="Observed bank487 kernel score agreed with exact BF16-value FP64 dot but not independently rounded FP32 matmul; original FP32 audit remains recorded, tolerances unchanged",
                  native_local_logit_units="base2; FP32 QK scaled by 128**(-0.5)/ln(2)",
                  limitations=["Sampled raw-score parity is not video quality or operator speed evidence.",
                               "Hard ownership is discontinuous at ties; failed scores remain recorded, not excused by aggregate averages."],
                  valid=bool(report["valid"] and all_rows["valid"] and global_check["valid"]))
    arrays = {"bank_coordinates": coordinates, "candidate_rows": candidate_rows,
              "anchors": anchor_table[row], "local_log2_logits_fp32": local_log2_numpy,
              "local_log2_logits_fp64": logits / math.log(2),
              "local_logits_natural": logits, "global_log2_normalizer": sampled_normalizer,
              "sampled_query_vectors_fp32": sampled_query_numpy,
              "sampled_key_vectors_fp32": sampled_key_numpy,
              "native_rows": sampled_rows, "native_scores": sampled_scores,
              "all_native_rows": full_rows, "candidate_table": table,
              "global_probe_logits_log2_fp32": global_logits2_numpy.astype(np.float32),
              "global_probe_native_log2_normalizer": global_native,
              "global_probe_reference_log2_normalizer": global_reference,
              **reference_arrays,
              **{f"former_fp32_{key}": value for key, value in fp32_reference_arrays.items()}}
    if collect_intermediate:
        witness_bank = 487
        _require(np.array_equal(coordinates[witness_bank], [3, 20, 10, 22]), "fixed witness bank moved")
        intermediate_report, intermediate_arrays = audit_recorded_intermediates(
            intermediate.cpu().numpy(), candidate_rows[witness_bank], sampled_rows[witness_bank],
            sampled_normalizer[witness_bank], int(column[witness_bank]), local_log2_numpy[witness_bank])
        # Exact FP64 products of original BF16 values identify accumulation error
        # independently of either GPU implementation's reduction order.
        witness_q = sampled_query_numpy[witness_bank].astype(np.float64)
        witness_k = sampled_key_numpy[witness_bank].astype(np.float64)
        exact_log2_logits = logits[witness_bank] / math.log(2)
        actual_values = intermediate_arrays["native_intermediate_planes"]
        intermediate_report["native_vs_exact_fp64_dot_logits"] = _comparison(actual_values[1], exact_log2_logits, ABS_TOL, REL_TOL)
        intermediate_report["independent_vs_exact_fp64_dot_logits"] = _comparison(local_log2_numpy[witness_bank], exact_log2_logits, ABS_TOL, REL_TOL)
        exact_replay, exact_arrays = audit_score_arrays(
            exact_log2_logits[None] * math.log(2), sampled_normalizer[witness_bank:witness_bank + 1],
            candidate_rows[witness_bank:witness_bank + 1], sampled_rows[witness_bank:witness_bank + 1],
            sampled_scores[witness_bank:witness_bank + 1], column[witness_bank:witness_bank + 1], anchor_table[row[witness_bank:witness_bank + 1]])
        intermediate_report["exact_fp64_dot_score_replay"] = exact_replay
        intermediate_report["diagnostic_store_invariance"] = trace_invariant
        report["witness_bank487"] = intermediate_report
        report["valid"] = bool(report["valid"] and trace_invariant["valid"])
        arrays.update(intermediate_arrays, witness_q_fp32=witness_q.astype(np.float32),
                      witness_k_fp32=witness_k.astype(np.float32), witness_exact_log2_logits=exact_log2_logits,
                      **{f"witness_exact_{key}": value for key, value in exact_arrays.items()})
    return report, arrays


__all__ = ["audit_score_arrays", "audit_all_rows", "audit_recorded_intermediates", "bank_coordinates", "reference_logits_from_vectors", "verify_native"]
