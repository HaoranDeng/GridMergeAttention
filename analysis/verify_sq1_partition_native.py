"""CPU-importable fixed-prefix audit of the isolated SQ1 Mass selector.

Only an explicit call to ``verify_native`` imports Torch/native attention or
uses CUDA. Import and array-level tests need only NumPy. This is a diagnostic,
not production dispatch, a launch authorization, or a quality/speed claim.
"""

from __future__ import annotations

import numpy as np

from analysis.affine_hellinger_seed import reference_logits_from_vectors
from analysis.sq1_partition_seed_reference import sq1_partition_metrics
from analysis.verify_conditional_jeffreys_native import (
    HEAD_LABELS, audit_all_rows, bank_coordinates,
)

SCORE_ATOL = 2e-5
SCORE_RTOL = 1e-4
LOCAL_COLUMNS = np.arange(0, 80, 5, dtype=np.int64)


def global_probe_indices():
    """The existing global sketch uses floor(N/G)*v, NOT floor(N*v/G)."""
    return np.arange(256, dtype=np.int64) * (75600 // 256)


def reference_global_normalizer(queries, global_keys, head_slots, scale):
    """True FP64 dot/LSE at G256; return base2 normalizers [banks,32].

    Queries are original-valued [banks,32,D], global keys [4,256,D]. FP32
    inputs are storage of exact BF16 source values, not already rounded QK.
    Only one head's temporary global logits is materialized at a time.
    """
    q, keys, heads = np.asarray(queries), np.asarray(global_keys), np.asarray(head_slots)
    if (q.ndim != 3 or q.shape[0] < 1 or q.shape[1] != 32
            or keys.shape != (4, 256, q.shape[-1]) or heads.shape != (q.shape[0],)
            or heads.dtype.kind not in "iu" or np.any((heads < 0) | (heads >= 4))):
        raise ValueError("require queries [banks,32,D], global_keys [4,256,D], integer head slots")
    if not np.isfinite(q).all() or not np.isfinite(keys).all() or not np.isfinite(scale):
        raise ValueError("global reference vectors and scale must be finite")
    result = np.empty(q.shape[:-1], dtype=np.float64)
    for head in range(4):
        mask = heads == head
        if mask.any():
            logits = reference_logits_from_vectors(q[mask], keys[head], scale)
            result[mask] = np.logaddexp.reduce(logits, axis=-1) / np.log(2.0)
    return result


def quadrature_arrays(local_logits, singleton_logits, query_columns):
    """Pack each U16 bulk + explicit SQ into at most 17 bins.

    The singleton is always the last valid bin, replacing an observed U16
    duplicate or appended when absent. Padding has zero measure and is NEVER
    passed to the positive-measure oracle. Zero padding is not a pseudo-key.
    """
    local = np.asarray(local_logits, dtype=np.float64)
    singleton = np.asarray(singleton_logits, dtype=np.float64)
    columns = np.asarray(query_columns)
    if (local.ndim != 3 or local.shape[0] < 1 or local.shape[1:] != (32, 16)
            or singleton.shape != local.shape[:2] or columns.shape != (local.shape[0],)
            or columns.dtype.kind not in "iu" or np.any((columns < 0) | (columns >= 80))):
        raise ValueError("require local [banks,32,16], singleton [banks,32], query columns in [0,80)")
    if not np.isfinite(local).all() or not np.isfinite(singleton).all():
        raise ValueError("local and explicit singleton logits must be finite")
    banks = local.shape[0]
    logits = np.zeros((banks, 32, 17), dtype=np.float64)
    measures = np.zeros((banks, 17), dtype=np.float64)
    support_columns = np.full((banks, 17), -1, dtype=np.int64)
    count = np.empty(banks, dtype=np.int64)
    for column in np.unique(columns):
        mask = columns == column
        keep = LOCAL_COLUMNS != column
        m = int(keep.sum())
        logits[mask, :, :m] = local[mask][:, :, keep]
        logits[mask, :, m] = singleton[mask]
        measures[mask, :m] = 79.0 / m
        measures[mask, m] = 1.0
        support_columns[mask, :m] = LOCAL_COLUMNS[keep]
        support_columns[mask, m] = column
        count[mask] = m + 1
    return {
        "quadrature_logits_natural_fp64": logits,
        "quadrature_measures": measures,
        "quadrature_columns": support_columns,
        "quadrature_support_count": count,
        "quadrature_singleton_index": count - 1,
    }


def audit_score_arrays(local_logits, singleton_logits, global_log2_normalizer,
                       candidate_rows, native_rows, native_scores, query_columns, anchors):
    """Check ALL32 scores x3 at actual prefixes and independent rank defects.

    Native rows are real grid rows, not candidate slots. Normalizers and QK
    must be independently computed in FP64 for the primary audit. The fixed
    tolerance is 2e-5 + 1e-4*abs(reference), with no override. Scores are
    dimensionless weighted deficits; NO ln(2) conversion is applied to them.
    Near ties may choose a different reference argmax, but each actual pick
    must satisfy the same tolerance on its nonnegative rank defect.
    """
    packed = quadrature_arrays(local_logits, singleton_logits, query_columns)
    logits = packed["quadrature_logits_natural_fp64"]
    norm = np.asarray(global_log2_normalizer, dtype=np.float64)
    rows, selected = np.asarray(candidate_rows), np.asarray(native_rows)
    scores = np.asarray(native_scores, dtype=np.float64)
    anchors = np.asarray(anchors)
    banks = logits.shape[0]
    if (norm.shape != (banks, 32) or rows.shape != (banks, 32)
            or selected.shape != (banks, 4) or scores.shape != (banks, 3, 32)
            or anchors.shape != (banks,)):
        raise ValueError("bad SQ1 audit shapes")
    if (not np.isfinite(norm).all() or rows.dtype.kind not in "iu"
            or selected.dtype.kind not in "iu" or anchors.dtype.kind not in "iu"
            or np.any((rows < 0) | (rows >= 45)) or np.any(np.diff(rows, axis=1) <= 0)
            or np.any((anchors < 0) | (anchors >= 32))):
        raise ValueError("invalid candidate geometry, anchor slots or normalizer")
    membership = selected[:, :, None] == rows[:, None, :]
    seed_checks = {
        "anchor_matches": bool(np.array_equal(selected[:, 0], rows[np.arange(banks), anchors])),
        "four_distinct": bool(np.all(np.diff(np.sort(selected, axis=1), axis=1) > 0)),
        "all_in_bank": bool(np.all(membership.sum(axis=-1) == 1)),
    }
    report = {
        "valid": False, "pass": False, "structure_pass": all(seed_checks.values()),
        "seed_checks": seed_checks, "banks": banks,
        "score_count": banks * 3 * 32, "rank_count": banks * 3,
        "score_atol": SCORE_ATOL, "score_rtol": SCORE_RTOL,
        "units": "dimensionless weighted partition-mass deficit; no log-base conversion",
        "reference": "FP64 original-value QK and G256 LSE; SQ1 quadrature at actual native prefixes",
        "tolerance_reason": "Fixed FP32 arithmetic allowance; failures are retained, never auto-relaxed",
    }
    if not all(seed_checks.values()):
        report["failure"] = "Invalid native rows prevent actual-prefix replay"
        return report, packed
    slots = membership.argmax(axis=-1)
    reference = np.empty_like(scores)
    deficits = np.empty_like(scores)
    log_mass = np.empty((banks, 32))
    importance = np.empty_like(log_mass)
    # Group 16- and 17-bin banks; all measures within one group are the same.
    for support in (16, 17):
        mask = packed["quadrature_support_count"] == support
        if not mask.any():
            continue
        z = logits[mask, :, :support]
        mu = packed["quadrature_measures"][mask, :support]
        log_mass[mask] = np.logaddexp.reduce(z + np.log(mu)[:, None, :], axis=-1) / np.log(2.0)
        mass = log_mass[mask] - norm[mask]
        importance[mask] = np.exp2(mass - mass.max(axis=-1, keepdims=True))
        group_slots = slots[mask]
        batch = np.arange(z.shape[0])[:, None]
        for step in range(1, 4):
            seed_logits = z[batch, group_slots[:, :step]]
            deficit = sq1_partition_metrics(z, seed_logits, support - 1, measures=mu)["mass_deficit"]
            deficits[mask, step - 1] = deficit
            reference[mask, step - 1] = importance[mask] * np.maximum(deficit, 0.0)
    finite = np.isfinite(scores)
    error = np.abs(np.where(finite, scores, 0.0) - reference)
    allowance = SCORE_ATOL + SCORE_RTOL * np.abs(reference)
    failed = ~finite | (error > allowance)
    rank_defect, rank_relative, rank_allowance = [np.empty((banks, 3)) for _ in range(3)]
    best_slots = np.empty((banks, 3), dtype=np.int64)
    for step in range(1, 4):
        used = np.any(np.arange(32)[None, :, None] == slots[:, None, :step], axis=-1)
        eligible = np.where(used, -np.inf, reference[:, step - 1])
        best_slots[:, step - 1] = np.argmax(eligible, axis=-1)
        best = np.max(eligible, axis=-1)
        actual = reference[np.arange(banks), step - 1, slots[:, step]]
        defect = np.maximum(best - actual, 0.0)
        rank_defect[:, step - 1] = defect
        rank_relative[:, step - 1] = defect / np.maximum(np.abs(best), np.finfo(float).tiny)
        rank_allowance[:, step - 1] = SCORE_ATOL + SCORE_RTOL * np.abs(best)
    rank_failed = rank_defect > rank_allowance
    witnesses = []
    for bank, step, candidate in np.argwhere(failed)[:20]:
        witnesses.append({"bank": int(bank), "step": int(step + 1), "candidate_slot": int(candidate),
                          "native": float(scores[bank, step, candidate]) if finite[bank, step, candidate] else None,
                          "reference": float(reference[bank, step, candidate]),
                          "allowed": float(allowance[bank, step, candidate])})
    passed = bool(not failed.any() and not rank_failed.any())
    report.update(
        valid=passed, **{"pass": passed}, score_failures=int(failed.sum()),
        rank_failures=int(rank_failed.sum()), nonfinite_scores=int((~finite).sum()),
        max_score_error=float(error[finite].max(initial=0)),
        max_score_tolerance_ratio=float((error / allowance)[finite].max(initial=0)),
        max_rank_defect=float(rank_defect.max(initial=0)),
        max_relative_rank_defect=float(rank_relative.max(initial=0)),
        max_rank_tolerance_ratio=float((rank_defect / rank_allowance).max(initial=0)),
        nonidentical_reference_argmax_choices=int(np.count_nonzero(best_slots != slots[:, 1:])),
        failure_witnesses=witnesses,
        rank_failure_coordinates=np.argwhere(rank_failed)[:20].tolist(),
        min_unclamped_reference_deficit=float(deficits.min()),
        negative_reference_deficit_count=int(np.count_nonzero(deficits < 0)),
    )
    packed.update(
        reference_scores=reference, reference_unclamped_deficits=deficits,
        reference_local_log2_normalizer=log_mass, reference_mass_weights=importance,
        native_candidate_slots=slots, reference_best_slots_at_native_prefix=best_slots,
        reference_rank_defect=rank_defect, reference_relative_rank_defect=rank_relative,
        reference_rank_tolerance=rank_allowance,
    )
    return report, packed


def verify_native(q, k):
    """Explicit GPU diagnostic of original CUDA BF16 [1,75600,4,128] Q/K.

    This function must only be called by a separately authorized, clean-GPU
    runner. One shared native G256 call, four selector calls (audit on/off for
    both owner-reuse paths), 600 fixed sampled banks/path, all302400 structural
    groups/path. Returns a JSON-serializable report and NPZ-ready raw arrays.
    Every path's raw-score/rank gate uses true FP64 local AND global QK.
    """
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    from attention.entropy_mass_seed_triton import global_probe_log_norm_triton
    from attention.sq1_partition_mass_seed_triton import sq1_partition_mass_seed_rows_triton

    if (tuple(q.shape) != (1, 75600, 4, 128) or tuple(k.shape) != tuple(q.shape)
            or q.dtype != torch.bfloat16 or k.dtype != q.dtype
            or not q.is_cuda or not k.is_cuda or q.device != k.device):
        raise ValueError("require original CUDA BF16 [1,75600,4,128] QK")
    scale = 128 ** -.5
    q_rf0 = q[:, :3600].reshape(1, 1, 45, 80, 4, 128).contiguous()
    keys = k.reshape(1, 21, 45, 80, 4, 128).contiguous()
    global_norm = global_probe_log_norm_triton(q_rf0, keys, scale=scale, global_probe_count=256)
    candidate_table, anchor_table = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    table = candidate_table.cpu().numpy().astype(np.int64)
    anchor_slots = anchor_table.cpu().numpy().astype(np.int64)
    coordinates = bank_coordinates()
    head, column, frame, row = coordinates.T
    gpu_head, gpu_column, gpu_frame, gpu_row = [torch.as_tensor(a, device=q.device) for a in coordinates.T]
    candidate_rows = table[row]
    gpu_candidate_rows = torch.as_tensor(candidate_rows, device=q.device)
    query_index = torch.as_tensor(candidate_rows * 80 + column[:, None], device=q.device)
    key_index = torch.as_tensor(frame[:, None] * 3600 + row[:, None] * 80 + LOCAL_COLUMNS[None, :], device=q.device)
    singleton_index = torch.as_tensor(frame * 3600 + row * 80 + column, device=q.device)
    # Cast BF16 values only for exact storage; every reference dot casts to
    # FP64 BEFORE accumulation. Never use a GPU FP32 matmul as ground truth.
    original_q = q[0, query_index, gpu_head[:, None]].float().cpu().numpy()
    original_k = k[0, key_index, gpu_head[:, None]].float().cpu().numpy()
    original_singleton_k = k[0, singleton_index, gpu_head].float().cpu().numpy()[:, None, :]
    global_indices = global_probe_indices()
    gpu_global_indices = torch.as_tensor(global_indices, device=q.device)
    gpu_all_heads = torch.arange(4, device=q.device)
    original_global_k = k[0, gpu_global_indices[:, None], gpu_all_heads[None, :]].permute(1, 0, 2).float().cpu().numpy()
    logits = reference_logits_from_vectors(original_q, original_k, scale)
    singleton_logits = reference_logits_from_vectors(original_q, original_singleton_k, scale)[..., 0]
    reference_norm = reference_global_normalizer(original_q, original_global_k, head, scale)
    native_norm = global_norm[0, 0, gpu_candidate_rows, gpu_column[:, None], gpu_head[:, None]].cpu().numpy()
    arrays = {
        "bank_coordinates": coordinates, "candidate_rows": candidate_rows,
        "anchors": anchor_slots[row], "candidate_table": table, "anchor_table": anchor_slots,
        "sampled_query_vectors_fp32": original_q, "sampled_key_vectors_fp32": original_k,
        "sampled_singleton_key_vectors_fp32": original_singleton_k,
        "global_probe_key_vectors_fp32_by_head": original_global_k,
        "global_probe_indices": global_indices, "global_probe_head_slots": np.arange(4),
        "local_logits_natural_fp64": logits, "singleton_logits_natural_fp64": singleton_logits,
        "reference_global_log2_normalizer": reference_norm, "native_global_log2_normalizer": native_norm,
        "all_native_global_log2_normalizers": global_norm.cpu().numpy(),
        "scale": np.asarray(scale, dtype=np.float64),
    }
    path_reports = {}
    for reuse in (False, True):
        name = "reuse_owner_logp" if reuse else "stored_seed_logp"
        common = dict(scale=scale, candidate_count=32, candidate_geometry_count=45,
                      block_j=4, probe_count=16, global_probe_count=256,
                      output_dtype=torch.uint8, gs_num_warps=4, num_stages=1,
                      global_normalizer=global_norm, reuse_owner_logp=reuse)
        untraced = sq1_partition_mass_seed_rows_triton(q_rf0, keys, **common)
        native, scores = sq1_partition_mass_seed_rows_triton(q_rf0, keys, **common, audit_scores=True)
        invariant = bool(torch.equal(native, untraced))
        full_rows, full_untraced = native.cpu().numpy(), untraced.cpu().numpy()
        sampled_rows = full_rows[0, 0, frame, row, column, head].astype(np.int64)
        sampled_scores = scores[0, 0, gpu_frame, gpu_row, gpu_column, gpu_head].cpu().numpy()
        del scores, native, untraced
        primary, reference = audit_score_arrays(
            logits, singleton_logits, reference_norm, candidate_rows, sampled_rows,
            sampled_scores, column, anchor_slots[row],
        )
        supplemental, supplemental_arrays = audit_score_arrays(
            logits, singleton_logits, native_norm, candidate_rows, sampled_rows,
            sampled_scores, column, anchor_slots[row],
        )
        full_report = audit_all_rows(full_rows, table)
        untraced_report = audit_all_rows(full_untraced, table)
        path_reports[name] = {
            "valid": bool(primary["valid"] and full_report["valid"] and untraced_report["valid"] and invariant),
            "raw_score_parity": primary, "all_native_rows": full_report,
            "all_audit_free_rows": untraced_report,
            "audit_on_off_all_rows_bitwise_equal": invariant,
            "supplemental_using_native_global_normalizer": supplemental,
        }
        arrays.update({name + "__" + key: value for key, value in reference.items()})
        arrays.update({name + "__native_rows": sampled_rows, name + "__native_scores": sampled_scores,
                       name + "__all_native_rows": full_rows, name + "__all_audit_free_rows": full_untraced})
        if "reference_scores" in supplemental_arrays:
            arrays[name + "__supplemental_scores_native_global_normalizer"] = supplemental_arrays["reference_scores"]
    cross_equal = np.array_equal(arrays["stored_seed_logp__all_native_rows"], arrays["reuse_owner_logp__all_native_rows"])
    report = {
        "valid": bool(all(path["valid"] for path in path_reports.values())),
        "paths": path_reports, "cross_reuse_all_rows_bitwise_equal": bool(cross_equal),
        "cross_reuse_semantics": "Each actual-prefix score/rank gate is mandatory; cross-path row identity is descriptive",
        "native_selector_calls": 4, "native_global_normalizer_calls": 1,
        "head_labels": list(HEAD_LABELS), "input_shape": list(q.shape), "rf0": True,
        "candidate_count": 32, "geometry_count": 45, "gs_num_warps": 4,
        "local_probes": LOCAL_COLUMNS.tolist(), "global_probe_stride": 295, "global_probe_count": 256,
        "primary_reference": "FP64 QK and G256 from original BF16 vectors; no independent FP32-dot reference",
        "native_global_norm_max_absolute_error_bits": float(np.max(np.abs(native_norm - reference_norm))),
        "score_atol": SCORE_ATOL, "score_rtol": SCORE_RTOL,
        "limitations": ["Not a video-quality or speed measurement.",
                        "All scores are audited only for the fixed 600 banks per path; structural checks cover all groups.",
                        "Audit on/off row equality is required separately because stores can alter native lowering.",
                        "U16+explicit SQ1 remains a quadrature approximation to full80 ownership and loss."],
    }
    return report, arrays


__all__ = ["SCORE_ATOL", "SCORE_RTOL", "global_probe_indices", "reference_global_normalizer",
           "quadrature_arrays", "audit_score_arrays", "verify_native"]
