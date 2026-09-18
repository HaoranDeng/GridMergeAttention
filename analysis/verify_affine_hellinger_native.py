"""Explicit native affine-selector parity diagnostic; never an inference path.

Root owns GPU launches and artifacts. Importing this module needs only NumPy;
verify_native is the only entry point that imports Torch or executes CUDA.
The coordinate/all-row checks are the existing 600-bank C32/g45 protocol.
"""

from __future__ import annotations

import numpy as np

from analysis.affine_hellinger_seed import audit_affine_scores, reference_logits_from_vectors
from analysis.verify_conditional_jeffreys_native import HEAD_LABELS, audit_all_rows, bank_coordinates


def rows_to_candidate_slots(candidate_rows, native_rows):
    """Map actual grid rows to candidate slots without silently accepting misses."""
    candidates, native = np.asarray(candidate_rows), np.asarray(native_rows)
    if candidates.ndim != 2 or candidates.shape[1] != 32 or native.shape != (candidates.shape[0], 4):
        raise ValueError("require candidate_rows [banks,32], native_rows [banks,4]")
    if candidates.dtype.kind not in "iu" or native.dtype.kind not in "iu":
        raise ValueError("rows must have integer dtype")
    match = native[..., None] == candidates[:, None, :]
    if not np.all(match.sum(-1) == 1):
        raise ValueError("every native row must match exactly one candidate")
    return np.argmax(match, axis=-1).astype(np.int64)


def verify_native(q, k):
    """Audit original CUDA BF16 [1,75600,4,128] QK; return report, NumPy arrays.

    Tests all native anchors/uniqueness/membership; 600 banks * 3 prefixes *
    32 scores and 1,800 pivot rank defects against true FP64 QK. The raw score
    gate remains atol=2e-5 and rtol=1e-4. Both scored and audit-free calls must
    select identical rows across all 302,400 groups. This is NOT timed and
    makes no attention-output/video-quality claim. No global normalizer call.
    """
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    from attention.affine_hellinger_seed_triton import affine_hellinger_seed_rows_triton

    if (tuple(q.shape) != (1, 75600, 4, 128) or tuple(k.shape) != tuple(q.shape)
            or q.dtype != torch.bfloat16 or k.dtype != q.dtype
            or not q.is_cuda or not k.is_cuda or q.device != k.device):
        raise ValueError("require original CUDA BF16 [1,75600,4,128] QK")
    scale = 128 ** -.5
    q_rf0 = q[:, :3600].reshape(1, 1, 45, 80, 4, 128).contiguous()
    keys = k.reshape(1, 21, 45, 80, 4, 128).contiguous()
    native_untraced = affine_hellinger_seed_rows_triton(q_rf0, keys, scale)
    scores = torch.empty((1, 1, 21, 45, 80, 4, 3, 32), device=q.device, dtype=torch.float32)
    native = affine_hellinger_seed_rows_triton(q_rf0, keys, scale, score_audit=scores)
    audit_invariance = bool(torch.equal(native, native_untraced))
    candidate_table, anchor_table = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    table = candidate_table.cpu().numpy().astype(np.int64)
    anchor_slots = anchor_table.cpu().numpy().astype(np.int64)
    full_rows = native.cpu().numpy()
    full_row_report = audit_all_rows(full_rows, table)
    coordinates = bank_coordinates()
    head, column, frame, row = coordinates.T
    gpu_head, gpu_column, gpu_frame, gpu_row = [torch.as_tensor(a, device=q.device) for a in coordinates.T]
    candidate_rows = table[row]
    query_index = torch.as_tensor(candidate_rows * 80 + column[:, None], device=q.device)
    key_index = torch.as_tensor(frame[:, None] * 3600 + row[:, None] * 80
                                + np.arange(0, 80, 5)[None, :], device=q.device)
    # Exact BF16 -> FP32 cast for storage only; dot is FP64, not GPU FP32.
    original_q = q[0, query_index, gpu_head[:, None]].float().cpu().numpy()
    original_k = k[0, key_index, gpu_head[:, None]].float().cpu().numpy()
    logits = reference_logits_from_vectors(original_q, original_k, scale)
    sampled_rows = full_rows[0, 0, frame, row, column, head].astype(np.int64)
    sampled_scores = scores[0, 0, gpu_frame, gpu_row, gpu_column, gpu_head].cpu().numpy()
    slots = rows_to_candidate_slots(candidate_rows, sampled_rows)
    report, reference = audit_affine_scores(logits, anchor_slots[row], slots, sampled_scores)
    # Supplemental only: do not treat separately rounded FP32 QK as truth.
    logits_fp32 = (original_q @ np.swapaxes(original_k, -1, -2)) * np.float32(scale)
    fp32_report, fp32_reference = audit_affine_scores(logits_fp32, anchor_slots[row], slots, sampled_scores)
    report.update(
        valid=bool(report["pass"] and full_row_report["valid"] and audit_invariance),
        all_native_rows=full_row_report,
        audit_on_off_all_rows_bitwise_equal=audit_invariance,
        native_selector_calls=2, native_global_normalizer_calls=0,
        head_labels=list(HEAD_LABELS), input_shape=list(q.shape),
        rf0=True, candidate_count=32, geometry_count=45, local_probes=list(range(0, 80, 5)),
        importance_weight="none", mean_centering=False,
        supplemental_fp32_cpu_dot=fp32_report,
        limitations=["Raw-score parity is not video quality or operator speed evidence.",
                     "Only the fixed 600 banks have all scores checked; structural checks cover every group.",
                     "Audit stores can alter lowering; all-row audit-on/off equality is separately required."],
    )
    arrays = {
        "bank_coordinates": coordinates, "candidate_rows": candidate_rows,
        "anchors": anchor_slots[row], "native_rows": sampled_rows,
        "native_candidate_slots": slots, "native_scores": sampled_scores,
        "reference_scores": reference, "local_logits_natural_fp64": logits,
        "sampled_query_vectors_fp32": original_q, "sampled_key_vectors_fp32": original_k,
        "supplemental_local_logits_fp32": logits_fp32,
        "supplemental_reference_scores_fp32_dot": fp32_reference,
        "candidate_table": table, "anchor_table": anchor_slots,
        "all_native_rows": full_rows,
    }
    return report, arrays


__all__ = ["rows_to_candidate_slots", "verify_native"]
