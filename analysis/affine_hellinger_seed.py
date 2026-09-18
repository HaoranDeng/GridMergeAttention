"""Torch-free FP64 reference and fixed-prefix audit for affine Hellinger-GS.

The entropy variational gap is KL(a || softmax(z)). Locally on the positive
simplex this equals twice squared root-probability distance to second order.
Affine root-probability span about a mandatory real anchor is consequently a
local Fisher surrogate, not a global optimum or a hard-partition guarantee.
"""

from __future__ import annotations

import numpy as np

GS_EPSILON = 1.0e-12
SCORE_ATOL = 2.0e-5
SCORE_RTOL = 1.0e-4


def root_profiles(logits):
    """Stable FP64 sqrt(softmax) of finite natural-unit [...,C,P] logits."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim < 2 or min(logits.shape[-2:]) < 1 or not np.isfinite(logits).all():
        raise ValueError("logits must be finite [..., candidates, probes]")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=-1, keepdims=True)
    return np.sqrt(probability)


def reference_logits_from_vectors(queries, keys, scale):
    """Cast original-valued Q/K to FP64 BEFORE the dot, never afterward.

    Inputs [...,C,D] and [...,P,D] may store BF16 values exactly in FP32.
    Returning a true FP64 QK reference avoids independent FP32-dot rounding.
    """
    q, k = np.asarray(queries, dtype=np.float64), np.asarray(keys, dtype=np.float64)
    if q.ndim < 2 or k.ndim < 2 or q.shape[-1] != k.shape[-1]:
        raise ValueError("queries/keys must have matching final feature dimension")
    if not np.isfinite(scale) or not np.isfinite(q).all() or not np.isfinite(k).all():
        raise ValueError("queries, keys and scale must be finite")
    return (q @ np.swapaxes(k, -1, -2)) * float(scale)


def _indices(values, leading_shape, length, count, name):
    original = np.asarray(values)
    if original.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer candidate slots")
    try:
        result = np.broadcast_to(original, (*leading_shape, length)).astype(np.int64)
    except ValueError as error:
        raise ValueError(f"{name} shape does not match bank shape") from error
    if np.any(result < 0) or np.any(result >= count):
        raise ValueError(f"{name} contains an out-of-range candidate slot")
    return result


def _project(residual, pivot):
    selected = np.take_along_axis(residual, pivot[..., None, None], axis=-2)[..., 0, :]
    basis = selected / np.sqrt(np.sum(selected * selected, axis=-1, keepdims=True) + GS_EPSILON)
    return residual - np.sum(residual * basis[..., None, :], axis=-1, keepdims=True) * basis[..., None, :]


def prefix_scores(logits, prefix):
    """All candidate residual² at an ACTUAL prefix, including used candidates.

    prefix [...,S] starts with the anchor. Replay the same regularized GS
    arithmetic in FP64; do not substitute a free-running reference prefix.
    Scores are dimensionless; unlike Jeffreys they need no log-base conversion.
    """
    profile = root_profiles(logits)
    prefix = np.asarray(prefix)
    if prefix.ndim < 1 or not 1 <= prefix.shape[-1] <= profile.shape[-2]:
        raise ValueError("prefix must have at least one seed and at most C seeds")
    prefix = _indices(prefix, profile.shape[:-2], prefix.shape[-1], profile.shape[-2], "prefix")
    if np.any(np.diff(np.sort(prefix, axis=-1), axis=-1) == 0):
        raise ValueError("prefix must contain distinct candidate slots")
    anchor_profile = np.take_along_axis(profile, prefix[..., :1, None], axis=-2)
    residual = profile - anchor_profile
    for step in range(1, prefix.shape[-1]):
        residual = _project(residual, prefix[..., step])
    return np.sum(residual * residual, axis=-1)


def select_affine_seeds(logits, anchor, budget=4, *, return_scores=False):
    """Greedy unweighted affine pivots, with lowest-slot deterministic ties."""
    profile = root_profiles(logits)
    count = profile.shape[-2]
    if not isinstance(budget, int) or not 1 <= budget <= count:
        raise ValueError("budget must be an integer from 1 through C")
    anchor = _indices(np.asarray(anchor)[..., None], profile.shape[:-2], 1, count, "anchor")
    residual = profile - np.take_along_axis(profile, anchor[..., None], axis=-2)
    selected = np.empty((*profile.shape[:-2], budget), dtype=np.int64)
    selected[..., 0] = anchor[..., 0]
    used = np.arange(count) == anchor
    traces = []
    for step in range(1, budget):
        score = np.sum(residual * residual, axis=-1)
        traces.append(score.copy())
        pivot = np.argmax(np.where(used, -np.inf, score), axis=-1)
        selected[..., step] = pivot
        used |= np.arange(count) == pivot[..., None]
        if step < budget - 1:
            residual = _project(residual, pivot)
    scores = np.stack(traces, axis=-2) if traces else np.empty((*profile.shape[:-2], 0, count))
    return (selected, scores) if return_scores else selected


def audit_affine_scores(logits, anchors, native_slots, native_scores):
    """JSON-serializable fixed-tolerance score/rank audit plus reference arrays.

    logits [banks,32,16] must be from true FP64 QK of original BF16 inputs;
    anchors [banks], native_slots [banks,4] are candidate-slot indices (NOT
    grid rows), and native_scores [banks,3,32] are pre-used-mask native scores.
    Check all 96 scores/bank, not only ranks. Near ties need not select the
    identical CPU argmax, but the actual pick's absolute rank defect must fit
    the same 2e-5 + 1e-4*|best| tolerance. No automatic tolerance relaxation.
    Return (report, reference_scores), preserving witnesses for root artifacts.
    """
    logits = np.asarray(logits, dtype=np.float64)
    slots, scores = np.asarray(native_slots), np.asarray(native_scores, dtype=np.float64)
    if logits.ndim != 3 or logits.shape[1:] != (32, 16):
        raise ValueError("logits must be [banks,32,16]")
    banks = logits.shape[0]
    slots = _indices(slots, (banks,), 4, 32, "native_slots")
    anchors = _indices(np.asarray(anchors)[..., None], (banks,), 1, 32, "anchors")[..., 0]
    if scores.shape != (banks, 3, 32):
        raise ValueError("native_scores must be [banks,3,32]")
    structural = bool(np.all(slots[:, 0] == anchors) and
                      np.all(np.diff(np.sort(slots, axis=-1), axis=-1) > 0))
    if not structural:
        return ({"pass": False, "structure_pass": False,
                 "error": "native rows must preserve anchor and contain four distinct candidates"}, None)
    reference = np.stack([prefix_scores(logits, slots[:, :step]) for step in range(1, 4)], axis=1)
    error = np.abs(scores - reference)
    allowance = SCORE_ATOL + SCORE_RTOL * np.abs(reference)
    ratio = error / allowance
    finite = np.isfinite(scores)
    failed = (~finite) | (error > allowance)
    rank_defect = np.zeros((banks, 3))
    rank_relative = np.zeros((banks, 3))
    rank_allowance = np.zeros((banks, 3))
    for step in range(1, 4):
        used = np.any(np.arange(32)[None, :, None] == slots[:, None, :step], axis=-1)
        best = np.max(np.where(used, -np.inf, reference[:, step - 1]), axis=-1)
        actual = reference[np.arange(banks), step - 1, slots[:, step]]
        rank_defect[:, step - 1] = np.maximum(best - actual, 0)
        rank_relative[:, step - 1] = np.maximum(best - actual, 0) / np.maximum(np.abs(best), np.finfo(float).tiny)
        rank_allowance[:, step - 1] = SCORE_ATOL + SCORE_RTOL * np.abs(best)
    rank_failed = rank_defect > rank_allowance
    witnesses = []
    for bank, step, candidate in np.argwhere(failed)[:20]:
        witnesses.append({"bank": int(bank), "step": int(step + 1), "candidate_slot": int(candidate),
                          "native": float(scores[bank, step, candidate]) if finite[bank, step, candidate] else None,
                          "reference": float(reference[bank, step, candidate]),
                          "allowed": float(allowance[bank, step, candidate])})
    report = {
        "pass": bool(not failed.any() and not rank_failed.any()), "structure_pass": True,
        "reference": "FP64 QK before scaling; natural-unit softmax; FP64 actual-prefix affine GS",
        "units": "squared root-probability distance (no log-base conversion)",
        "score_atol": SCORE_ATOL, "score_rtol": SCORE_RTOL,
        "banks": banks, "score_count": int(scores.size), "rank_count": int(rank_defect.size),
        "score_failures": int(failed.sum()), "rank_failures": int(rank_failed.sum()),
        "nonfinite_scores": int((~finite).sum()),
        "max_score_error": float(error[finite].max(initial=0)),
        "max_score_tolerance_ratio": float(ratio[finite].max(initial=0)),
        "max_rank_defect": float(rank_defect.max(initial=0)),
        "max_relative_rank_defect": float(rank_relative.max(initial=0)),
        "max_rank_tolerance_ratio": float((rank_defect / rank_allowance).max(initial=0)),
        "failure_witnesses": witnesses,
    }
    return report, reference
