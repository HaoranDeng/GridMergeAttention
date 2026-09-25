"""One SQ1-aware, owner-preserving real-seed medoid pass (NumPy FP64).

This is an isolated reference, not a native selector or a new theorem. The
incumbent comes from Anchor32 outside this module. Only slots 1,2,3 may move.
The input contains the already-computed RF0 logits on Uniform16 plus the
actually observed query-column singleton; no Q/K/V or other frame is read.

For bins l with positive measures mu_l, p_i=softmax(z_i+log(mu)). Singleton
measure is one and the other m bins each represent 79/m keys. Ownership is
argmax of the normalized seed profiles before peeling the singleton. Each
bulk cell uses the seed's conditional distribution w_c. On this support,

  u_ic = w_c^T log(p_i) + H_bin(w_c),  u_i,SQ = log(p_i,SQ),
  beta_i = softmax(u_i),   E = -mean_i logsumexp_c(u_ic).

E is exactly min reverse KL of the disjoint compressed family on these
positive-measure bins in ideal arithmetic. Equivalently, on raw z one uses
measure entropy -sum_l w_l log(w_l/mu_l), not plain entropy a second time.

Before EACH cell update freeze its CURRENT beta (uniform query demand).
With alpha_ic=beta_ic/sum_j beta_jc and
rho_c=softmax_cell(sum_i alpha_ic log(p_i)), select the real candidate s
minimizing KL(p_s(.|cell)||rho_c), constrained to distinct seeds and unchanged
bulk-bin owner labels. This is exactly the discrete fixed-beta conditional
KL subproblem. The old singleton owner may change: its final cell is always
the singleton. Ties keep the incumbent. Recompute beta after each update.

The joint variational objective cannot decrease in ideal arithmetic: the
current seed is feasible, the fixed-beta component is optimized, then beta
is reoptimized. A single pass is NOT convergence of joint seed optimization.
The guarantee is not for unsampled keys, global block importance, temporal
row reuse, BF16 merge moments, output error, or generated videos.

Cost per bank excluding upstream QK/Anchor32: four route evaluations cost
4*C*R*P weighted terms; three conditional barycenters/KL scans and ownership
checks cost O(3*R*P) each. R=32,P=17,C=4 gives 8,704 route weighted terms,
1,632 barycenter terms and 1,632 KL terms, plus softmax/owner operations.
There is no R-by-R pairwise matrix or candidate-set repartition sweep.
Working storage is O(RP+RC+CP), plus bounded three-step audit snapshots;
one FP64 [32,17] array is 4,352 bytes. This CPU reference has no speed claim.
Owner constraints may leave very few useful replacements, especially when
probe cells are empty/singletons. It is not a commitment to a video run.
"""

import numpy as np


def uniform16_sq1_support(singleton_column):
    """Return the distinct physical columns and exact positive base measures."""
    raw = np.asarray(singleton_column)
    if raw.ndim != 0 or raw.dtype.kind not in "iu" or not 0 <= int(raw) < 80:
        raise ValueError("singleton_column must be an integer in [0,80)")
    column = int(raw)
    columns = np.arange(0, 80, 5, dtype=np.int64)
    if column % 5:
        columns = np.append(columns, column)
    singleton_index = int(np.flatnonzero(columns == column)[0])
    measures = np.full(len(columns), 79.0 / (len(columns) - 1), dtype=np.float64)
    measures[singleton_index] = 1.0
    return {"probe_columns": columns, "singleton_index": singleton_index,
            "measures": measures}


def _integer_vector(value, length, name, upper):
    raw = np.asarray(value)
    if raw.shape != (length,) or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer vector of length {length}")
    if np.any(raw >= upper) or np.any(raw < 0):
        raise ValueError(f"{name} entries must be in [0,{upper})")
    if len(np.unique(raw)) != length:
        raise ValueError(f"{name} entries must be distinct")
    return raw.astype(np.int64, copy=True)


def _log_softmax(value):
    shifted = value - np.max(value, axis=-1, keepdims=True)
    return shifted - np.logaddexp.reduce(shifted, axis=-1, keepdims=True)


def _inputs(logits, candidate_rows, incumbent_slots, probe_columns, singleton_column):
    raw = np.asarray(logits)
    if raw.dtype.kind != "f":
        raise TypeError("logits must be a real floating-point array")
    if raw.ndim != 2 or not 4 <= raw.shape[0] <= 45 or raw.shape[1] not in (16, 17):
        raise ValueError("one bank of logits [R,P], 4<=R<=45 and P=16/17, is required")
    z = raw.astype(np.float64, copy=True)
    if not np.isfinite(z).all():
        raise ValueError("logits must be finite")
    rows, probes = z.shape
    candidate_rows = _integer_vector(candidate_rows, rows, "candidate_rows", 45)
    selected = _integer_vector(incumbent_slots, 4, "incumbent_slots", rows)
    columns = _integer_vector(probe_columns, probes, "probe_columns", 80)
    support = uniform16_sq1_support(singleton_column)
    if not np.array_equal(np.sort(columns), np.sort(support["probe_columns"])):
        raise ValueError("probe_columns must contain exactly Uniform16 plus observed SQ, without duplicates")
    singleton = int(np.flatnonzero(columns == int(singleton_column))[0])
    measures = np.full(probes, 79.0 / (probes - 1), dtype=np.float64)
    measures[singleton] = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        # Center before adding the base measure to preserve logit gauge.
        logp = _log_softmax((z - z.max(axis=-1, keepdims=True)) + np.log(measures))
    if not np.isfinite(logp).all():
        raise ValueError("logit dynamic range exceeds FP64 arithmetic")
    return logp, candidate_rows, selected, columns, singleton, measures


def _route_state(logp, selected, singleton):
    """Rebuild the actual four bulk conditionals plus singleton, never overlap."""
    rows, probes = logp.shape
    owner_before = np.argmax(logp[selected], axis=0)
    owner = owner_before.copy()
    owner[singleton] = 4
    weights = np.zeros((5, probes), dtype=np.float64)
    log_weights = np.full((5, probes), -np.inf, dtype=np.float64)
    component = np.full((rows, 5), -np.inf, dtype=np.float64)
    for cell, slot in enumerate(selected):
        mask = owner == cell
        if not mask.any():
            continue
        logw = _log_softmax(logp[slot, mask])
        w = np.exp(logw)
        weights[cell, mask] = w
        log_weights[cell, mask] = logw
        component[:, cell] = (logp[:, mask] - logw) @ w
    weights[4, singleton] = 1.0
    log_weights[4, singleton] = 0.0
    component[:, 4] = logp[:, singleton]
    represented = np.logaddexp.reduce(component, axis=-1)
    log_beta = component - represented[:, None]
    gap = -represented
    if not np.isfinite(gap).all():
        raise ValueError("restricted objective exceeds FP64 arithmetic")
    return {"owner_before_singleton": owner_before, "owner": owner,
            "weights": weights, "log_weights": log_weights,
            "component_logits": component, "log_beta": log_beta,
            "beta": np.exp(log_beta), "gap": gap, "mean_gap": float(gap.mean())}


def _feasible_candidates(logp, selected, cell, owner, singleton):
    """Vectorized winner comparisons; no trial cell/route construction."""
    other_cells = np.delete(np.arange(4), cell)
    other_profiles = logp[selected[other_cells]]
    other_winner = np.argmax(other_profiles, axis=0)
    competitor = np.max(other_profiles, axis=0)
    competitor_owner = other_cells[other_winner]
    wins = (logp > competitor) | ((logp == competitor) & (cell < competitor_owner))
    trial_owner = np.where(wins, cell, competitor_owner)
    bulk = np.arange(logp.shape[-1]) != singleton
    feasible = np.all(trial_owner[:, bulk] == owner[bulk], axis=-1)
    feasible[selected[other_cells]] = False
    if not feasible[selected[cell]]:
        raise AssertionError("the incumbent must remain feasible")
    return feasible


def refine_sq1_conditional_medoids(
    logits, candidate_rows, incumbent_slots, probe_columns, singleton_column,
):
    """Return a one-bank FP64 refinement and complete per-cell audit.

    logits: already-computed, finite natural-log QK scores [R,P]. Production
      uses R=32, with BF16 operands/FP32 accumulation then FP64 scaling;
      this reference does not compute QK or establish input provenance.
    candidate_rows: distinct physical row IDs [R] in 0..44.
    incumbent_slots: four distinct candidate-table slots. The caller supplies
      Anchor32; this function preserves slot zero's row but does not rerun or
      verify the Fisher algorithm or know the geometric matched key row.
    probe_columns: distinct physical U16+SQ columns [P], in any explicit order.
    singleton_column: actual query-column index, whose QK score is present.

    Returns seeds (candidate slots), seed_rows (original physical rows),
    initial/final restricted states and three audit steps. Every step records
    the CURRENT beta, fixed-beta conditional KL scores, feasibility mask and
    chosen slot, then the reoptimized route. Scores/gaps are not clipped, and
    there is no fitted acceptance threshold or full-gap fallback rule.
    """
    logp, row_ids, selected, columns, singleton, measures = _inputs(
        logits, candidate_rows, incumbent_slots, probe_columns, singleton_column,
    )
    incumbent = selected.copy()
    state = _route_state(logp, selected, singleton)
    initial = state
    fixed_owner = initial["owner"].copy()
    steps = []
    for cell in (1, 2, 3):
        before_slots = selected.copy()
        before = state
        feasible = _feasible_candidates(logp, selected, cell, fixed_owner, singleton)
        mask = fixed_owner == cell
        active = bool(mask.any())
        alpha = np.zeros(logp.shape[0], dtype=np.float64)
        barycenter = np.zeros(logp.shape[1], dtype=np.float64)
        costs = np.zeros(logp.shape[0], dtype=np.float64)
        old_slot = int(selected[cell])
        best_slot = old_slot
        accepted = False
        if active:
            # Normalize log-beta directly so very small finite demand is not
            # lost merely because exp(log_beta) underflows for every query.
            alpha = np.exp(_log_softmax(before["log_beta"][:, cell]))
            mean_logp = alpha @ logp[:, mask]
            log_rho = _log_softmax(mean_logp)
            barycenter[mask] = np.exp(log_rho)
            candidate_logw = _log_softmax(logp[:, mask])
            candidate_w = np.exp(candidate_logw)
            costs = np.sum(candidate_w * (candidate_logw - log_rho), axis=-1)
            if not np.isfinite(costs).all():
                raise ValueError("conditional KL exceeds FP64 arithmetic")
            minimizer = int(np.argmin(np.where(feasible, costs, np.inf)))
            # Exact ties, including a duplicated conditional profile, keep
            # the old row. A different physical row is required for a change.
            accepted = minimizer != old_slot and costs[minimizer] < costs[old_slot]
            if accepted:
                best_slot = minimizer
                selected[cell] = best_slot
        state = _route_state(logp, selected, singleton)
        if not np.array_equal(state["owner"], fixed_owner):
            raise AssertionError("a bulk probe owner changed")
        steps.append({"cell": cell, "active": active,
                      "before_slots": before_slots, "after_slots": selected.copy(),
                      "old_slot": old_slot, "chosen_slot": best_slot,
                      "accepted": accepted, "feasible": feasible,
                      "alpha": alpha, "conditional_barycenter": barycenter,
                      "candidate_conditional_kl": costs,
                      "before": before, "after": state})
    if selected[0] != incumbent[0] or len(np.unique(selected)) != 4:
        raise AssertionError("anchor/real-row distinctness invariant violated")
    return {"seeds": selected.copy(), "seed_rows": row_ids[selected].copy(),
            "incumbent": incumbent, "incumbent_rows": row_ids[incumbent].copy(),
            "candidate_rows": row_ids, "probe_columns": columns,
            "singleton_index": singleton, "measures": measures, "log_p": logp,
            "initial": initial, "final": state, "steps": steps,
            "changed_seed_count": int(np.count_nonzero(selected != incumbent))}
