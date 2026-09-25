"""CPU-only, globally frozen-route SQ1 conditional-medoid reference.

Inputs are already computed RF0 logits, not Q/K/V. One target population of
all 45 physical rows is shared across every supplied bank. This module does
not establish that the caller supplied every physical bank or original RF0
data. Its ideal-arithmetic monotonicity concerns only the supplied positive-
measure quadrature and fixed bulk ownership, not full80 or video quality.

The bank-local control freezes its INITIAL local beta for the whole pass,
unlike the older per-coordinate route-refresh reference. Both modes use the
same candidates, ownership constraints and strict conditional-KL decisions.
"""

import numpy as np

from analysis.sq1_conditional_medoid_reference import (
    _feasible_candidates, _integer_vector, _log_softmax, _route_state,
    uniform16_sq1_support,
)


def _integer_rows(value, banks, count, name):
    raw = np.asarray(value)
    if raw.shape != (banks, count):
        raise ValueError(f"{name} must have shape [{banks},{count}]")
    return np.stack([_integer_vector(row, count, name, 45) for row in raw])


def _prepare(logits, seed_rows, probe_columns, singleton_column):
    raw = np.asarray(logits)
    if raw.dtype.kind != "f":
        raise TypeError("logits must be a real floating-point array")
    if raw.ndim != 3 or raw.shape[0] < 1 or raw.shape[1] != 45 or raw.shape[2] not in (16, 17):
        raise ValueError("logits must have shape [banks,45,P], P=16 or17")
    z = raw.astype(np.float64, copy=True)
    if not np.isfinite(z).all():
        raise ValueError("logits must be finite")
    banks, _, probes = z.shape
    seeds = _integer_rows(seed_rows, banks, 4, "seed_rows")
    columns = _integer_vector(probe_columns, probes, "probe_columns", 80)
    support = uniform16_sq1_support(singleton_column)
    if not np.array_equal(np.sort(columns), np.sort(support["probe_columns"])):
        raise ValueError("probe_columns must be exactly distinct Uniform16 plus observed SQ")
    singleton = int(np.flatnonzero(columns == int(singleton_column))[0])
    measures = np.full(probes, 79.0 / (probes - 1), dtype=np.float64)
    measures[singleton] = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        # Only a QUERY-wide gauge is removable. A bank-specific shift changes
        # global demand, even though it leaves that bank's profiles unchanged.
        shift = z.max(axis=(0, 2))
        weighted = (z - shift[None, :, None]) + np.log(measures)
        bank_log_z = np.logaddexp.reduce(weighted, axis=-1)
        # Center locally before normalizing, avoiding loss of conditional
        # detail when the whole bank has a very small global probability.
        log_p = _log_softmax((z - z.max(axis=-1, keepdims=True)) + np.log(measures))
    if not np.isfinite(weighted).all() or not np.isfinite(log_p).all():
        raise ValueError("logit dynamic range exceeds FP64 arithmetic")
    return {
        "log_p": log_p, "bank_log_z": bank_log_z,
        "dense_log_z": np.logaddexp.reduce(bank_log_z, axis=0),
        "seed_rows": seeds, "probe_columns": columns,
        "singleton_index": singleton, "measures": measures,
        "query_gauge_shift": shift,
    }


def _state(prepared, seed_rows):
    log_p = prepared["log_p"]
    local = [_route_state(p, rows, prepared["singleton_index"])
             for p, rows in zip(log_p, seed_rows)]
    local_component = np.stack([s["component_logits"] for s in local])
    # Old helper components have subtracted log Z_bank. Restore it before
    # computing a route across banks; concatenating local components is wrong.
    component = local_component + prepared["bank_log_z"][:, :, None]
    represented = np.logaddexp.reduce(np.logaddexp.reduce(component, axis=-1), axis=0)
    global_log_beta = component - represented[None, :, None]
    gap = prepared["dense_log_z"] - represented
    if len(local) == 1:
        # Algebraically identical; reuse the same normalization order so the
        # one-bank global/control identity also holds bit-for-bit.
        global_log_beta = local[0]["log_beta"][None].copy()
        gap = local[0]["gap"].copy()
    if not np.isfinite(gap).all():
        raise ValueError("global restricted objective exceeds FP64 arithmetic")
    return {
        "seed_rows": seed_rows.copy(),
        "owners": np.stack([s["owner"] for s in local]),
        "owners_before_singleton": np.stack([s["owner_before_singleton"] for s in local]),
        "weights": np.stack([s["weights"] for s in local]),
        "log_weights": np.stack([s["log_weights"] for s in local]),
        "component_logits": component,
        "local_log_beta": np.stack([s["log_beta"] for s in local]),
        "global_log_beta": global_log_beta,
        "global_gap": gap, "mean_global_gap": float(gap.mean()),
        "local_gap": np.stack([s["gap"] for s in local]),
    }


def global_quadrature_state(logits, seed_rows, probe_columns, singleton_column):
    """Evaluate [B,45,P] against all B*5 components, in NumPy FP64.

    seed_rows contains four distinct PHYSICAL row IDs per bank, not candidate
    slots. P is strictly U16+SQ; this is not an API for full80 evaluation.
    """
    prepared = _prepare(logits, seed_rows, probe_columns, singleton_column)
    result = _state(prepared, prepared["seed_rows"])
    result.update({k: v for k, v in prepared.items() if k != "seed_rows"})
    return result


def refine_global_route_medoids(
    logits, candidate_rows, incumbent_rows, probe_columns, singleton_column,
    *, weighting="global",
):
    """One three-slot medoid pass with initially frozen global/local beta.

    logits: finite natural logits [B,45,P], axis1 is physical RF0 row0..44.
    candidate_rows: [B,32], distinct IDs0..44; table order breaks new ties.
    incumbent_rows: [B,4], distinct PHYSICAL IDs present in each candidate
      table. Its first row remains fixed. The caller must verify Anchor32
      provenance; the reference neither selects nor assumes a matched row.
    probe_columns/singleton_column: exact physical U16+SQ support, any order.
    weighting: 'global' (default) or 'bank_local', frozen for the ENTIRE pass.

    No probability threshold or all-cell exp(beta)==0 test is used. Each
    active cell normalizes finite log-beta ACROSS TARGETS before exponentiation.
    Steps retain log-demand and log-gain, so an absolute surrogate improvement
    below FP64's exponent range is not reported as a missing-demand cell.

    The global mode minimizes its frozen-beta surrogate and, in ideal
    arithmetic, cannot increase mean global quadrature gap after re-routing.
    The local control only has the analogous mean BANK-LOCAL guarantee.
    Neither promises nonincrease for each of the 45 individual targets.
    """
    if weighting not in ("global", "bank_local"):
        raise ValueError("weighting must be 'global' or 'bank_local'")
    prepared = _prepare(logits, incumbent_rows, probe_columns, singleton_column)
    log_p = prepared["log_p"]
    banks = len(log_p)
    candidates = _integer_rows(candidate_rows, banks, 32, "candidate_rows")
    selected = prepared["seed_rows"].copy()
    incumbent = selected.copy()
    if any(not np.isin(rows, table).all() for rows, table in zip(selected, candidates)):
        raise ValueError("all incumbent physical rows must occur in their candidate table")
    initial = _state(prepared, selected)
    fixed_owners = initial["owners"]
    frozen = initial["global_log_beta" if weighting == "global" else "local_log_beta"].copy()
    # Both a query mean and a bank mean are used for the local control. This
    # normalization has no effect on its medoid decisions.
    log_population = np.log(45.0) + (np.log(float(banks)) if weighting == "bank_local" else 0.0)
    steps = []
    log_gains = []
    for bank in range(banks):
        p = log_p[bank]
        table = candidates[bank]
        for cell in (1, 2, 3):
            before = selected[bank].copy()
            mask = fixed_owners[bank] == cell
            active = bool(mask.any())
            feasible_all = _feasible_candidates(
                p, selected[bank], cell, fixed_owners[bank], prepared["singleton_index"])
            feasible = feasible_all[table]
            old_slot = int(np.flatnonzero(table == before[cell])[0])
            chosen_slot = old_slot
            costs = np.zeros(32, dtype=np.float64)
            log_alpha = np.full(45, -np.inf, dtype=np.float64)
            log_rho = np.full(p.shape[-1], -np.inf, dtype=np.float64)
            log_mean_demand = -np.inf
            normalized_gain = 0.0
            log_gain = -np.inf
            if active:
                demand = frozen[bank, :, cell]
                if not np.isfinite(demand).all():
                    raise ValueError("a structurally nonempty cell must have finite log demand")
                log_alpha = _log_softmax(demand)
                alpha = np.exp(log_alpha)
                # Conditional target logs remove candidate-independent mass
                # terms before the barycenter reduction, improving stability.
                target_logw = _log_softmax(p[:, mask])
                log_rho[mask] = _log_softmax(alpha @ target_logw)
                candidate_logw = target_logw[table]
                costs = np.sum(np.exp(candidate_logw) *
                               (candidate_logw - log_rho[mask]), axis=-1)
                if not np.isfinite(costs).all():
                    raise ValueError("conditional KL exceeds FP64 arithmetic")
                best = int(np.argmin(np.where(feasible, costs, np.inf)))
                if best != old_slot and costs[best] < costs[old_slot]:
                    chosen_slot = best
                    selected[bank, cell] = table[best]
                    normalized_gain = float(costs[old_slot] - costs[best])
                log_mean_demand = float(np.logaddexp.reduce(demand) - log_population)
                if normalized_gain > 0:
                    log_gain = log_mean_demand + float(np.log(normalized_gain))
                    log_gains.append(log_gain)
            after = _route_state(p, selected[bank], prepared["singleton_index"])
            if not np.array_equal(after["owner"], fixed_owners[bank]):
                raise AssertionError("a bulk quadrature owner changed")
            steps.append({
                "bank": bank, "cell": cell, "active": active,
                "before_rows": before, "after_rows": selected[bank].copy(),
                "old_slot": old_slot, "chosen_slot": chosen_slot,
                "accepted": chosen_slot != old_slot,
                "feasible_count": int(feasible.sum()),
                "feasible_candidates": feasible, "candidate_conditional_kl": costs,
                "frozen_log_demand": frozen[bank, :, cell].copy(),
                "log_alpha": log_alpha, "conditional_log_barycenter": log_rho,
                "normalized_surrogate_improvement": normalized_gain,
                "log_mean_frozen_demand": log_mean_demand,
                "log_surrogate_improvement": log_gain,
            })
    final = _state(prepared, selected)
    if not np.array_equal(final["owners"], fixed_owners):
        raise AssertionError("fixed-partition invariant violated")
    if not np.array_equal(selected[:, 0], incumbent[:, 0]):
        raise AssertionError("anchor row changed")
    if any(len(np.unique(row)) != 4 for row in selected):
        raise AssertionError("four-real-row uniqueness violated")
    log_total_gain = float(np.logaddexp.reduce(log_gains)) if log_gains else -np.inf
    return {
        "weighting": weighting, "seed_rows": selected,
        "incumbent_rows": incumbent, "candidate_rows": candidates,
        "changed_slots": selected != incumbent,
        "changed_seed_count": int(np.count_nonzero(selected != incumbent)),
        "probe_columns": prepared["probe_columns"],
        "singleton_index": prepared["singleton_index"], "measures": prepared["measures"],
        "old_global_gap": initial["global_gap"].copy(),
        "new_global_gap": final["global_gap"].copy(),
        "old_mean_global_gap": initial["mean_global_gap"],
        "new_mean_global_gap": final["mean_global_gap"],
        "frozen_log_beta": frozen, "steps": steps,
        "log_fixed_beta_surrogate_improvement": log_total_gain,
        "fixed_beta_surrogate_improvement": float(np.exp(log_total_gain)),
        "surrogate_scope": "mean_global" if weighting == "global" else "mean_bank_local",
        "guarantee_scope": "ideal RF0 supplied quadrature, fixed bulk owners, target mean only",
        "initial": initial, "final": final,
    }
