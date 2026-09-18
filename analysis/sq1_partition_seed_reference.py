"""CPU-only SQ1 correction of the existing partition-deficit farthest pivot.

This is not a new selector family, an optimal C4 solver, or a PSNR/runtime
claim. The anchor is fixed; later real rows maximize their CURRENT weighted
missing partition mass, not the improvement after trial insertion. Hard
ownership is recomputed on the entire supplied support BEFORE SQ1 peeling.
Adding a seed can worsen the objective because the hard cells change.

Finite logits use natural units. Positive shared quadrature measures mu model
replicated keys: Z=sum_j mu_j exp(z_j), with unit measure for the explicitly
observed singleton. Equivalently, bin logits are z+log(mu). Bin entropy H(w)
therefore corresponds to H(w)+sum_j w_j log(mu_j) on represented keys. Merely
adding a singleton to U16 with equal bin weights is not the 80-key quadrature.

Identities are exact for the supplied weighted-support model, up to float64
roundoff. P16+SQ is still a quadrature surrogate for full80 ownership and loss;
RF0 transfer, global-mass estimates and native BF16 rounding remain separate
approximations. This module neither runs nor modifies a production operator.
"""

from __future__ import annotations

import numpy as np


def _finite(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _index(value, shape, size, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be explicit integer indices")
    try:
        result = np.broadcast_to(raw, shape)
    except ValueError as error:
        raise ValueError(f"{name} does not broadcast to batch shape") from error
    if np.any(result < 0) or np.any(result >= size):
        raise ValueError(f"{name} must be in [0, {size})")
    return result.astype(np.int64, copy=False)


def _lse(value, axis=-1, keepdims=False):
    maximum = np.max(value, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(maximum), maximum, 0.0)
    with np.errstate(under="ignore", divide="ignore"):
        result = safe + np.log(np.exp(value - safe).sum(axis=axis, keepdims=True))
    return result if keepdims else np.squeeze(result, axis=axis)


def _weighted_logp(logits, log_measure):
    # Remove the row offset before adding log(mu), preserving small measure
    # corrections even when an otherwise harmless logit offset is large.
    with np.errstate(over="ignore", invalid="ignore"):
        centered = logits - logits.max(axis=-1, keepdims=True)
        weighted = centered + log_measure[..., None, :]
        centered = weighted - weighted.max(axis=-1, keepdims=True)
    if not np.isfinite(centered).all():
        raise ValueError("logit dynamic range exceeds float64 arithmetic")
    return centered - _lse(centered, keepdims=True)


def _measures(value, batch_shape, probes, singleton):
    measure = np.ones((*batch_shape, probes)) if value is None else _finite(value, "measures")
    try:
        measure = np.broadcast_to(measure, (*batch_shape, probes))
    except ValueError as error:
        raise ValueError("measures must broadcast to [..., probes]") from error
    if np.any(measure <= 0):
        raise ValueError("measures must be strictly positive")
    singleton_measure = np.take_along_axis(measure, singleton[..., None], axis=-1)
    if np.any(singleton_measure != 1.0):
        raise ValueError("the explicit singleton must have unit measure")
    return measure


def sq1_partition_metrics(
    target_logits, seed_logits, singleton_index, *, measures=None, return_details=False,
):
    """Return per-target ``reverse_gap`` and ``mass_deficit`` as float64.

    Targets have shape ``[..., R, P]`` and seeds ``[..., C, P]``; leading
    dimensions may broadcast. The explicit singleton index broadcasts to
    that batch shape and must lie in [0,P). None/-1 are rejected: an unseen
    singleton must be evaluated explicitly, not treated as zero mass.
    ``measures`` is positive, shared by all targets/seeds, and broadcasts to
    ``[..., P]``; it defaults to one and must be one at the singleton.

    For the normalized bin distribution p_i, fixed bulk weights w_c and SQ,
    represented_i = p_i[SQ] + sum_c exp(w_c @ log(p_i) + H(w_c)).
    The returned gap is -log(represented), and the deficit is
    1-represented, evaluated with expm1. Empty cells contribute zero mass.
    Tiny negative roundoff is preserved in metrics, not silently clipped.
    Details include normalized/reconstructed distributions and cell data;
    singleton occupies slot C, and lower seed slots win ownership ties.
    """
    target, seeds = _finite(target_logits, "target_logits"), _finite(seed_logits, "seed_logits")
    if target.ndim < 2 or seeds.ndim < 2:
        raise ValueError("target_logits and seed_logits need rank >= 2")
    if min(target.shape[-2:]) < 1 or min(seeds.shape[-2:]) < 1:
        raise ValueError("target, seed, and support dimensions must be nonempty")
    if target.shape[-1] != seeds.shape[-1]:
        raise ValueError("target and seed support sizes differ")
    try:
        batch = np.broadcast_shapes(target.shape[:-2], seeds.shape[:-2])
    except ValueError as error:
        raise ValueError("target and seed batch shapes do not broadcast") from error
    rows, probes = target.shape[-2:]
    count = seeds.shape[-2]
    target = np.broadcast_to(target, (*batch, rows, probes))
    seeds = np.broadcast_to(seeds, (*batch, count, probes))
    singleton = _index(singleton_index, batch, probes, "singleton_index")
    measure = _measures(measures, batch, probes, singleton)
    log_mu = np.log(measure)
    log_p = _weighted_logp(target, log_mu)
    seed_log_p = _weighted_logp(seeds, log_mu)
    owner_before = np.argmax(seed_log_p, axis=-2)
    owner = np.where(np.arange(probes) == singleton[..., None], count, owner_before)
    cells = owner[..., None, :] == np.arange(count + 1)[:, None]
    active = cells.any(axis=-1)
    source = np.concatenate((seed_log_p, np.zeros((*batch, 1, probes))), axis=-2)
    norm = _lse(np.where(cells, source, -np.inf), keepdims=True)
    log_w = np.where(cells, source - np.where(active[..., None], norm, 0.0), -np.inf)
    finite_log_w = np.where(cells, log_w, 0.0)
    with np.errstate(under="ignore"):
        weights = np.exp(log_w)
    entropy = -(weights * finite_log_w).sum(axis=-1)
    with np.errstate(over="ignore", invalid="ignore"):
        component = np.einsum("...rp,...cp->...rc", log_p, weights, optimize=True)
        component += entropy[..., None, :]
    component = np.where(active[..., None, :], component, -np.inf)
    log_represented = _lse(component)
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        deficit = -np.expm1(log_represented)
    result = {"reverse_gap": -log_represented, "mass_deficit": deficit}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("metric arithmetic exceeds the finite float64 range")
    if return_details:
        log_beta = component - log_represented[..., None]
        target_owner = np.broadcast_to(owner[..., None, :], log_p.shape)
        owner_log_w = np.take_along_axis(log_w, owner[..., None, :], axis=-2)[..., 0, :]
        log_hat = np.take_along_axis(log_beta, target_owner, axis=-1) + owner_log_w[..., None, :]
        with np.errstate(under="ignore"):
            p, p_hat = np.exp(log_p), np.exp(log_hat)
        result.update(
            log_p=log_p, p=p, seed_log_p=seed_log_p, log_p_hat=log_hat, p_hat=p_hat,
            owner_before_singleton=owner_before, owner=owner, weights=weights,
            log_weights=log_w, active_cells=active, bin_entropy=entropy,
            measure_entropy=entropy + (weights * log_mu[..., None, :]).sum(axis=-1),
            component_log_mass=component, log_represented_mass=log_represented,
        )
    return result


def select_sq1_partition_seeds(
    logits, anchor, singleton_index, *, importance=None, num_seeds=4,
    measures=None, return_history=False,
):
    """Anchored current-deficit farthest pivot; no all-trial lookahead.

    ``logits[..., R, P]`` contains the actual real-query candidate rows.
    Return int64 positions ``[..., num_seeds]`` into those rows. Importance
    is finite/nonnegative, broadcasts to ``[..., R]``, and defaults to one.
    It is divided by its bank maximum, a common ranking-invariant factor;
    all-zero importance yields unused low-index ties, not a hidden fallback.
    Ranking alone clips roundoff-negative deficits to zero. Exact score ties
    choose the lowest unused position. No marginal-improvement guarantee.
    Optional history contains pre-insertion metrics and UNMASKED scores;
    the second-last dimension indexes the num_seeds-1 pivot steps.
    """
    raw = _finite(logits, "logits")
    if raw.ndim < 2 or min(raw.shape[-2:]) < 1:
        raise ValueError("logits must have nonempty [..., R, P] shape")
    count, probes = raw.shape[-2:]
    if isinstance(num_seeds, (bool, np.bool_)) or not isinstance(num_seeds, (int, np.integer)):
        raise ValueError("num_seeds must be an integer")
    if not 1 <= num_seeds <= count:
        raise ValueError("num_seeds must be between one and the candidate count")
    batch = raw.shape[:-2]
    first = _index(anchor, batch, count, "anchor")
    singleton = _index(singleton_index, batch, probes, "singleton_index")
    measure = _measures(measures, batch, probes, singleton)
    # Validate arithmetic even when num_seeds=1 bypasses metric evaluation.
    _weighted_logp(raw, np.log(measure))
    demand = np.ones(raw.shape[:-1]) if importance is None else _finite(importance, "importance")
    try:
        demand = np.broadcast_to(demand, raw.shape[:-1])
    except ValueError as error:
        raise ValueError("importance does not broadcast to candidate shape") from error
    if np.any(demand < 0):
        raise ValueError("importance must be nonnegative")
    maximum = demand.max(axis=-1, keepdims=True)
    demand = demand / np.where(maximum > 0, maximum, 1.0)
    chosen = first[..., None].copy()
    used = np.zeros(raw.shape[:-1], dtype=bool)
    np.put_along_axis(used, chosen, True, axis=-1)
    history = {name: [] for name in ("reverse_gap", "mass_deficit", "scores")}
    for _ in range(1, num_seeds):
        indices = np.broadcast_to(chosen[..., :, None], (*batch, chosen.shape[-1], probes))
        selected_logits = np.take_along_axis(raw, indices, axis=-2)
        metrics = sq1_partition_metrics(raw, selected_logits, singleton, measures=measure)
        score = demand * np.maximum(metrics["mass_deficit"], 0.0)
        picked = np.argmax(np.where(used, -np.inf, score), axis=-1)[..., None]
        if return_history:
            history["reverse_gap"].append(metrics["reverse_gap"])
            history["mass_deficit"].append(metrics["mass_deficit"])
            history["scores"].append(score)
        chosen = np.concatenate((chosen, picked), axis=-1)
        np.put_along_axis(used, picked, True, axis=-1)
    if not return_history:
        return chosen
    result = {"seeds": chosen}
    result.update({name: np.stack(values, axis=-2) if values else np.empty((*batch, 0, count))
                   for name, values in history.items()})
    return result


def uniform16_sq1_quadrature(singleton_column, width=80):
    """Describe U16 plus an explicitly observed SQ; does not evaluate QK.

    Keep all original floor(v*width/16) columns in order. If SQ is absent,
    append it (17 bins); otherwise reuse its original slot (16 bins). Give
    SQ unit measure and the m remaining bulk probes (width-1)/m each, so
    total measure equals width. A caller must supply the actual singleton
    logit, including its additional QK when it was absent from U16.
    """
    if isinstance(width, (bool, np.bool_)) or not isinstance(width, (int, np.integer)) or width < 16:
        raise ValueError("width must be an integer >= 16")
    singleton = int(_index(singleton_column, (), width, "singleton_column"))
    columns = np.arange(16, dtype=np.int64) * width // 16
    match = np.flatnonzero(columns == singleton)
    if match.size:
        position = int(match[0])
    else:
        position = 16
        columns = np.concatenate((columns, [singleton]))
    bulk_count = len(columns) - 1
    measure = np.full(len(columns), (width - 1) / bulk_count, dtype=np.float64)
    measure[position] = 1.0
    return {"probe_columns": columns, "singleton_index": position, "measures": measure,
            "bulk_probe_count": bulk_count, "represented_key_count": int(width)}


__all__ = ["sq1_partition_metrics", "select_sq1_partition_seeds", "uniform16_sq1_quadrature"]
