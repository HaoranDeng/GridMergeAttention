"""Isolated NumPy reference for conditional-Jeffreys seed selection.

This is an unpromoted, training-free CPU diagnostic, not a production path.
Logits use natural-log units and must be finite. A seed first receives its
full-block softmax; hard ownership uses the largest normalized log probability,
with lower seed slots winning ties. An optional singleton is peeled only after
that ownership decision. The remaining seed conditionals and singleton are
disjoint components of one shared entropy-regularized projection.

For dense target p, cell masses P_c, conditionals p_c and fixed components w_c,
the reverse-KL projection p_hat satisfies

    Gamma = sum_c P_c [KL(w_c || p_c) + KL(p_c || w_c)]
          = KL(p_hat || p) + KL(p || p_hat).

The ideal identity concerns these supplied logits and cells, not full attention
when the supplied keys are only probes. Small negative numerical residuals are
not hidden in returned metrics. Log probabilities stay finite on owned support
even if their exponentials underflow. Float64 representable arithmetic is
required; astronomically large finite logits can exceed that arithmetic range.

The anchored selector chooses the largest weighted current Gamma. Adding a
seed repartitions cells and need not lower Gamma or the entropy objective.
Neither video-quality improvement nor unchanged runtime follows from this rule.
"""

from __future__ import annotations

import numpy as np


def _finite_real(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _logsumexp(value, axis, *, keepdims=False):
    """Stable internal LSE, allowing all-minus-infinity empty-cell slices."""
    maximum = np.max(value, axis=axis, keepdims=True)
    safe_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    with np.errstate(divide="ignore", under="ignore", invalid="ignore"):
        total = np.exp(value - safe_maximum).sum(axis=axis, keepdims=True)
        result = safe_maximum + np.log(total)
    return result if keepdims else np.squeeze(result, axis=axis)


def _log_softmax(value):
    # Center first rather than subtracting an LSE rounded at a large offset.
    with np.errstate(over="ignore", invalid="ignore"):
        centered = value - np.max(value, axis=-1, keepdims=True)
    if not np.isfinite(centered).all():
        raise ValueError("logit dynamic range is not representable in float64")
    return centered - _logsumexp(centered, -1, keepdims=True)


def _indices(value, batch_shape, lower, upper, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer indices")
    try:
        indices = np.broadcast_to(raw, batch_shape)
    except ValueError as error:
        raise ValueError(f"{name} does not broadcast to batch shape {batch_shape}") from error
    if np.any(indices < lower) or np.any(indices >= upper):
        raise ValueError(f"{name} indices must be in [{lower}, {upper})")
    return indices.astype(np.int64, copy=False)


def partition_metrics(
    target_logits,
    seed_logits,
    singleton_index=None,
    *,
    return_distributions=False,
):
    """Evaluate fixed hard cells and their exact-arithmetic projection identity.

    Args:
        target_logits: Finite real ``[..., R, P]`` target scores, R/P >= 1.
        seed_logits: Finite real ``[..., C, P]`` seed scores, C >= 1. The
            leading batch dimensions of targets and seeds may broadcast.
        singleton_index: Integer scalar or array broadcasting to the resulting
            batch shape. ``None`` or ``-1`` means absent; otherwise 0 <= u < P.
            Ownership is computed on all P keys before u is peeled.
        return_distributions: Include reconstruction/cell diagnostics below.

    Returns:
        A dict with ``reverse_gap``, ``forward_kl``, and ``jeffreys``, each
        float64 ``[..., R]``. Reverse gap comes from the projected log partition,
        forward KL from reconstructed log p_hat, and Jeffreys from the additive
        conditional identity. Tiny roundoff-negative metrics are returned as-is.
        If requested, also return ``log_p``, ``p``, ``log_p_hat``, ``p_hat``
        (``[..., R, P]``), ``owner_before_singleton`` and ``owner`` (``[..., P]``),
        ``log_weights``/``weights`` (``[..., C+1, P]``), ``active_cells``
        (``[..., C+1]``), and ``cell_mass``/``log_beta`` (``[..., R, C+1]``).
        Slot C is reserved for the singleton and is empty when absent. Empty
        cells have zero weights and minus-infinity log weights/log beta.
    """
    target = _finite_real(target_logits, "target_logits")
    seeds = _finite_real(seed_logits, "seed_logits")
    if target.ndim < 2 or seeds.ndim < 2:
        raise ValueError("target_logits and seed_logits must have rank at least two")
    if min(target.shape[-2:]) < 1 or min(seeds.shape[-2:]) < 1:
        raise ValueError("target, seed, and key dimensions must be nonempty")
    if target.shape[-1] != seeds.shape[-1]:
        raise ValueError("target_logits and seed_logits must have the same key count")
    try:
        batch_shape = np.broadcast_shapes(target.shape[:-2], seeds.shape[:-2])
    except ValueError as error:
        raise ValueError("target and seed batch dimensions do not broadcast") from error
    targets, probes = target.shape[-2:]
    seed_count = seeds.shape[-2]
    target = np.broadcast_to(target, (*batch_shape, targets, probes))
    seeds = np.broadcast_to(seeds, (*batch_shape, seed_count, probes))
    singleton = _indices(
        -1 if singleton_index is None else singleton_index,
        batch_shape, -1, probes, "singleton_index",
    )

    log_p = _log_softmax(target)
    seed_log_p = _log_softmax(seeds)
    owner_before = np.argmax(seed_log_p, axis=-2)
    owner = np.where(
        np.arange(probes) == singleton[..., None], seed_count, owner_before,
    )
    cells = owner[..., None, :] == np.arange(seed_count + 1)[:, None]
    active = cells.any(axis=-1)
    source_log_p = np.concatenate(
        (seed_log_p, np.zeros((*batch_shape, 1, probes), dtype=np.float64)), axis=-2,
    )
    masked_source = np.where(cells, source_log_p, -np.inf)
    cell_norm = _logsumexp(masked_source, -1, keepdims=True)
    safe_norm = np.where(active[..., None], cell_norm, 0.0)
    log_weights = np.where(cells, source_log_p - safe_norm, -np.inf)
    finite_log_weights = np.where(cells, log_weights, 0.0)
    with np.errstate(under="ignore"):
        weights = np.exp(log_weights)
        p = np.exp(log_p)
    entropy = -(weights * finite_log_weights).sum(axis=-1)
    u = np.einsum("...rp,...cp->...rc", log_p, weights, optimize=True)
    u += entropy[..., None, :]
    u = np.where(active[..., None, :], u, -np.inf)
    log_bound = _logsumexp(u, -1)
    reverse_gap = -log_bound

    log_beta = u - log_bound[..., None]
    log_owner_weight = np.take_along_axis(
        log_weights, owner[..., None, :], axis=-2,
    )[..., 0, :]
    owner_for_targets = np.broadcast_to(owner[..., None, :], log_p.shape)
    log_p_hat = np.take_along_axis(log_beta, owner_for_targets, axis=-1)
    log_p_hat += log_owner_weight[..., None, :]
    forward_kl = (p * (log_p - log_p_hat)).sum(axis=-1)

    cell_mass = np.einsum(
        "...rp,...cp->...rc", p, cells.astype(np.float64), optimize=True,
    )
    first = (p * (log_p - log_owner_weight[..., None, :])).sum(axis=-1)
    safe_u = np.where(active[..., None, :], u, 0.0)
    jeffreys = first - (cell_mass * safe_u).sum(axis=-1)
    result = {
        "reverse_gap": reverse_gap,
        "forward_kl": forward_kl,
        "jeffreys": jeffreys,
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("metric arithmetic exceeded the finite float64 range")
    if return_distributions:
        with np.errstate(under="ignore"):
            p_hat = np.exp(log_p_hat)
        result.update(
            log_p=log_p, p=p, log_p_hat=log_p_hat, p_hat=p_hat,
            owner_before_singleton=owner_before, owner=owner,
            log_weights=log_weights, weights=weights, active_cells=active,
            cell_mass=cell_mass, log_beta=log_beta,
        )
    return result


def select_jeffreys_seeds(
    logits, anchor, weights=None, num_seeds=4, singleton_index=None,
):
    """Choose anchored real-candidate indices by weighted current Jeffreys gap.

    ``logits`` is finite ``[..., R, P]``; ``anchor`` is an integer scalar or
    array broadcasting to ``...``. Nonnegative finite ``weights`` broadcast to
    ``[..., R]`` and default to one. All-zero weights give deterministic unused
    low-index choices, not a hidden fallback criterion. Ties always choose the
    lowest unused candidate index. Return int64 ``[..., num_seeds]``.

    We divide weights by each bank's maximum (a common positive factor), and
    clamp roundoff-negative Gamma to zero only when ranking candidates. No
    mean/synthetic query is inserted. Adding seeds does not guarantee a lower
    loss because ownership and within-cell weights change.
    """
    scores = _finite_real(logits, "logits")
    if scores.ndim < 2 or min(scores.shape[-2:]) < 1:
        raise ValueError("logits must have nonempty [..., R, P] shape")
    if isinstance(num_seeds, (bool, np.bool_)) or not isinstance(num_seeds, (int, np.integer)):
        raise ValueError("num_seeds must be an integer")
    count, probes = scores.shape[-2:]
    if not 1 <= num_seeds <= count:
        raise ValueError("num_seeds must be between one and the candidate count")
    batch_shape = scores.shape[:-2]
    first = _indices(anchor, batch_shape, 0, count, "anchor")
    singleton = _indices(
        -1 if singleton_index is None else singleton_index,
        batch_shape, -1, probes, "singleton_index",
    )
    if weights is None:
        demand = np.ones(scores.shape[:-1], dtype=np.float64)
    else:
        demand = _finite_real(weights, "weights")
        try:
            demand = np.broadcast_to(demand, scores.shape[:-1])
        except ValueError as error:
            raise ValueError("weights do not broadcast to candidate shape") from error
        if np.any(demand < 0):
            raise ValueError("weights must be nonnegative")
    maximum = demand.max(axis=-1, keepdims=True)
    demand = demand / np.where(maximum > 0.0, maximum, 1.0)
    used = np.zeros(scores.shape[:-1], dtype=bool)
    chosen = first[..., None].copy()
    np.put_along_axis(used, chosen, True, axis=-1)
    for _ in range(1, num_seeds):
        selected = np.take_along_axis(
            scores, np.broadcast_to(chosen[..., None], (*chosen.shape, probes)), axis=-2,
        )
        gamma = partition_metrics(scores, selected, singleton)["jeffreys"]
        priority = demand * np.maximum(gamma, 0.0)
        next_index = np.argmax(np.where(used, -np.inf, priority), axis=-1)
        chosen = np.concatenate((chosen, next_index[..., None]), axis=-1)
        np.put_along_axis(used, next_index[..., None], True, axis=-1)
    return chosen.astype(np.int64, copy=False)


__all__ = ["partition_metrics", "select_jeffreys_seeds"]
