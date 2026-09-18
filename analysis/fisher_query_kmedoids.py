"""Fixed-reference Fisher query k-medoids, using four distinct real queries.

NumPy reference only. The shared metric is frozen before clustering; no
parameters are learned across inference calls. Lloyd updates run to a fixed
point. The iteration limit is a failure guard, never a truncated candidate.
"""
from __future__ import annotations

import numpy as np


def fisher_features(logits):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 3 or min(z.shape) < 1 or not np.isfinite(z).all():
        raise ValueError("finite nonempty [bank, candidate, probe] logits required")
    reference = z.mean(axis=1)
    rho = np.exp(reference - np.logaddexp.reduce(reference, axis=-1)[:, None])
    delta = z - reference[:, None, :]
    return (delta - (delta * rho[:, None, :]).sum(-1, keepdims=True)) * np.sqrt(rho[:, None, :])


def validate_features(features, count):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("finite nonempty feature banks required")
    if not isinstance(count, int) or not 1 <= count <= x.shape[1]:
        raise ValueError("invalid real-seed count")
    return x


def squared_distances(x, seeds):
    centers = x[np.arange(len(x))[:, None], seeds]
    return ((x[:, :, None, :] - centers[:, None, :, :]) ** 2).sum(-1)


def coverage(x, seeds):
    return squared_distances(x, seeds).min(-1).sum(-1)


def direct_initialization(features, count=4):
    """Closest real point to the mean, then farthest-first; no spatial anchor."""
    x = validate_features(features, count)
    batch = np.arange(len(x))
    chosen = np.empty((len(x), count), dtype=np.int64)
    chosen[:, 0] = ((x - x.mean(1, keepdims=True)) ** 2).sum(-1).argmin(-1)
    used = np.zeros(x.shape[:2], dtype=bool)
    used[batch, chosen[:, 0]] = True
    nearest = squared_distances(x, chosen[:, :1])[..., 0]
    for slot in range(1, count):
        pick = np.where(used, -np.inf, nearest).argmax(-1)
        chosen[:, slot] = pick
        used[batch, pick] = True
        nearest = np.minimum(nearest, squared_distances(x, chosen[:, slot:slot + 1])[..., 0])
    return chosen


def fisher_gs(features, anchor, count=4):
    """Paper's anchored Fisher-GS in FP64 on the same features."""
    x = validate_features(features, count)
    anchor = np.asarray(anchor, dtype=np.int64)
    if anchor.shape != (len(x),) or np.any((anchor < 0) | (anchor >= x.shape[1])):
        raise ValueError("invalid anchor slots")
    batch = np.arange(len(x))
    residual = x - x[batch, anchor, None, :]
    used = np.zeros(x.shape[:2], dtype=bool)
    used[batch, anchor] = True
    chosen = np.empty((len(x), count), dtype=np.int64)
    chosen[:, 0] = anchor
    for slot in range(1, count):
        pick = np.where(used, -np.inf, (residual ** 2).sum(-1)).argmax(-1)
        chosen[:, slot] = pick
        used[batch, pick] = True
        basis = residual[batch, pick].copy()
        basis /= np.sqrt((basis ** 2).sum(-1, keepdims=True) + 1e-12)
        residual -= (residual * basis[:, None, :]).sum(-1, keepdims=True) * basis[:, None, :]
    return chosen


def lloyd_medoids(features, initial=None, *, count=4, fixed_first=False, max_iterations=256):
    """Exact cluster-medoid updates; retain the old seed on numerical ties.

    A selected point owns itself on a zero-distance tie. Consequently each
    cluster is nonempty and disjoint, even for identical feature vectors.
    Simultaneous within-cluster medoid updates retain distinct real indices.
    """
    x = validate_features(features, count)
    seeds = direct_initialization(x, count) if initial is None else np.asarray(initial).copy()
    if (seeds.shape != (len(x), count) or seeds.dtype.kind not in "iu"
            or np.any((seeds < 0) | (seeds >= x.shape[1]))
            or np.any(np.diff(np.sort(seeds, axis=-1), axis=-1) == 0)):
        raise ValueError("initialization must contain distinct valid real indices")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("positive convergence guard required")
    seeds = seeds.astype(np.int64)
    batch = np.arange(len(x))
    converged = np.zeros(len(x), dtype=bool)
    iterations = np.zeros(len(x), dtype=np.int64)
    history = [coverage(x, seeds)]
    for iteration in range(1, max_iterations + 1):
        distance = squared_distances(x, seeds)
        labels = distance.argmin(-1)
        labels[batch[:, None], seeds] = np.arange(count)[None, :]
        members = labels[:, None, :] == np.arange(count)[None, :, None]
        sizes = members.sum(-1)
        means = np.einsum("bcr,brp->bcp", members.astype(np.float64), x) / sizes[..., None]
        costs = ((x[:, None, :, :] - means[:, :, None, :]) ** 2).sum(-1)
        best = np.where(members, costs, np.inf).argmin(-1)
        current_cost = np.take_along_axis(costs, seeds[..., None], axis=2)[..., 0]
        best_cost = np.take_along_axis(costs, best[..., None], axis=2)[..., 0]
        improve = current_cost - best_cost > 1e-12 * (1 + np.abs(current_cost))
        if fixed_first:
            improve[:, 0] = False
        updated = np.where(improve & ~converged[:, None], best, seeds)
        objective = coverage(x, updated)
        if np.any(objective > history[-1] + 1e-10 * (1 + np.abs(history[-1]))):
            raise ArithmeticError("nearest-medoid objective increased")
        history.append(objective)
        stable = np.all(updated == seeds, axis=-1)
        iterations[~converged & stable] = iteration
        converged |= stable
        seeds = updated
        if converged.all():
            return {"seeds": seeds, "iterations": iterations,
                    "objective_history": np.stack(history), "converged": converged}
    raise RuntimeError("Fisher k-medoids did not converge; refusing truncated results")
