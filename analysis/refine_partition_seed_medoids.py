"""Reference-only conditional seed-medoid refinement on a fixed probe partition.

The first seed is fixed.  At most one update is made to each other cell.
Original local compressed route probabilities are held fixed during this
pass.  A replacement must preserve every probe key's owner and remain a
distinct real candidate row.  It minimizes candidate-first conditional KL to
the route-mass-weighted logit barycenter of that cell.

This is coordinate ascent for the fixed-rho, probe-local variational objective,
not a guarantee about full-key/global attention or generated-video PSNR.
Nothing in this module is integrated into the production inference path.
"""

from __future__ import annotations

import json

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _cell_distributions(log_probability, cells):
    """Normalize row log probabilities within possibly empty Boolean cells."""
    masked = log_probability.masked_fill(~cells, -float("inf"))
    maximum = masked.amax(dim=-1, keepdim=True)
    active = cells.any(dim=-1, keepdim=True)
    maximum = torch.where(active, maximum, 0.0)
    exponential = torch.where(cells, (log_probability - maximum).exp(), 0.0)
    total = exponential.sum(dim=-1, keepdim=True)
    total = torch.where(active, total, 1.0)
    probability = exponential / total
    log_weight = torch.where(cells, log_probability - maximum - total.log(), 0.0)
    return probability, log_weight, active.squeeze(-1)


def _route_state(log_probability, selected, owner):
    count = selected.size(-1)
    selected_logp = log_probability.gather(
        1, selected[..., None].expand(-1, -1, log_probability.size(-1))
    )
    cell_ids = torch.arange(count, device=selected.device)[None, :, None]
    cells = owner[:, None, :] == cell_ids
    weights, log_weights, active = _cell_distributions(selected_logp, cells)
    entropy = -(weights * log_weights).sum(dim=-1)
    bound = torch.einsum("brp,bcp->brc", log_probability, weights)
    bound = bound + entropy[:, None, :]
    bound = bound.masked_fill(~active[:, None, :], -float("inf"))
    local_gap = -torch.logsumexp(bound, dim=-1)
    return bound.softmax(dim=-1), local_gap, active


@torch.no_grad() if torch is not None else (lambda function: function)
def refine_partition_seed_medoids(
    logits: torch.Tensor,
    selected_indices: torch.Tensor,
    query_weights: torch.Tensor,
    *,
    return_diagnostics: bool = False,
):
    """Refine C4 candidate slots while preserving the current probe ownership.

    Args:
        logits: Finite natural-log scores, shape ``[..., R, P]``.
        selected_indices: Four distinct candidate slots, broadcastable to
            ``[..., 4]``.  Slot zero of this seed list is never changed.
        query_weights: Fixed finite nonnegative rho, broadcastable to
            ``[..., R]``.  It is not recomputed during refinement.
        return_diagnostics: Return ``(indices, diagnostics)`` instead of indices.

    Returns long candidate slots with shape ``[..., 4]``.  All batch dimensions
    are flattened internally and restored.  Low-precision inputs accumulate
    in float32; float64 is preserved for mathematical checks.

    The original local compressed lambda is computed once. For cell c, the
    barycenter logit is sum_i rho_i lambda_ic z_i / sum_i rho_i lambda_ic.
    Candidate conditionals use the original cell support, and the candidate
    must leave all argmax-normalized-likelihood owner labels unchanged.
    Exact ties favor the current row.  Empty or zero-demand cells are skipped.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for conditional medoid refinement")
    if logits.ndim < 2 or not logits.is_floating_point():
        raise ValueError("logits must be floating-point with shape [...,R,P]")
    rows, probes = logits.shape[-2:]
    if rows < 4 or probes < 1:
        raise ValueError("require at least four candidate rows and one probe")
    prefix = logits.shape[:-2]
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    work = logits.to(dtype=dtype).reshape(-1, rows, probes)
    selected = torch.broadcast_to(
        torch.as_tensor(selected_indices, device=logits.device, dtype=torch.long),
        (*prefix, 4),
    ).reshape(-1, 4).clone()
    rho = torch.broadcast_to(
        torch.as_tensor(query_weights, device=logits.device, dtype=dtype),
        (*prefix, rows),
    ).reshape(-1, rows)
    if not bool(torch.isfinite(work).all()) or not bool(torch.isfinite(rho).all()):
        raise ValueError("logits and query_weights must be finite")
    if bool((rho < 0).any()):
        raise ValueError("query_weights must be nonnegative")
    if bool(((selected < 0) | (selected >= rows)).any()):
        raise ValueError("selected candidate index is outside [0,R)")
    ordered = selected.sort(dim=-1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise ValueError("initial seed rows must be distinct")

    initial = selected.clone()
    logp = work.log_softmax(dim=-1)
    selected_logp = logp.gather(1, selected[..., None].expand(-1, -1, probes))
    owner = selected_logp.argmax(dim=1)
    original_lambda, initial_gap, active = _route_state(logp, selected, owner)
    candidate_ids = torch.arange(rows, device=logits.device)[None, :]
    seed_ids = torch.arange(4, device=logits.device)[None, :]

    for cell in range(1, 4):
        cells = owner[:, None, :] == cell
        demand = rho * original_lambda[:, :, cell]
        total_demand = demand.sum(dim=-1, keepdim=True)
        denominator = total_demand.clamp_min(torch.finfo(dtype).tiny)
        barycenter = (demand[..., None] * logp).sum(dim=1) / denominator
        _, barycenter_logp, _ = _cell_distributions(barycenter[:, None, :], cells)
        conditional, conditional_logp, _ = _cell_distributions(logp, cells)
        divergence = (
            conditional * (conditional_logp - barycenter_logp)
        ).sum(dim=-1)

        # Argmax uses the first seed on equal normalized probabilities.  The
        # following comparison reproduces that rule for every trial row.
        selected_logp = logp.gather(1, selected[..., None].expand(-1, -1, probes))
        other_logp = selected_logp.masked_fill(seed_ids[..., None] == cell, -float("inf"))
        competitor_value, competitor_owner = other_logp.max(dim=1)
        wins = (logp > competitor_value[:, None, :]) | (
            (logp == competitor_value[:, None, :])
            & (cell < competitor_owner[:, None, :])
        )
        trial_owner = torch.where(wins, cell, competitor_owner[:, None, :])
        allowed = (trial_owner == owner[:, None, :]).all(dim=-1)
        duplicate = (
            (candidate_ids[:, :, None] == selected[:, None, :])
            & (seed_ids[:, None, :] != cell)
        ).any(dim=-1)
        allowed = allowed & ~duplicate
        current = selected[:, cell]
        if not bool(allowed.gather(-1, current[:, None]).all()):
            raise AssertionError("the current seed must remain a feasible candidate")
        candidate_loss = divergence.masked_fill(~allowed, float("inf"))
        best_loss, best = candidate_loss.min(dim=-1)
        current_loss = divergence.gather(-1, current[:, None]).squeeze(-1)
        change = (
            active[:, cell] & (total_demand[:, 0] > 0.0)
            & (best_loss < current_loss)
        )
        selected[:, cell] = torch.where(change, best, current)

    selected_logp = logp.gather(1, selected[..., None].expand(-1, -1, probes))
    final_owner = selected_logp.argmax(dim=1)
    _, final_gap, _ = _route_state(logp, selected, final_owner)
    if not bool((final_owner == owner).all()):
        raise AssertionError("refinement changed a probe owner label")
    result = selected.reshape(*prefix, 4)
    if not return_diagnostics:
        return result

    local_log_norm = work.logsumexp(dim=-1)
    before_objective = (rho * (local_log_norm - initial_gap)).sum(dim=-1)
    after_objective = (rho * (local_log_norm - final_gap)).sum(dim=-1)
    diagnostics = {
        "weighted_local_lse_before": before_objective.reshape(prefix),
        "weighted_local_lse_after": after_objective.reshape(prefix),
        "weighted_local_lse_gain": (after_objective - before_objective).reshape(prefix),
        "weighted_local_gap_before": (rho * initial_gap).sum(dim=-1).reshape(prefix),
        "weighted_local_gap_after": (rho * final_gap).sum(dim=-1).reshape(prefix),
        "changed_seed_count": (selected != initial).sum(dim=-1).reshape(prefix),
        "owner_preserved": (final_owner == owner).all(dim=-1).reshape(prefix),
        "anchor_preserved": (selected[:, 0] == initial[:, 0]).reshape(prefix),
    }
    return result, diagnostics


def _numpy_reference(logits, selected, rho):
    """Small independent CPU oracle; explicitly evaluates each trial owner."""
    import numpy as np

    def logsoftmax(value, axis=-1):
        maximum = value.max(axis=axis, keepdims=True)
        return value - maximum - np.log(np.exp(value - maximum).sum(axis=axis, keepdims=True))

    def conditional(logp, mask):
        output = np.zeros_like(logp)
        log_output = np.zeros_like(logp)
        if mask.any():
            log_output[..., mask] = logsoftmax(logp[..., mask])
            output[..., mask] = np.exp(log_output[..., mask])
        return output, log_output

    def routes(logp, seeds, owner):
        bound = np.full((logp.shape[0], 4), -np.inf)
        for c in range(4):
            mask = owner == c
            if not mask.any():
                continue
            weight, log_weight = conditional(logp[seeds[c]], mask)
            bound[:, c] = logp @ weight - np.sum(weight * log_weight)
        finite_max = bound.max(axis=-1)
        local_lse = finite_max + np.log(np.exp(bound - finite_max[:, None]).sum(axis=-1))
        return np.exp(logsoftmax(bound)), local_lse

    output, before, after = [], [], []
    for z, initial, demand_rho in zip(logits, selected, rho):
        logp = logsoftmax(z)
        seeds = initial.copy()
        owner = logp[seeds].argmax(axis=0)
        route, initial_lse = routes(logp, seeds, owner)
        for c in range(1, 4):
            mask = owner == c
            weights = demand_rho * route[:, c]
            if not mask.any() or weights.sum() == 0.0:
                continue
            barycenter = (weights[:, None] * logp).sum(axis=0) / weights.sum()
            _, barycenter_logp = conditional(barycenter, mask)
            candidate_p, candidate_logp = conditional(logp, mask)
            kl = np.sum(candidate_p * (candidate_logp - barycenter_logp), axis=-1)
            allowed = np.zeros(logp.shape[0], dtype=bool)
            for candidate in range(logp.shape[0]):
                trial = seeds.copy()
                trial[c] = candidate
                allowed[candidate] = (
                    len(set(trial.tolist())) == 4
                    and np.array_equal(logp[trial].argmax(axis=0), owner)
                )
            assert allowed[seeds[c]]
            best = np.where(allowed, kl, np.inf).argmin()
            if kl[best] < kl[seeds[c]]:
                seeds[c] = best
        assert seeds[0] == initial[0]
        assert np.array_equal(logp[seeds].argmax(axis=0), owner)
        _, final_lse = routes(logp, seeds, owner)
        output.append(seeds)
        # Dense local LSE is a fixed additive term; normalized LSE suffices
        # for the monotonicity test and avoids irrelevant large offsets.
        before.append(float(demand_rho @ initial_lse))
        after.append(float(demand_rho @ final_lse))
    return np.stack(output), np.array(before), np.array(after)


def _self_check():
    import numpy as np

    generator = np.random.default_rng(4617)
    cases, changes, min_gain = 0, 0, float("inf")
    torch_cases = 0
    for trial in range(20):
        z = generator.normal(size=(6, 11, 9)) * (8.0 if trial == 1 else 1.0)
        initial = np.stack([generator.choice(11, 4, replace=False) for _ in range(6)])
        rho = generator.random((6, 11))
        if trial == 2:
            z[:] = 0.0  # Empty non-anchor cells, exact ties.
        if trial == 3:
            rho[:] = 0.0  # Zero-demand cells must not move.
        expected, before, after = _numpy_reference(z, initial, rho)
        gain = after - before
        assert gain.min() >= -1.0e-10
        if trial in (2, 3):
            assert np.array_equal(expected, initial)
        min_gain = min(min_gain, float(gain.min()))
        cases += 6
        changes += int(np.count_nonzero(expected != initial))
        if torch is not None:
            result, info = refine_partition_seed_medoids(
                torch.from_numpy(z.reshape(2, 3, 11, 9)),
                torch.from_numpy(initial.reshape(2, 3, 4)),
                torch.from_numpy(rho.reshape(2, 3, 11)),
                return_diagnostics=True,
            )
            assert np.array_equal(result.numpy().reshape(6, 4), expected)
            assert bool(info["owner_preserved"].all())
            assert bool(info["anchor_preserved"].all())
            assert float(info["weighted_local_lse_gain"].min()) >= -1.0e-10
            torch_cases += 6
    return {
        "numpy_float64_cases": cases,
        "torch_float64_cases": torch_cases,
        "changed_seed_slots": changes,
        "minimum_weighted_local_lse_gain": min_gain,
        "scope": "fixed-rho local probes only; no global-gap or PSNR guarantee",
    }


if __name__ == "__main__":
    print(json.dumps(_self_check(), indent=2))
