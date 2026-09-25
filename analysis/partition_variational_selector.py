"""Probe-only seed selection using the actual hard-partition variational gap.

For selected seed distributions p_c, a probe belongs to argmax_c log p_c.
Within its disjoint cell, w_c is the conditional distribution of p_c.  The
compressed log-partition bound for target logits z_i is

    lower_i = logsumexp_c(<w_c, z_i> + H(w_c)).

``logsumexp(z_i) - lower_i`` is the reverse-KL projection error of that
partition family.  This module evaluates it directly, without a fitted
metric, learned parameter, or additional QK products.  It is a PyTorch
screening reference, not a claim of production selector performance.
"""

from __future__ import annotations

import math

import torch


VARIANTS = (
    "gap_farthest",
    "gap_weighted",
    "gap_gain",
    "gap_gain_weighted",
    "gap_deficit_weighted",
    "gap_deficit_gain_weighted",
)


def _prepare(
    logits: torch.Tensor,
    probe_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if logits.ndim < 2 or logits.size(-2) == 0 or logits.size(-1) == 0:
        raise ValueError("logits must have nonempty [..., candidates, probes] shape")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    batch_shape = tuple(logits.shape[:-2])
    batch_size = math.prod(batch_shape)
    candidates, probes = logits.shape[-2:]
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    raw = logits.to(dtype).reshape(batch_size, candidates, probes)
    # A shared support is necessary for the finite reverse-KL objective.
    valid = torch.isfinite(raw).all(dim=-2)
    if probe_mask is not None:
        mask = torch.broadcast_to(
            probe_mask.to(device=logits.device, dtype=torch.bool),
            (*batch_shape, probes),
        ).reshape(batch_size, probes)
        valid = valid & mask
    masked = raw.masked_fill(~valid[:, None, :], -torch.inf)
    log_normalizer = torch.logsumexp(masked, dim=-1, keepdim=True)
    log_normalizer = torch.where(valid.any(-1)[:, None, None], log_normalizer, 0.0)
    log_probability = masked - log_normalizer
    return log_probability, valid, batch_shape


def _gather(log_probability: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    return log_probability.gather(
        1, selected[..., None].expand(*selected.shape, log_probability.size(-1))
    )


def _gaps_from_seeds(
    log_probability: torch.Tensor,
    seed_log_probability: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return [batch, trials, targets] gaps, vectorized over all trials.

    Targets are [batch, targets, probes]; seeds are
    [batch, trials, selected_seeds, probes].  Empty cells contribute -inf
    to the bound, whereas a block with no valid probes gets zero gap.
    """
    seed_count = seed_log_probability.size(-2)
    owner = seed_log_probability.argmax(dim=-2)
    slots = torch.arange(seed_count, device=log_probability.device)
    cell = (owner[..., None, :] == slots[None, None, :, None])
    cell = cell & valid[:, None, None, :]
    active = cell.any(dim=-1)
    masked = seed_log_probability.masked_fill(~cell, -torch.inf)
    normalizer = torch.logsumexp(masked, dim=-1, keepdim=True)
    normalizer = torch.where(active[..., None], normalizer, 0.0)
    log_weight = torch.where(cell, masked - normalizer, 0.0)
    weight = torch.where(cell, log_weight.exp(), 0.0)
    entropy = -(weight * log_weight).sum(dim=-1)
    target = torch.where(valid[:, None, :], log_probability, 0.0)
    # No [batch, trials, targets, seeds, probes] intermediate is materialized.
    bound = torch.einsum("brcp,btp->brtc", weight, target)
    bound = bound + entropy[:, :, None, :]
    bound = bound.masked_fill(~active[:, :, None, :], -torch.inf)
    gap = -torch.logsumexp(bound, dim=-1)
    gap = torch.where(valid.any(-1)[:, None, None], gap, 0.0)
    return gap.clamp_min(0.0)


@torch.inference_mode()
def partition_variational_gap(
    logits: torch.Tensor,
    selected: torch.Tensor,
    *,
    probe_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one gap per target candidate, shape [..., candidates].

    ``selected`` contains candidate positions and broadcasts to
    ``logits.shape[:-2] + (selected_count,)``. At least one seed is required.
    ``probe_mask`` broadcasts to [..., probes]. Probes with nonfinite
    logits for any candidate are excluded from the shared support. A block
    with no remaining support has zero diagnostic loss by convention.
    Empty selected-seed cells are excluded, rather than assigned fake mass.
    """
    log_probability, valid, batch_shape = _prepare(logits, probe_mask)
    if selected.ndim < 1 or selected.size(-1) == 0:
        raise ValueError("selected must contain at least one seed")
    positions = torch.broadcast_to(
        selected.to(device=logits.device, dtype=torch.long),
        (*batch_shape, selected.size(-1)),
    ).reshape(log_probability.size(0), selected.size(-1))
    seed_log_probability = _gather(log_probability, positions)[:, None, :, :]
    gaps = _gaps_from_seeds(log_probability, seed_log_probability, valid)[:, 0, :]
    return gaps.reshape(*batch_shape, logits.size(-2))


@torch.inference_mode()
def select_partition_variational_seeds(
    logits: torch.Tensor,
    anchor: torch.Tensor,
    *,
    variant: str = "gap_farthest",
    weights: torch.Tensor | None = None,
    num_seeds: int = 4,
    probe_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select distinct seed positions, returning [..., num_seeds] int64.

    Args:
        logits: Probe QK logits [..., candidates, probes], normally [...,32,16].
        anchor: First candidate position; broadcasts to logits.shape[:-2].
        variant: ``gap_farthest`` chooses the largest current target gap;
            ``gap_weighted`` multiplies it by that target's importance.
            ``gap_gain`` tries every possible next seed and chooses the
            partition with the smallest mean target gap; ``gap_gain_weighted``
            uses the importance-weighted mean. ``gap_deficit_weighted`` uses
            the bounded partition-mass deficit ``1 - exp(-gap)`` in place of
            gap; ``gap_deficit_gain_weighted`` minimizes its weighted mean
            over all targets after each trial addition. The lookahead is
            batched over all candidates and recomputes ownership for every trial.
        weights: Existing nonnegative importance weights, broadcasting to
            [..., candidates]. Omission means uniform weighting. All-zero
            weights fall back to uniform weighting. No weight is fitted here.
        num_seeds: Includes the fixed anchor; cannot exceed candidate count.
        probe_mask: Optional shared valid-probe mask, as in the diagnostic API.

    Equal scores resolve to the smallest unused candidate position, including
    zero-logit and all-masked blocks. The anchor is always output first.
    Adding a seed repartitions all cells: this exact heuristic makes no claim
    that the global loss decreases at every addition or that C4 is optimal.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if not 1 <= num_seeds <= logits.size(-2):
        raise ValueError("num_seeds must be between one and candidate count")
    log_probability, valid, batch_shape = _prepare(logits, probe_mask)
    batch_size, candidate_count, probes = log_probability.shape
    first = torch.broadcast_to(
        anchor.to(device=logits.device, dtype=torch.long), batch_shape
    ).reshape(batch_size)
    used = torch.zeros(
        (batch_size, candidate_count), device=logits.device, dtype=torch.bool
    )
    used.scatter_(1, first[:, None], True)
    chosen = first[:, None]
    importance = torch.ones_like(log_probability[..., 0])
    if weights is not None and variant.endswith("weighted"):
        importance = torch.broadcast_to(
            weights.to(device=logits.device, dtype=log_probability.dtype),
            (*batch_shape, candidate_count),
        ).reshape(batch_size, candidate_count)
        if not bool(torch.isfinite(importance).all()) or bool((importance < 0).any()):
            raise ValueError("weights must be finite and nonnegative")
        importance = torch.where(
            importance.sum(-1, keepdim=True) > 0, importance, 1.0
        )
    importance = importance / importance.sum(-1, keepdim=True)
    lookahead = variant in (
        "gap_gain", "gap_gain_weighted", "gap_deficit_gain_weighted"
    )
    deficit = variant in ("gap_deficit_weighted", "gap_deficit_gain_weighted")
    for _ in range(1, num_seeds):
        selected_log_probability = _gather(log_probability, chosen)
        if lookahead:
            existing = selected_log_probability[:, None, :, :].expand(
                batch_size, candidate_count, chosen.size(-1), probes
            )
            trial_seeds = torch.cat((existing, log_probability[:, :, None, :]), dim=-2)
            gap = _gaps_from_seeds(log_probability, trial_seeds, valid)
            loss = -torch.expm1(-gap) if deficit else gap
            score = -(loss * importance[:, None, :]).sum(-1)
        else:
            gap = _gaps_from_seeds(
                log_probability, selected_log_probability[:, None, :, :], valid
            )[:, 0, :]
            loss = -torch.expm1(-gap) if deficit else gap
            score = loss * importance
        next_position = score.masked_fill(used, -torch.inf).argmax(-1)
        used.scatter_(1, next_position[:, None], True)
        chosen = torch.cat((chosen, next_position[:, None]), dim=-1)
    return chosen.reshape(*batch_shape, num_seeds)


def _selfcheck() -> None:
    """CPU checks of the variational identity, lookahead, and empty cells."""
    generator = torch.Generator(device="cpu").manual_seed(1024)
    logits = torch.randn((2, 3, 7, 11), generator=generator, dtype=torch.float64)
    anchor = torch.tensor([[0, 2, 5], [6, 4, 1]])
    logp = logits.log_softmax(-1)
    p_anchor = logp.gather(
        -2, anchor[..., None, None].expand(2, 3, 1, 11)
    ).squeeze(-2)
    kl = (p_anchor.exp()[..., None, :] * (p_anchor[..., None, :] - logp)).sum(-1)
    actual = partition_variational_gap(logits, anchor[..., None])
    torch.testing.assert_close(actual, kl.clamp_min(0), atol=1e-12, rtol=1e-12)
    shifted = logits + 100 * torch.randn((2, 3, 7, 1), generator=generator)
    torch.testing.assert_close(
        partition_variational_gap(shifted, anchor[..., None]), actual,
        atol=1e-12, rtol=1e-12,
    )
    weight = torch.rand((2, 3, 7), generator=generator, dtype=torch.float64)
    for variant in VARIANTS:
        selected = select_partition_variational_seeds(
            logits, anchor, variant=variant, weights=weight
        )
        assert torch.equal(selected[..., 0], anchor)
        assert bool((selected.sort(-1).values.diff(dim=-1) > 0).all())
        if variant in ("gap_gain", "gap_gain_weighted", "gap_deficit_gain_weighted"):
            uniform = torch.ones_like(weight)
            importance = weight if variant.endswith("weighted") else uniform
            for step in range(1, 4):
                scores = []
                for trial in range(7):
                    candidate = torch.full_like(anchor[..., None], trial)
                    prefix = torch.cat((selected[..., :step], candidate), dim=-1)
                    trial_gap = partition_variational_gap(logits, prefix)
                    trial_loss = (
                        -torch.expm1(-trial_gap) if "deficit" in variant else trial_gap
                    )
                    scores.append(-(trial_loss * importance).sum(-1))
                score = torch.stack(scores, dim=-1)
                score.scatter_(-1, selected[..., :step], -torch.inf)
                assert torch.equal(score.argmax(-1), selected[..., step])
        for empty_logits, mask in (
            (torch.zeros_like(logits), None),
            (logits, torch.zeros((2, 3, 11), dtype=torch.bool)),
            (torch.full_like(logits, -torch.inf), None),
        ):
            empty_selected = select_partition_variational_seeds(
                empty_logits, anchor, variant=variant, weights=torch.zeros_like(weight),
                probe_mask=mask,
            )
            assert torch.equal(empty_selected[..., 0], anchor)
            assert bool((empty_selected.sort(-1).values.diff(dim=-1) > 0).all())
            gap = partition_variational_gap(empty_logits, empty_selected, probe_mask=mask)
            torch.testing.assert_close(gap, torch.zeros_like(gap), atol=1e-12, rtol=0)
    # A single candidate with duplicate seeds yields one nonempty cell only.
    duplicated = anchor[..., None].expand(2, 3, 4)
    torch.testing.assert_close(partition_variational_gap(logits, duplicated), actual)
    scalar_selection = select_partition_variational_seeds(logits[0, 0], anchor[0, 0])
    assert scalar_selection.shape == (4,)
    print("partition_variational_selector: all CPU selfchecks passed")


if __name__ == "__main__":
    _selfcheck()
