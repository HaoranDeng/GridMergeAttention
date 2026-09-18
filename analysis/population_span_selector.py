"""Population reconstruction, rather than volume, for variational GCA seeds.

For residual candidate features ``X`` and nonnegative demand weights ``w``,
adding row ``s`` to their span decreases the population reconstruction loss

    sum_i w_i ||X_i||^2

by ``sum_i w_i <X_i, X_s>^2 / ||X_s||^2``.  This is an exact marginal
reduction of this *span surrogate*, not of GCA's hard-partition KL gap.
The fixed geometric anchor is retained.  No additional QK products are used.

Run this file on a machine with PyTorch for a small CPU-only projector check.
"""

from __future__ import annotations

import torch


def _population_features(
    logits: torch.Tensor,
    weights: torch.Tensor,
    geometry: str,
) -> torch.Tensor:
    probability = torch.softmax(logits, dim=-1)
    if geometry == "hellinger":
        root = probability.sqrt()
        return root - root.mean(dim=-1, keepdim=True)
    if geometry == "fisher":
        # Around a positive common distribution m, KL(p || q) is locally
        # one half of ||(p-q)/sqrt(m)||^2.  The weighted population mean
        # supplies m without a temperature or covariance regularizer.
        weight_sum = weights.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(weights, 1.0 / weights.size(-1))
        normalized = torch.where(
            weight_sum > 0.0,
            weights / weight_sum.clamp_min(torch.finfo(logits.dtype).tiny),
            uniform,
        )
        center = (probability * normalized.unsqueeze(-1)).sum(
            dim=-2, keepdim=True
        )
        return (probability - center) / center.clamp_min(
            torch.finfo(logits.dtype).tiny
        ).sqrt()
    raise ValueError("geometry must be 'hellinger' or 'fisher'")


@torch.no_grad()
def select_population_span(
    logits: torch.Tensor,
    anchor: torch.Tensor | int,
    weights: torch.Tensor | None = None,
    num_seeds: int = 4,
    *,
    geometry: str = "hellinger",
) -> torch.Tensor:
    """Select anchored seeds by maximum population span-loss reduction.

    Args:
        logits: Finite natural-log probe scores, shape ``[..., R, P]``.
        anchor: Candidate *slot* (not absolute query row), broadcastable to
            ``logits.shape[:-2]`` and in ``[0, R)``.
        weights: Optional finite nonnegative demand weights, broadcastable
            to ``logits.shape[:-1]``.  Defaults to equal query demand.  These
            weight reconstruction targets, not candidate novelty.
        num_seeds: Number of distinct candidate slots, including the anchor.
        geometry: Centered square-root probabilities by default; ``fisher``
            uses probability deviations whitened by their population mean.

    Returns:
        Long candidate slots of shape ``[..., num_seeds]``, anchor first.
        Exact ties use the smallest unselected candidate slot.

    The Gram update is equivalent to orthogonally removing each chosen
    residual feature.  Only the initial R-by-R Gram product and three
    rank-one updates are needed for C4.  Float64 inputs retain their dtype
    for verification; low-precision inference inputs accumulate in float32.
    """
    if logits.ndim < 2:
        raise ValueError("logits must have candidate and probe dimensions")
    candidate_count, probe_count = logits.shape[-2:]
    if probe_count < 1 or not 1 <= num_seeds <= candidate_count:
        raise ValueError("require P >= 1 and 1 <= num_seeds <= R")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    scores = logits.to(dtype=dtype)
    prefix = scores.shape[:-2]
    if weights is None:
        demand = torch.ones_like(scores[..., 0])
    else:
        demand = torch.broadcast_to(
            torch.as_tensor(weights, device=scores.device, dtype=dtype),
            scores.shape[:-1],
        )
    anchor_slot = torch.broadcast_to(
        torch.as_tensor(anchor, device=scores.device, dtype=torch.long), prefix
    )
    features = _population_features(scores, demand, geometry)
    gram = features @ features.transpose(-1, -2)
    used = torch.zeros(scores.shape[:-1], dtype=torch.bool, device=scores.device)
    selected = [anchor_slot]
    position = anchor_slot
    denominator_floor = 1.0e-12

    for _ in range(1, num_seeds):
        used.scatter_(-1, position.unsqueeze(-1), True)
        column = gram.gather(
            -1,
            position[..., None, None].expand(*prefix, candidate_count, 1),
        ).squeeze(-1)
        pivot = column.gather(-1, position.unsqueeze(-1)).squeeze(-1)
        gram = gram - (
            column.unsqueeze(-1) * column.unsqueeze(-2)
            / pivot.clamp_min(denominator_floor)[..., None, None]
        )
        # Roundoff can make a theoretically zero residual norm negative.
        diagonal = gram.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
        gain = (gram.square() * demand.unsqueeze(-1)).sum(dim=-2)
        gain = gain / diagonal.clamp_min(denominator_floor)
        gain = torch.where(diagonal > denominator_floor, gain, 0.0)
        position = gain.masked_fill(used, -float("inf")).argmax(dim=-1)
        selected.append(position)
    return torch.stack(selected, dim=-1)


def _projector_loss(
    features: torch.Tensor,
    weights: torch.Tensor,
    selected: list[int],
) -> torch.Tensor:
    basis = features[selected]
    projection = torch.linalg.pinv(basis) @ basis
    residual = features - features @ projection
    return (weights * residual.square().sum(dim=-1)).sum()


def _self_check() -> None:
    """Check all greedy insertions against independently formed projectors."""
    generator = torch.Generator(device="cpu").manual_seed(731)
    logits = torch.randn(2, 9, 7, generator=generator, dtype=torch.float64)
    weights = torch.rand(2, 9, generator=generator, dtype=torch.float64)
    anchors = torch.tensor([2, 5])
    for geometry in ("hellinger", "fisher"):
        for weighted in (False, True):
            demand = weights if weighted else torch.ones_like(weights)
            chosen = select_population_span(
                logits, anchors, demand, geometry=geometry
            )
            features = _population_features(logits, demand, geometry)
            for batch in range(logits.size(0)):
                expected = [int(anchors[batch])]
                for _ in range(1, 4):
                    before = _projector_loss(
                        features[batch], demand[batch], expected
                    )
                    basis = features[batch, expected]
                    projector = torch.linalg.pinv(basis) @ basis
                    residual = features[batch] - features[batch] @ projector
                    gram = residual @ residual.T
                    analytic = (gram.square() * demand[batch, :, None]).sum(0)
                    analytic /= gram.diagonal().clamp_min(1.0e-12)
                    gains = torch.full((9,), -float("inf"), dtype=torch.float64)
                    for candidate in range(9):
                        if candidate in expected:
                            continue
                        gains[candidate] = before - _projector_loss(
                            features[batch], demand[batch], expected + [candidate]
                        )
                        torch.testing.assert_close(
                            gains[candidate], analytic[candidate],
                            rtol=1.0e-8, atol=1.0e-10,
                        )
                    expected.append(int(gains.argmax()))
                assert chosen[batch].tolist() == expected
                assert len(set(expected)) == 4
    zero = select_population_span(torch.zeros(3, 6, 4), torch.tensor([0, 2, 5]))
    assert zero.tolist() == [[0, 1, 2, 3], [2, 0, 1, 3], [5, 0, 1, 2]]
    single = select_population_span(logits[0], 2, num_seeds=1)
    assert single.tolist() == [2]
    print("population span: CPU projector and degenerate-case checks passed")


if __name__ == "__main__":
    _self_check()
