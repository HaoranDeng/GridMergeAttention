"""Screen simpler alternatives to greedy weighted-span seed selection.

The input tensors come from ``oracle_wan14b_720p_hard_c4.py --tensor_output``:
one real Wan2.1-14B attention slice with all 80 query columns, all 45 query
rows, and all 21x45 key-row blocks.  Every selector uses the same production
probe construction, importance score, and unchanged hard-C4 merge.  Only the
choice of the three rows after the index-matched anchor changes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch

from attention.adaptive_seed_pytorch import multiscale_candidate_table
from oracle_wan14b_720p_hard_c4 import compressed_attention, error_metrics


C4 = 4
ANCHOR_OFFSETS = (-2, 0, 1, 3)
ANCHOR_OFFSET_SETS = {
    1: (0,),
    2: (-1, 1),
    3: (-2, 0, 2),
    4: ANCHOR_OFFSETS,
}
NEG_INF = torch.finfo(torch.float32).min


def _gather_rows(source: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Gather ``source[J, R, D]`` with ``rows[J, B, C]``."""
    columns = torch.arange(source.size(0), device=source.device)[:, None, None]
    return source[columns, rows]


def _gather_candidates(source: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    return source.gather(
        -2,
        position[..., None, None].expand(*position.shape, 1, source.size(-1)),
    ).squeeze(-2)


def _project_out(residual: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    basis = _gather_candidates(residual, position)
    basis = basis * torch.rsqrt(basis.square().sum(-1, keepdim=True) + 1e-12)
    projection = (residual * basis.unsqueeze(-2)).sum(-1, keepdim=True)
    return residual - projection * basis.unsqueeze(-2)


def _candidate_windows(
    key_rows: torch.Tensor,
    query_rows: int,
    candidate_count: int,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layout == "multiscale":
        rows, anchors = multiscale_candidate_table(
            query_rows,
            candidate_count,
            key_rows.device,
            geometry_count=query_rows,
        )
        return rows[key_rows].long(), anchors[key_rows].long()

    if layout == "local":
        start = (key_rows - candidate_count // 2).clamp(
            0, query_rows - candidate_count
        )
        rows = start[:, None] + torch.arange(candidate_count, device=key_rows.device)
        return rows, key_rows - start

    if layout not in {"uniform", "local_uniform"}:
        raise ValueError(f"unknown candidate layout {layout!r}")

    def fill_farthest(chosen: set[int]) -> None:
        while len(chosen) < candidate_count:
            remaining = [row for row in range(query_rows) if row not in chosen]
            row = max(
                remaining,
                key=lambda item: (min(abs(item - seed) for seed in chosen), -item),
            )
            chosen.add(row)

    table = []
    anchors = []
    for key_row in range(query_rows):
        if layout == "uniform":
            count = candidate_count
            chosen = {
                round(index * (query_rows - 1) / (count - 1))
                for index in range(count)
            }
            if key_row not in chosen:
                replaced = min(chosen, key=lambda item: (abs(item - key_row), item))
                chosen.remove(replaced)
                chosen.add(key_row)
        else:
            chosen = {key_row}
            radius = max(1, candidate_count // 4)
            chosen.update(
                range(max(0, key_row - radius), min(query_rows, key_row + radius + 1))
            )
            count = max(4, candidate_count // 4)
        if layout == "local_uniform" and count > 1:
            chosen.update(
                round(index * (query_rows - 1) / (count - 1))
                for index in range(count)
            )
        fill_farthest(chosen)
        selected = sorted(chosen)[:candidate_count]
        table.append(selected)
        anchors.append(selected.index(key_row))

    candidate_table = torch.tensor(table, device=key_rows.device, dtype=torch.long)
    anchor_table = torch.tensor(anchors, device=key_rows.device, dtype=torch.long)
    return candidate_table[key_rows], anchor_table[key_rows]


def _probe_indices(
    query: torch.Tensor,
    key: torch.Tensor,
    key_rows: torch.Tensor,
    scale: float,
    block_j: int,
    top_probes: int,
    uniform_probes: int,
    probe_anchor_count: int,
) -> torch.Tensor:
    """Production-style probes, shared by ``block_j`` adjacent columns."""
    if top_probes < 0 or uniform_probes < 0 or top_probes + uniform_probes == 0:
        raise ValueError("probe counts must be non-negative and sum to at least one")
    groups = math.ceil(query.size(0) / block_j)
    parts = []
    uniform = torch.empty(0, device=query.device, dtype=torch.long)
    if uniform_probes:
        stride = key.size(1) // uniform_probes
        uniform = torch.arange(uniform_probes, device=query.device) * stride

    if top_probes:
        owner_columns = torch.arange(groups, device=query.device) * block_j
        if probe_anchor_count not in ANCHOR_OFFSET_SETS:
            raise ValueError("probe anchor count must be between one and four")
        offsets = torch.tensor(
            ANCHOR_OFFSET_SETS[probe_anchor_count], device=query.device
        )
        anchor_rows = (key_rows[:, None] + offsets).clamp(0, query.size(1) - 1)
        anchors = query[owner_columns[:, None, None], anchor_rows[None, :, :]]
        response = torch.einsum("gbad,bld->gbal", anchors.float(), key.float()) * scale
        importance = torch.log_softmax(response, dim=-1).amax(dim=-2)
        eligible = torch.ones(key.size(1), device=query.device, dtype=torch.bool)
        eligible[uniform] = False
        parts.append(
            importance.masked_fill(~eligible, NEG_INF).topk(top_probes, dim=-1).indices
        )

    if uniform_probes:
        parts.append(uniform.view(1, 1, -1).expand(groups, key.size(0), -1))
    return torch.cat(parts, dim=-1)


def _selector_features(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_rows: torch.Tensor,
    *,
    alpha: float,
    candidate_count: int,
    block_j: int,
    top_probes: int,
    uniform_probes: int,
    probe_anchor_count: int,
    value_dims: int,
    attention_dim: int,
    candidate_layout: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return profile geometries, weights, candidate rows, and anchor."""
    query_rows, query_columns, selector_dim = (
        query.size(1),
        query.size(0),
        query.size(2),
    )
    scale = attention_dim**-0.5
    candidate_rows, anchor_b = _candidate_windows(
        key_rows, query_rows, candidate_count, candidate_layout
    )
    candidates = query[:, candidate_rows].float()

    probe_g = _probe_indices(
        query,
        key,
        key_rows,
        scale,
        block_j,
        top_probes=top_probes,
        uniform_probes=uniform_probes,
        probe_anchor_count=probe_anchor_count,
    )
    group_for_column = torch.arange(query_columns, device=query.device) // block_j
    probe = probe_g[group_for_column]
    expanded_key = key.float().unsqueeze(0).expand(probe.size(0), -1, -1, -1)
    key_probe = expanded_key.gather(
        2, probe[..., None].expand(*probe.shape, selector_dim)
    )
    logits = torch.einsum("jbrd,jbpd->jbrp", candidates, key_probe) * scale
    probability = torch.softmax(logits, dim=-1)

    log_mass = torch.logsumexp(logits, dim=-1)
    z = log_mass - log_mass.mean(-1, keepdim=True)
    z = z * torch.rsqrt(z.var(-1, correction=0, keepdim=True) + 1e-6)
    weight = torch.exp(2.0 * alpha * z)

    value_dim = value.size(-1)
    pooled_value = value.float().reshape(
        value.size(0), value.size(1), value_dims, value_dim // value_dims
    ).mean(-1)
    expanded_value = pooled_value.unsqueeze(0).expand(probe.size(0), -1, -1, -1)
    value_probe = expanded_value.gather(
        2, probe[..., None].expand(*probe.shape, value_dims)
    )
    value_profile = torch.einsum("jbrp,jbpd->jbrd", probability, value_probe)
    value_profile = value_profile - value_profile.mean(-1, keepdim=True)
    probability_profile = probability - probability.mean(-1, keepdim=True)
    hellinger_profile = probability.sqrt()
    hellinger_profile = hellinger_profile - hellinger_profile.mean(-1, keepdim=True)
    log_probability_profile = torch.log_softmax(logits, dim=-1)
    log_probability_profile = (
        log_probability_profile - log_probability_profile.mean(-1, keepdim=True)
    )
    anchor = anchor_b.unsqueeze(0).expand(query_columns, -1)
    return (
        value_profile,
        probability_profile,
        hellinger_profile,
        log_probability_profile,
        logits,
        weight,
        candidate_rows,
        anchor,
    )


def _greedy_span(profile: torch.Tensor, weight: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    residual = _project_out(profile, anchor)
    used = torch.zeros_like(weight, dtype=torch.bool)
    used.scatter_(-1, anchor[..., None], True)
    chosen = [anchor]
    for _ in range(C4 - 1):
        score = residual.square().sum(-1) * weight
        position = score.masked_fill(used, NEG_INF).argmax(-1)
        chosen.append(position)
        used.scatter_(-1, position[..., None], True)
        residual = _project_out(residual, position)
    return torch.stack(chosen, dim=-1)


def _pivoted_cholesky_span(
    profile: torch.Tensor,
    weight: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    """Stable Gram-matrix equivalent of weighted pivoted Gram-Schmidt.

    In exact arithmetic this chooses the same pivots as ``_greedy_span``.  It
    updates only the Schur-complement diagonal and clips roundoff below zero,
    avoiding repeated mutation of the feature vectors themselves.
    """
    gram = torch.einsum("jbrd,jbsd->jbrs", profile.float(), profile.float())
    diagonal = gram.diagonal(dim1=-2, dim2=-1).clone()
    used = torch.zeros_like(weight, dtype=torch.bool)
    columns: list[torch.Tensor] = []
    chosen: list[torch.Tensor] = []
    candidate_count = profile.size(-2)

    for step in range(C4):
        if step == 0:
            pivot = anchor
        else:
            score = (diagonal * weight).masked_fill(used, NEG_INF)
            pivot = score.argmax(-1)
        chosen.append(pivot)
        used.scatter_(-1, pivot[..., None], True)

        gram_pivot = gram.gather(
            -1,
            pivot[..., None, None].expand(*pivot.shape, candidate_count, 1),
        ).squeeze(-1)
        correction = torch.zeros_like(gram_pivot)
        for column in columns:
            pivot_value = column.gather(-1, pivot[..., None]).squeeze(-1)
            correction = correction + column * pivot_value[..., None]
        pivot_diagonal = diagonal.gather(-1, pivot[..., None]).squeeze(-1)
        column = (gram_pivot - correction) * torch.rsqrt(
            pivot_diagonal.clamp_min(1e-12)
        )[..., None]
        columns.append(column)
        diagonal = (diagonal - column.square()).clamp_min(0.0)

    return torch.stack(chosen, dim=-1)


def _coarse_to_fine_span(
    profile: torch.Tensor,
    weight: torch.Tensor,
    anchor: torch.Tensor,
    coarse_profile: torch.Tensor,
    coarse_weight: torch.Tensor,
    *,
    shortlist_size: int,
) -> torch.Tensor:
    """Shortlist globally with cheap probes, then run full-profile greedy span."""
    coarse_residual = _project_out(coarse_profile, anchor)
    coarse_score = coarse_residual.square().sum(-1) * coarse_weight
    coarse_score.scatter_(-1, anchor[..., None], NEG_INF)
    shortlist = torch.cat(
        (
            anchor[..., None],
            coarse_score.topk(shortlist_size - 1, dim=-1).indices,
        ),
        dim=-1,
    )
    fine_profile = profile.gather(
        -2, shortlist[..., None].expand(*shortlist.shape, profile.size(-1))
    )
    fine_weight = weight.gather(-1, shortlist)
    fine_anchor = torch.zeros_like(anchor)
    selected = _greedy_span(fine_profile, fine_weight, fine_anchor)
    return shortlist.gather(-1, selected)


def _coarse_hellinger(
    logits: torch.Tensor,
    *,
    alpha: float,
    uniform_probes: int,
    coarse_probes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coarse_probes > uniform_probes:
        raise ValueError("coarse probes must come from the uniform probe set")
    uniform_logits = logits[..., -uniform_probes:]
    slots = torch.linspace(
        0,
        uniform_probes - 1,
        coarse_probes,
        device=logits.device,
    ).round().long()
    coarse_logits = uniform_logits[..., slots]
    probability = torch.softmax(coarse_logits, dim=-1)
    profile = probability.sqrt()
    profile = profile - profile.mean(-1, keepdim=True)
    log_mass = torch.logsumexp(coarse_logits, dim=-1)
    z = log_mass - log_mass.mean(-1, keepdim=True)
    z = z * torch.rsqrt(z.var(-1, correction=0, keepdim=True) + 1e-6)
    return profile, torch.exp(2.0 * alpha * z)


def _anchor_topk(profile: torch.Tensor, weight: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    residual = _project_out(profile, anchor)
    score = residual.square().sum(-1) * weight
    score.scatter_(-1, anchor[..., None], NEG_INF)
    return torch.cat((anchor[..., None], score.topk(C4 - 1, dim=-1).indices), dim=-1)


def _two_stage_span(
    profile: torch.Tensor, weight: torch.Tensor, anchor: torch.Tensor
) -> torch.Tensor:
    """Choose one pivot after the anchor, then choose the last two in parallel."""
    residual = _project_out(profile, anchor)
    used = torch.zeros_like(weight, dtype=torch.bool)
    used.scatter_(-1, anchor[..., None], True)
    score = residual.square().sum(-1) * weight
    second = score.masked_fill(used, NEG_INF).argmax(-1)
    used.scatter_(-1, second[..., None], True)
    residual = _project_out(residual, second)
    score = (residual.square().sum(-1) * weight).masked_fill(used, NEG_INF)
    tail = score.topk(C4 - 2, dim=-1).indices
    return torch.cat((anchor[..., None], second[..., None], tail), dim=-1)


def _stratified_top1(
    profile: torch.Tensor, weight: torch.Tensor, anchor: torch.Tensor
) -> torch.Tensor:
    residual = _project_out(profile, anchor)
    score = residual.square().sum(-1) * weight
    score.scatter_(-1, anchor[..., None], NEG_INF)
    boundaries = torch.linspace(
        0, profile.size(-2), C4, device=profile.device, dtype=torch.int64
    )
    picks = []
    for begin, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
        picks.append(score[..., begin:end].argmax(-1) + begin)
    return torch.stack((anchor, *picks), dim=-1)


def _global_redundancy_topk(
    profile: torch.Tensor,
    weight: torch.Tensor,
    anchor: torch.Tensor,
    *,
    beta: float,
    penalty: float,
) -> torch.Tensor:
    residual = _project_out(profile, anchor)
    norm2 = residual.square().sum(-1)
    unit = residual * torch.rsqrt(norm2[..., None] + 1e-12)
    similarity2 = torch.einsum("jbrd,jbsd->jbrs", unit, unit).square()
    redundancy = torch.logsumexp(beta * similarity2, dim=-1) / beta
    score = torch.log(norm2.clamp_min(1e-12)) + torch.log(weight) - penalty * redundancy
    score.scatter_(-1, anchor[..., None], NEG_INF)
    return torch.cat((anchor[..., None], score.topk(C4 - 1, dim=-1).indices), dim=-1)


def _joint_volume(
    profile: torch.Tensor,
    weight: torch.Tensor,
    anchor: torch.Tensor,
    *,
    shortlist_size: int,
) -> torch.Tensor:
    """Jointly choose three rows by weighted volume from a short parallel list."""
    residual = _project_out(profile, anchor)
    first_score = residual.square().sum(-1) * weight
    first_score.scatter_(-1, anchor[..., None], NEG_INF)
    shortlist = first_score.topk(shortlist_size, dim=-1).indices
    vectors = residual.gather(
        -2, shortlist[..., None].expand(*shortlist.shape, residual.size(-1))
    )
    gram = torch.einsum("jbmd,jbnd->jbmn", vectors, vectors)
    quality = torch.log(weight.gather(-1, shortlist).clamp_min(1e-12))

    combinations = torch.tensor(
        list(itertools.combinations(range(shortlist_size), C4 - 1)),
        device=profile.device,
        dtype=torch.long,
    )
    a, b, c = combinations.unbind(-1)
    g11, g22, g33 = gram[..., a, a], gram[..., b, b], gram[..., c, c]
    g12, g13, g23 = gram[..., a, b], gram[..., a, c], gram[..., b, c]
    determinant = (
        g11 * g22 * g33
        + 2.0 * g12 * g13 * g23
        - g11 * g23.square()
        - g22 * g13.square()
        - g33 * g12.square()
    )
    objective = torch.log(determinant.clamp_min(1e-20))
    objective = objective + quality[..., a] + quality[..., b] + quality[..., c]
    best = objective.argmax(-1)
    selected_slots = combinations[best]
    selected = shortlist.gather(-1, selected_slots)
    return torch.cat((anchor[..., None], selected), dim=-1)


def _pair_first_volume(
    profile: torch.Tensor,
    weight: torch.Tensor,
    anchor: torch.Tensor,
    *,
    shortlist_size: int | None = None,
) -> torch.Tensor:
    """Choose the best anchored pair jointly, then add the best third row.

    The method uses Gram determinants only: there is no iterative vector
    orthogonalization.  Both the all-pairs pass and the final-candidate pass
    are parallel over candidates.
    """
    residual = _project_out(profile.float(), anchor)
    candidate_count = profile.size(-2)
    shortlist_size = shortlist_size or (candidate_count - 1)
    if not 2 <= shortlist_size < candidate_count:
        raise ValueError("pair-first shortlist must be in [2, candidate_count)")
    ids = torch.arange(candidate_count, device=profile.device)
    diagonal = residual.square().sum(-1).clamp_min(1e-12)
    valid = ids[None, None, :] != anchor[..., None]
    anchor_score = (diagonal * weight).masked_fill(~valid, NEG_INF)
    shortlist = anchor_score.topk(shortlist_size, dim=-1).indices
    vectors = residual.gather(
        -2,
        shortlist[..., None].expand(*shortlist.shape, residual.size(-1)),
    )
    shortlist_weight = weight.gather(-1, shortlist)
    gram = torch.einsum("jbmd,jbnd->jbmn", vectors, vectors)
    shortlist_diagonal = gram.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
    shortlist_ids = torch.arange(shortlist_size, device=profile.device)

    scale = shortlist_diagonal[..., :, None] * shortlist_diagonal[..., None, :]
    correlation2 = gram.square() / scale.clamp_min(1e-20)
    log_area = torch.log(scale) + torch.log1p(
        -correlation2.clamp(max=1.0 - 1e-7)
    )
    log_weight = torch.log(weight.clamp_min(1e-12))
    shortlist_log_weight = torch.log(shortlist_weight.clamp_min(1e-12))
    pair_score = (
        log_area
        + shortlist_log_weight[..., :, None]
        + shortlist_log_weight[..., None, :]
    )
    upper = shortlist_ids[:, None] < shortlist_ids[None, :]
    pair_score = pair_score.masked_fill(~upper, NEG_INF)
    pair = pair_score.flatten(-2).argmax(-1)
    first_slot = pair // shortlist_size
    second_slot = pair % shortlist_size
    first = shortlist.gather(-1, first_slot[..., None]).squeeze(-1)
    second = shortlist.gather(-1, second_slot[..., None]).squeeze(-1)

    def gather_vector(slot: torch.Tensor) -> torch.Tensor:
        return vectors.gather(
            -2,
            slot[..., None, None].expand(*slot.shape, 1, vectors.size(-1)),
        ).squeeze(-2)

    first_vector = gather_vector(first_slot)
    second_vector = gather_vector(second_slot)
    g11 = first_vector.square().sum(-1).clamp_min(1e-12)
    g22 = second_vector.square().sum(-1).clamp_min(1e-12)
    g12 = (first_vector * second_vector).sum(-1)
    g13 = (residual * first_vector[..., None, :]).sum(-1)
    g23 = (residual * second_vector[..., None, :]).sum(-1)
    determinant = (
        g11[..., None] * g22[..., None] * diagonal
        + 2.0 * g12[..., None] * g13 * g23
        - g11[..., None] * g23.square()
        - g22[..., None] * g13.square()
        - diagonal * g12[..., None].square()
    )
    third_score = torch.log(determinant.clamp_min(1e-20)) + log_weight
    used = (
        (ids[None, None, :] == anchor[..., None])
        | (ids[None, None, :] == first[..., None])
        | (ids[None, None, :] == second[..., None])
    )
    third = third_score.masked_fill(used, NEG_INF).argmax(-1)
    return torch.stack((anchor, first, second, third), dim=-1)


def _pivot_pair_volume(
    profile: torch.Tensor,
    weight: torch.Tensor,
    anchor: torch.Tensor,
    *,
    shortlist_size: int,
) -> torch.Tensor:
    """Choose one pivot, then jointly choose the final pair by spanned area."""
    residual = _project_out(profile, anchor)
    used = torch.zeros_like(weight, dtype=torch.bool)
    used.scatter_(-1, anchor[..., None], True)

    score = residual.square().sum(-1) * weight
    pivot = score.masked_fill(used, NEG_INF).argmax(-1)
    used.scatter_(-1, pivot[..., None], True)
    residual = _project_out(residual, pivot)

    score = (residual.square().sum(-1) * weight).masked_fill(used, NEG_INF)
    shortlist = score.topk(shortlist_size, dim=-1).indices
    vectors = residual.gather(
        -2, shortlist[..., None].expand(*shortlist.shape, residual.size(-1))
    )
    gram = torch.einsum("jbmd,jbnd->jbmn", vectors, vectors)
    quality = torch.log(weight.gather(-1, shortlist).clamp_min(1e-12))

    combinations = torch.tensor(
        list(itertools.combinations(range(shortlist_size), C4 - 2)),
        device=profile.device,
        dtype=torch.long,
    )
    a, b = combinations.unbind(-1)
    determinant = gram[..., a, a] * gram[..., b, b] - gram[..., a, b].square()
    objective = (
        torch.log(determinant.clamp_min(1e-20)) + quality[..., a] + quality[..., b]
    )
    selected_slots = combinations[objective.argmax(-1)]
    selected = shortlist.gather(-1, selected_slots)
    return torch.cat((anchor[..., None], pivot[..., None], selected), dim=-1)


@torch.inference_mode()
def select_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    alpha: float,
    candidate_count: int,
    block_j: int,
    block_chunk: int,
    top_probes: int = 8,
    uniform_probes: int = 8,
    probe_anchor_count: int = 4,
    sketch_dim: int = 0,
    sketch_seed: int = 20260825,
    candidate_layout: str = "local",
    variant_names: set[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    attention_dim = query.size(-1)
    if sketch_dim and sketch_dim < attention_dim:
        generator = torch.Generator(device=query.device).manual_seed(sketch_seed)
        projection = torch.randn(
            attention_dim,
            sketch_dim,
            generator=generator,
            device=query.device,
            dtype=torch.float32,
        ) * (sketch_dim**-0.5)
        selector_query = torch.matmul(query.float(), projection)
        selector_key = torch.matmul(key.float(), projection)
    else:
        selector_query = query
        selector_key = key

    positions: dict[str, list[torch.Tensor]] = {}
    elapsed: dict[str, float] = {}
    total_blocks = key.size(0)
    for begin in range(0, total_blocks, block_chunk):
        end = min(begin + block_chunk, total_blocks)
        block_rows = torch.arange(begin, end, device=query.device) % query.size(1)
        (
            value_profile,
            probability_profile,
            hellinger_profile,
            log_probability_profile,
            logits,
            weight,
            candidate_rows,
            anchor,
        ) = _selector_features(
            selector_query,
            selector_key[begin:end],
            value[begin:end],
            block_rows,
            alpha=alpha,
            candidate_count=candidate_count,
            block_j=block_j,
            top_probes=top_probes,
            uniform_probes=uniform_probes,
            probe_anchor_count=probe_anchor_count,
            value_dims=32,
            attention_dim=attention_dim,
            candidate_layout=candidate_layout,
        )
        variants = {
            "weighted_gs": lambda: _greedy_span(value_profile, weight, anchor),
            "probability_gs": lambda: _greedy_span(probability_profile, weight, anchor),
            "hellinger_gs": lambda: _greedy_span(hellinger_profile, weight, anchor),
            "hellinger_pivoted_cholesky": lambda: _pivoted_cholesky_span(
                hellinger_profile, weight, anchor
            ),
            "log_probability_gs": lambda: _greedy_span(
                log_probability_profile, weight, anchor
            ),
            "value_two_stage": lambda: _two_stage_span(value_profile, weight, anchor),
            "probability_two_stage": lambda: _two_stage_span(
                probability_profile, weight, anchor
            ),
            "hellinger_two_stage": lambda: _two_stage_span(
                hellinger_profile, weight, anchor
            ),
            "hellinger_anchor_top3": lambda: _anchor_topk(
                hellinger_profile, weight, anchor
            ),
            "hellinger_stratified_top1": lambda: _stratified_top1(
                hellinger_profile, weight, anchor
            ),
            "hellinger_pivot_pair_m6": lambda: _pivot_pair_volume(
                hellinger_profile, weight, anchor, shortlist_size=6
            ),
            "hellinger_pivot_pair_m8": lambda: _pivot_pair_volume(
                hellinger_profile, weight, anchor, shortlist_size=8
            ),
            "hellinger_pair_first": lambda: _pair_first_volume(
                hellinger_profile, weight, anchor
            ),
            "hellinger_pair_first_m4": lambda: _pair_first_volume(
                hellinger_profile, weight, anchor, shortlist_size=4
            ),
            "hellinger_pair_first_m8": lambda: _pair_first_volume(
                hellinger_profile, weight, anchor, shortlist_size=8
            ),
            "hellinger_pair_first_m12": lambda: _pair_first_volume(
                hellinger_profile, weight, anchor, shortlist_size=12
            ),
            "hellinger_pair_first_m16": lambda: _pair_first_volume(
                hellinger_profile, weight, anchor, shortlist_size=16
            ),
            "anchor_top3": lambda: _anchor_topk(value_profile, weight, anchor),
            "probability_anchor_top3": lambda: _anchor_topk(
                probability_profile, weight, anchor
            ),
            "stratified_top1": lambda: _stratified_top1(value_profile, weight, anchor),
            "redundancy_b4_l025": lambda: _global_redundancy_topk(
                value_profile, weight, anchor, beta=4.0, penalty=0.25
            ),
            "redundancy_b4_l050": lambda: _global_redundancy_topk(
                value_profile, weight, anchor, beta=4.0, penalty=0.5
            ),
            "redundancy_b8_l050": lambda: _global_redundancy_topk(
                value_profile, weight, anchor, beta=8.0, penalty=0.5
            ),
            "joint_volume_m6": lambda: _joint_volume(
                value_profile, weight, anchor, shortlist_size=6
            ),
            "joint_volume_m8": lambda: _joint_volume(
                value_profile, weight, anchor, shortlist_size=8
            ),
            "hellinger_joint_volume_m6": lambda: _joint_volume(
                hellinger_profile, weight, anchor, shortlist_size=6
            ),
            "hellinger_joint_volume_m8": lambda: _joint_volume(
                hellinger_profile, weight, anchor, shortlist_size=8
            ),
            "hellinger_joint_volume_m10": lambda: _joint_volume(
                hellinger_profile, weight, anchor, shortlist_size=10
            ),
        }
        for shortlist_size in (12, 16, candidate_count - 1):
            if shortlist_size <= C4 - 1 or shortlist_size >= candidate_count:
                continue
            name = (
                "hellinger_joint_volume_full"
                if shortlist_size == candidate_count - 1
                else f"hellinger_joint_volume_m{shortlist_size}"
            )
            variants[name] = (
                lambda size=shortlist_size: _joint_volume(
                    hellinger_profile,
                    weight,
                    anchor,
                    shortlist_size=size,
                )
            )
        # Joint/pivot shortlists exclude the fixed anchor, so they must be
        # strictly smaller than the candidate bank. Small-tile screens (C8)
        # intentionally exercise this boundary.
        for shortlist_size in (6, 8, 10):
            if shortlist_size >= candidate_count:
                variants.pop(f"joint_volume_m{shortlist_size}", None)
                variants.pop(f"hellinger_joint_volume_m{shortlist_size}", None)
                variants.pop(f"hellinger_pivot_pair_m{shortlist_size}", None)
        if uniform_probes >= 4 and candidate_count >= 29:
            for coarse_probes, shortlist_size in ((2, 29), (3, 28), (4, 25), (4, 26)):
                coarse_profile, coarse_weight = _coarse_hellinger(
                    logits,
                    alpha=alpha,
                    uniform_probes=uniform_probes,
                    coarse_probes=coarse_probes,
                )
                name = f"hellinger_coarse{coarse_probes}_short{shortlist_size}"
                variants[name] = (
                    lambda cp=coarse_profile, cw=coarse_weight, size=shortlist_size: (
                        _coarse_to_fine_span(
                            hellinger_profile,
                            weight,
                            anchor,
                            cp,
                            cw,
                            shortlist_size=size,
                        )
                    )
                )
        if variant_names is not None:
            missing = variant_names.difference(variants)
            if missing:
                raise ValueError(f"unknown selector variants: {sorted(missing)}")
            variants = {name: variants[name] for name in variant_names}
        for name, function in variants.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            chosen = function()
            torch.cuda.synchronize()
            elapsed[name] = elapsed.get(name, 0.0) + time.perf_counter() - start
            rows = candidate_rows.unsqueeze(0).expand(query.size(0), -1, -1)
            positions.setdefault(name, []).append(rows.gather(-1, chosen).cpu())
    return (
        {name: torch.cat(chunks, dim=1).to(query.device) for name, chunks in positions.items()},
        elapsed,
    )


@torch.inference_mode()
def evaluate_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dense: torch.Tensor,
    rows: torch.Tensor,
) -> dict[str, float]:
    seeds = _gather_rows(query, rows).float()
    logits = torch.einsum("jbcd,bld->jbcl", seeds, key.float()) * (query.size(-1) ** -0.5)
    assignment_score = torch.log_softmax(logits, dim=-1)
    assignment = torch.nn.functional.one_hot(
        assignment_score.argmax(dim=-2), num_classes=C4
    ).permute(0, 1, 3, 2).bool()
    selected = logits.masked_fill(~assignment, float("-inf"))
    log_weight = torch.log_softmax(selected, dim=-1)
    log_weight = torch.where(assignment, log_weight, 0.0)
    weight = torch.where(assignment, log_weight.exp(), 0.0)
    active = assignment.any(-1)
    output = compressed_attention(query.float(), key.float(), value.float(), weight, active)
    metrics = error_metrics(output, dense.float())
    difference = output.float() - dense.float()
    metrics["squared_error"] = float(difference.square().sum())
    metrics["reference_squared"] = float(dense.float().square().sum())
    metrics["numel"] = difference.numel()
    return metrics


def _fixed_rows(query: torch.Tensor, blocks: int) -> torch.Tensor:
    key_row = torch.arange(blocks, device=query.device) % query.size(1)
    start = (key_row - 1).clamp(0, query.size(1) - C4)
    rows = start[:, None] + torch.arange(C4, device=query.device)
    return rows.unsqueeze(0).expand(query.size(0), -1, -1)


def _shared_seed_count(rows: torch.Tensor, reference: torch.Tensor) -> float:
    shared = (rows[..., :, None] == reference[..., None, :]).any(-1).float().sum(-1)
    return float(shared.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--candidate_count", type=int, default=45)
    parser.add_argument(
        "--candidate_layout",
        choices=("local", "uniform", "local_uniform", "multiscale"),
        default="local",
    )
    parser.add_argument("--block_j", type=int, default=2)
    parser.add_argument("--block_chunk", type=int, default=21)
    parser.add_argument("--top_probes", type=int, default=8)
    parser.add_argument("--uniform_probes", type=int, default=8)
    parser.add_argument("--probe_anchor_count", type=int, default=4)
    parser.add_argument("--sketch_dim", type=int, default=0)
    parser.add_argument("--sketch_seed", type=int, default=20260825)
    parser.add_argument(
        "--variants",
        help="Comma-separated selector names; omit to run the full historical sweep.",
    )
    args = parser.parse_args()

    per_sample: dict[str, dict[str, dict[str, float]]] = {}
    for input_path in args.inputs:
        payload = torch.load(input_path, map_location=args.device, weights_only=False)
        query = payload["query"].to(args.device)
        key = payload["key"].to(args.device)
        value = payload["value"].to(args.device)
        dense = payload["dense"].to(args.device)

        rows, selector_seconds = select_rows(
            query,
            key,
            value,
            alpha=args.alpha,
            candidate_count=args.candidate_count,
            block_j=args.block_j,
            block_chunk=args.block_chunk,
            top_probes=args.top_probes,
            uniform_probes=args.uniform_probes,
            probe_anchor_count=args.probe_anchor_count,
            sketch_dim=args.sketch_dim,
            sketch_seed=args.sketch_seed,
            candidate_layout=args.candidate_layout,
            variant_names=(
                {name.strip() for name in args.variants.split(",") if name.strip()}
                if args.variants
                else None
            ),
        )
        rows["fixed_c4"] = _fixed_rows(query, key.size(0))
        reference = rows.get("hellinger_gs", next(iter(rows.values())))
        sample = {}
        for name, seed_rows in rows.items():
            metrics = evaluate_rows(query, key, value, dense, seed_rows)
            metrics["shared_seeds_vs_weighted_gs"] = _shared_seed_count(seed_rows, reference)
            if name in selector_seconds:
                metrics["selection_seconds_python_screen"] = selector_seconds[name]
            sample[name] = metrics
            print(
                f"{Path(input_path).parent.name:>10s} {name:>22s} "
                f"rel_l2={metrics['rel_l2_vs_dense']:.6f} "
                f"shared={metrics['shared_seeds_vs_weighted_gs']:.3f}/4",
                flush=True,
            )
        per_sample[str(input_path)] = sample
        del payload, query, key, value, dense, rows
        torch.cuda.empty_cache()

    names = sorted(next(iter(per_sample.values())))
    aggregate = {}
    for name in names:
        aggregate[name] = {
            metric: sum(sample[name][metric] for sample in per_sample.values()) / len(per_sample)
            for metric in next(iter(per_sample.values()))[name]
        }
    ordered = sorted(aggregate, key=lambda name: aggregate[name]["rel_l2_vs_dense"])
    print("\naggregate")
    for name in ordered:
        result = aggregate[name]
        print(
            f"{name:>22s} rel_l2={result['rel_l2_vs_dense']:.6f} "
            f"mse={result['mse_vs_dense']:.8f} "
            f"shared={result['shared_seeds_vs_weighted_gs']:.3f}/4"
        )

    output = {
        "settings": {
            "alpha": args.alpha,
            "candidate_count": args.candidate_count,
            "candidate_layout": args.candidate_layout,
            "block_j": args.block_j,
            "top_probes": args.top_probes,
            "uniform_probes": args.uniform_probes,
            "probe_anchor_count": args.probe_anchor_count,
            "sketch_dim": args.sketch_dim,
            "sketch_seed": args.sketch_seed,
            "inputs": args.inputs,
            "variants": args.variants,
        },
        "per_sample": per_sample,
        "aggregate": aggregate,
        "ranking": ordered,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
