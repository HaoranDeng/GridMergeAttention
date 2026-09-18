"""Adaptive seed selection for hard-C4 GridMerge.

``hard_route_pytorch.py`` picks the four seed query rows with index arithmetic:
for key row ``k`` it always takes the rows nearest ``k``. This module picks them
from the data instead, then hands the result to the *unchanged* merge in
``hard_route_pytorch``. Only the seed choice differs; the key-softmax hard
assignment, the entropy routing bias, and the full second-stage attention are
identical, so any quality change is attributable to seed selection alone.

Selection runs in three stages for each ``(query frame, key frame, key row,
query column, head)`` block:

1. Probe keys. Four fixed anchor rows score all ``L`` keys; the ``top`` highest
   plus a uniform spread become the probe set. This keeps the scoring cost near
   the merge's own instead of scoring every candidate against every key.
2. Response profiles. Each candidate query row is scored against the probes and
   converted to a value-output profile -- what that row would actually retrieve
   -- which ranks candidates better than raw logits do.
3. Weighted span. Greedy Gram-Schmidt over those profiles, with each candidate's
   residual scaled by ``exp(alpha * z)`` where ``z`` is its z-scored log
   attention mass. ``alpha = 0`` is pure diversity; larger values trade
   diversity for importance.

Seeds are chosen per head, so this returns seed *queries* rather than the shared
row indices that ``gather_seed_queries`` produces.
"""

from __future__ import annotations

import re

import torch
from einops import rearrange

from .hard_route_pytorch import (
    biased_sdpa,
    cached_local_c4_seed_rows,
    gather_seed_queries,
    hard_route_merge_kv_pytorch,
)

C4 = 4
ANCHOR_OFFSETS = (-2, 0, 1, 3)
FLOAT_NEG_INF = torch.finfo(torch.float32).min


_MULTISCALE_TABLE_CACHE = {}


def _unit_rows(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _gather_position(matrix, position):
    """Row ``position`` of a ``[..., candidates, X]`` tensor -> ``[..., X]``."""
    return matrix.gather(
        dim=-2,
        index=position[..., None, None].expand(*position.shape, 1, matrix.size(-1)),
    ).squeeze(-2)


def _project_position_out(residual, position):
    basis = _unit_rows(_gather_position(residual, position))
    return residual - (
        (residual * basis.unsqueeze(-2)).sum(dim=-1, keepdim=True) * basis.unsqueeze(-2)
    )


def _candidate_window(key_rows, query_rows, candidate_count, device):
    """Candidate rows per key row, plus the index-matched row inside the window."""
    key_row = torch.arange(key_rows, device=device)
    start = (key_row - candidate_count // 2).clamp(0, query_rows - candidate_count)
    candidate_rows = start[:, None] + torch.arange(candidate_count, device=device)
    return candidate_rows, key_row - start


def _anchor_rows(key_rows, query_rows, device):
    key_row = torch.arange(key_rows, device=device)
    anchor = key_row[:, None] + torch.tensor(ANCHOR_OFFSETS, device=device)
    valid = ((anchor >= 0) & (anchor < query_rows)).all(dim=-1, keepdim=True)
    fallback = (key_row - 1).clamp(0, query_rows - C4)[:, None] + torch.arange(
        C4, device=device
    )
    return torch.where(valid, anchor, fallback)


def query_l2_kcenter_seed_rows_pytorch(
    q, *, candidate_count=6, sketch_dimensions=None
):
    """Select four real query rows by anchored local L2 k-center.

    The selector is strictly query-only. For every query frame, key row,
    query column, and head, it considers a six-row window with the index-
    matched row as a mandatory first center. Three greedy farthest-point steps
    choose the remaining rows. The returned native table has shape
    ``[B, A, row, column, H, 4]`` and is independent of key frame.
    """

    if q.ndim != 6:
        raise ValueError("query must have shape [B, A, row, column, H, D]")
    batch, query_outer, query_rows, columns, heads, head_dim = q.shape
    candidate_count = int(candidate_count)
    if not C4 <= candidate_count <= query_rows:
        raise ValueError(
            f"candidate_count must be in [{C4}, {query_rows}], got {candidate_count}"
        )

    candidate_rows, anchor = _candidate_window(
        query_rows, query_rows, candidate_count, q.device
    )
    sketch_dimensions = int(sketch_dimensions or head_dim)
    if not 1 <= sketch_dimensions <= head_dim:
        raise ValueError(
            f"sketch_dimensions must be in [1, {head_dim}], got {sketch_dimensions}"
        )
    points = q[:, :, candidate_rows, :, :, :sketch_dimensions]
    points = points.permute(0, 1, 2, 4, 5, 3, 6).float()
    prefix = points.shape[:-2]
    anchor = anchor.view(1, 1, query_rows, 1, 1).expand(prefix)
    used = torch.zeros(
        *prefix, candidate_count, device=q.device, dtype=torch.bool
    )
    used.scatter_(-1, anchor.unsqueeze(-1), True)
    chosen = [anchor]

    def gather(position):
        return points.gather(
            -2,
            position[..., None, None].expand(
                *position.shape, 1, points.size(-1)
            ),
        ).squeeze(-2)

    center = gather(anchor)
    minimum = (points - center.unsqueeze(-2)).square().sum(-1)
    for _ in range(C4 - 1):
        position = minimum.masked_fill(used, FLOAT_NEG_INF).argmax(dim=-1)
        chosen.append(position)
        used.scatter_(-1, position.unsqueeze(-1), True)
        center = gather(position)
        distance = (points - center.unsqueeze(-2)).square().sum(-1)
        minimum = torch.minimum(minimum, distance)

    positions = torch.stack(chosen, dim=-1)
    window = candidate_rows.view(1, 1, query_rows, 1, 1, candidate_count)
    window = window.expand(batch, query_outer, query_rows, columns, heads, -1)
    return torch.gather(window, -1, positions).to(torch.uint8)


def multiscale_candidate_table(
    query_rows,
    candidate_count,
    device=None,
    *,
    geometry_count=None,
):
    """Deterministic local/global candidate rows and matched-anchor positions."""
    query_rows = int(query_rows)
    candidate_count = int(candidate_count)
    geometry_count = int(geometry_count or candidate_count)
    if not 4 <= candidate_count <= query_rows:
        raise ValueError(
            f"candidate_count must be in [4, {query_rows}], got {candidate_count}"
        )
    if not candidate_count <= geometry_count <= query_rows:
        raise ValueError(
            "geometry_count must be between candidate_count and query_rows"
        )
    device = torch.device("cpu" if device is None else device)
    cache_key = (str(device), query_rows, candidate_count, geometry_count)
    cached = _MULTISCALE_TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    table = []
    anchors = []
    radius = max(1, geometry_count // 4)
    uniform_count = max(4, geometry_count // 4)
    for key_row in range(query_rows):
        chosen = {key_row}
        chosen.update(
            range(
                max(0, key_row - radius),
                min(query_rows, key_row + radius + 1),
            )
        )
        chosen.update(
            round(index * (query_rows - 1) / (uniform_count - 1))
            for index in range(uniform_count)
        )
        if len(chosen) > candidate_count:
            raise ValueError(
                "candidate geometry creates more mandatory rows than the target pool"
            )
        while len(chosen) < candidate_count:
            remaining = [row for row in range(query_rows) if row not in chosen]
            row = max(
                remaining,
                key=lambda item: (min(abs(item - seed) for seed in chosen), -item),
            )
            chosen.add(row)
        selected = sorted(chosen)[:candidate_count]
        if key_row not in selected:
            raise RuntimeError("multiscale candidate truncation removed the anchor row")
        table.append(selected)
        anchors.append(selected.index(key_row))

    result = (
        torch.tensor(table, device=device, dtype=torch.int32),
        torch.tensor(anchors, device=device, dtype=torch.int32),
    )
    _MULTISCALE_TABLE_CACHE[cache_key] = result
    return result


def _kl_facility_distance_column(distance, index):
    demand_count = distance.size(-2)
    gather_index = index[..., None, None].expand(*index.shape, demand_count, 1)
    return distance.gather(-1, gather_index).squeeze(-1)


def _anchored_reverse_kl_facility(distance, anchor, demand_weight, num_seeds):
    """Greedily reduce weighted KL facility cost after a mandatory anchor."""

    prefix = distance.shape[:-2]
    candidate_count = distance.size(-1)
    anchor = anchor.expand(prefix)
    selected = [anchor]
    used = torch.zeros(
        *prefix, candidate_count, device=distance.device, dtype=torch.bool
    )
    used.scatter_(-1, anchor.unsqueeze(-1), True)
    minimum = _kl_facility_distance_column(distance, anchor)
    for _ in range(1, num_seeds):
        improved = torch.minimum(minimum.unsqueeze(-1), distance)
        gain = (
            (minimum.unsqueeze(-1) - improved) * demand_weight.unsqueeze(-1)
        ).sum(-2)
        position = gain.masked_fill(used, FLOAT_NEG_INF).argmax(dim=-1)
        selected.append(position)
        used.scatter_(-1, position.unsqueeze(-1), True)
        minimum = torch.minimum(
            minimum, _kl_facility_distance_column(distance, position)
        )
    return torch.stack(selected, dim=-1)


@torch.inference_mode()
def mass_weighted_reverse_kl_seed_rows(
    q,
    k,
    scale=None,
    *,
    candidate_count=32,
    candidate_geometry_count=45,
    num_seeds=4,
    head_chunk=4,
    block_chunk=21,
    output_dtype=torch.uint8,
):
    """All-key oracle for entropy-objective C4 seed selection.

    For every native 80-key block, real query rows minimize
    ``sum_i m_ib min_s KL(p_s,b || p_i,b)``. This reads every key and exact
    dense block masses for the router frame, so it is an oracle diagnostic,
    not a deployable pure-GCA selector. The returned native seed table has
    shape ``[B,A,key_frame,key_row,column,H,C]``.
    """

    if q.ndim != 6 or k.ndim != 6:
        raise ValueError("q and k must have shape [B,outer,row,column,H,D]")
    batch, query_outer, query_rows, columns, heads, dimension = q.shape
    key_batch, key_outer, key_rows, key_columns, key_heads, key_dimension = k.shape
    if (batch, columns, heads, dimension) != (
        key_batch,
        key_columns,
        key_heads,
        key_dimension,
    ):
        raise ValueError("q and k batch/column/head/dimension shapes must match")
    if num_seeds != C4:
        raise ValueError("the entropy seed oracle currently requires C4")
    if not C4 <= candidate_count <= query_rows:
        raise ValueError("candidate_count must be between C4 and query rows")
    if head_chunk <= 0 or block_chunk <= 0:
        raise ValueError("head_chunk and block_chunk must be positive")
    if output_dtype not in (torch.uint8, torch.int32, torch.int64):
        raise ValueError("output_dtype must be uint8, int32, or int64")

    scale = float(dimension**-0.5 if scale is None else scale)
    candidate_rows_by_key_row, anchor_by_key_row = multiscale_candidate_table(
        query_rows,
        candidate_count,
        q.device,
        geometry_count=candidate_geometry_count,
    )
    key_blocks = k.reshape(
        batch, key_outer * key_rows, key_columns, heads, dimension
    )
    total_blocks = key_blocks.size(1)
    block_key_rows = torch.arange(total_blocks, device=q.device) % key_rows
    candidate_rows = candidate_rows_by_key_row[block_key_rows].to(torch.long)
    anchors = anchor_by_key_row[block_key_rows].to(torch.long)
    result = torch.empty(
        batch,
        query_outer,
        total_blocks,
        columns,
        heads,
        num_seeds,
        device=q.device,
        dtype=output_dtype,
    )

    for head_start in range(0, heads, head_chunk):
        head_end = min(head_start + head_chunk, heads)
        query = (
            q[..., head_start:head_end, :]
            .permute(0, 1, 3, 4, 2, 5)
            .float()
        )
        active_heads = head_end - head_start
        global_log_partition = torch.full(
            (batch, query_outer, columns, active_heads, query_rows),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )

        for block_start in range(0, total_blocks, block_chunk):
            block_end = min(block_start + block_chunk, total_blocks)
            key_chunk = (
                key_blocks[
                    :, block_start:block_end, :, head_start:head_end, :
                ]
                .permute(0, 1, 3, 2, 4)
                .float()
            )
            target_logits = torch.einsum(
                "bajhrd,bnhld->bajhnrl", query, key_chunk
            ).mul_(scale)
            chunk_log_partition = torch.logsumexp(
                target_logits, dim=(-3, -1)
            )
            global_log_partition = torch.logaddexp(
                global_log_partition, chunk_log_partition
            )

        for block_start in range(0, total_blocks, block_chunk):
            block_end = min(block_start + block_chunk, total_blocks)
            blocks = block_end - block_start
            rows_chunk = candidate_rows[block_start:block_end]
            anchors_chunk = anchors[block_start:block_end]
            key_chunk = (
                key_blocks[
                    :, block_start:block_end, :, head_start:head_end, :
                ]
                .permute(0, 1, 3, 2, 4)
                .float()
            )
            target_logits = torch.einsum(
                "bajhrd,bnhld->bajhnrl", query, key_chunk
            ).mul_(scale)
            target_log_partition = torch.logsumexp(target_logits, dim=-1)
            target_log_probability = torch.log_softmax(target_logits, dim=-1)

            expanded_query = query.unsqueeze(-3).expand(
                batch,
                query_outer,
                columns,
                active_heads,
                blocks,
                query_rows,
                dimension,
            )
            candidate_index = rows_chunk.view(
                1, 1, 1, 1, blocks, candidate_count, 1
            ).expand(
                batch,
                query_outer,
                columns,
                active_heads,
                blocks,
                candidate_count,
                dimension,
            )
            candidate_query = expanded_query.gather(-2, candidate_index)
            candidate_logits = torch.einsum(
                "bajhncd,bnhld->bajhncl", candidate_query, key_chunk
            ).mul_(scale)
            candidate_log_probability = torch.log_softmax(
                candidate_logits, dim=-1
            )
            candidate_probability = candidate_log_probability.exp()
            candidate_negative_entropy = (
                candidate_probability * candidate_log_probability
            ).sum(-1)
            candidate_under_target = torch.einsum(
                "bajhncl,bajhnrl->bajhnrc",
                candidate_probability,
                target_log_probability,
            )
            reverse_kl = (
                candidate_negative_entropy.unsqueeze(-2)
                - candidate_under_target
            ).clamp_min_(0.0)
            block_mass = torch.exp(
                target_log_partition - global_log_partition.unsqueeze(-2)
            )
            anchor = anchors_chunk.view(1, 1, 1, 1, blocks).expand(
                batch, query_outer, columns, active_heads, blocks
            )
            positions = _anchored_reverse_kl_facility(
                reverse_kl, anchor, block_mass, num_seeds
            )
            expanded_rows = rows_chunk.view(
                1, 1, 1, 1, blocks, candidate_count
            ).expand(
                batch,
                query_outer,
                columns,
                active_heads,
                blocks,
                candidate_count,
            )
            selected_rows = expanded_rows.gather(-1, positions)
            result[
                :, :, block_start:block_end, :, head_start:head_end, :
            ] = selected_rows.permute(0, 1, 4, 2, 3, 5).to(output_dtype)

    return result.reshape(
        batch,
        query_outer,
        key_outer,
        key_rows,
        columns,
        heads,
        num_seeds,
    )


def _weighted_gram_schmidt_rows(
    gram,
    log_mass,
    candidate_rows,
    anchor_indices,
    *,
    alpha,
    num_seeds,
):
    """Select candidate rows from a centered Hellinger Gram matrix."""

    batch, heads, columns, rows, _ = gram.shape
    key_rows, candidates = candidate_rows.shape
    prefix = (batch, heads, columns, key_rows)

    expanded_gram = gram.unsqueeze(-3).expand(*gram.shape[:-2], key_rows, rows, rows)
    row_index = candidate_rows.view(1, 1, 1, key_rows, candidates, 1).expand(
        *prefix, candidates, rows
    )
    candidate_gram = torch.gather(expanded_gram, -2, row_index)
    column_index = candidate_rows.view(1, 1, 1, key_rows, 1, candidates).expand(
        *prefix, candidates, candidates
    )
    residual_gram = torch.gather(candidate_gram, -1, column_index)

    mass_index = candidate_rows.view(1, 1, 1, key_rows, candidates).expand(
        *prefix, candidates
    )
    candidate_mass = torch.gather(
        log_mass.unsqueeze(-2).expand(*log_mass.shape[:-1], key_rows, rows),
        -1,
        mass_index,
    )
    deviation = candidate_mass - candidate_mass.mean(dim=-1, keepdim=True)
    z_score = deviation * torch.rsqrt(
        deviation.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    weight = torch.exp(2.0 * float(alpha) * z_score)

    candidate_rows_expanded = candidate_rows.view(
        1, 1, 1, key_rows, candidates
    ).expand(*prefix, candidates)
    pivot = anchor_indices.view(1, 1, 1, key_rows).expand(*prefix).long()
    used = torch.zeros((*prefix, candidates), device=gram.device, dtype=torch.bool)
    selected_rows = []

    for step in range(int(num_seeds)):
        if step:
            objective = residual_gram.diagonal(dim1=-2, dim2=-1).clamp_min_(0)
            pivot = (objective * weight).masked_fill(used, -1.0).argmax(dim=-1)
        used.scatter_(-1, pivot.unsqueeze(-1), True)
        selected_rows.append(
            torch.gather(candidate_rows_expanded, -1, pivot.unsqueeze(-1)).squeeze(-1)
        )
        if step + 1 == int(num_seeds):
            continue
        pivot_column = torch.gather(
            residual_gram,
            -1,
            pivot[..., None, None].expand(*prefix, candidates, 1),
        ).squeeze(-1)
        pivot_norm = torch.gather(
            residual_gram.diagonal(dim1=-2, dim2=-1),
            -1,
            pivot.unsqueeze(-1),
        ).squeeze(-1).clamp_min(1e-12)
        residual_gram = residual_gram - (
            pivot_column.unsqueeze(-1) * pivot_column.unsqueeze(-2)
            / pivot_norm[..., None, None]
        )

    return torch.stack(selected_rows, dim=-1)


@torch.inference_mode()
def first_frame_global_hellinger_seed_rows(
    q,
    k,
    scale=None,
    *,
    alpha=0.2,
    candidate_count=32,
    candidate_geometry_count=None,
    num_seeds=4,
    head_chunk=8,
    column_chunk=4,
    profile_dtype=torch.float32,
    output_dtype=torch.uint8,
):
    """Choose shared seed rows from first-frame attention to every video key.

    ``q`` contains exactly one router frame in blocked ``[B,1,R,C,H,D]``
    layout. For each query column and head, this computes the complete
    attention distribution of every query row over all ``F*R*C`` keys. The
    centered square-root distributions define a Hellinger Gram matrix. A
    weighted Gram-Schmidt selector then chooses four rows from each key row's
    multiscale candidate pool. The resulting table is shared across key frames;
    it guides GMA routing only and is never used as a dense output rescue.
    """

    if q.ndim != 6 or k.ndim != 6:
        raise ValueError("first-frame global routing expects blocked 6D Q and K")
    if q.size(1) != 1:
        raise ValueError("first-frame global routing requires exactly one query frame")
    if q.size(0) != k.size(0) or q.shape[2:] != k.shape[2:]:
        raise ValueError("first-frame global routing requires matching Q/K block shapes")
    if output_dtype not in (torch.uint8, torch.int32):
        raise ValueError("seed rows must use uint8 or int32 output")
    if profile_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Hellinger profiles must use float32, float16, or bfloat16")
    rows, columns, heads, head_dim = q.shape[2:]
    if int(num_seeds) < 2 or int(num_seeds) > int(candidate_count):
        raise ValueError("num_seeds must be between 2 and candidate_count")
    if scale is None:
        scale = head_dim**-0.5

    candidate_rows, anchor_indices = multiscale_candidate_table(
        rows,
        candidate_count,
        q.device,
        geometry_count=candidate_geometry_count,
    )
    key_count = k.size(1) * rows * columns
    key_flat = k.permute(0, 4, 1, 2, 3, 5).reshape(
        k.size(0), heads, key_count, head_dim
    )
    query_frame = q[:, 0].permute(0, 3, 2, 1, 4).contiguous()
    table = torch.empty(
        (q.size(0), 1, k.size(1), rows, columns, heads, int(num_seeds)),
        device=q.device,
        dtype=output_dtype,
    )

    head_chunk = max(1, min(int(head_chunk), heads))
    column_chunk = max(1, min(int(column_chunk), columns))
    for head_start in range(0, heads, head_chunk):
        head_end = min(head_start + head_chunk, heads)
        active_keys = key_flat[:, head_start:head_end]
        transposed_keys = active_keys.transpose(-1, -2).unsqueeze(2)
        for column_start in range(0, columns, column_chunk):
            column_end = min(column_start + column_chunk, columns)
            active_queries = query_frame[
                :, head_start:head_end, column_start:column_end
            ]
            logits = torch.matmul(active_queries, transposed_keys)
            logits.mul_(float(scale))
            if profile_dtype == torch.float32:
                logits = logits.float()
                log_partition = torch.logsumexp(logits, dim=-1)
                profile = torch.exp(
                    0.5 * (logits - log_partition.unsqueeze(-1))
                )
            else:
                row_max = logits.amax(dim=-1, keepdim=True)
                centered_logits = logits - row_max
                half_exp = torch.exp(centered_logits * 0.5).to(profile_dtype)
                normalizer = torch.sum(
                    half_exp.square(), dim=-1, dtype=torch.float32
                ).clamp_min_(1e-20)
                profile = half_exp * torch.rsqrt(normalizer).to(half_exp.dtype).unsqueeze(-1)
                log_partition = row_max.squeeze(-1).float() + torch.log(normalizer)
            profile_sum = profile.sum(dim=-1, dtype=torch.float32)
            gram = torch.matmul(profile, profile.transpose(-1, -2)).float()
            gram.sub_(
                profile_sum.unsqueeze(-1) * profile_sum.unsqueeze(-2) / key_count
            )
            selected = _weighted_gram_schmidt_rows(
                gram,
                log_partition,
                candidate_rows,
                anchor_indices,
                alpha=alpha,
                num_seeds=num_seeds,
            )
            selected = selected.permute(0, 3, 2, 1, 4).contiguous()
            table[
                :, 0, :, :, column_start:column_end, head_start:head_end
            ] = selected.unsqueeze(1).expand(-1, k.size(1), -1, -1, -1, -1)

    return table


def _probe_columns(q, k, scale, probe_count, top_count):
    """Pick probe key columns from four fixed anchor rows."""
    key_cols = k.size(3)
    anchor = _anchor_rows(k.size(2), q.size(2), q.device)
    anchor_queries = q[:, :, anchor, :, :, :]
    response = torch.einsum(
        "bakcjhd,bfklhd->bafkjhcl", anchor_queries.float(), k.float()
    )
    importance = torch.log_softmax(response * float(scale), dim=-1).amax(dim=-2)
    top_index = importance.topk(top_count, dim=-1).indices
    uniform_count = probe_count - top_count
    if uniform_count <= 0:
        return top_index
    uniform = (
        torch.linspace(0, key_cols - 1, uniform_count, device=k.device).round().long()
    )
    uniform = uniform.view(*(1 for _ in top_index.shape[:-1]), uniform_count).expand(
        *top_index.shape[:-1], uniform_count
    )
    return torch.cat((top_index, uniform), dim=-1)


def _gather_probes(source, probe_index, query_frames, query_cols):
    """Gather ``[b, f, k, l, h, d]`` at probe columns -> ``[b,a,f,k,j,h,probe,d]``."""
    expanded = rearrange(source.float(), "b f k l h d -> b 1 f k 1 h l d").expand(
        -1, query_frames, -1, -1, query_cols, -1, -1, -1
    )
    return torch.gather(
        expanded, -2, probe_index[..., None].expand(*probe_index.shape, source.size(-1))
    )


def _adaptive_selection(
    q, k, v, scale, *, alpha, candidate_count, probe_count, value_dimensions
):
    """Run the selector once; return chosen positions, the window, and the bank.

    Positions index into the candidate window, so they are only meaningful
    alongside ``candidate_rows``. Both public entry points build on this.
    """
    query_rows, query_cols = q.size(2), q.size(3)
    candidate_count = int(candidate_count or query_rows)
    candidate_count = max(C4, min(candidate_count, query_rows))
    top_count = max(1, probe_count // 2)

    candidate_rows, center = _candidate_window(
        k.size(2), query_rows, candidate_count, q.device
    )
    probe_index = _probe_columns(q, k, scale, probe_count, top_count)

    candidates = q[:, :, candidate_rows, :, :, :].float()
    bank = rearrange(candidates, "b a k p j h d -> b a 1 k j h p d").expand(
        -1, -1, k.size(1), -1, -1, -1, -1, -1
    )
    key_probes = _gather_probes(k, probe_index, q.size(1), query_cols)
    logits = torch.einsum("bafkjhpd,bafkjhsd->bafkjhps", bank, key_probes) * float(scale)

    # Importance: log attention mass on the probes, z-scored across candidates so
    # one alpha is comparable across blocks with different logit scales.
    log_mass = torch.logsumexp(logits, dim=-1)
    z = log_mass - log_mass.mean(dim=-1, keepdim=True)
    z = z * torch.rsqrt(z.var(dim=-1, keepdim=True, correction=0) + 1e-6)

    # Value-output profile: what each candidate would actually retrieve.
    feature_value = v.float()
    if value_dimensions and value_dimensions < feature_value.size(-1):
        feature_value = feature_value.reshape(
            *feature_value.shape[:-1],
            value_dimensions,
            feature_value.size(-1) // value_dimensions,
        ).mean(dim=-1)
    value_probes = _gather_probes(feature_value, probe_index, q.size(1), query_cols)
    profiles = torch.einsum(
        "bafkjhps,bafkjhsd->bafkjhpd", torch.softmax(logits, dim=-1), value_probes
    )

    residual = profiles - profiles.mean(dim=-1, keepdim=True)
    residual = residual * torch.exp(float(alpha) * z)[..., None]

    anchor = center[None, None, None, :, None, None].expand(residual.shape[:-2])
    used = torch.zeros_like(residual[..., 0], dtype=torch.bool)
    used.scatter_(-1, anchor.unsqueeze(-1), True)
    chosen = [anchor]
    residual = _project_position_out(residual, anchor)
    for _ in range(C4 - 1):
        position = (
            residual.square().sum(dim=-1).masked_fill(used, FLOAT_NEG_INF).argmax(dim=-1)
        )
        chosen.append(position)
        used.scatter_(-1, position.unsqueeze(-1), True)
        residual = _project_position_out(residual, position)

    return chosen, candidate_rows, bank


def adaptive_seed_queries(
    q,
    k,
    v,
    scale,
    *,
    alpha=0.3,
    candidate_count=None,
    probe_count=16,
    value_dimensions=32,
):
    """Seed queries shaped [B, A, F, K, J, C, H, D] for the unchanged merge.

    q:   [B, A, I, J, H, D]
    k/v: [B, F, K, L, H, D]
    """
    chosen, _, bank = _adaptive_selection(
        q,
        k,
        v,
        scale,
        alpha=alpha,
        candidate_count=candidate_count,
        probe_count=probe_count,
        value_dimensions=value_dimensions,
    )
    # Stacking puts the seed axis after the head axis; the merge wants
    # [..., C, H, D], so swap them rather than relying on H != C to catch it.
    seeds = torch.stack(
        [_gather_position(bank, position) for position in chosen], dim=-2
    ).transpose(-3, -2)
    return seeds.to(q.dtype)


def adaptive_seed_row_indices(
    q,
    k,
    v,
    scale,
    *,
    alpha=0.3,
    candidate_count=None,
    probe_count=16,
    value_dimensions=32,
):
    """Chosen query-row indices shaped [B, A, F, K, J, C, H].

    This is what a seed table would hold: the selector's output reduced to four
    small integers per block and head, which is what makes caching or a
    table-driven Triton merge possible.
    """
    chosen, candidate_rows, _ = _adaptive_selection(
        q,
        k,
        v,
        scale,
        alpha=alpha,
        candidate_count=candidate_count,
        probe_count=probe_count,
        value_dimensions=value_dimensions,
    )
    positions = torch.stack(chosen, dim=-1)
    window = candidate_rows[None, None, None, :, None, None, :].expand(
        *positions.shape[:-1], candidate_rows.size(-1)
    )
    return torch.gather(window, -1, positions)


def adaptive_hard_route_attention_pytorch(
    q,
    k,
    v,
    scale,
    query_outer_offset=0,
    *,
    alpha=0.3,
    candidate_count=None,
    probe_count=16,
    value_dimensions=32,
    head_chunk=None,
):
    """Hard C4 with adaptive seeds. Same signature as the fixed-seed reference.

    ``alpha=None`` selects the fixed local seed rows instead, which makes this a
    drop-in equivalent of ``hard_route_attention_pytorch`` and gives the port a
    self-check.
    """
    heads = q.size(4)
    head_chunk = heads if not head_chunk or head_chunk >= heads else int(head_chunk)
    outputs = []
    for start in range(0, heads, head_chunk):
        stop = min(start + head_chunk, heads)
        q_h, k_h, v_h = q[..., start:stop, :], k[..., start:stop, :], v[..., start:stop, :]
        if alpha is None:
            seed_rows = cached_local_c4_seed_rows(
                q.device,
                query_outer_offset,
                q.size(1),
                k.size(1),
                k.size(2),
                q.size(3),
                q.size(2),
            )
            seeds = gather_seed_queries(q_h, seed_rows)
        else:
            seeds = adaptive_seed_queries(
                q_h,
                k_h,
                v_h,
                scale,
                alpha=alpha,
                candidate_count=candidate_count,
                probe_count=probe_count,
                value_dimensions=value_dimensions,
            )
        merged_k, merged_v, bias = hard_route_merge_kv_pytorch(seeds, k_h, v_h, scale)
        out = biased_sdpa(
            rearrange(q_h, "b a i j h d -> (b h a j) i d"),
            rearrange(merged_k, "b a f j k c h d -> (b h a j) (f k c) d"),
            rearrange(merged_v, "b a f j k c h d -> (b h a j) (f k c) d"),
            rearrange(bias, "b h a f k j c -> (b h a j) (f k c)"),
            scale=scale,
        )
        outputs.append(
            rearrange(
                out,
                "(b h a j) i d -> b a i j h d",
                b=q.size(0),
                h=stop - start,
                a=q.size(1),
                j=q.size(3),
            )
        )
    return torch.cat(outputs, dim=4)


def parse_weighted_span_seed_mode(seed_mode):
    """Parse ``weighted_span_alpha<A>_pool<N>`` into ``(alpha, pool)``.

    ``weighted_span_alpha0p3_pool30`` is alpha 0.3 over a 30-row candidate
    pool; ``weighted_span_alpha0p0_pool30`` is the unweighted control. The
    words are spelled out because the short form put "alpha 0.3" next to
    "pool 30" and was misread as 0.03. Returns ``None`` for any other seed
    mode so callers fall through to their own table.
    """
    if not seed_mode.startswith("weighted_span_alpha"):
        return None
    try:
        alpha_text, pool_text = seed_mode[len("weighted_span_alpha"):].split("_pool")
        return float(alpha_text.replace("p", ".")), int(pool_text)
    except ValueError:
        raise ValueError(
            f"malformed weighted-span seed mode {seed_mode!r}; expected "
            "weighted_span_alpha<A>_pool<N>, e.g. weighted_span_alpha0p3_pool30"
        )


def parse_hellinger_span_seed_mode(seed_mode):
    """Parse Hellinger-span settings and optional head/time sharing."""
    match = re.fullmatch(
        r"hellinger_span_alpha([0-9]+(?:p[0-9]+)?)_c([0-9]+)_p([0-9]+)"
        r"_a([0-9]+)_j([0-9]+)(?:_g([0-9]+)(?:_r([0-9]+))?)?"
        r"(?:_t([0-9]+))?",
        str(seed_mode),
    )
    if match is None:
        if str(seed_mode).startswith("hellinger_span_"):
            raise ValueError(
                f"malformed Hellinger-span seed mode {seed_mode!r}; expected "
                "hellinger_span_alpha0p3_c22_p16_a2_j4 or "
                "hellinger_span_alpha0p3_c22_p16_a2_j4_g8_r2_t3"
            )
        return None
    (
        alpha_text,
        candidates,
        probes,
        anchors,
        block_j,
        head_group,
        router_offset,
        temporal_frames,
    ) = match.groups()
    head_group = int(head_group or 1)
    router_offset = int(router_offset or 0)
    if router_offset >= head_group:
        raise ValueError("routing-head offset must be smaller than its group")
    temporal_frames = int(temporal_frames or 0)
    if temporal_frames < 0 or temporal_frames > 21:
        raise ValueError("temporal routing frames must be in [0, 21]")
    return (
        float(alpha_text.replace("p", ".")),
        int(candidates),
        int(probes),
        int(anchors),
        int(block_j),
        head_group,
        router_offset,
        temporal_frames,
    )


__all__ = [
    "adaptive_hard_route_attention_pytorch",
    "adaptive_seed_queries",
    "adaptive_seed_row_indices",
    "first_frame_global_hellinger_seed_rows",
    "mass_weighted_reverse_kl_seed_rows",
    "multiscale_candidate_table",
    "parse_hellinger_span_seed_mode",
    "parse_weighted_span_seed_mode",
    "query_l2_kcenter_seed_rows_pytorch",
]
