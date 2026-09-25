"""Maintained Wan hard-C4 attention plus causal one-token GridMerge."""

from __future__ import annotations

import math
import re

import torch

from .core import flash_attention
from .capacity_contract import require_exact_merged_capacity
from ._grid_utils import (
    blocks_to_video as general_blocks_to_video,
    crop_video_tokens_from_multiples,
    pad_video_tokens_to_multiples,
    video_to_blocks as general_video_to_blocks,
)
from .causal_gridmerge import causal_gridmerge_one
from .hard_c4_wan21_1_3b_480p_triton import HARD_C4_BACKEND, hard_c4_rt_attention_triton_blocks
from .hard_c4_wan21_14b_720p_triton import (
    hard_c4_720p_exact_query_top3_attention_triton_blocks,
    hard_c4_720p_exact_intra_frame_attention_triton_blocks,
    hard_c4_720p_query_centroid_attention_triton_blocks,
    hard_c4_720p_query_response_attention_triton_blocks,
    hard_c4_720p_responsibility_centroid_attention_triton_blocks,
    hard_c4_720p_native_seed_rows_attention_triton_blocks,
    hard_c4_720p_seed_rows_attention_triton_blocks,
    hard_c4_720p_temporal_compact_attention_triton_blocks,
    hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks,
)
from .hard_c5_c6_wan21_14b_720p_triton import (
    hard_c_720p_native_temporal_seed_rows_attention_triton_blocks,
    hard_c3_720p_attention_triton_blocks,
    hard_c5_720p_attention_triton_blocks,
    hard_c5_720p_query_centroid_norm_attention_triton_blocks,
    hard_c5_720p_temporal_shared_seed_rows_attention_triton_blocks,
    hard_c6_720p_attention_triton_blocks,
    hard_c6_720p_compact_attention_triton_blocks,
    hard_c6_720p_fp8_key_attention_triton_blocks,
    hard_c6_720p_gaussian_partition_attention_triton_blocks,
    hard_c6_720p_split_kv_attention_triton_blocks,
    hard_c6_720p_max_partition_attention_triton_blocks,
    hard_c8_720p_attention_triton_blocks,
    hard_c8_720p_fp8_key_attention_triton_blocks,
    hard_c8_720p_gaussian_partition_attention_triton_blocks,
    hard_c8_720p_split_kv_attention_triton_blocks,
    hard_c8_720p_max_partition_attention_triton_blocks,
)
from .hard_c4_wan21_14b_720p_transpose_triton import (
    hard_c3_720p_transpose_attention_triton_head_major,
)
from .hard_c4_grid_wan21_14b_720p_triton import (
    hard_c4_grid_720p_attention_triton_head_major,
)
from .hierarchical_c4_wan21_14b_720p_triton import (
    hard_c4_720p_hierarchical_mean_split_attention_triton_blocks,
)
from .singleton_c4_wan21_14b_720p_triton import (
    hard_c4_720p_singleton_leverage_attention_triton_blocks,
    hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks,
    hard_c4_720p_singleton_leverage_top3_exact_intra_frame_attention_triton_blocks,
    hard_c4_720p_singleton_leverage_top6_attention_triton_blocks,
)
from .adaptive_seed_pytorch import (
    adaptive_hard_route_attention_pytorch,
    adaptive_seed_row_indices,
    first_frame_global_hellinger_seed_rows,
    mass_weighted_reverse_kl_seed_rows,
    parse_hellinger_span_seed_mode,
    parse_weighted_span_seed_mode,
)
from .adaptive_seed_triton import (
    adaptive_seed_attention_triton,
    hellinger_span_seed_rows_triton,
    query_l2_kcenter_seed_rows_triton,
    span_seed_rows_triton_split,
    span_seed_rows_triton_temporal_consensus2,
    span_seed_rows_triton_temporal_probes_per_frame,
    temporal_local_refine_seed_rows_triton,
    temporal_simpson_query_triton,
    temporal_query_row_map_triton,
)
from .hard_route_pytorch import hard_route_attention_pytorch
from .entropy_mass_seed_triton import entropy_mass_seed_rows_triton
from .joint_frame_mass_seed_triton import joint_frame_mass_seed_rows_triton
from .partition_mass_seed_triton import partition_mass_seed_rows_triton
from .monarch_rt_wan21_1_3b_480p_triton import MONARCH_RT_BACKEND, monarch_rt_attention_triton_blocks


LAYOUT_HARD_ROUTE = "hard_route"
BIAXIAL_C3_BACKEND = "biaxial_c3"
_FIXED_CONTIGUOUS_SEED_CACHE = {}


def _strip_hellinger_probe_policy(seed_mode):
    """Return the seed mode and its explicit P16 probe-policy override."""

    if seed_mode.endswith("_u16"):
        return seed_mode[:-4], "uniform"
    if seed_mode.endswith("_a8u8"):
        return seed_mode[:-5], "mixed"
    return seed_mode, "default"


def _use_uniform_hellinger_probes(probes, probe_policy):
    """Use Uniform-16 by default while retaining explicit legacy replay."""

    if probe_policy == "uniform":
        return True
    if probe_policy == "mixed":
        return False
    return int(probes) == 16


def _hard_c4_video_to_blocks(x, height, width):
    batch, seq_len, heads, dim = x.shape
    frames = seq_len // (height * width)
    return x.reshape(batch, frames, height, width, heads, dim)


def _hard_c4_blocks_to_video(x):
    batch, frames, height, width, heads, dim = x.shape
    return x.reshape(batch, frames * height * width, heads, dim)


def _fixed_contiguous_seed_rows(
    device, batch, columns, active_heads, key_outer, rows, clusters
):
    key = (
        device.type,
        device.index,
        batch,
        columns,
        active_heads,
        key_outer,
        rows,
        clusters,
    )
    table = _FIXED_CONTIGUOUS_SEED_CACHE.get(key)
    if table is None:
        key_rows = torch.arange(rows, device=device)
        start = (key_rows - 1).clamp(0, rows - clusters)
        local_rows = start[:, None] + torch.arange(clusters, device=device)
        table = (
            local_rows.to(torch.uint8)
            .view(1, 1, 1, 1, 1, rows, clusters)
            .expand(batch, 1, columns, active_heads, key_outer, rows, clusters)
            .contiguous()
        )
        _FIXED_CONTIGUOUS_SEED_CACHE[key] = table
    return table


def _hard_head_chunks(heads, head_chunk, dense_heads=()):
    """Return contiguous C4 head chunks while excluding fixed dense heads."""

    dense_heads = set(int(head) for head in dense_heads)
    chunks = []
    start = 0
    while start < heads:
        if start in dense_heads:
            start += 1
            continue
        run_end = start + 1
        while run_end < heads and run_end not in dense_heads:
            run_end += 1
        chunk_start = start
        while chunk_start < run_end:
            chunk_end = min(chunk_start + head_chunk, run_end)
            chunks.append((chunk_start, chunk_end - chunk_start))
            chunk_start = chunk_end
        start = run_end
    return chunks


def _uniform_anchor_indices(length, count, device):
    """Return deterministic, endpoint-covering anchors for one grid axis."""

    length = int(length)
    count = int(count)
    if count <= 0 or count > length:
        raise ValueError(f"anchor count must be in [1, {length}], got {count}.")
    if count == 1:
        return torch.tensor([length // 2], device=device, dtype=torch.long)
    return torch.linspace(0, length - 1, count, device=device).round().long()


def _linear_interpolation_map(length, anchors):
    positions = torch.arange(length, device=anchors.device, dtype=torch.long)
    right = torch.searchsorted(anchors, positions).clamp(max=anchors.numel() - 1)
    left = (right - 1).clamp(min=0)
    denominator = (anchors[right] - anchors[left]).clamp_min(1)
    alpha = (positions - anchors[left]).float() / denominator.float()
    return left, right, alpha


def _interpolate_anchor_grid_residual(
    anchor_residual,
    time_anchors,
    row_anchors,
    *,
    frames,
    rows,
    width_anchors=None,
    width=None,
):
    """Extend sparse residual anchors to the full video query grid."""

    if width_anchors is not None:
        if width is None:
            raise ValueError("width is required when width anchors are provided.")
        width_left, width_right, width_alpha = _linear_interpolation_map(
            width, width_anchors
        )
        residual_left = anchor_residual.index_select(3, width_left)
        residual_right = anchor_residual.index_select(3, width_right)
        width_alpha = width_alpha.to(anchor_residual.dtype).view(
            1, 1, 1, width, 1, 1
        )
        anchor_residual = torch.lerp(
            residual_left, residual_right, width_alpha
        )

    row_left, row_right, row_alpha = _linear_interpolation_map(rows, row_anchors)
    residual_left = anchor_residual.index_select(2, row_left)
    residual_right = anchor_residual.index_select(2, row_right)
    row_alpha = row_alpha.to(anchor_residual.dtype).view(1, 1, rows, 1, 1, 1)
    row_residual = torch.lerp(residual_left, residual_right, row_alpha)

    time_left, time_right, time_alpha = _linear_interpolation_map(
        frames, time_anchors
    )
    residual_left = row_residual.index_select(1, time_left)
    residual_right = row_residual.index_select(1, time_right)
    time_alpha = time_alpha.to(anchor_residual.dtype).view(
        1, frames, 1, 1, 1, 1
    )
    return torch.lerp(residual_left, residual_right, time_alpha)


def _apply_full_attention_anchor_grid(
    output_blocks,
    query_blocks,
    key,
    value,
    *,
    sm_scale,
    time_count,
    row_count,
    width_count=0,
    correction_scale=1.0,
):
    """Correct Hard C4 with a sparse grid of exact dense-query residuals."""

    batch, frames, rows, width, heads, dim = query_blocks.shape
    time_anchors = _uniform_anchor_indices(frames, time_count, query_blocks.device)
    row_anchors = _uniform_anchor_indices(rows, row_count, query_blocks.device)
    anchor_query = query_blocks.index_select(1, time_anchors).index_select(
        2, row_anchors
    )
    width_anchors = None
    if width_count:
        width_anchors = _uniform_anchor_indices(
            width, width_count, query_blocks.device
        )
        anchor_query = anchor_query.index_select(3, width_anchors)
    sampled_width = width_count or width
    dense_anchor = flash_attention(
        anchor_query.reshape(batch, -1, heads, dim),
        key,
        value,
        softmax_scale=sm_scale,
    ).reshape(batch, time_count, row_count, sampled_width, heads, dim)
    approximate_anchor = output_blocks.index_select(1, time_anchors).index_select(
        2, row_anchors
    )
    if width_anchors is not None:
        approximate_anchor = approximate_anchor.index_select(3, width_anchors)
    anchor_residual = dense_anchor - approximate_anchor
    interpolated_residual = _interpolate_anchor_grid_residual(
        anchor_residual,
        time_anchors,
        row_anchors,
        frames=frames,
        rows=rows,
        width_anchors=width_anchors,
        width=width,
    )
    output_blocks.add_(interpolated_residual, alpha=float(correction_scale))


def _hard_c4_attention(
    query,
    key,
    value,
    *,
    seed_query=None,
    h,
    w,
    h_reduce,
    w_reduce,
    sm_scale,
    query_outer_chunk,
    head_chunk,
    hard_route_backend,
    exact_intra_frame_attention,
    exact_neighbor_window_columns,
    exact_query_rows,
    first_frame_full_attention,
    full_attention_anchor_time_count,
    full_attention_anchor_row_count,
    full_attention_anchor_scale,
    seed_mode,
    full_attention_anchor_width_count=0,
    dense_heads=(),
    seed_table_cache=None,
    cache_timestep=None,
):
    """Execute the reference, native-grid, or h-reduced Hard-C4 path."""

    require_exact_merged_capacity(4, 0)
    h = int(h)
    w = int(w)
    h_reduce = int(h_reduce)
    w_reduce = int(w_reduce)
    if h_reduce not in {1, 2, 4} or w_reduce != 1:
        raise ValueError(
            "Wan Hard C4 supports h_reduce=1, 2, or 4 with w_reduce=1."
        )
    if h_reduce != 1 and (h, w) != (45, 80):
        raise ValueError("h-reduced Hard C4 is specialized for the 45x80 720p grid.")
    query_blocks = _hard_c4_video_to_blocks(query, h, w)
    key_blocks = _hard_c4_video_to_blocks(key, h, w)
    value_blocks = _hard_c4_video_to_blocks(value, h, w)
    seed_query_blocks = (
        None
        if seed_query is None
        else _hard_c4_video_to_blocks(seed_query, h, w)
    )
    backend = (hard_route_backend or HARD_C4_BACKEND).lower()
    use_reference = backend == "sdpa"
    use_triton = backend == HARD_C4_BACKEND
    if not use_reference and not use_triton:
        raise ValueError(
            "Wan hard-route supports hard_route_backend='sdpa' or 'hard_c4'."
        )
    seed_mode = str(seed_mode or "local_with_ith").lower()
    requested_seed_mode = seed_mode
    common_c_ablation = seed_mode.endswith("_commonc")
    if common_c_ablation:
        seed_mode = seed_mode[:-8]
    merge_loop_stages = 1
    merge_disable_licm = None
    merge_pipeline_suffix = "_ml2"
    if seed_mode.endswith(("_ml1", merge_pipeline_suffix)):
        # Explicit stage counts preserve the tuned LICM setting. An omitted
        # suffix still retains the historical default compiler behavior.
        merge_loop_stages = int(seed_mode[-1])
        merge_disable_licm = False
        seed_mode = seed_mode[: -len(merge_pipeline_suffix)]
    native_seed_layout_suffix = "_nl"
    direct_native_seed_layout = seed_mode.endswith(native_seed_layout_suffix)
    if direct_native_seed_layout:
        seed_mode = seed_mode[: -len(native_seed_layout_suffix)]
    split_kv_accum = seed_mode.endswith("_skv")
    if split_kv_accum:
        seed_mode = seed_mode[:-4]
    merge_frame_group = 1
    merge_frame_group_match = re.search(r"_mg(3|7|21)$", seed_mode)
    if merge_frame_group_match is not None:
        merge_frame_group = int(merge_frame_group_match.group(1))
        seed_mode = seed_mode[: merge_frame_group_match.start()]
    merge_col_group = 1
    merge_col_group_match = re.search(r"_mcg(2|4|8|16)$", seed_mode)
    if merge_col_group_match is not None:
        merge_col_group = int(merge_col_group_match.group(1))
        seed_mode = seed_mode[: merge_col_group_match.start()]
    tuned_launch_suffix = "_gt32gw8"
    tuned_hellinger_launch = seed_mode.endswith(tuned_launch_suffix)
    if tuned_hellinger_launch:
        seed_mode = seed_mode[: -len(tuned_launch_suffix)]
    materialize_singleton = seed_mode.endswith("_mat")
    if materialize_singleton:
        seed_mode = seed_mode[:-4]
    split_merge = seed_mode.endswith("_splitmerge")
    if split_merge:
        seed_mode = seed_mode[:-11]
    query_frame_group = 21
    query_frame_group_match = re.search(r"_qg(1|3|7)$", seed_mode)
    if query_frame_group_match is not None:
        query_frame_group = int(query_frame_group_match.group(1))
        seed_mode = seed_mode[: query_frame_group_match.start()]
    fp8_score_dot = seed_mode.endswith("_fp8score")
    if fp8_score_dot:
        seed_mode = seed_mode[:-9]
    fp8_merge_k_dot = False
    fp8_merge_v_dot = False
    for fp8_merge_suffix, merge_k_flag, merge_v_flag in (
        ("_fp8mkv", True, True),
        ("_fp8mk", True, False),
        ("_fp8mv", False, True),
    ):
        if seed_mode.endswith(fp8_merge_suffix):
            fp8_merge_k_dot = merge_k_flag
            fp8_merge_v_dot = merge_v_flag
            seed_mode = seed_mode[: -len(fp8_merge_suffix)]
            break
    direct_bulk_normalization = seed_mode.endswith("_directbulk")
    if direct_bulk_normalization:
        seed_mode = seed_mode[:-11]
    reuse_partition_exp = seed_mode.endswith("_reuseexp")
    if reuse_partition_exp:
        seed_mode = seed_mode[:-9]
    compressed_fp8_kv = seed_mode.endswith("_cfp8")
    if compressed_fp8_kv:
        seed_mode = seed_mode[:-5]
    scaled_fp8_kv = seed_mode.endswith("_fp8")
    if scaled_fp8_kv:
        seed_mode = seed_mode[:-4]
    gaussian_partition_routing = seed_mode.endswith("_gaussianroute")
    if gaussian_partition_routing:
        seed_mode = seed_mode[:-14]
    mean_centered_routing = seed_mode.endswith("_meanroute")
    if mean_centered_routing:
        seed_mode = seed_mode[:-10]
    packed_singleton_kv = seed_mode.endswith("_pack")
    if packed_singleton_kv:
        seed_mode = seed_mode[:-5]
    query_column_major = packed_singleton_kv or seed_mode.endswith("_qm")
    if not packed_singleton_kv and query_column_major:
        seed_mode = seed_mode[:-3]
    query_column_primary_row_mod4_omit2 = seed_mode.endswith("_pr4o2")
    if query_column_primary_row_mod4_omit2:
        seed_mode = seed_mode[:-6]
    query_column_secondary_frame_stride = 1
    secondary_frame_match = re.search(r"_sf(2|3|4)$", seed_mode)
    if secondary_frame_match is not None:
        query_column_secondary_frame_stride = int(
            secondary_frame_match.group(1)
        )
        seed_mode = seed_mode[: secondary_frame_match.start()]
    singleton_residual = 0
    singleton_selector = "routing_leverage"
    singleton_residual_match = re.search(r"_s([rmcq])([1234])$", seed_mode)
    if singleton_residual_match is not None:
        singleton_selector = {
            "r": "routing_leverage",
            "m": "mass",
            "c": "cluster_mass",
            "q": "query_column",
        }[singleton_residual_match.group(1)]
        singleton_residual = int(singleton_residual_match.group(2))
        seed_mode = seed_mode[: singleton_residual_match.start()]
    candidate_geometry_count = None
    candidate_geometry_match = re.search(r"_cg([0-9]+)$", seed_mode)
    if candidate_geometry_match is not None:
        candidate_geometry_count = int(candidate_geometry_match.group(1))
        seed_mode = seed_mode[: candidate_geometry_match.start()]
    temporal_simpson3 = seed_mode.endswith("_ts3")
    if temporal_simpson3:
        seed_mode = seed_mode[:-4]
    seed_mode, probe_policy = _strip_hellinger_probe_policy(seed_mode)
    first_frame_global_routing = seed_mode.endswith("_ffg")
    if first_frame_global_routing:
        seed_mode = seed_mode[:-4]
    router_frame_override = None
    router_frame_match = re.search(r"_rf([0-9]|1[0-9]|20)$", seed_mode)
    if router_frame_match is not None:
        router_frame_override = int(router_frame_match.group(1))
        seed_mode = seed_mode[: router_frame_match.start()]
    hellinger_selection_mode = "gs"
    for selection_suffix, selection_name in (
        ("_pair_m4", "pair_first_m4"),
        ("_pair_m8", "pair_first_m8"),
        ("_chol", "pivoted_cholesky"),
        ("_two_stage", "two_stage"),
        ("_anchor_top3", "anchor_top3"),
    ):
        if seed_mode.endswith(selection_suffix):
            hellinger_selection_mode = selection_name
            seed_mode = seed_mode[: -len(selection_suffix)]
            break
    logit_variance_bias_scale = 0.0
    logit_variance_match = re.search(
        r"_lv([0-9]+(?:p[0-9]+)?)$", seed_mode
    )
    if logit_variance_match is not None:
        logit_variance_bias_scale = float(
            logit_variance_match.group(1).replace("p", ".")
        )
        seed_mode = seed_mode[: logit_variance_match.start()]
    force_local_minus_one = seed_mode.endswith("_lm1")
    if force_local_minus_one:
        seed_mode = seed_mode[:-4]
    temporal_refine_suffixes = {
        "_lr2p8m1": False,
        "_lc8p8m1": True,
    }
    temporal_refine_suffix = next(
        (suffix for suffix in temporal_refine_suffixes if seed_mode.endswith(suffix)),
        "",
    )
    temporal_local_refine = bool(temporal_refine_suffix)
    temporal_compact_refine = temporal_refine_suffixes.get(
        temporal_refine_suffix, False
    )
    entropy_mass_rule = None
    entropy_global_probes = 256
    entropy_mass_match = re.fullmatch(
        r"entropy_(mass_span|partition_mass|joint_mass_span)_c([0-9]+)_p([0-9]+)_g([0-9]+)_j([0-9]+)_t1",
        seed_mode,
    )
    if entropy_mass_match is not None:
        entropy_mass_rule = entropy_mass_match.group(1)
        entropy_global_probes = int(entropy_mass_match.group(4))
        seed_mode = (
            f"hellinger_span_alpha0_c{entropy_mass_match.group(2)}"
            f"_p{entropy_mass_match.group(3)}_a2_j{entropy_mass_match.group(5)}_t1"
        )
    hellinger_seed_mode = (
        seed_mode[: -len(temporal_refine_suffix)]
        if temporal_local_refine
        else seed_mode
    )
    entropy_seed_match = re.fullmatch(
        r"entropy_reverse_kl_mass_c([0-9]+)_t1", seed_mode
    )
    if entropy_seed_match is not None:
        if (
            use_reference
            or h_reduce != 1
            or w_reduce != 1
            or (h, w) != (45, 80)
            or query_blocks.size(1) != 21
            or int(query_outer_chunk) != 21
            or seed_query_blocks is not None
            or exact_intra_frame_attention
            or exact_neighbor_window_columns
            or exact_query_rows
            or first_frame_full_attention
            or full_attention_anchor_time_count
            or dense_heads
            or singleton_residual
        ):
            raise ValueError(
                "entropy reverse-KL oracle requires native 45x80 all-layer "
                "Triton C4 without dense output rescue paths"
            )
        candidates = int(entropy_seed_match.group(1))
        router_frame = 10 if router_frame_override is None else router_frame_override
        if not 0 <= router_frame < query_blocks.size(1):
            raise ValueError("entropy reverse-KL router frame is outside the video")
        router_table = mass_weighted_reverse_kl_seed_rows(
            query_blocks[:, router_frame : router_frame + 1],
            key_blocks,
            sm_scale,
            candidate_count=candidates,
            candidate_geometry_count=(candidate_geometry_count or candidates),
            num_seeds=4,
            head_chunk=min(4, query_blocks.size(4)),
            block_chunk=21,
            output_dtype=torch.uint8,
        )
        output = torch.empty_like(query_blocks)
        heads = query_blocks.size(4)
        selector_head_chunk = (
            heads if not head_chunk or head_chunk >= heads else int(head_chunk)
        )
        for h_start in range(0, heads, selector_head_chunk):
            h_end = min(h_start + selector_head_chunk, heads)
            shared_seed_rows = (
                router_table[..., h_start:h_end, :]
                .permute(0, 1, 4, 5, 2, 3, 6)
                .contiguous()
            )
            hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks(
                query_blocks,
                key_blocks,
                value_blocks,
                shared_seed_rows,
                sm_scale,
                output=output,
                head_start=h_start,
                head_count=h_end - h_start,
                logit_variance_bias_scale=logit_variance_bias_scale,
            )
        return _hard_c4_blocks_to_video(output)
    hellinger_span = parse_hellinger_span_seed_mode(hellinger_seed_mode)
    if hellinger_span is not None:
        if use_reference:
            raise ValueError(
                "Hellinger-span generation currently requires hard_route_backend='hard_c4'."
            )
        (
            alpha,
            candidates,
            probes,
            anchors,
            block_j,
            routing_head_group,
            routing_head_offset,
            temporal_frame_count,
        ) = hellinger_span
        if probes % 2:
            raise ValueError("Hellinger-span requires an even probe count")
        uniform_only_probes = _use_uniform_hellinger_probes(
            probes, probe_policy
        )
        if entropy_mass_rule == "joint_mass_span" and (
            temporal_frame_count != 1
            or routing_head_group != 1
            or router_frame_override is not None
            or temporal_simpson3
            or first_frame_global_routing
            or first_frame_full_attention
            or not uniform_only_probes
        ):
            raise ValueError(
                "joint-frame mass selection requires per-head shared-table t1 "
                "routing, fixed midpoint frames, Uniform16 probes, and no dense rescue"
            )
        if router_frame_override is not None and temporal_frame_count != 1:
            raise ValueError("an explicit Hellinger router frame requires t1 routing")
        if temporal_simpson3 and temporal_frame_count != 1:
            raise ValueError("a temporal-Simpson router requires t1 routing")
        if first_frame_global_routing and (
            temporal_frame_count != 1
            or router_frame_override != 0
            or routing_head_group != 1
        ):
            raise ValueError(
                "first-frame global Hellinger routing requires per-head t1_rf0 routing"
            )
        if temporal_local_refine and (
            routing_head_group != 1
            or temporal_frame_count != 1
            or first_frame_full_attention
        ):
            raise ValueError(
                "temporal local refinement requires per-head t1 routing and no dense rescue"
            )
        output = torch.empty_like(query_blocks)
        heads = query_blocks.size(4)
        selector_head_chunk = (
            heads if not head_chunk or head_chunk >= heads else int(head_chunk)
        )
        if routing_head_group > 1 or temporal_frame_count:
            if (h, w) != (45, 80) or h_reduce != 1 or w_reduce != 1:
                raise ValueError(
                    "routing-head Hellinger C4 requires the native 45x80 grid"
                )
            if query_blocks.size(1) != 21:
                raise ValueError(
                    "routing-head Hellinger C4 requires all 21 latent frames"
                )
            if any(
                (
                    exact_intra_frame_attention,
                    exact_neighbor_window_columns,
                    exact_query_rows,
                    full_attention_anchor_time_count,
                    dense_heads,
                )
            ):
                raise ValueError(
                    "routing-head Hellinger C4 does not compose with rescue paths"
                )

            group_starts = torch.arange(
                0, heads, routing_head_group, device=query_blocks.device
            )
            router_heads = torch.minimum(
                group_starts + routing_head_offset,
                torch.full_like(group_starts, heads - 1),
            )
            if temporal_frame_count:
                temporal_frame_count = min(temporal_frame_count, query_blocks.size(1))
                if router_frame_override is not None:
                    if router_frame_override >= query_blocks.size(1):
                        raise ValueError("Hellinger router frame is outside the video")
                    router_frame_values = [router_frame_override]
                elif temporal_frame_count == 4 and query_blocks.size(1) == 21:
                    # Cover both temporal endpoints; QKV screening shows that
                    # centered T4 routers under-represent boundary frames.
                    router_frame_values = [0, 7, 13, 20]
                else:
                    router_frame_values = [
                        round(
                            (slot + 0.5)
                            * query_blocks.size(1)
                            / temporal_frame_count
                            - 0.5
                        )
                        for slot in range(temporal_frame_count)
                    ]
                router_frames = torch.tensor(
                    router_frame_values, device=query_blocks.device, dtype=torch.long
                )
                frame_distance = (
                    torch.arange(query_blocks.size(1), device=query_blocks.device)[:, None]
                    - router_frames[None, :]
                ).abs()
                frame_to_router = frame_distance.argmin(dim=1)
            else:
                router_frames = torch.arange(
                    query_blocks.size(1), device=query_blocks.device
                )
                frame_to_router = router_frames
            identity_router_heads = (
                routing_head_group == 1
                and routing_head_offset == 0
                and router_heads.numel() == heads
            )
            if entropy_mass_rule == "joint_mass_span":
                # t1 means one shared row-index table, not one sampled frame.
                # Two midpoint query frames construct that table jointly; the
                # current-frame merge and global route below stay unchanged.
                router_query = query_blocks[:, 5:16:10]
            elif temporal_frame_count == 1:
                router_frame = int(router_frame_values[0])
                router_query = query_blocks[:, router_frame : router_frame + 1]
                if temporal_simpson3:
                    # Simpson quadrature approximates the continuous-time
                    # reverse-KL barycenter with fixed 1:4:1 endpoint/center
                    # weights, avoiding a tunable temporal mixing parameter.
                    router_query = temporal_simpson_query_triton(
                        query_blocks,
                        center_frame=router_frame,
                    )
            else:
                router_query = query_blocks.index_select(1, router_frames)
            if not identity_router_heads:
                router_query = router_query.index_select(4, router_heads)
                router_key = key_blocks.index_select(4, router_heads)
            else:
                # The common per-head T1 path already uses every head in its
                # native order. Avoid copying the full 21-frame K tensor.
                router_key = key_blocks
            if first_frame_global_routing:
                # The global profile Gram is the selector bottleneck. A 16x16
                # tile amortizes launches while FP16 profiles use Tensor Cores;
                # normalization and the greedy selection remain FP32.
                router_table = first_frame_global_hellinger_seed_rows(
                    router_query,
                    router_key,
                    sm_scale,
                    alpha=alpha,
                    candidate_count=candidates,
                    candidate_geometry_count=candidate_geometry_count,
                    num_seeds=4,
                    head_chunk=min(16, heads),
                    column_chunk=min(16, w),
                    profile_dtype=torch.float16,
                    output_dtype=torch.uint8,
                )
            elif entropy_mass_rule is not None:
                entropy_selector = (
                    joint_frame_mass_seed_rows_triton
                    if entropy_mass_rule == "joint_mass_span"
                    else entropy_mass_seed_rows_triton
                    if entropy_mass_rule == "mass_span"
                    else partition_mass_seed_rows_triton
                )
                entropy_launch_options = {
                    "gs_num_warps": 8 if entropy_mass_rule == "partition_mass" else 4,
                }
                if (
                    entropy_mass_rule == "partition_mass"
                    and requested_seed_mode
                    == "entropy_partition_mass_c16_p16_g256_j4_t1_rf0_u16_cg16"
                    and (h, w) == (45, 80)
                    and query_blocks.size(1) == 21
                ):
                    # Match the measured RF0 candidate exactly. Other modes
                    # retain their historical launches; the cell/deficit
                    # mathematics and the downstream merge/route are unchanged.
                    entropy_launch_options.update(gs_num_warps=4, reuse_owner_logp=True)
                router_table = entropy_selector(
                    router_query,
                    router_key,
                    sm_scale,
                    candidate_count=candidates,
                    candidate_geometry_count=(
                        candidate_geometry_count or candidates
                        if entropy_mass_rule == "joint_mass_span"
                        else candidate_geometry_count
                    ),
                    block_j=block_j,
                    probe_count=probes,
                    global_probe_count=entropy_global_probes,
                    output_dtype=torch.uint8,
                    **entropy_launch_options,
                )
            else:
                router_table = hellinger_span_seed_rows_triton(
                    router_query,
                    router_key,
                    sm_scale,
                    alpha=alpha,
                    candidate_count=candidates,
                    candidate_geometry_count=candidate_geometry_count,
                    top_probes=0 if uniform_only_probes else probes // 2,
                    uniform_probes=probes if uniform_only_probes else probes // 2,
                    probe_anchor_count=anchors,
                    block_j=block_j,
                    group_tile=32 if tuned_hellinger_launch else 16,
                    output_dtype=torch.uint8,
                    gs_num_warps=8 if tuned_hellinger_launch else 4,
                    selection_mode=hellinger_selection_mode,
                    force_local_minus_one=force_local_minus_one,
                )
            if common_c_ablation:
                for h_start in range(0, heads, selector_head_chunk):
                    h_end = min(h_start + selector_head_chunk, heads)
                    hard_c_720p_native_temporal_seed_rows_attention_triton_blocks(
                        query_blocks,
                        key_blocks,
                        value_blocks,
                        router_table,
                        4,
                        sm_scale,
                        output=output,
                        head_start=h_start,
                        head_count=h_end - h_start,
                    )
                return _hard_c4_blocks_to_video(output)
            if temporal_compact_refine:
                for h_start in range(0, heads, selector_head_chunk):
                    h_end = min(h_start + selector_head_chunk, heads)
                    hard_c4_720p_temporal_compact_attention_triton_blocks(
                        query_blocks,
                        key_blocks,
                        value_blocks,
                        router_table,
                        sm_scale,
                        output=output,
                        head_start=h_start,
                        head_count=h_end - h_start,
                        alpha=alpha,
                    )
                return _hard_c4_blocks_to_video(output)
            if temporal_local_refine:
                active_table = temporal_local_refine_seed_rows_triton(
                    query_blocks,
                    key_blocks,
                    router_table,
                    sm_scale,
                    radius=2,
                    probes=8,
                    alpha=alpha,
                    block_j=block_j,
                    compact=False,
                )
                native_seed_layout = False
                shared_seed_layout = False
            elif 1 <= temporal_frame_count <= 4:
                active_table = router_table
                if routing_head_group > 1:
                    head_to_router = torch.arange(
                        heads, device=query_blocks.device
                    ).div(routing_head_group, rounding_mode="floor")
                    active_table = active_table.index_select(5, head_to_router)
                native_seed_layout = False
                shared_seed_layout = True
            else:
                head_to_router = torch.arange(
                    heads, device=query_blocks.device
                ).div(routing_head_group, rounding_mode="floor")
                active_table = router_table.index_select(
                    1, frame_to_router
                ).index_select(5, head_to_router)
                native_seed_layout = False
                shared_seed_layout = False
            for h_start in range(0, heads, selector_head_chunk):
                h_end = min(h_start + selector_head_chunk, heads)
                if shared_seed_layout:
                    if singleton_residual and direct_native_seed_layout:
                        shared_seed_rows = active_table
                    else:
                        shared_seed_rows = (
                            active_table[..., h_start:h_end, :]
                            .permute(0, 1, 4, 5, 2, 3, 6)
                            .contiguous()
                        )
                    if singleton_residual:
                        hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks(
                            query_blocks,
                            key_blocks,
                            value_blocks,
                            shared_seed_rows,
                            sm_scale,
                            output=output,
                            head_start=h_start,
                            head_count=h_end - h_start,
                            top_r=singleton_residual,
                            selector=singleton_selector,
                            compressed_fp8_kv=compressed_fp8_kv,
                            scaled_fp8_kv=scaled_fp8_kv,
                            mean_centered_routing=mean_centered_routing,
                            gaussian_partition_routing=(
                                gaussian_partition_routing
                            ),
                            reuse_partition_exp=reuse_partition_exp,
                            direct_bulk_normalization=(
                                direct_bulk_normalization
                            ),
                            fp8_score_dot=fp8_score_dot,
                            fp8_merge_k_dot=fp8_merge_k_dot,
                            fp8_merge_v_dot=fp8_merge_v_dot,
                            query_frame_group=query_frame_group,
                            split_merge=split_merge,
                            materialize_singleton=materialize_singleton,
                            query_column_major=query_column_major,
                            query_column_secondary_frame_stride=(
                                query_column_secondary_frame_stride
                            ),
                            query_column_primary_row_mod4_omit2=(
                                query_column_primary_row_mod4_omit2
                            ),
                            packed_singleton_kv=packed_singleton_kv,
                            seed_rows_native_layout=direct_native_seed_layout,
                            merge_frame_group=merge_frame_group,
                            merge_col_group=merge_col_group,
                            merge_loop_stages=merge_loop_stages,
                            merge_disable_licm=merge_disable_licm,
                            split_kv_accum=split_kv_accum,
                        )
                    else:
                        hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks(
                            query_blocks,
                            key_blocks,
                            value_blocks,
                            shared_seed_rows,
                            sm_scale,
                            output=output,
                            head_start=h_start,
                            head_count=h_end - h_start,
                            logit_variance_bias_scale=logit_variance_bias_scale,
                        )
                    continue
                if native_seed_layout:
                    hard_c4_720p_native_seed_rows_attention_triton_blocks(
                        query_blocks,
                        key_blocks,
                        value_blocks,
                        active_table,
                        sm_scale,
                        output=output,
                        head_start=h_start,
                        head_count=h_end - h_start,
                    )
                    continue
                active_seed_rows = (
                    active_table[..., h_start:h_end, :]
                    .permute(0, 1, 4, 5, 2, 3, 6)
                    .contiguous()
                )
                hard_c4_720p_seed_rows_attention_triton_blocks(
                    query_blocks,
                    key_blocks,
                    value_blocks,
                    active_seed_rows,
                    sm_scale,
                    output=output,
                    head_start=h_start,
                    head_count=h_end - h_start,
                )
            output_video = _hard_c4_blocks_to_video(output)
            if first_frame_full_attention:
                frame_tokens = h * w
                output_video[:, :frame_tokens] = flash_attention(
                    query[:, :frame_tokens],
                    key,
                    value,
                    softmax_scale=sm_scale,
                )
            return output_video

        for start in range(0, query_blocks.size(1), query_outer_chunk):
            end = min(start + query_outer_chunk, query_blocks.size(1))
            query_chunk = query_blocks[:, start:end]
            full_table = hellinger_span_seed_rows_triton(
                query_chunk,
                key_blocks,
                sm_scale,
                alpha=alpha,
                candidate_count=candidates,
                candidate_geometry_count=candidate_geometry_count,
                top_probes=0 if uniform_only_probes else probes // 2,
                uniform_probes=probes if uniform_only_probes else probes // 2,
                probe_anchor_count=anchors,
                block_j=block_j,
                selection_mode=hellinger_selection_mode,
                force_local_minus_one=force_local_minus_one,
            )
            for h_start in range(0, heads, selector_head_chunk):
                h_end = min(h_start + selector_head_chunk, heads)
                if singleton_residual:
                    active_seed_rows = (
                        full_table[..., h_start:h_end, :]
                        .permute(0, 1, 4, 5, 2, 3, 6)
                        .contiguous()
                    )
                    hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks(
                        query_chunk,
                        key_blocks,
                        value_blocks,
                        active_seed_rows,
                        sm_scale,
                        output=output[:, start:end],
                        head_start=h_start,
                        head_count=h_end - h_start,
                        top_r=singleton_residual,
                        query_outer_offset=start,
                        selector=singleton_selector,
                        compressed_fp8_kv=compressed_fp8_kv,
                        scaled_fp8_kv=scaled_fp8_kv,
                        mean_centered_routing=mean_centered_routing,
                        gaussian_partition_routing=(
                            gaussian_partition_routing
                        ),
                        reuse_partition_exp=reuse_partition_exp,
                        direct_bulk_normalization=direct_bulk_normalization,
                        fp8_score_dot=fp8_score_dot,
                        fp8_merge_k_dot=fp8_merge_k_dot,
                        fp8_merge_v_dot=fp8_merge_v_dot,
                        query_frame_group=query_frame_group,
                        split_merge=split_merge,
                        materialize_singleton=materialize_singleton,
                        query_column_major=query_column_major,
                        query_column_secondary_frame_stride=(
                            query_column_secondary_frame_stride
                        ),
                        query_column_primary_row_mod4_omit2=(
                            query_column_primary_row_mod4_omit2
                        ),
                        packed_singleton_kv=packed_singleton_kv,
                        merge_frame_group=merge_frame_group,
                        merge_col_group=merge_col_group,
                        split_kv_accum=split_kv_accum,
                    )
                    continue
                output[:, start:end, ..., h_start:h_end, :] = (
                    adaptive_seed_attention_triton(
                        query_chunk[..., h_start:h_end, :],
                        key_blocks[..., h_start:h_end, :],
                        value_blocks[..., h_start:h_end, :],
                        full_table[..., h_start:h_end, :],
                        scale=sm_scale,
                    )
                )
        return _hard_c4_blocks_to_video(output)
    if seed_mode == "query_l2_kcenter_c6_d32":
        if (
            use_reference
            or h_reduce != 1
            or (h, w) != (45, 80)
            or query_blocks.size(1) != 21
            or int(query_outer_chunk) != 21
            or seed_query_blocks is not None
            or exact_intra_frame_attention
            or exact_neighbor_window_columns
            or exact_query_rows
            or first_frame_full_attention
            or full_attention_anchor_time_count
            or dense_heads
        ):
            raise ValueError(
                "query L2 k-center requires pure all-layer native 45x80 "
                "Triton C4 with all 21 query frames"
            )
        output = torch.empty_like(query_blocks)
        base_seed_rows = query_l2_kcenter_seed_rows_triton(
            query_blocks,
            candidate_count=6,
            sketch_dimensions=32,
            block_j=8,
        )
        native_seed_rows = (
            base_seed_rows.unsqueeze(2)
            .expand(-1, -1, key_blocks.size(1), -1, -1, -1, -1)
            .contiguous()
        )
        active_chunk = (
            query_blocks.size(4)
            if not head_chunk or head_chunk >= query_blocks.size(4)
            else int(head_chunk)
        )
        for h_start in range(0, query_blocks.size(4), active_chunk):
            h_count = min(active_chunk, query_blocks.size(4) - h_start)
            hard_c4_720p_native_seed_rows_attention_triton_blocks(
                query_blocks,
                key_blocks,
                value_blocks,
                native_seed_rows,
                sm_scale,
                output=output,
                head_start=h_start,
                head_count=h_count,
            )
        return _hard_c4_blocks_to_video(output)
    if seed_mode == "fixed_contiguous":
        if (
            use_reference
            or h_reduce != 1
            or (h, w) != (45, 80)
            or query_blocks.size(1) != 21
            or int(query_outer_chunk) != 21
            or singleton_residual != 1
            or singleton_selector != "query_column"
            or any(
                (
                    exact_intra_frame_attention,
                    exact_neighbor_window_columns,
                    exact_query_rows,
                    first_frame_full_attention,
                    full_attention_anchor_time_count,
                    full_attention_anchor_row_count,
                    full_attention_anchor_width_count,
                    dense_heads,
                )
            )
        ):
            raise ValueError(
                "fixed_contiguous_sq1 requires pure all-layer native 45x80 "
                "Triton C4 with all 21 query frames"
            )
        output = torch.empty_like(query_blocks)
        heads = query_blocks.size(4)
        active_chunk = heads if not head_chunk or head_chunk >= heads else int(head_chunk)
        for h_start in range(0, heads, active_chunk):
            h_count = min(active_chunk, heads - h_start)
            shared_seed_rows = _fixed_contiguous_seed_rows(
                query_blocks.device,
                query_blocks.size(0),
                w,
                h_count,
                key_blocks.size(1),
                h,
                4,
            )
            hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks(
                query_blocks,
                key_blocks,
                value_blocks,
                shared_seed_rows,
                sm_scale,
                output=output,
                head_start=h_start,
                head_count=h_count,
                top_r=1,
                selector="query_column",
                compressed_fp8_kv=compressed_fp8_kv,
                scaled_fp8_kv=scaled_fp8_kv,
                mean_centered_routing=mean_centered_routing,
                gaussian_partition_routing=gaussian_partition_routing,
                reuse_partition_exp=reuse_partition_exp,
                direct_bulk_normalization=direct_bulk_normalization,
                fp8_score_dot=fp8_score_dot,
                fp8_merge_k_dot=fp8_merge_k_dot,
                fp8_merge_v_dot=fp8_merge_v_dot,
                query_frame_group=query_frame_group,
                split_merge=split_merge,
                materialize_singleton=materialize_singleton,
                query_column_major=query_column_major,
                query_column_secondary_frame_stride=(
                    query_column_secondary_frame_stride
                ),
                query_column_primary_row_mod4_omit2=(
                    query_column_primary_row_mod4_omit2
                ),
                packed_singleton_kv=packed_singleton_kv,
                merge_frame_group=merge_frame_group,
                merge_col_group=merge_col_group,
                split_kv_accum=split_kv_accum,
            )
        return _hard_c4_blocks_to_video(output)
    if singleton_residual:
        raise ValueError(
            "singleton residual suffix requires a temporally shared Hellinger mode"
        )
    # weighted_span_alpha<A>_pool<N>: importance-weighted response span. Unlike
    # query_response_span15_s8 it carries an importance weight and runs at any
    # grid shape, so it is dispatched before the fixed-name table below.
    # ``_tN`` computes the selector from N temporal router frames and reuses
    # each table for its nearest query frames. ``_j4`` trades a small amount
    # of selector fidelity for the latency frontier; ``_j2`` remains the
    # default quality setting. The merge and route stages still replace
    # attention in every layer and frame, so this is a pure all-layer GMA path.
    weighted_temporal_count = 0
    weighted_temporal_mean = False
    weighted_temporal_consensus = False
    weighted_temporal_probe_shared = False
    weighted_head_group = 1
    weighted_head_offset = 0
    weighted_routing_heads = False
    weighted_generic_adaptive = False
    weighted_selector_exact = False
    weighted_key_frame_count = 0
    weighted_track_radius = 0
    weighted_block_j = 2
    weighted_probe_count = 16
    weighted_value_dims = 32
    weighted_denoise_cache_steps = 0
    weighted_denoise_cache_late = False
    weighted_denoise_cache_match = re.search(r"_dc(2l|2|3|4)$", seed_mode)
    if weighted_denoise_cache_match is not None:
        cache_mode = weighted_denoise_cache_match.group(1)
        weighted_denoise_cache_late = cache_mode == "2l"
        weighted_denoise_cache_steps = (
            2 if weighted_denoise_cache_late else int(cache_mode)
        )
        seed_mode = seed_mode[: weighted_denoise_cache_match.start()]
    weighted_key_frame_match = re.search(r"_kf(3|5|7|11)$", seed_mode)
    if weighted_key_frame_match is not None:
        weighted_key_frame_count = int(weighted_key_frame_match.group(1))
        seed_mode = seed_mode[: weighted_key_frame_match.start()]
    weighted_head_match = re.search(
        r"_hg(1|2|4|5|8|10)o([0-9]+)_j(2|4)"
        r"(?:_p(4|8|16)_v(8|16|32))?$",
        seed_mode,
    )
    if weighted_head_match is not None:
        weighted_routing_heads = True
        weighted_head_group = int(weighted_head_match.group(1))
        weighted_head_offset = int(weighted_head_match.group(2))
        weighted_block_j = int(weighted_head_match.group(3))
        if weighted_head_match.group(4) is not None:
            weighted_probe_count = int(weighted_head_match.group(4))
            weighted_value_dims = int(weighted_head_match.group(5))
        seed_mode = seed_mode[: weighted_head_match.start()]
    if seed_mode.endswith("_ga"):
        weighted_generic_adaptive = True
        seed_mode = seed_mode[:-3]
    if seed_mode.endswith("_sx"):
        weighted_selector_exact = True
        seed_mode = seed_mode[:-3]
    weighted_track_match = re.search(r"_tr([1234])$", seed_mode)
    if weighted_track_match is not None:
        weighted_track_radius = int(weighted_track_match.group(1))
        seed_mode = seed_mode[: weighted_track_match.start()]
    weighted_selector_match = re.search(
        r"_p(4|8|16)_v(8|16|32)$", seed_mode
    )
    if weighted_selector_match is not None:
        weighted_probe_count = int(weighted_selector_match.group(1))
        weighted_value_dims = int(weighted_selector_match.group(2))
        seed_mode = seed_mode[: weighted_selector_match.start()]
    weighted_temporal_probe_match = re.search(
        r"_tp3(?:_j(2|4))?$", seed_mode
    )
    if weighted_temporal_probe_match is not None:
        weighted_temporal_probe_shared = True
        weighted_block_j = int(weighted_temporal_probe_match.group(1) or 4)
        seed_mode = seed_mode[: weighted_temporal_probe_match.start()]
    weighted_temporal_match = re.search(
        r"_(?:t([1-4])|(tm)|(tc2))(?:_j(2|4))?$", seed_mode
    )
    if weighted_temporal_match is not None:
        weighted_temporal_mean = weighted_temporal_match.group(2) is not None
        weighted_temporal_consensus = weighted_temporal_match.group(3) is not None
        weighted_temporal_count = (
            2
            if weighted_temporal_consensus
            else (1 if weighted_temporal_mean else int(weighted_temporal_match.group(1)))
        )
        weighted_block_j = int(weighted_temporal_match.group(4) or 2)
        seed_mode = seed_mode[: weighted_temporal_match.start()]
    weighted_span = parse_weighted_span_seed_mode(seed_mode)
    if weighted_span is not None:
        alpha, pool = weighted_span
        output = torch.empty_like(query_blocks)
        heads = query_blocks.size(4)
        span_head_chunk = (
            heads if not head_chunk or head_chunk >= heads else int(head_chunk)
        )
        if weighted_routing_heads:
            if not 0 <= weighted_head_offset < weighted_head_group:
                raise ValueError("weighted-span routing-head offset is outside its group")
            if heads % weighted_head_group:
                raise ValueError(
                    "weighted-span routing-head groups must divide the head count"
                )
            if (
                (h, w) != (45, 80)
                or h_reduce != 1
                or w_reduce != 1
                or query_blocks.size(1) != 21
                or int(query_outer_chunk) != 21
            ):
                raise ValueError(
                    "weighted-span routing heads require the full native 21x45x80 grid."
                )
            if seed_query_blocks is not None or any(
                (
                    exact_intra_frame_attention,
                    exact_neighbor_window_columns,
                    exact_query_rows,
                    first_frame_full_attention,
                    full_attention_anchor_time_count,
                    dense_heads,
                )
            ):
                raise ValueError(
                    "weighted-span routing heads do not compose with dense rescue paths."
                )
            router_heads = torch.arange(
                weighted_head_offset,
                heads,
                weighted_head_group,
                device=query_blocks.device,
            )
            cache_key = "weighted_span_routing_heads"
            if weighted_denoise_cache_steps:
                if seed_table_cache is None:
                    raise ValueError(
                        "denoise-cached weighted-span requires a mutable seed cache"
                    )
                if cache_timestep is None:
                    raise ValueError(
                        "denoise-cached weighted-span requires the current timestep"
                    )
            refresh_indices = (
                ({0, 3} if weighted_denoise_cache_late else {0, 2})
                if weighted_denoise_cache_steps == 2
                else (
                    {0, 1, 2}
                    if weighted_denoise_cache_steps == 3
                    else {0}
                )
            )
            cache_step_key = f"{cache_key}_step"
            cache_timestep_key = f"{cache_key}_timestep"
            current_timestep = (
                int(cache_timestep) if cache_timestep is not None else None
            )
            previous_timestep = (
                seed_table_cache.get(cache_timestep_key)
                if seed_table_cache is not None
                else None
            )
            if weighted_denoise_cache_steps:
                if current_timestep >= 999 or previous_timestep is None:
                    cache_step = 0
                elif current_timestep != previous_timestep:
                    cache_step = int(seed_table_cache.get(cache_step_key, 0)) + 1
                else:
                    cache_step = int(seed_table_cache.get(cache_step_key, 0))
                seed_table_cache[cache_step_key] = cache_step
                seed_table_cache[cache_timestep_key] = current_timestep
            else:
                cache_step = 0
            refresh_cache = (
                not weighted_denoise_cache_steps
                or cache_step in refresh_indices
                or cache_key not in seed_table_cache
            )
            if refresh_cache:
                selector_key_blocks = key_blocks
                selector_value_blocks = value_blocks
                key_frame_map = None
                if weighted_key_frame_count:
                    key_frame_values = [
                        round(
                            (slot + 0.5)
                            * key_blocks.size(1)
                            / weighted_key_frame_count
                            - 0.5
                        )
                        for slot in range(weighted_key_frame_count)
                    ]
                    key_frames = torch.tensor(
                        key_frame_values,
                        device=key_blocks.device,
                        dtype=torch.long,
                    )
                    selector_key_blocks = key_blocks.index_select(1, key_frames)
                    selector_value_blocks = value_blocks.index_select(1, key_frames)
                    key_frame_map = (
                        torch.arange(key_blocks.size(1), device=key_blocks.device)[:, None]
                        - key_frames[None, :]
                    ).abs().argmin(dim=1)
                router_table = span_seed_rows_triton_split(
                    query_blocks.index_select(4, router_heads),
                    selector_key_blocks.index_select(4, router_heads),
                    selector_value_blocks.index_select(4, router_heads),
                    sm_scale,
                    alpha=alpha,
                    candidate_count=pool,
                    top_probes=weighted_probe_count // 2,
                    uniform_probes=weighted_probe_count // 2,
                    value_dims=weighted_value_dims,
                    block_j=weighted_block_j,
                    group_tile=16,
                    # Generic-adaptive is the quality reference. Preserve the
                    # original selector's full key-frame reduction and warp
                    # geometry so this branch isolates merge semantics rather
                    # than changing two numerical components at once.
                    key_outer_tile=(
                        None
                        if weighted_generic_adaptive or weighted_selector_exact
                        else 7
                    ),
                    output_dtype=(
                        torch.int32 if weighted_generic_adaptive else torch.uint8
                    ),
                    probe_num_warps=4,
                    gs_num_warps=(
                        4
                        if weighted_generic_adaptive or weighted_selector_exact
                        else 8
                    ),
                )
                if key_frame_map is not None:
                    router_table = router_table.index_select(2, key_frame_map)
                if weighted_denoise_cache_steps:
                    seed_table_cache[cache_key] = router_table
            else:
                router_table = seed_table_cache[cache_key]
            if weighted_head_group == 1:
                native_table = router_table
            else:
                head_to_router = torch.arange(
                    heads, device=query_blocks.device
                ).div(weighted_head_group, rounding_mode="floor")
                native_table = router_table.index_select(5, head_to_router)
            for h_start in range(0, heads, span_head_chunk):
                h_end = min(h_start + span_head_chunk, heads)
                if weighted_generic_adaptive:
                    output[..., h_start:h_end, :] = adaptive_seed_attention_triton(
                        query_blocks[..., h_start:h_end, :],
                        key_blocks[..., h_start:h_end, :],
                        value_blocks[..., h_start:h_end, :],
                        native_table[..., h_start:h_end, :],
                        scale=sm_scale,
                    )
                else:
                    hard_c4_720p_native_seed_rows_attention_triton_blocks(
                        query_blocks,
                        key_blocks,
                        value_blocks,
                        native_table,
                        sm_scale,
                        output=output,
                        head_start=h_start,
                        head_count=h_end - h_start,
                    )
            return _hard_c4_blocks_to_video(output)
        if weighted_temporal_probe_shared:
            if use_reference:
                raise ValueError(
                    "temporal-probe weighted-span requires hard_route_backend='hard_c4'."
                )
            if (
                (h, w) != (45, 80)
                or h_reduce != 1
                or w_reduce != 1
                or query_blocks.size(1) != 21
                or int(query_outer_chunk) != 21
            ):
                raise ValueError(
                    "temporal-probe weighted-span requires the full native 21x45x80 grid."
                )
            if seed_query_blocks is not None or any(
                (
                    exact_intra_frame_attention,
                    exact_neighbor_window_columns,
                    exact_query_rows,
                    first_frame_full_attention,
                    full_attention_anchor_time_count,
                    dense_heads,
                )
            ):
                raise ValueError(
                    "temporal-probe weighted-span does not compose with dense rescue paths."
                )
            native_table = span_seed_rows_triton_temporal_probes_per_frame(
                query_blocks,
                key_blocks,
                value_blocks,
                sm_scale,
                alpha=alpha,
                candidate_count=pool,
                top_probes=weighted_probe_count // 2,
                uniform_probes=weighted_probe_count // 2,
                value_dims=weighted_value_dims,
                block_j=weighted_block_j,
                router_frames=3,
            )
            for h_start in range(0, heads, span_head_chunk):
                h_end = min(h_start + span_head_chunk, heads)
                hard_c4_720p_native_seed_rows_attention_triton_blocks(
                    query_blocks,
                    key_blocks,
                    value_blocks,
                    native_table,
                    sm_scale,
                    output=output,
                    head_start=h_start,
                    head_count=h_end - h_start,
                )
            return _hard_c4_blocks_to_video(output)
        if weighted_temporal_count:
            if use_reference:
                raise ValueError(
                    "temporally shared weighted-span requires hard_route_backend='hard_c4'."
                )
            if (
                (h, w) != (45, 80)
                or h_reduce != 1
                or w_reduce != 1
                or query_blocks.size(1) != 21
                or int(query_outer_chunk) != 21
            ):
                raise ValueError(
                    "temporally shared weighted-span requires the full native 21x45x80 grid."
                )
            if seed_query_blocks is not None or any(
                (
                    exact_intra_frame_attention,
                    exact_neighbor_window_columns,
                    exact_query_rows,
                    first_frame_full_attention,
                    full_attention_anchor_time_count,
                    dense_heads,
                )
            ):
                raise ValueError(
                    "temporally shared weighted-span does not compose with dense rescue paths."
                )
            if router_frame_override is not None and (
                weighted_temporal_count != 1 or weighted_temporal_mean
            ):
                raise ValueError(
                    "an explicit weighted-span router frame requires t1 routing"
                )
            if weighted_temporal_mean:
                router_query = query_blocks.mean(dim=1, keepdim=True)
            elif router_frame_override is not None:
                router_frame_values = [router_frame_override]
            elif weighted_temporal_count == 4:
                router_frame_values = [0, 7, 13, 20]
            else:
                router_frame_values = [
                    round(
                        (slot + 0.5)
                        * query_blocks.size(1)
                        / weighted_temporal_count
                        - 0.5
                    )
                    for slot in range(weighted_temporal_count)
                ]
            if not weighted_temporal_mean:
                if any(
                    not 0 <= router_frame < query_blocks.size(1)
                    for router_frame in router_frame_values
                ):
                    raise ValueError("weighted-span router frame is outside the video")
                router_frames = torch.tensor(
                    router_frame_values,
                    device=query_blocks.device,
                    dtype=torch.long,
                )
                router_query = query_blocks.index_select(1, router_frames)
            if weighted_track_radius and (
                weighted_temporal_count != 1 or weighted_temporal_mean
            ):
                raise ValueError(
                    "temporal query row tracking requires a concrete t1 router frame"
                )
            selector = (
                span_seed_rows_triton_temporal_consensus2
                if weighted_temporal_consensus
                else span_seed_rows_triton_split
            )
            full_table = selector(
                router_query.contiguous(),
                key_blocks,
                value_blocks,
                sm_scale,
                alpha=alpha,
                candidate_count=pool,
                top_probes=weighted_probe_count // 2,
                uniform_probes=weighted_probe_count // 2,
                value_dims=weighted_value_dims,
                block_j=weighted_block_j,
            )
            temporal_query_row_map = (
                temporal_query_row_map_triton(
                    query_blocks,
                    router_frame=router_frame_values[0],
                    radius=weighted_track_radius,
                )
                if weighted_track_radius
                else None
            )
            for h_start in range(0, heads, span_head_chunk):
                h_end = min(h_start + span_head_chunk, heads)
                shared_seed_rows = (
                    full_table[..., h_start:h_end, :]
                    .permute(0, 1, 4, 5, 2, 3, 6)
                    .contiguous()
                )
                hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks(
                    query_blocks,
                    key_blocks,
                    value_blocks,
                    shared_seed_rows,
                    sm_scale,
                    output=output,
                    head_start=h_start,
                    head_count=h_end - h_start,
                    temporal_query_row_map=temporal_query_row_map,
                )
            return _hard_c4_blocks_to_video(output)
        for start in range(0, query_blocks.size(1), query_outer_chunk):
            end = min(start + query_outer_chunk, query_blocks.size(1))
            query_chunk = query_blocks[:, start:end]
            if use_reference:
                output[:, start:end] = adaptive_hard_route_attention_pytorch(
                    query_chunk,
                    key_blocks,
                    value_blocks,
                    sm_scale,
                    query_outer_offset=start,
                    alpha=alpha,
                    candidate_count=pool,
                    head_chunk=span_head_chunk,
                )
                continue
            # Triton path: the seed table comes from the Triton selector, then
            # the merge and route kernels consume it. Selection in PyTorch is
            # roughly 46x slower and dominates the whole call, so the backend
            # only makes sense with the kernel selector.
            #
            # The selector runs ONCE over all heads: its kernel is generic over
            # the head count (every stride uses HEADS, head = pid % HEADS), its
            # int32 offsets stay far below the 2^31 limit that forces the
            # MERGE stage to chunk heads (the largest selector offset is the
            # full Q tensor, ~50M elements, vs merge's 4.2G intermediate), and
            # query_chunk/key_blocks are dim-1 slices of contiguous tensors so
            # no .contiguous() copies are needed. One launch also puts the
            # whole grid in flight at once instead of three sequential waves.
            # A 12-head launch is bit-identical to three 4-head launches
            # because each program depends only on its own head index.
            # Split probe/Gram-Schmidt selector at BLOCK_J=2: measured 63.1 ms
            # vs the fused J=4 kernel's 99.6 ms (isolated, 12 heads, A100)
            # at rel-L2 0.06670 vs 0.06789 -- strictly better on both axes.
            full_table = span_seed_rows_triton_split(
                query_chunk.contiguous(),
                key_blocks,
                value_blocks,
                sm_scale,
                alpha=alpha,
                candidate_count=pool,
            )
            # The merge/route stage still chunks heads (int32 offsets on its
            # merged K/V intermediate), so slice the table on the same
            # boundaries; an int32 table slice is a cheap copy.
            for h_start in range(0, heads, span_head_chunk):
                h_end = min(h_start + span_head_chunk, heads)
                q_heads = query_chunk[..., h_start:h_end, :]
                k_heads = key_blocks[..., h_start:h_end, :]
                v_heads = value_blocks[..., h_start:h_end, :]
                seed_table = full_table[..., h_start:h_end, :]
                output[:, start:end, ..., h_start:h_end, :] = (
                    adaptive_seed_attention_triton(
                        q_heads, k_heads, v_heads, seed_table, scale=sm_scale
                    )
                )
        return _hard_c4_blocks_to_video(output)
    if seed_mode not in {
        "local_with_ith",
        "local_centroid_p11",
        "query_response_span15_s8",
        "responsibility_centroid_top2",
        "hierarchical_mean_split_top2",
        "singleton_leverage_top2",
        "singleton_leverage_top3",
        "singleton_leverage_top4",
        "singleton_leverage_top6",
        "singleton_leverage_top8",
        "singleton_leverage_top12",
        "singleton_leverage_top16",
        "singleton_leverage_top20",
        "singleton_leverage_top24",
        "singleton_leverage_top28",
        "singleton_leverage_top32",
    }:
        raise ValueError(
            "Wan Hard C4 supports seed_mode='local_with_ith', "
            "'local_centroid_p11', 'query_response_span15_s8', "
            "'responsibility_centroid_top2', "
            "'hierarchical_mean_split_top2', or a supported "
            "'singleton_leverage_topR' budget."
        )
    use_query_centroids = seed_mode == "local_centroid_p11"
    use_query_response = seed_mode == "query_response_span15_s8"
    use_responsibility_centroids = seed_mode == "responsibility_centroid_top2"
    use_hierarchical_split = seed_mode == "hierarchical_mean_split_top2"
    singleton_top_r = {
        "singleton_leverage_top2": 2,
        "singleton_leverage_top3": 3,
        "singleton_leverage_top4": 4,
        "singleton_leverage_top6": 6,
        "singleton_leverage_top8": 8,
        "singleton_leverage_top12": 12,
        "singleton_leverage_top16": 16,
        "singleton_leverage_top20": 20,
        "singleton_leverage_top24": 24,
        "singleton_leverage_top28": 28,
        "singleton_leverage_top32": 32,
    }.get(seed_mode, 0)
    use_singleton_peeling = bool(singleton_top_r)
    if seed_query_blocks is not None and (
        use_reference
        or h_reduce != 1
        or (h, w) != (45, 80)
        or seed_mode != "local_with_ith"
        or exact_intra_frame_attention
        or exact_neighbor_window_columns
        or exact_query_rows
        or first_frame_full_attention
        or full_attention_anchor_time_count
        or dense_heads
    ):
        raise ValueError(
            "separate seed_query requires plain 45x80 Triton Hard C4"
        )
    dense_heads = tuple(sorted(int(head) for head in dense_heads))
    heads = query_blocks.size(4)
    if len(set(dense_heads)) != len(dense_heads):
        raise ValueError("dense_heads must not contain duplicates.")
    if any(head < 0 or head >= heads for head in dense_heads):
        raise ValueError(f"dense_heads must be in [0, {heads - 1}].")
    if dense_heads and (
        use_reference
        or h_reduce != 1
        or (h, w) != (45, 80)
        or seed_mode != "local_with_ith"
        or exact_intra_frame_attention
        or exact_neighbor_window_columns
        or exact_query_rows
        or full_attention_anchor_time_count
    ):
        raise ValueError(
            "fixed dense-head rescue requires plain 45x80 Triton Hard C4."
        )
    exact_query_rows = int(exact_query_rows or 0)
    if use_query_centroids and (
        use_reference
        or h_reduce != 1
        or (h, w) != (45, 80)
        or exact_intra_frame_attention
        or exact_neighbor_window_columns
        or first_frame_full_attention
        or full_attention_anchor_time_count
    ):
        raise ValueError(
            "local-centroid P11 seeds require the plain 45x80 Triton Hard-C4 path."
        )
    if use_query_response and (
        use_reference
        or h_reduce != 1
        or (h, w) != (45, 80)
        or exact_intra_frame_attention
        or exact_neighbor_window_columns
        or exact_query_rows
        or first_frame_full_attention
        or full_attention_anchor_time_count
        or dense_heads
    ):
        raise ValueError(
            "query-response seeds require the plain 45x80 Triton Hard-C4 path."
        )
    if use_responsibility_centroids and (
        use_reference
        or h_reduce != 1
        or (h, w) != (45, 80)
        or exact_intra_frame_attention
        or exact_neighbor_window_columns
        or first_frame_full_attention
        or full_attention_anchor_time_count
    ):
        raise ValueError(
            "responsibility-centroid Top-2 requires the plain 45x80 Triton "
            "Hard-C4 path."
        )
    if use_hierarchical_split and (
        use_reference
        or h_reduce != 1
        or (h, w) != (45, 80)
        or exact_intra_frame_attention
        or exact_neighbor_window_columns
        or first_frame_full_attention
        or full_attention_anchor_time_count
    ):
        raise ValueError(
            "hierarchical mean-split Top-2 requires the plain 45x80 Triton "
            "Hard-C4 path."
        )
    if use_singleton_peeling and (
        use_reference
        or h_reduce != 1
        or (h, w) != (45, 80)
        or exact_neighbor_window_columns
        or first_frame_full_attention
        or full_attention_anchor_time_count
    ):
        raise ValueError(
            "singleton-leverage refinement requires the plain 45x80 Triton "
            "Hard-C4 path."
        )
    if use_singleton_peeling and exact_intra_frame_attention and singleton_top_r != 3:
        raise ValueError(
            "exact intra-frame singleton refinement requires exactly three atoms."
        )
    if exact_query_rows not in {0, 3}:
        raise ValueError(
            "budgeted exact-query refinement supports exactly 3 rows."
        )
    if exact_query_rows and (
        use_reference
        or h_reduce != 1
        or (h, w) != (45, 80)
        or exact_intra_frame_attention
        or exact_neighbor_window_columns
        or first_frame_full_attention
        or full_attention_anchor_time_count
        or use_query_centroids
        or use_responsibility_centroids
        or use_hierarchical_split
        or use_singleton_peeling
    ):
        raise ValueError(
            "budgeted exact-query refinement requires plain 45x80 Triton Hard C4."
        )
    if exact_intra_frame_attention:
        if not use_triton:
            raise ValueError(
                "exact intra-frame Hard C4 requires hard_route_backend='hard_c4'."
            )
        if h_reduce != 1 or (h, w) != (45, 80):
            raise ValueError(
                "exact intra-frame Hard C4 is specialized for h_reduce=1 on "
                "the 45x80 720p grid."
            )
        if query_blocks.size(1) != 21 or int(query_outer_chunk) != 21:
            raise ValueError(
                "exact intra-frame Hard C4 requires all 21 latent frames in "
                "one query chunk."
            )

    output = torch.empty_like(query_blocks)
    head_chunk = heads if not head_chunk or head_chunk >= heads else int(head_chunk)
    for start in range(0, query_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, query_blocks.size(1))
        query_chunk = query_blocks[:, start:end]
        seed_query_chunk = (
            None
            if seed_query_blocks is None
            else seed_query_blocks[:, start:end]
        )
        output_slice = output[:, start:end]
        if use_reference:
            chunk = hard_route_attention_pytorch(
                query_chunk,
                key_blocks,
                value_blocks,
                sm_scale,
                query_outer_offset=start,
            )
        elif use_triton and h_reduce == 1:
            for h_start, h_count in _hard_head_chunks(
                heads, head_chunk, dense_heads
            ):
                if use_responsibility_centroids:
                    kernel = (
                        hard_c4_720p_responsibility_centroid_attention_triton_blocks
                    )
                elif use_hierarchical_split:
                    kernel = (
                        hard_c4_720p_hierarchical_mean_split_attention_triton_blocks
                    )
                elif use_singleton_peeling:
                    if exact_intra_frame_attention:
                        kernel = (
                            hard_c4_720p_singleton_leverage_top3_exact_intra_frame_attention_triton_blocks
                        )
                    else:
                        kernel = (
                            hard_c4_720p_singleton_leverage_top6_attention_triton_blocks
                            if singleton_top_r == 6
                            else hard_c4_720p_singleton_leverage_attention_triton_blocks
                        )
                elif use_query_centroids:
                    kernel = hard_c4_720p_query_centroid_attention_triton_blocks
                elif use_query_response:
                    kernel = hard_c4_720p_query_response_attention_triton_blocks
                elif exact_query_rows:
                    kernel = hard_c4_720p_exact_query_top3_attention_triton_blocks
                elif exact_intra_frame_attention:
                    kernel = hard_c4_720p_exact_intra_frame_attention_triton_blocks
                else:
                    kernel = hard_c4_rt_attention_triton_blocks
                kernel(
                    query_chunk,
                    key_blocks,
                    value_blocks,
                    sm_scale,
                    **(
                        {"seed_q": seed_query_chunk}
                        if seed_query_chunk is not None
                        else {}
                    ),
                    output=output_slice,
                    head_start=h_start,
                    head_count=h_count,
                    **(
                        {"top_r": singleton_top_r}
                        if (
                            use_singleton_peeling
                            and not exact_intra_frame_attention
                            and singleton_top_r != 6
                        )
                        else {}
                    ),
                    **(
                        {"query_outer_offset": start}
                        if exact_intra_frame_attention
                        else {}
                    ),
                    **(
                        {
                            "exact_neighbor_window_columns": (
                                exact_neighbor_window_columns
                            )
                        }
                        if exact_neighbor_window_columns
                        else {}
                    ),
                )
            chunk = output_slice
        elif use_triton:
            query_head_major = query_chunk.reshape(
                query_chunk.size(0), -1, heads, query_chunk.size(-1)
            ).permute(0, 2, 1, 3).contiguous()
            key_head_major = key_blocks.reshape(
                key_blocks.size(0), -1, heads, key_blocks.size(-1)
            ).permute(0, 2, 1, 3).contiguous()
            value_head_major = value_blocks.reshape(
                value_blocks.size(0), -1, heads, value_blocks.size(-1)
            ).permute(0, 2, 1, 3).contiguous()
            output_head_major = torch.empty_like(query_head_major)
            query_block_rows = 24 if h_reduce == 2 else 12
            for h_start in range(0, heads, head_chunk):
                h_end = min(h_start + head_chunk, heads)
                hard_c4_grid_720p_attention_triton_head_major(
                    query_head_major,
                    key_head_major,
                    value_head_major,
                    query_block_rows=query_block_rows,
                    key_block_cols=80,
                    scale=sm_scale,
                    output=output_head_major,
                    head_start=h_start,
                    head_count=h_end - h_start,
                )
            chunk = output_head_major.permute(0, 2, 1, 3).reshape_as(query_chunk)
        if chunk is not output_slice:
            output_slice.copy_(chunk)
    if full_attention_anchor_time_count:
        _apply_full_attention_anchor_grid(
            output,
            query_blocks,
            key,
            value,
            sm_scale=sm_scale,
            time_count=full_attention_anchor_time_count,
            row_count=full_attention_anchor_row_count,
            width_count=full_attention_anchor_width_count,
            correction_scale=full_attention_anchor_scale,
        )
    output = _hard_c4_blocks_to_video(output)
    if dense_heads:
        head_index = torch.tensor(dense_heads, device=query.device)
        dense_output = flash_attention(
            query.index_select(2, head_index),
            key.index_select(2, head_index),
            value.index_select(2, head_index),
            softmax_scale=sm_scale,
        )
        output.index_copy_(2, head_index, dense_output)
    if first_frame_full_attention:
        frame_tokens = h * w
        output[:, :frame_tokens] = flash_attention(
            query[:, :frame_tokens],
            key,
            value,
            softmax_scale=sm_scale,
        )
    return output


def _hard_c568_attention(
    query,
    key,
    value,
    *,
    h,
    w,
    sm_scale,
    query_outer_chunk,
    head_chunk,
    num_merge_tokens,
    seed_mode,
):
    query_blocks = _hard_c4_video_to_blocks(query, int(h), int(w))
    key_blocks = _hard_c4_video_to_blocks(key, int(h), int(w))
    value_blocks = _hard_c4_video_to_blocks(value, int(h), int(w))
    output = torch.empty_like(query_blocks)
    heads = query_blocks.size(4)
    head_chunk = heads if not head_chunk or head_chunk >= heads else int(head_chunk)
    seed_mode = str(seed_mode or "local_with_ith").lower()
    common_c_ablation = seed_mode.endswith("_commonc")
    if common_c_ablation:
        seed_mode = seed_mode[:-8]
    candidate_geometry_count = None
    candidate_geometry_match = re.search(r"_cg([0-9]+)$", seed_mode)
    if candidate_geometry_match is not None:
        candidate_geometry_count = int(candidate_geometry_match.group(1))
        seed_mode = seed_mode[: candidate_geometry_match.start()]
    seed_mode, probe_policy = _strip_hellinger_probe_policy(seed_mode)
    router_frame_override = None
    router_frame_match = re.search(r"_rf([0-9]|1[0-9]|20)$", seed_mode)
    if router_frame_match is not None:
        router_frame_override = int(router_frame_match.group(1))
        seed_mode = seed_mode[: router_frame_match.start()]
    tuned_launch_suffix = "_gt32gw8"
    tuned_hellinger_launch = seed_mode.endswith(tuned_launch_suffix)
    if tuned_hellinger_launch:
        seed_mode = seed_mode[: -len(tuned_launch_suffix)]
    hellinger_span = parse_hellinger_span_seed_mode(seed_mode)
    if hellinger_span is not None:
        if num_merge_tokens not in {5, 6, 8}:
            raise ValueError("dynamic Hellinger seeds require Hard C5/C6/C8")
        if num_merge_tokens in {6, 8} and not common_c_ablation:
            raise ValueError("dynamic Hard C6/C8 requires the common-C ablation path")
        (
            alpha,
            candidates,
            probes,
            anchors,
            block_j,
            routing_head_group,
            routing_head_offset,
            temporal_frame_count,
        ) = hellinger_span
        if routing_head_group != 1 or routing_head_offset != 0:
            raise ValueError("dynamic Hard C5 currently requires per-head routing")
        if not 1 <= temporal_frame_count <= 3:
            raise ValueError("dynamic Hard C5 requires one to three temporal routers")
        if query_blocks.size(1) != 21 or int(query_outer_chunk) != 21:
            raise ValueError("dynamic Hard C5 requires all 21 latent frames per chunk")
        if probes % 2:
            raise ValueError("Hellinger-span requires an even probe count")
        uniform_only_probes = _use_uniform_hellinger_probes(
            probes, probe_policy
        )

        if router_frame_override is not None:
            if temporal_frame_count != 1:
                raise ValueError("an explicit Hellinger router frame requires t1 routing")
            router_frame_values = [router_frame_override]
        else:
            router_frame_values = [
                round(
                    (slot + 0.5) * query_blocks.size(1) / temporal_frame_count - 0.5
                )
                for slot in range(temporal_frame_count)
            ]
        router_frames = torch.tensor(
            router_frame_values, device=query_blocks.device, dtype=torch.long
        )
        router_query = query_blocks.index_select(1, router_frames).contiguous()
        router_table = hellinger_span_seed_rows_triton(
            router_query,
            key_blocks,
            sm_scale,
            alpha=alpha,
            candidate_count=candidates,
            candidate_geometry_count=candidate_geometry_count,
            top_probes=0 if uniform_only_probes else probes // 2,
            uniform_probes=probes if uniform_only_probes else probes // 2,
            probe_anchor_count=anchors,
            block_j=block_j,
            group_tile=32 if tuned_hellinger_launch else 16,
            output_dtype=torch.uint8,
            gs_num_warps=8 if tuned_hellinger_launch else 4,
            num_seeds=num_merge_tokens,
        )
        if common_c_ablation:
            for h_start in range(0, heads, head_chunk):
                h_end = min(h_start + head_chunk, heads)
                hard_c_720p_native_temporal_seed_rows_attention_triton_blocks(
                    query_blocks,
                    key_blocks,
                    value_blocks,
                    router_table,
                    num_merge_tokens,
                    sm_scale,
                    output=output,
                    head_start=h_start,
                    head_count=h_end - h_start,
                )
            return _hard_c4_blocks_to_video(output)
        for h_start in range(0, heads, head_chunk):
            h_end = min(h_start + head_chunk, heads)
            shared_seed_rows = (
                router_table[..., h_start:h_end, :]
                .permute(0, 1, 4, 5, 2, 3, 6)
                .contiguous()
            )
            hard_c5_720p_temporal_shared_seed_rows_attention_triton_blocks(
                query_blocks,
                key_blocks,
                value_blocks,
                shared_seed_rows,
                sm_scale,
                output=output,
                head_start=h_start,
                head_count=h_end - h_start,
            )
        return _hard_c4_blocks_to_video(output)

    if num_merge_tokens == 5:
        if seed_mode == "local_centroid_norm_p11":
            kernel = hard_c5_720p_query_centroid_norm_attention_triton_blocks
        elif seed_mode == "local_with_ith":
            kernel = hard_c5_720p_attention_triton_blocks
        else:
            raise ValueError(
                "Hard C5 supports seed_mode='local_with_ith' or "
                "'local_centroid_norm_p11'."
            )
    elif num_merge_tokens == 6:
        if seed_mode == "local_with_ith_compact":
            kernel = hard_c6_720p_compact_attention_triton_blocks
        elif seed_mode == "local_with_ith_fp8_key":
            kernel = hard_c6_720p_fp8_key_attention_triton_blocks
        elif seed_mode == "local_with_ith_gaussian_partition":
            kernel = hard_c6_720p_gaussian_partition_attention_triton_blocks
        elif seed_mode == "local_with_ith_split_kv":
            kernel = hard_c6_720p_split_kv_attention_triton_blocks
        elif seed_mode == "local_with_ith_max_partition":
            kernel = hard_c6_720p_max_partition_attention_triton_blocks
        else:
            kernel = hard_c6_720p_attention_triton_blocks
    else:
        if seed_mode == "local_with_ith_fp8_key":
            kernel = hard_c8_720p_fp8_key_attention_triton_blocks
        elif seed_mode == "local_with_ith_gaussian_partition":
            kernel = hard_c8_720p_gaussian_partition_attention_triton_blocks
        elif seed_mode == "local_with_ith_split_kv":
            kernel = hard_c8_720p_split_kv_attention_triton_blocks
        elif seed_mode == "local_with_ith_max_partition":
            kernel = hard_c8_720p_max_partition_attention_triton_blocks
        else:
            kernel = hard_c8_720p_attention_triton_blocks
    for start in range(0, query_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, query_blocks.size(1))
        output_slice = output[:, start:end]
        for h_start in range(0, heads, head_chunk):
            h_end = min(h_start + head_chunk, heads)
            kernel(
                query_blocks[:, start:end],
                key_blocks,
                value_blocks,
                sm_scale,
                output=output_slice,
                head_start=h_start,
                head_count=h_end - h_start,
            )
    return _hard_c4_blocks_to_video(output)


def _biaxial_c3_attention(
    query,
    key,
    value,
    *,
    h,
    w,
    sm_scale,
    query_outer_chunk,
    head_chunk,
):
    """Fuse row and column C3 estimators through their partition functions."""

    h = int(h)
    w = int(w)
    if (h, w) != (45, 80):
        raise ValueError("Bi-Axial C3 is specialized for the 45x80 Wan grid.")
    query_blocks = _hard_c4_video_to_blocks(query, h, w)
    key_blocks = _hard_c4_video_to_blocks(key, h, w)
    value_blocks = _hard_c4_video_to_blocks(value, h, w)
    if int(query_outer_chunk) != 21 or query_blocks.size(1) != 21:
        raise ValueError("Bi-Axial C3 requires all 21 latent frames in one chunk.")

    batch, _, _, _, heads, dim = query_blocks.shape
    head_chunk = heads if not head_chunk or head_chunk >= heads else int(head_chunk)
    output = torch.empty_like(query_blocks)
    key_head_major = key.reshape(batch, -1, heads, dim).permute(0, 2, 1, 3).contiguous()
    value_head_major = value.reshape(batch, -1, heads, dim).permute(0, 2, 1, 3).contiguous()
    query_head_major = query.reshape(batch, -1, heads, dim).permute(0, 2, 1, 3).contiguous()
    row_output = torch.empty_like(query_blocks)
    column_output_head_major = torch.empty_like(query_head_major)

    for h_start in range(0, heads, head_chunk):
        h_end = min(h_start + head_chunk, heads)
        _, row_log_z = hard_c3_720p_attention_triton_blocks(
            query_blocks,
            key_blocks,
            value_blocks,
            sm_scale,
            output=row_output,
            head_start=h_start,
            head_count=h_end - h_start,
            return_log_partition=True,
        )
        _, column_log_z = hard_c3_720p_transpose_attention_triton_head_major(
            query_head_major,
            key_head_major,
            value_head_major,
            sm_scale,
            output=column_output_head_major,
            head_start=h_start,
            head_count=h_end - h_start,
            return_log_partition=True,
        )
        column_output = column_output_head_major[:, h_start:h_end]
        column_output = column_output.permute(0, 2, 1, 3).reshape(
            batch,
            query_blocks.size(1),
            h,
            w,
            h_end - h_start,
            dim,
        )
        row_weight = torch.sigmoid(row_log_z - column_log_z).unsqueeze(-1)
        fused = row_weight * row_output[..., h_start:h_end, :].float()
        fused.add_((1.0 - row_weight) * column_output.float())
        output[..., h_start:h_end, :].copy_(fused.to(output.dtype))
    return _hard_c4_blocks_to_video(output)


def _causal_one_token_attention(
    query,
    key,
    value,
    *,
    f_tied,
    h_reduce,
    w_reduce,
    h,
    w,
    sm_scale,
    q_init,
    query_outer_chunk,
    backend,
    head_chunk,
):
    """Execute cache-compatible one-token GridMerge with unequal Q and K/V."""

    if key.size(1) != value.size(1):
        raise ValueError("causal one-token GridMerge expects K/V lengths to match.")
    if query.size(0) != key.size(0) or query.shape[2:] != key.shape[2:]:
        raise ValueError(
            "causal one-token GridMerge expects Q and K/V batch, head, and "
            "head-dimension shapes to match."
        )

    f_tied = int(f_tied)
    h_reduce = int(h_reduce)
    w_reduce = int(w_reduce)
    original_h = int(h)
    original_w = int(w)
    query, padded_h, padded_w, _ = pad_video_tokens_to_multiples(
        query, original_h, original_w, h_reduce, w_reduce
    )
    key, _, _, key_valid = pad_video_tokens_to_multiples(
        key, original_h, original_w, h_reduce, w_reduce
    )
    value, _, _, _ = pad_video_tokens_to_multiples(
        value, original_h, original_w, h_reduce, w_reduce
    )

    query_blocks = general_video_to_blocks(
        query, f_tied, h_reduce, w_reduce, padded_h, padded_w
    )
    key_blocks = general_video_to_blocks(
        key, f_tied, h_reduce, w_reduce, padded_h, padded_w
    )
    value_blocks = general_video_to_blocks(
        value, f_tied, h_reduce, w_reduce, padded_h, padded_w
    )
    key_valid_blocks = None
    if key_valid is not None:
        key_valid_blocks = general_video_to_blocks(
            key_valid, f_tied, h_reduce, w_reduce, padded_h, padded_w
        )[..., 0, 0]

    output_blocks = torch.empty_like(query_blocks)
    use_monarch_rt_kernel = (backend or "").lower() == MONARCH_RT_BACKEND
    if use_monarch_rt_kernel:
        if (q_init or "ith").lower() not in {"ith", "i", "identity"}:
            raise ValueError("monarch_rt Triton backend currently supports q_init=ith only.")
        head_chunk = max(1, int(head_chunk or query_blocks.size(-2)))

    for start in range(0, query_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, query_blocks.size(1))
        if use_monarch_rt_kernel:
            for head_start in range(0, query_blocks.size(-2), head_chunk):
                head_end = min(head_start + head_chunk, query_blocks.size(-2))
                head_slice = slice(head_start, head_end)
                kernel_kwargs = (
                    {"key_valid": key_valid_blocks}
                    if key_valid_blocks is not None
                    else {}
                )
                output_blocks[:, start:end, ..., head_slice, :] = (
                    monarch_rt_attention_triton_blocks(
                        query_blocks[:, start:end, ..., head_slice, :],
                        key_blocks[..., head_slice, :],
                        value_blocks[..., head_slice, :],
                        sm_scale,
                        **kernel_kwargs,
                    )
                )
        else:
            output_blocks[:, start:end] = causal_gridmerge_one(
                query_blocks[:, start:end],
                key_blocks,
                value_blocks,
                sm_scale,
                q_init,
                key_valid=key_valid_blocks,
            )
    output = general_blocks_to_video(
        output_blocks, f_tied, h_reduce, w_reduce
    )
    return crop_video_tokens_from_multiples(
        output, original_h, original_w, padded_h, padded_w
    )


def token_merging_attn(
    q,
    k,
    v,
    f_tied,
    h_reduce,
    w_reduce,
    h,
    w,
    sm_scale=None,
    q_init=None,
    query_outer_chunk=None,
    layout=LAYOUT_HARD_ROUTE,
    num_merge_tokens=4,
    seed_mode="local_with_ith",
    seed=0,
    hard_route_backend=HARD_C4_BACKEND,
    hard_route_routing_backend=None,
    head_chunk=None,
    exact_intra_frame_attention=False,
    exact_neighbor_window_columns=0,
    exact_query_rows=0,
    first_frame_full_attention=False,
    full_attention_anchor_time_count=0,
    full_attention_anchor_row_count=0,
    full_attention_anchor_width_count=0,
    full_attention_anchor_scale=1.0,
    dense_heads=(),
    seed_query=None,
    seed_table_cache=None,
    cache_timestep=None,
):
    """Dispatch the ordinary-Wan C4 and causal Self-Forcing C1 paths.

    Ordinary Wan supports main's maintained hard-C4 implementations:
      - hard_route_backend="sdpa": readable PyTorch reference.
      - hard_route_backend="hard_c4": fastest C4 Triton kernel.

    Causal Self-Forcing uses one-token GridMerge because its current-block Q is
    shorter than its visible cached K/V prefix.
    """

    del seed, hard_route_routing_backend
    if (layout or LAYOUT_HARD_ROUTE) != LAYOUT_HARD_ROUTE:
        raise ValueError("the merged Wan implementation supports only hard_route.")
    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5
    query_outer_chunk = max(1, int(query_outer_chunk or 1))
    num_merge_tokens = int(num_merge_tokens)
    dense_heads = tuple(dense_heads or ())
    first_frame_full_attention = bool(first_frame_full_attention)
    exact_intra_frame_attention = bool(exact_intra_frame_attention)
    exact_neighbor_window_columns = int(exact_neighbor_window_columns or 0)
    exact_query_rows = int(exact_query_rows or 0)
    full_attention_anchor_time_count = int(full_attention_anchor_time_count or 0)
    full_attention_anchor_row_count = int(full_attention_anchor_row_count or 0)
    full_attention_anchor_width_count = int(full_attention_anchor_width_count or 0)
    full_attention_anchor_scale = float(full_attention_anchor_scale)

    if dense_heads and num_merge_tokens != 4:
        raise ValueError("dense rescue is supported only for Hard C4.")
    if seed_query is not None and num_merge_tokens != 4:
        raise ValueError("separate seed_query is supported only for Hard C4.")

    if first_frame_full_attention and num_merge_tokens != 4:
        raise ValueError(
            "first-frame full-attention recomputation is supported only for Hard C4."
        )
    if exact_intra_frame_attention and num_merge_tokens != 4:
        raise ValueError(
            "exact intra-frame attention is supported only for Hard C4."
        )
    if exact_query_rows and num_merge_tokens != 4:
        raise ValueError(
            "budgeted exact-query refinement is supported only for Hard C4."
        )
    if exact_neighbor_window_columns and not exact_intra_frame_attention:
        raise ValueError(
            "exact neighbor windows require exact intra-frame attention."
        )
    if not 0 <= exact_neighbor_window_columns <= 30:
        raise ValueError(
            "exact neighbor windows support 0 to 30 columns per neighbor."
        )
    if bool(full_attention_anchor_time_count) != bool(
        full_attention_anchor_row_count
    ):
        raise ValueError(
            "full-attention residual anchors require both time and row counts."
        )
    if full_attention_anchor_width_count and not full_attention_anchor_time_count:
        raise ValueError(
            "width anchors require full-attention residual time and row anchors."
        )
    if first_frame_full_attention and full_attention_anchor_time_count:
        raise ValueError(
            "first-frame recomputation and residual-anchor correction are mutually exclusive."
        )
    if exact_intra_frame_attention and (
        first_frame_full_attention or full_attention_anchor_time_count
    ):
        raise ValueError(
            "exact intra-frame attention, first-frame recomputation, and "
            "residual-anchor correction are mutually exclusive."
        )
    if exact_query_rows and (
        exact_intra_frame_attention
        or first_frame_full_attention
        or full_attention_anchor_time_count
    ):
        raise ValueError(
            "budgeted exact-query refinement and other full-attention "
            "corrections are mutually exclusive."
        )
    if full_attention_anchor_time_count and num_merge_tokens != 4:
        raise ValueError(
            "full-attention residual anchors are supported only for Hard C4."
        )
    if not math.isfinite(full_attention_anchor_scale):
        raise ValueError("full-attention residual-anchor scale must be finite.")

    if num_merge_tokens == 1:
        return _causal_one_token_attention(
            q,
            k,
            v,
            f_tied=f_tied,
            h_reduce=h_reduce,
            w_reduce=w_reduce,
            h=h,
            w=w,
            sm_scale=float(sm_scale),
            q_init=q_init,
            query_outer_chunk=query_outer_chunk,
            backend=hard_route_backend,
            head_chunk=head_chunk,
        )
    if num_merge_tokens == 3:
        if (hard_route_backend or "").lower() != BIAXIAL_C3_BACKEND:
            raise ValueError(
                "Bi-Axial C3 requires hard_route_backend='biaxial_c3'."
            )
        if int(h_reduce) != 1 or int(w_reduce) != 1:
            raise ValueError("Bi-Axial C3 requires h_reduce=w_reduce=1.")
        if str(seed_mode or "local_with_ith").lower() != "local_with_ith":
            raise ValueError("Bi-Axial C3 uses fixed symmetric local seeds.")
        return _biaxial_c3_attention(
            q,
            k,
            v,
            h=h,
            w=w,
            sm_scale=float(sm_scale),
            query_outer_chunk=query_outer_chunk,
            head_chunk=head_chunk,
        )
    if num_merge_tokens in {5, 6, 8}:
        if (hard_route_backend or HARD_C4_BACKEND).lower() != HARD_C4_BACKEND:
            raise ValueError(
                "Wan hard C5/C6/C8 requires hard_route_backend='hard_c4'."
            )
        return _hard_c568_attention(
            q,
            k,
            v,
            h=h,
            w=w,
            sm_scale=float(sm_scale),
            query_outer_chunk=query_outer_chunk,
            head_chunk=head_chunk,
            num_merge_tokens=num_merge_tokens,
            seed_mode=seed_mode,
        )
    if num_merge_tokens != 4:
        raise ValueError(
            "Wan supports num_merge_tokens=3, 4, 5, 6, or 8 for Bi-Axial "
            "C3/C4/C5/C6/C8, "
            "or num_merge_tokens=1 for causal Self-Forcing."
        )
    return _hard_c4_attention(
        q,
        k,
        v,
        seed_query=seed_query,
        h=h,
        w=w,
        h_reduce=h_reduce,
        w_reduce=w_reduce,
        sm_scale=float(sm_scale),
        query_outer_chunk=query_outer_chunk,
        head_chunk=head_chunk,
        hard_route_backend=hard_route_backend,
        exact_intra_frame_attention=exact_intra_frame_attention,
        exact_neighbor_window_columns=exact_neighbor_window_columns,
        exact_query_rows=exact_query_rows,
        first_frame_full_attention=first_frame_full_attention,
        full_attention_anchor_time_count=full_attention_anchor_time_count,
        full_attention_anchor_row_count=full_attention_anchor_row_count,
        full_attention_anchor_width_count=full_attention_anchor_width_count,
        full_attention_anchor_scale=full_attention_anchor_scale,
        seed_mode=seed_mode,
        dense_heads=dense_heads,
        seed_table_cache=seed_table_cache,
        cache_timestep=cache_timestep,
    )


def token_merging_attn_with_kv_cache(
    q,
    k_cache,
    v_cache,
    new_k,
    new_v,
    start_idx,
    end_idx,
    f_tied,
    h_reduce,
    w_reduce,
    h,
    w,
    sm_scale=None,
    q_init=None,
    query_outer_chunk=None,
    layout=LAYOUT_HARD_ROUTE,
    num_merge_tokens=1,
    seed_mode="random",
    seed=0,
    hard_route_backend="sdpa",
    hard_route_routing_backend="sdpa",
    head_chunk=None,
):
    """Overwrite the current cache block and attend only to its visible prefix."""

    if (layout or LAYOUT_HARD_ROUTE) != LAYOUT_HARD_ROUTE or int(
        num_merge_tokens
    ) != 1:
        raise ValueError(
            "causal cached token merging supports only layout='hard_route' "
            "with num_merge_tokens=1."
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shapes.")
    if new_k.shape != new_v.shape:
        raise ValueError("new_k and new_v must have identical shapes.")
    if q.size(0) != k_cache.size(0) or q.shape[2:] != k_cache.shape[2:]:
        raise ValueError(
            "Q and cached K/V batch, head, and head-dimension shapes must match."
        )
    if new_k.size(0) != k_cache.size(0) or new_k.shape[2:] != k_cache.shape[2:]:
        raise ValueError(
            "new and cached K/V batch, head, and head-dimension shapes must match."
        )

    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if not 0 <= start_idx <= end_idx <= k_cache.size(1):
        raise ValueError(
            f"invalid cache interval [{start_idx}, {end_idx}) for capacity "
            f"{k_cache.size(1)}."
        )
    if end_idx - start_idx != new_k.size(1):
        raise ValueError(
            "cache interval length must equal the number of new K/V tokens."
        )

    k_cache[:, start_idx:end_idx].copy_(new_k)
    v_cache[:, start_idx:end_idx].copy_(new_v)
    return token_merging_attn(
        q,
        k_cache[:, :end_idx],
        v_cache[:, :end_idx],
        f_tied=f_tied,
        h_reduce=h_reduce,
        w_reduce=w_reduce,
        h=h,
        w=w,
        sm_scale=sm_scale,
        q_init=q_init,
        query_outer_chunk=query_outer_chunk,
        layout=layout,
        num_merge_tokens=num_merge_tokens,
        seed_mode=seed_mode,
        seed=seed,
        hard_route_backend=hard_route_backend,
        hard_route_routing_backend=hard_route_routing_backend,
        head_chunk=head_chunk,
    )


__all__ = [
    "LAYOUT_HARD_ROUTE",
    "token_merging_attn",
    "token_merging_attn_with_kv_cache",
]
