"""Probe leverage-aware singleton peeling for Hard-C4 on native Wan QKV."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

from analysis.compare_hard_c4_backends_dense import load_pipeline, run_denoise_only, sync
from analysis.probe_hierarchical_split_gma_real_qkv import (
    _conditional_child,
    output_from_representatives,
    summarize,
)
from analysis.probe_query_centroid_gma_real_qkv import (
    ErrorStats,
    fixed_seed_queries,
    parse_ints,
)
from attention.hard_c4_wan21_14b_720p_triton import (
    hard_c4_720p_attention_triton_blocks,
)
from attention.token_merging import (
    _hard_c4_blocks_to_video,
    _hard_c4_video_to_blocks,
)
from generate import load_prompts


SELECTORS = (
    "mass",
    "entropy",
    "routing_leverage",
    "routing_leverage_static",
)


def fixed_contiguous_seed_queries(query, key_frames, key_rows, clusters):
    """Gather the contiguous local seeds used by the production C5 family."""

    clusters = int(clusters)
    key_row = torch.arange(key_rows, device=query.device).view(1, 1, key_rows, 1)
    start = (key_row - clusters // 2).clamp(0, query.size(2) - clusters)
    offsets = torch.arange(clusters, device=query.device).view(1, 1, 1, 1, clusters)
    seed_rows = (start.unsqueeze(-1) + offsets).long().expand(
        query.size(1), key_frames, key_rows, query.size(3), clusters
    )
    query_outer = torch.arange(query.size(1), device=query.device).view(
        -1, 1, 1, 1, 1
    )
    query_col = torch.arange(query.size(3), device=query.device).view(
        1, 1, 1, -1, 1
    )
    return query[:, query_outer, seed_rows, query_col]


def hard_c4_parent_state(seeds, key, value, scale):
    seed_scores = torch.einsum(
        "bafkjchd,bfklhd->bhafkjcl", seeds, key
    ).float() * float(scale)
    assignment = torch.zeros_like(seed_scores, dtype=torch.bool)
    best_seed = torch.log_softmax(seed_scores, dim=-1).argmax(
        dim=-2, keepdim=True
    )
    assignment.scatter_(dim=-2, index=best_seed, value=True)
    routed_scores = seed_scores.masked_fill(~assignment, float("-inf"))
    log_weights = torch.log_softmax(routed_scores, dim=-1).masked_fill(
        ~assignment, 0.0
    )
    weights = log_weights.exp().masked_fill(~assignment, 0.0)
    parent_key, parent_value, parent_bias, _, _ = _conditional_child(
        weights, log_weights, assignment, key, value
    )
    return {
        "seed_scores": seed_scores,
        "assignment": assignment,
        "log_weights": log_weights,
        "weights": weights,
        "parent_key": parent_key,
        "parent_value": parent_value,
        "parent_bias": parent_bias,
    }


def greedy_singletons(state, count, selector):
    static_topk = selector == "routing_leverage_static"
    if static_topk:
        selector = "routing_leverage"
    active = state["assignment"].clone()
    weights = state["weights"]
    seed_scores = state["seed_scores"]
    width = seed_scores.size(-1)
    selected = []

    for _ in range(int(count)):
        active_mass = (weights * active).sum(dim=-1)
        conditional = weights * active / active_mass.unsqueeze(-1).clamp_min(1e-20)

        if selector == "mass":
            score = conditional
        elif selector == "entropy":
            score = -conditional * conditional.clamp_min(1e-20).log()
        elif selector == "routing_leverage":
            mean_score = torch.einsum(
                "...cl,...sl->...cs", conditional, seed_scores
            )
            point_norm = seed_scores.square().sum(dim=-2)
            mean_norm = mean_score.square().sum(dim=-1)
            cross = torch.einsum("...cs,...sl->...cl", mean_score, seed_scores)
            distance = (
                mean_norm.unsqueeze(-1) + point_norm.unsqueeze(-2) - 2.0 * cross
            ).clamp_min(0.0)
            score = conditional / (1.0 - conditional).clamp_min(1e-6) * distance
        else:
            raise ValueError(f"unknown selector {selector!r}")

        score = score.masked_fill(~active, float("-inf"))
        key_score = score.amax(dim=-2)
        if static_topk:
            return key_score.topk(int(count), dim=-1).indices
        index = key_score.argmax(dim=-1)
        selected.append(index)
        selected_mask = F.one_hot(index, num_classes=width).bool().unsqueeze(-2)
        active = active & ~selected_mask

    return torch.stack(selected, dim=-1)


def static_response_singletons(state, value, count, selector):
    """Rank singleton splits by routing-centered conditional-value response."""

    if selector not in {
        "routing_leverage_centered",
        "routing_cross_energy",
        "routing_cross_energy_uncentered",
    }:
        raise ValueError(f"unknown response selector {selector!r}")
    weights = state["weights"]
    seed_scores = state["seed_scores"]
    assignment = state["assignment"]
    mean_score = torch.einsum("...cl,...sl->...cs", weights, seed_scores)
    point_norm = seed_scores.square().sum(dim=-2)
    mean_norm = mean_score.square().sum(dim=-1)
    cross = torch.einsum("...cs,...sl->...cl", mean_score, seed_scores)
    uncentered_distance = (
        mean_norm.unsqueeze(-1) + point_norm.unsqueeze(-2) - 2.0 * cross
    ).clamp_min(0.0)

    owner_mean = torch.diagonal(mean_score, dim1=-2, dim2=-1)
    owner_delta = seed_scores - owner_mean.unsqueeze(-1)
    delta_sum = seed_scores.sum(dim=-2).unsqueeze(-2) - mean_score.sum(
        dim=-1
    ).unsqueeze(-1)
    centered_distance = (
        uncentered_distance
        - 2.0 * owner_delta * delta_sum
        + seed_scores.size(-2) * owner_delta.square()
    ).clamp_min(0.0)
    ratio = weights / (1.0 - weights).clamp_min(1e-6)

    if selector == "routing_leverage_centered":
        score = ratio * centered_distance
    else:
        parent_value = rearrange(
            state["parent_value"], "b a f j k c h d -> b h a f k j c d"
        ).float()
        value_h = rearrange(value, "b f k l h d -> b h f k l d").float()
        value_cross = torch.einsum(
            "bhafkjcd,bhfkld->bhafkjcl", parent_value, value_h
        )
        value_distance = (
            parent_value.square().sum(dim=-1).unsqueeze(-1)
            + value_h.square().sum(dim=-1)[:, :, None, :, :, None, None, :]
            - 2.0 * value_cross
        ).clamp_min(0.0)
        query_distance = (
            uncentered_distance
            if selector == "routing_cross_energy_uncentered"
            else centered_distance
        )
        score = ratio.square() * query_distance * value_distance

    score = score.masked_fill(~assignment, float("-inf"))
    key_score = score.amax(dim=-2)
    return key_score.topk(int(count), dim=-1).indices


def peeled_representatives(state, key, value, selected_index):
    width = state["seed_scores"].size(-1)
    selected_mask = F.one_hot(selected_index, num_classes=width).bool().any(dim=-2)
    bulk_mask = state["assignment"] & ~selected_mask.unsqueeze(-2)
    bulk_key, bulk_value, bulk_bias, _, _ = _conditional_child(
        state["weights"],
        state["log_weights"],
        bulk_mask,
        key,
        value,
    )
    bulk_key = rearrange(bulk_key, "b a f j k c h d -> b h a f k j c d").float()
    bulk_value = rearrange(
        bulk_value, "b a f j k c h d -> b h a f k j c d"
    ).float()

    batch, heads, query_frames, key_frames, rows, query_cols = selected_index.shape[:6]
    key_h = rearrange(key, "b f k l h d -> b h f k l d")
    value_h = rearrange(value, "b f k l h d -> b h f k l d")
    key_view = key_h[:, :, None, :, :, None, :, :].expand(
        batch,
        heads,
        query_frames,
        key_frames,
        rows,
        query_cols,
        width,
        key_h.size(-1),
    )
    value_view = value_h[:, :, None, :, :, None, :, :].expand_as(key_view)
    gather_index = selected_index.unsqueeze(-1).expand(
        *selected_index.shape, key_h.size(-1)
    )
    singleton_key = key_view.gather(dim=-2, index=gather_index).float()
    singleton_value = value_view.gather(dim=-2, index=gather_index).float()
    singleton_bias = torch.zeros_like(selected_index, dtype=bulk_bias.dtype)
    return (
        torch.cat((bulk_key, singleton_key), dim=-2),
        torch.cat((bulk_value, singleton_value), dim=-2),
        torch.cat((bulk_bias, singleton_bias), dim=-1),
    )


def per_parent_leverage_candidates(state, key, value):
    """Choose one splittable singleton candidate inside every C4 parent."""

    weights = state["weights"]
    assignment = state["assignment"]
    seed_scores = state["seed_scores"]
    mean_score = torch.einsum("...cl,...sl->...cs", weights, seed_scores)
    point_norm = seed_scores.square().sum(dim=-2)
    mean_norm = mean_score.square().sum(dim=-1)
    cross = torch.einsum("...cs,...sl->...cl", mean_score, seed_scores)
    distance = (
        mean_norm.unsqueeze(-1) + point_norm.unsqueeze(-2) - 2.0 * cross
    ).clamp_min(0.0)
    leverage = weights / (1.0 - weights).clamp_min(1e-6) * distance
    leverage = leverage.masked_fill(~assignment, float("-inf"))
    candidate_index = leverage.argmax(dim=-1)
    candidate_score = leverage.gather(
        dim=-1, index=candidate_index.unsqueeze(-1)
    ).squeeze(-1)

    width = seed_scores.size(-1)
    candidate_mask = F.one_hot(candidate_index, num_classes=width).bool()
    candidate_mask &= assignment
    bulk_mask = assignment & ~candidate_mask
    bulk_key, bulk_value, bulk_bias, bulk_mass, _ = _conditional_child(
        weights,
        state["log_weights"],
        bulk_mask,
        key,
        value,
    )
    singleton_key = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", candidate_mask.to(key.dtype), key
    )
    singleton_value = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", candidate_mask.to(value.dtype), value
    )
    candidate_weight = weights.gather(
        dim=-1, index=candidate_index.unsqueeze(-1)
    ).squeeze(-1)
    valid = assignment.sum(dim=-1) > 1
    valid &= bulk_mass > 0
    return {
        "bulk_key": rearrange(
            bulk_key, "b a f j k c h d -> b h a f k j c d"
        ).float(),
        "bulk_value": rearrange(
            bulk_value, "b a f j k c h d -> b h a f k j c d"
        ).float(),
        "bulk_bias": bulk_bias,
        "singleton_key": rearrange(
            singleton_key, "b a f j k c h d -> b h a f k j c d"
        ).float(),
        "singleton_value": rearrange(
            singleton_value, "b a f j k c h d -> b h a f k j c d"
        ).float(),
        "candidate_weight": candidate_weight,
        "candidate_score": candidate_score,
        "valid": valid,
    }


def query_adaptive_parent_split_output(
    query,
    state,
    candidates,
    scale,
    refinements=2,
    allocation="mass_gain",
):
    """Allocate a fixed split budget independently for every query row."""

    parent_key = rearrange(
        state["parent_key"], "b a f j k c h d -> b h a f k j c d"
    ).float()
    parent_value = rearrange(
        state["parent_value"], "b a f j k c h d -> b h a f k j c d"
    ).float()
    query_h = rearrange(query, "b a i j h d -> b h a j i d").float()

    parent_dot = torch.einsum(
        "bhajid,bhafkjcd->bhajifkc", query_h, parent_key
    ) * float(scale)
    singleton_dot = torch.einsum(
        "bhajid,bhafkjcd->bhajifkc", query_h, candidates["singleton_key"]
    ) * float(scale)
    parent_bias = rearrange(
        state["parent_bias"], "b h a f k j c -> b h a j f k c"
    ).unsqueeze(-4)
    bulk_bias = rearrange(
        candidates["bulk_bias"], "b h a f k j c -> b h a j f k c"
    ).unsqueeze(-4)
    alpha = rearrange(
        candidates["candidate_weight"],
        "b h a f k j c -> b h a j f k c",
    ).unsqueeze(-4)
    valid = rearrange(
        candidates["valid"], "b h a f k j c -> b h a j f k c"
    ).unsqueeze(-4)

    parent_logit = parent_dot - parent_bias
    bulk_dot = (parent_dot - alpha * singleton_dot) / (1.0 - alpha).clamp_min(
        1e-6
    )
    bulk_logit = bulk_dot - bulk_bias
    split_logit = torch.logaddexp(bulk_logit, singleton_dot)

    if allocation == "mass_gain":
        common_shift = torch.maximum(parent_logit, split_logit).amax(
            dim=-1, keepdim=True
        )
        allocation_score = (
            torch.exp(split_logit - common_shift)
            - torch.exp(parent_logit - common_shift)
        ).clamp_min(0.0)
    elif allocation == "log_gain":
        allocation_score = split_logit - parent_logit
    elif allocation == "parent_mass":
        allocation_score = parent_logit
    elif allocation == "static_leverage":
        allocation_score = rearrange(
            candidates["candidate_score"],
            "b h a f k j c -> b h a j f k c",
        ).unsqueeze(-4).expand_as(parent_logit)
    else:
        raise ValueError(f"unknown allocation {allocation!r}")
    allocation_score = allocation_score.masked_fill(~valid, float("-inf"))
    selected_index = allocation_score.topk(int(refinements), dim=-1).indices
    selected = torch.zeros_like(allocation_score, dtype=torch.bool)
    selected.scatter_(dim=-1, index=selected_index, value=True)
    selected &= valid

    base_logit = torch.where(selected, bulk_logit, parent_logit)
    extra_logit = singleton_dot.masked_fill(~selected, float("-inf"))
    logits = torch.cat((base_logit, extra_logit), dim=-1)
    probability = torch.softmax(
        rearrange(logits, "b h a j i f k r -> b h a j i (f k r)"), dim=-1
    )
    probability = rearrange(
        probability,
        "b h a j i (f k r) -> b h a j i f k r",
        f=parent_key.size(3),
        k=parent_key.size(4),
    )
    clusters = parent_key.size(-2)
    base_probability = probability[..., :clusters]
    extra_probability = probability[..., clusters:]
    output = torch.einsum(
        "bhajifkc,bhafkjcd->bhajid",
        base_probability * (~selected),
        parent_value,
    )
    output += torch.einsum(
        "bhajifkc,bhafkjcd->bhajid",
        base_probability * selected,
        candidates["bulk_value"],
    )
    output += torch.einsum(
        "bhajifkc,bhafkjcd->bhajid",
        extra_probability,
        candidates["singleton_value"],
    )
    return rearrange(output, "b h a j i d -> b a i j h d").to(query.dtype)


def singleton_peeling_outputs(query, key, value, seeds, scale, count=2):
    state = hard_c4_parent_state(seeds, key, value, scale)
    parent_key = rearrange(
        state["parent_key"], "b a f j k c h d -> b h a f k j c d"
    ).float()
    parent_value = rearrange(
        state["parent_value"], "b a f j k c h d -> b h a f k j c d"
    ).float()
    outputs = {
        "fixed_wide": output_from_representatives(
            query, parent_key, parent_value, state["parent_bias"], scale
        )
    }
    for selector in SELECTORS:
        selected = greedy_singletons(state, count, selector)
        representatives = peeled_representatives(state, key, value, selected)
        outputs[f"singleton_{selector}_r{count}"] = output_from_representatives(
            query, *representatives, scale
        )
    for selector in (
        "routing_leverage_centered",
        "routing_cross_energy",
        "routing_cross_energy_uncentered",
    ):
        selected = static_response_singletons(state, value, count, selector)
        representatives = peeled_representatives(state, key, value, selected)
        outputs[f"singleton_{selector}_static_r{count}"] = (
            output_from_representatives(query, *representatives, scale)
        )
    if count <= state["seed_scores"].size(-2):
        candidates = per_parent_leverage_candidates(state, key, value)
        for allocation in (
            "mass_gain",
            "log_gain",
            "parent_mass",
            "static_leverage",
        ):
            outputs[f"query_adaptive_{allocation}_r{count}"] = (
                query_adaptive_parent_split_output(
                    query,
                    state,
                    candidates,
                    scale,
                    refinements=count,
                    allocation=allocation,
                )
            )
    return outputs


def singleton_budget_curve_outputs(query, key, value, seeds, scale, counts):
    """Evaluate one production-compatible static selector at several budgets."""

    state = hard_c4_parent_state(seeds, key, value, scale)
    parent_key = rearrange(
        state["parent_key"], "b a f j k c h d -> b h a f k j c d"
    ).float()
    parent_value = rearrange(
        state["parent_value"], "b a f j k c h d -> b h a f k j c d"
    ).float()
    outputs = {
        "fixed_wide": output_from_representatives(
            query, parent_key, parent_value, state["parent_bias"], scale
        )
    }
    for count in counts:
        selected = greedy_singletons(
            state, int(count), "routing_leverage_static"
        )
        representatives = peeled_representatives(
            state, key, value, selected
        )
        outputs[f"singleton_routing_leverage_static_r{int(count)}"] = (
            output_from_representatives(query, *representatives, scale)
        )
    return outputs


def install_probe(stats, per_call_stats, per_head_stats, per_call_head_stats, args):
    import wan.modules.model as wan_model

    selected_heads = parse_ints(args.heads)
    record_calls = set(parse_ints(args.record_calls))
    curve_counts = parse_ints(args.curve_counts)
    state = {"calls_total": 0, "grid_size": None, "evaluated": []}

    def probed_self_attention(self, x, grid_size, rope_cache):
        batch, seq_len, heads, dim = *x.shape[:2], self.num_heads, self.head_dim
        q = wan_model.rmsnorm(self.q(x), self.norm_q.weight, eps=self.norm_q.eps).view(
            batch, seq_len, heads, dim
        )
        k = wan_model.rmsnorm(self.k(x), self.norm_k.weight, eps=self.norm_k.eps).view(
            batch, seq_len, heads, dim
        )
        v = self.v(x).view(batch, seq_len, heads, dim)
        q, k = wan_model.apply_rope_video_tokens(q, k, rope_cache)
        dense = wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

        state["calls_total"] += 1
        call = state["calls_total"]
        frames, rows, width = [int(item) for item in grid_size]
        state["grid_size"] = [frames, rows, width]
        if call in record_calls:
            step = (call - 1) // 40
            layer = (call - 1) % 40
            query_frame = (step * 7 + layer * 3) % frames
            scale = dim**-0.5
            if args.fixed_only:
                head_chunks = [
                    selected_heads[start : start + args.probe_head_chunk]
                    for start in range(0, len(selected_heads), args.probe_head_chunk)
                ]
                for chunk_heads in head_chunks:
                    head_index = torch.tensor(chunk_heads, device=q.device)
                    q_blocks = _hard_c4_video_to_blocks(
                        q.index_select(2, head_index).contiguous(), rows, width
                    )
                    k_blocks = _hard_c4_video_to_blocks(
                        k.index_select(2, head_index).contiguous(), rows, width
                    )
                    v_blocks = _hard_c4_video_to_blocks(
                        v.index_select(2, head_index).contiguous(), rows, width
                    )
                    dense_blocks = _hard_c4_video_to_blocks(
                        dense.index_select(2, head_index).contiguous(), rows, width
                    )
                    query = q_blocks[:, query_frame : query_frame + 1]
                    reference = dense_blocks[:, query_frame : query_frame + 1]
                    seeds = fixed_seed_queries(query, frames, rows)
                    output = singleton_budget_curve_outputs(
                        query, k_blocks, v_blocks, seeds, scale, ()
                    )["fixed_wide"]
                    stats["fixed_wide"].add(output, reference)
                    per_call_stats[call]["fixed_wide"].add(output, reference)
                    for position, head in enumerate(chunk_heads):
                        head_output = output[..., position : position + 1, :]
                        head_reference = reference[..., position : position + 1, :]
                        per_head_stats[head]["fixed_wide"].add(
                            head_output, head_reference
                        )
                        per_call_head_stats[call][head]["fixed_wide"].add(
                            head_output, head_reference
                        )
                    del q_blocks, k_blocks, v_blocks, dense_blocks, output
                state["evaluated"].append(
                    {
                        "call": call,
                        "step": step,
                        "layer": layer,
                        "query_frame": query_frame,
                    }
                )
                if args.trajectory == "dense":
                    return self.o(dense.flatten(2))

            head_index = torch.tensor(selected_heads, device=q.device)
            q_blocks = _hard_c4_video_to_blocks(
                q.index_select(2, head_index).contiguous(), rows, width
            )
            k_blocks = _hard_c4_video_to_blocks(
                k.index_select(2, head_index).contiguous(), rows, width
            )
            v_blocks = _hard_c4_video_to_blocks(
                v.index_select(2, head_index).contiguous(), rows, width
            )
            dense_blocks = _hard_c4_video_to_blocks(
                dense.index_select(2, head_index).contiguous(), rows, width
            )
            query = q_blocks[:, query_frame : query_frame + 1]
            reference = dense_blocks[:, query_frame : query_frame + 1]
            seeds = fixed_seed_queries(query, frames, rows)
            if curve_counts:
                candidates = singleton_budget_curve_outputs(
                    query,
                    k_blocks,
                    v_blocks,
                    seeds,
                    scale,
                    curve_counts,
                )
                if args.compare_c5_r2:
                    c5_seeds = fixed_contiguous_seed_queries(
                        query, frames, rows, clusters=5
                    )
                    c5_candidates = singleton_budget_curve_outputs(
                        query,
                        k_blocks,
                        v_blocks,
                        c5_seeds,
                        scale,
                        (2,),
                    )
                    candidates["c5_fixed_contiguous"] = c5_candidates["fixed_wide"]
                    candidates["c5_singleton_routing_leverage_static_r2"] = (
                        c5_candidates["singleton_routing_leverage_static_r2"]
                    )
            else:
                candidates = singleton_peeling_outputs(
                    query,
                    k_blocks,
                    v_blocks,
                    seeds,
                    scale,
                    count=args.singletons,
                )
            for name, output in candidates.items():
                stats[name].add(output, reference)
                per_call_stats[call][name].add(output, reference)
                for position, head in enumerate(selected_heads):
                    head_output = output[..., position : position + 1, :]
                    head_reference = reference[..., position : position + 1, :]
                    per_head_stats[head][name].add(head_output, head_reference)
                    per_call_head_stats[call][head][name].add(
                        head_output, head_reference
                    )
            state["evaluated"].append(
                {"call": call, "step": step, "layer": layer, "query_frame": query_frame}
            )
        if args.trajectory == "dense":
            trajectory_output = dense
        elif args.trajectory == "c4":
            q_blocks = _hard_c4_video_to_blocks(q, rows, width)
            k_blocks = _hard_c4_video_to_blocks(k, rows, width)
            v_blocks = _hard_c4_video_to_blocks(v, rows, width)
            c4_blocks = torch.empty_like(q_blocks)
            for head_start in range(heads):
                hard_c4_720p_attention_triton_blocks(
                    q_blocks,
                    k_blocks,
                    v_blocks,
                    dim**-0.5,
                    output=c4_blocks,
                    head_start=head_start,
                    head_count=1,
                )
            trajectory_output = _hard_c4_blocks_to_video(c4_blocks)
        else:
            raise ValueError(f"unknown trajectory {args.trajectory!r}")
        return self.o(trajectory_output.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


def theoretical_flops(singletons, svg_density):
    rows, width, clusters = 45, 80, 4
    representatives = clusters + int(singletons)
    conservative_ratio = (
        clusters * width
        + 2 * representatives * width
        + 2 * rows * representatives
    ) / (2 * rows * width)
    ratio = (
        3 * clusters * width + 2 * rows * representatives
    ) / (2 * rows * width)
    cross_leverage_ratio = ratio + width / (2 * rows * width)
    return {
        "representatives": representatives,
        "ratio_vs_dense": ratio,
        "ratio_vs_svg_ear": ratio / svg_density,
        "conservative_ratio_vs_dense": conservative_ratio,
        "cross_leverage_ratio_vs_dense": cross_leverage_ratio,
        "cross_leverage_ratio_vs_svg_ear": cross_leverage_ratio / svg_density,
        "limit_ratio_vs_dense": 0.70 * svg_density,
        "passes_limit": ratio <= 0.70 * svg_density,
        "cross_leverage_passes_limit": cross_leverage_ratio <= 0.70 * svg_density,
        "selector_note": "Routing-score leverage reuses the four seed logits.",
        "cross_leverage_note": (
            "Value residuals add one owner-segmented width pass."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dense_wan_14b_720p_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument(
        "--record_calls",
        default="1,20,40,41,60,80,81,100,120,121,140,160",
    )
    parser.add_argument("--singletons", type=int, default=2)
    parser.add_argument(
        "--curve_counts",
        default="",
        help="Comma-separated singleton counts for a static-leverage budget curve.",
    )
    parser.add_argument(
        "--compare_c5_r2",
        action="store_true",
        help="Also evaluate production-style contiguous C5 and C5+2S.",
    )
    parser.add_argument(
        "--fixed_only",
        action="store_true",
        help="Evaluate only fixed C4, processing selected heads in bounded chunks.",
    )
    parser.add_argument("--probe_head_chunk", type=int, default=4)
    parser.add_argument("--trajectory", choices=("dense", "c4"), default="dense")
    parser.add_argument("--svg_ear_density", type=float, default=0.37906453310046345)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config, pipeline = load_pipeline(
        args.config, args.checkpoint, dtype, torch.device("cuda"), args.use_ema
    )
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]
    stats = defaultdict(ErrorStats)
    per_call_stats = defaultdict(lambda: defaultdict(ErrorStats))
    per_head_stats = defaultdict(lambda: defaultdict(ErrorStats))
    per_call_head_stats = defaultdict(
        lambda: defaultdict(lambda: defaultdict(ErrorStats))
    )
    state = install_probe(
        stats, per_call_stats, per_head_stats, per_call_head_stats, args
    )

    sync()
    started = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()
    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "prompt_index": args.prompt_index,
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "grid_size": state["grid_size"],
        "calls_total": state["calls_total"],
        "record_calls": parse_ints(args.record_calls),
        "heads": parse_ints(args.heads),
        "wall_seconds": time.perf_counter() - started,
        "errors_vs_dense": summarize(stats),
        "per_call_errors_vs_dense": {
            str(call): summarize(values)
            for call, values in sorted(per_call_stats.items())
        },
        "per_head_errors_vs_dense": {
            str(head): summarize(values)
            for head, values in sorted(per_head_stats.items())
        },
        "per_call_head_errors_vs_dense": {
            str(call): {
                str(head): summarize(values)
                for head, values in sorted(heads.items())
            }
            for call, heads in sorted(per_call_head_stats.items())
        },
        "flops": (
            {
                str(count): theoretical_flops(count, args.svg_ear_density)
                for count in parse_ints(args.curve_counts)
            }
            if parse_ints(args.curve_counts)
            else theoretical_flops(args.singletons, args.svg_ear_density)
        ),
        "c5_r2_flops": (
            {
                "ratio_vs_dense": (3 * 5 * 80 + 2 * 45 * (5 + 2))
                / (2 * 45 * 80),
                "ratio_vs_svg_ear": (
                    (3 * 5 * 80 + 2 * 45 * (5 + 2))
                    / (2 * 45 * 80)
                    / args.svg_ear_density
                ),
            }
            if args.compare_c5_r2
            else None
        ),
        "derivation": {
            "mass_order": "Z_parent_hat(q) <= Z_peeled_hat(q) <= Z_exact(q)",
            "leverage": "w/(1-w) * ||routing_score - cluster_mean||^2",
            "trajectory": args.trajectory,
            "intended_schedule": "all 40 layers at every denoise step",
        },
        "per_call": state["evaluated"],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
