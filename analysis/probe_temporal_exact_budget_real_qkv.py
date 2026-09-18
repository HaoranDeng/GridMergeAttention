"""Screen temporal exact-frame budgets on real Wan2.1-14B StepDistill QKV."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange

from analysis.compare_hard_c4_backends_dense import load_pipeline, run_denoise_only, sync
from analysis.probe_singleton_peeling_gma_real_qkv import (
    greedy_singletons,
    hard_c4_parent_state,
    peeled_representatives,
)
from attention.hard_route_pytorch import (
    gather_seed_queries,
    hard_route_merge_kv_pytorch,
    wide_local_c4_seed_rows,
)
from attention.hard_c4_wan21_14b_720p_triton import (
    hard_c4_720p_attention_triton_blocks,
    hard_c4_720p_exact_intra_frame_attention_triton_blocks,
)
from attention.singleton_c4_wan21_14b_720p_triton import (
    hard_c4_720p_singleton_leverage_attention_triton_blocks,
    hard_c4_720p_singleton_leverage_top3_exact_intra_frame_attention_triton_blocks,
    hard_c4_720p_singleton_leverage_top6_attention_triton_blocks,
)
from attention.token_merging import _hard_c4_video_to_blocks
from generate import load_prompts


CANDIDATES = (
    {
        "name": "c4_exact0",
        "offsets": (-2, 0, 1, 3),
        "production_c4": True,
        "exact_frames": 0,
        "temporal_order": "symmetric",
    },
    {
        "name": "c4_exact1",
        "offsets": (-2, 0, 1, 3),
        "production_c4": True,
        "exact_frames": 1,
        "temporal_order": "symmetric",
    },
    {
        "name": "c3_exact2_past",
        "offsets": (-2, 0, 2),
        "exact_frames": 2,
        "temporal_order": "past_first",
    },
    {
        "name": "c3_exact2_future",
        "offsets": (-2, 0, 2),
        "exact_frames": 2,
        "temporal_order": "future_first",
    },
    {
        "name": "c2_exact3_centered",
        "offsets": (-1, 1),
        "exact_frames": 3,
        "temporal_order": "symmetric",
    },
    {
        "name": "c2_exact3_past_seed",
        "offsets": (-2, 0),
        "exact_frames": 3,
        "temporal_order": "symmetric",
    },
    {
        "name": "c2_exact3_future_seed",
        "offsets": (0, 2),
        "exact_frames": 3,
        "temporal_order": "symmetric",
    },
    {
        "name": "c1_exact4",
        "offsets": (0,),
        "exact_frames": 4,
        "temporal_order": "symmetric",
    },
)

REFINED_CANDIDATES = (
    {
        "name": "c4_exact0_singleton6",
        "exact_frames": 0,
        "singletons": 6,
    },
    {
        "name": "c4_exact1_singleton3",
        "exact_frames": 1,
        "singletons": 3,
    },
)


class ErrorStats:
    def __init__(self):
        self.calls = 0
        self.sqerr = 0.0
        self.ref_sq = 0.0
        self.numel = 0
        self.max_abs = 0.0

    def add(self, output, reference):
        difference = output.float() - reference.float()
        self.calls += 1
        self.sqerr += float(difference.square().sum())
        self.ref_sq += float(reference.float().square().sum())
        self.numel += difference.numel()
        self.max_abs = max(self.max_abs, float(difference.abs().max()))

    def summary(self):
        return {
            "calls": self.calls,
            "numel": self.numel,
            "mse_vs_dense": self.sqerr / self.numel,
            "rel_l2_vs_dense": math.sqrt(self.sqerr / max(self.ref_sq, 1e-20)),
            "max_abs": self.max_abs,
        }


def parse_ints(value):
    return [int(item) for item in value.replace(",", " ").split()]


def spaced_seed_rows(query_frame, frames, rows, width, offsets, device):
    query_outer = torch.arange(query_frame, query_frame + 1, device=device)
    key_outer = torch.arange(frames, device=device)
    if tuple(offsets) == (-2, 0, 1, 3):
        return wide_local_c4_seed_rows(
            query_outer, key_outer, rows, width, rows
        )

    offsets = torch.tensor(offsets, device=device)
    minimum = int(offsets.min())
    maximum = int(offsets.max())
    span = maximum - minimum + 1
    key_row = torch.arange(rows, device=device).view(1, 1, rows, 1, 1)
    start = (key_row + minimum).clamp(0, rows - span)
    positions = (offsets - minimum).view(1, 1, 1, 1, -1)
    seed_rows = (start + positions).long()
    return seed_rows.expand(1, frames, rows, width, offsets.numel())


def ordered_frames(query_frame, frames, mode):
    order = [query_frame]
    for distance in range(1, frames):
        directions = (-distance, distance)
        if mode == "future_first":
            directions = (distance, -distance)
        for offset in directions:
            frame = query_frame + offset
            if 0 <= frame < frames:
                order.append(frame)
    return order


def mixed_output(
    query,
    key_blocks,
    value_blocks,
    merged_key,
    merged_value,
    routing_bias,
    exact_frames,
    scale,
):
    exact_frames = torch.tensor(
        exact_frames, device=query.device, dtype=torch.long
    )
    remote_mask = torch.ones(key_blocks.size(1), device=query.device, dtype=torch.bool)
    remote_mask[exact_frames] = False
    remote_frames = remote_mask.nonzero(as_tuple=False).flatten()

    query_h = rearrange(query, "b a i j h d -> b h a j i d").float()
    remote_key_h = rearrange(
        merged_key.index_select(2, remote_frames),
        "b a f j k c h d -> b h a j (f k c) d",
    ).float()
    remote_value_h = rearrange(
        merged_value.index_select(2, remote_frames),
        "b a f j k c h d -> b h a j (f k c) d",
    ).float()
    remote_bias_h = rearrange(
        routing_bias.index_select(3, remote_frames),
        "b h a f k j c -> b h a j (f k c)",
    ).float()
    exact_key_h = rearrange(
        key_blocks.index_select(1, exact_frames),
        "b f k l h d -> b h (f k l) d",
    ).float()
    exact_value_h = rearrange(
        value_blocks.index_select(1, exact_frames),
        "b f k l h d -> b h (f k l) d",
    ).float()

    remote_logits = (
        torch.einsum("bhajid,bhajrd->bhajir", query_h, remote_key_h) * scale
        - remote_bias_h.unsqueeze(-2)
    )
    exact_logits = torch.einsum(
        "bhajid,bhld->bhajil", query_h, exact_key_h
    ) * scale
    probabilities = torch.softmax(
        torch.cat((remote_logits, exact_logits), dim=-1), dim=-1
    )
    remote_count = remote_logits.size(-1)
    output = torch.einsum(
        "bhajir,bhajrd->bhajid",
        probabilities[..., :remote_count],
        remote_value_h,
    )
    output += torch.einsum(
        "bhajil,bhld->bhajid",
        probabilities[..., remote_count:],
        exact_value_h,
    )
    return rearrange(output, "b h a j i d -> b a i j h d").to(query.dtype)


def install_probe(stats, triton_validation, args):
    import wan.modules.model as wan_model

    selected_heads = parse_ints(args.heads)
    record_calls = set(parse_ints(args.record_calls))
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
        dense = wan_model.flash_attention(
            q=q, k=k, v=v, window_size=self.window_size
        )

        state["calls_total"] += 1
        call = state["calls_total"]
        frames, rows, width = [int(item) for item in grid_size]
        state["grid_size"] = [frames, rows, width]
        if call in record_calls:
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
            step = (call - 1) // 40
            layer = (call - 1) % 40
            query_frame = (step * 7 + layer * 3) % frames
            query = q_blocks[:, query_frame : query_frame + 1]
            reference = dense_blocks[:, query_frame : query_frame + 1]
            scale = dim**-0.5

            grouped = defaultdict(list)
            for candidate in CANDIDATES:
                grouped[tuple(candidate["offsets"])].append(candidate)
            for offsets, candidates in grouped.items():
                seed_rows = spaced_seed_rows(
                    query_frame, frames, rows, width, offsets, query.device
                )
                seed_queries = gather_seed_queries(query, seed_rows)
                merged_key, merged_value, routing_bias = (
                    hard_route_merge_kv_pytorch(
                        seed_queries, k_blocks, v_blocks, scale
                    )
                )
                for candidate in candidates:
                    exact_frames = ordered_frames(
                        query_frame, frames, candidate["temporal_order"]
                    )[: candidate["exact_frames"]]
                    output = mixed_output(
                        query,
                        k_blocks,
                        v_blocks,
                        merged_key,
                        merged_value,
                        routing_bias,
                        exact_frames,
                        scale,
                    )
                    stats[candidate["name"]].add(output, reference)
                    if (
                        candidate["name"] == "c4_exact0"
                        and call == args.validate_triton_call
                    ):
                        triton_hard_c4 = hard_c4_720p_attention_triton_blocks(
                            q_blocks,
                            k_blocks,
                            v_blocks,
                            scale=scale,
                        )
                        triton_validation["hard_c4_vs_pytorch"].add(
                            triton_hard_c4[:, query_frame : query_frame + 1],
                            output,
                        )
                    if (
                        candidate["name"] == "c4_exact1"
                        and call == args.validate_triton_call
                    ):
                        triton_exact = (
                            hard_c4_720p_exact_intra_frame_attention_triton_blocks(
                                q_blocks,
                                k_blocks,
                                v_blocks,
                                scale=scale,
                                query_outer_offset=0,
                            )
                        )
                        triton_validation["same_frame_exact_vs_pytorch"].add(
                            triton_exact[:, query_frame : query_frame + 1],
                            output,
                        )

                if tuple(offsets) == (-2, 0, 1, 3):
                    parent_state = hard_c4_parent_state(
                        seed_queries, k_blocks, v_blocks, scale
                    )
                    for candidate in REFINED_CANDIDATES:
                        selected = greedy_singletons(
                            parent_state,
                            candidate["singletons"],
                            "routing_leverage_static",
                        )
                        refined_key, refined_value, refined_bias = (
                            peeled_representatives(
                                parent_state, k_blocks, v_blocks, selected
                            )
                        )
                        refined_key = rearrange(
                            refined_key,
                            "b h a f k j c d -> b a f j k c h d",
                        )
                        refined_value = rearrange(
                            refined_value,
                            "b h a f k j c d -> b a f j k c h d",
                        )
                        exact_frames = ordered_frames(
                            query_frame, frames, "symmetric"
                        )[: candidate["exact_frames"]]
                        output = mixed_output(
                            query,
                            k_blocks,
                            v_blocks,
                            refined_key,
                            refined_value,
                            refined_bias,
                            exact_frames,
                            scale,
                        )
                        stats[candidate["name"]].add(output, reference)
                        if (
                            candidate["name"] == "c4_exact0_singleton6"
                            and call == args.validate_triton_call
                        ):
                            triton_singleton6 = (
                                hard_c4_720p_singleton_leverage_top6_attention_triton_blocks(
                                    q_blocks,
                                    k_blocks,
                                    v_blocks,
                                    scale=scale,
                                )
                            )
                            triton_validation["singleton6_vs_pytorch"].add(
                                triton_singleton6[
                                    :, query_frame : query_frame + 1
                                ],
                                output,
                            )
                        if (
                            candidate["name"] == "c4_exact1_singleton3"
                            and call == args.validate_triton_call
                        ):
                            triton_output = (
                                hard_c4_720p_singleton_leverage_top3_exact_intra_frame_attention_triton_blocks(
                                    q_blocks,
                                    k_blocks,
                                    v_blocks,
                                    scale=scale,
                                    query_outer_offset=0,
                                )
                            )
                            triton_validation[
                                "combined_vs_pytorch_definition"
                            ].add(
                                triton_output[
                                    :, query_frame : query_frame + 1
                                ],
                                output,
                            )
                            output_without_exact_frame = mixed_output(
                                query,
                                k_blocks,
                                v_blocks,
                                refined_key,
                                refined_value,
                                refined_bias,
                                [],
                                scale,
                            )
                            triton_singleton3 = (
                                hard_c4_720p_singleton_leverage_attention_triton_blocks(
                                    q_blocks,
                                    k_blocks,
                                    v_blocks,
                                    top_r=3,
                                    scale=scale,
                                )
                            )
                            triton_validation["singleton3_vs_pytorch"].add(
                                triton_singleton3[
                                    :, query_frame : query_frame + 1
                                ],
                                output_without_exact_frame,
                            )

            state["evaluated"].append(
                {
                    "call": call,
                    "step": step,
                    "layer": layer,
                    "query_frame": query_frame,
                }
            )
        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


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
    parser.add_argument("--heads", default="0,20")
    parser.add_argument(
        "--record_calls",
        default="1,14,27,40,41,54,67,80,81,94,107,120,121,134,147,160",
    )
    parser.add_argument("--svg_ear_density", type=float, default=0.37906453310046345)
    parser.add_argument(
        "--validate_triton_call",
        type=int,
        default=0,
        help="Compare the fused combined route with its PyTorch definition.",
    )
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required.")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config, pipeline = load_pipeline(
        args.config, args.checkpoint, dtype, torch.device("cuda"), args.use_ema
    )
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]
    stats = defaultdict(ErrorStats)
    triton_validation = defaultdict(ErrorStats)
    state = install_probe(stats, triton_validation, args)
    sync()
    started = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()

    errors = {name: value.summary() for name, value in sorted(stats.items())}
    baseline_mse = errors["c4_exact0"]["mse_vs_dense"]
    for value in errors.values():
        value["mse_ratio_vs_c4_exact0"] = value["mse_vs_dense"] / baseline_mse
    frames, rows, width = state["grid_size"]
    flops = {}
    for candidate in CANDIDATES:
        c = len(candidate["offsets"])
        exact = candidate["exact_frames"]
        merge_ratio = 3 * c / (2 * rows)
        route_ratio = exact / frames + (frames - exact) / frames * c / width
        total_ratio = merge_ratio + route_ratio
        flops[candidate["name"]] = {
            "merge_ratio_vs_dense": merge_ratio,
            "route_ratio_vs_dense": route_ratio,
            "total_ratio_vs_dense": total_ratio,
            "ratio_vs_svg_ear": total_ratio / args.svg_ear_density,
            "passes_70_percent_svg_ear_limit": (
                total_ratio <= 0.7 * args.svg_ear_density
            ),
        }
    for candidate in REFINED_CANDIDATES:
        c = 4
        exact = candidate["exact_frames"]
        representatives = c + candidate["singletons"]
        merge_ratio = 3 * c / (2 * rows)
        route_ratio = (
            exact / frames
            + (frames - exact) / frames * representatives / width
        )
        total_ratio = merge_ratio + route_ratio
        flops[candidate["name"]] = {
            "merge_ratio_vs_dense": merge_ratio,
            "route_ratio_vs_dense": route_ratio,
            "total_ratio_vs_dense": total_ratio,
            "ratio_vs_svg_ear": total_ratio / args.svg_ear_density,
            "passes_70_percent_svg_ear_limit": (
                total_ratio <= 0.7 * args.svg_ear_density
            ),
        }

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
        "errors_vs_dense": errors,
        "production_triton_validation": {
            name: value.summary()
            for name, value in sorted(triton_validation.items())
        },
        "flops": flops,
        "candidates": CANDIDATES,
        "refined_candidates": REFINED_CANDIDATES,
        "invariants": {
            "dense_trajectory_preserved": True,
            "all_exact_and_compressed_key_sets_are_disjoint": True,
            "all_key_frames_are_retained": True,
            "single_joint_softmax": True,
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
