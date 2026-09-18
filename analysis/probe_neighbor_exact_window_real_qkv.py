"""Screen disjoint exact-neighbor windows plus remote C4 on real Wan QKV."""

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
from attention.hard_c4_wan21_14b_720p_triton import (
    hard_c4_720p_exact_intra_frame_attention_triton_blocks,
)
from attention.hard_route_pytorch import (
    FLOAT_NEG_INF,
    gather_seed_queries,
    wide_local_c4_seed_rows,
)
from attention.token_merging import _hard_c4_video_to_blocks
from generate import load_prompts


CANDIDATES = (
    {"name": "local_only", "windows": ()},
    {"name": "past_center48", "windows": (("past", 48, "center"),)},
    {"name": "past_center56", "windows": (("past", 56, "center"),)},
    {"name": "past_center61", "windows": (("past", 61, "center"),)},
    {"name": "future_center61", "windows": (("future", 61, "center"),)},
    {"name": "past_uniform61", "windows": (("past", 61, "uniform"),)},
    {
        "name": "bidirectional_center30",
        "windows": (("nearest_0", 30, "center"), ("nearest_1", 30, "center")),
    },
    {
        "name": "bidirectional_uniform30",
        "windows": (("nearest_0", 30, "uniform"), ("nearest_1", 30, "uniform")),
    },
)


class ErrorStats:
    def __init__(self):
        self.calls = 0
        self.sqerr = 0.0
        self.ref_sq = 0.0
        self.numel = 0

    def add(self, output, reference):
        difference = output.float() - reference.float()
        self.calls += 1
        self.sqerr += float(difference.square().sum())
        self.ref_sq += float(reference.float().square().sum())
        self.numel += difference.numel()

    def summary(self):
        return {
            "calls": self.calls,
            "numel": self.numel,
            "mse_vs_dense": self.sqerr / self.numel,
            "rel_l2_vs_dense": math.sqrt(self.sqerr / max(self.ref_sq, 1e-20)),
        }


def parse_ints(value):
    return [int(item) for item in value.replace(",", " ").split()]


def neighbor_frames(query_frame, frames):
    lower = [frame for frame in range(query_frame - 1, -1, -1)]
    upper = [frame for frame in range(query_frame + 1, frames)]
    nearest = sorted(
        [*lower, *upper], key=lambda frame: (abs(frame - query_frame), frame)
    )
    return {
        "past": lower[0] if lower else upper[0],
        "future": upper[0] if upper else lower[0],
        "nearest_0": nearest[0],
        "nearest_1": nearest[1],
    }


def selected_columns(width, count, mode, device):
    query_columns = torch.arange(width, device=device)
    if mode == "uniform":
        columns = torch.linspace(0, width - 1, count, device=device).round().long()
        return columns.view(1, count).expand(width, count)
    starts = (query_columns - count // 2).clamp(0, width - count)
    return starts[:, None] + torch.arange(count, device=device)[None, :]


def exact_windows(candidate, query_frame, frames, width, device):
    neighbors = neighbor_frames(query_frame, frames)
    return [
        (
            neighbors[role],
            selected_columns(width, count, mode, device),
        )
        for role, count, mode in candidate["windows"]
    ]


def masked_hard_route_merge(seed_queries, key, value, scale, windows):
    logits = torch.einsum(
        "bafkjchd,bfklhd->bhafkjcl", seed_queries, key
    ).float()
    logits *= float(scale)
    _, _, query_frames, key_frames, key_rows, query_cols, _, key_cols = logits.shape
    compress_mask = torch.ones(
        query_frames,
        key_frames,
        key_rows,
        query_cols,
        key_cols,
        device=key.device,
        dtype=torch.bool,
    )
    for frame, columns in windows:
        selected = torch.zeros(
            query_cols, key_cols, device=key.device, dtype=torch.bool
        )
        selected.scatter_(1, columns, True)
        compress_mask[:, frame] &= ~selected.view(1, query_cols, key_cols)
    valid = compress_mask[None, None].unsqueeze(-2)

    assignment_logits = logits.masked_fill(~valid, FLOAT_NEG_INF)
    assignment_scores = torch.log_softmax(assignment_logits, dim=-1)
    best_seed = assignment_scores.argmax(dim=-2, keepdim=True)
    assignment = torch.zeros_like(logits, dtype=torch.bool)
    assignment.scatter_(dim=-2, index=best_seed, value=True)
    assignment &= valid

    routed_logits = logits.masked_fill(~assignment, FLOAT_NEG_INF)
    log_weights = torch.log_softmax(routed_logits, dim=-1).masked_fill(
        ~assignment, 0.0
    )
    weights = log_weights.exp().masked_fill(~assignment, 0.0)
    routing_bias = (weights * log_weights).sum(dim=-1)
    routing_bias = routing_bias.masked_fill(
        ~assignment.any(dim=-1), float("inf")
    )
    merge_weights = weights.to(key.dtype)
    merged_key = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", merge_weights, key
    )
    merged_value = torch.einsum(
        "bhafkjcl,bfklhe->bafjkche", merge_weights, value
    )
    return merged_key, merged_value, routing_bias


def gather_window_tokens(blocks, frame, columns):
    batch, rows, width, heads, dim = blocks[:, frame].shape
    query_cols, count = columns.shape
    source = blocks[:, frame].unsqueeze(2).expand(
        batch, rows, query_cols, width, heads, dim
    )
    index = columns.view(1, 1, query_cols, count, 1, 1).expand(
        batch, rows, query_cols, count, heads, dim
    )
    gathered = source.gather(3, index)
    return rearrange(gathered, "b k j l h d -> b h 1 j (k l) d")


def mixed_output(
    query,
    key_blocks,
    value_blocks,
    merged_key,
    merged_value,
    routing_bias,
    query_frame,
    windows,
    scale,
):
    frames = key_blocks.size(1)
    remote_frames = torch.tensor(
        [frame for frame in range(frames) if frame != query_frame],
        device=query.device,
    )
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

    local_key = rearrange(
        key_blocks[:, query_frame], "b k l h d -> b h 1 1 (k l) d"
    ).expand(-1, -1, 1, query.size(3), -1, -1)
    local_value = rearrange(
        value_blocks[:, query_frame], "b k l h d -> b h 1 1 (k l) d"
    ).expand(-1, -1, 1, query.size(3), -1, -1)
    exact_keys = [local_key]
    exact_values = [local_value]
    for frame, columns in windows:
        exact_keys.append(gather_window_tokens(key_blocks, frame, columns))
        exact_values.append(gather_window_tokens(value_blocks, frame, columns))
    exact_key_h = torch.cat(exact_keys, dim=-2).float()
    exact_value_h = torch.cat(exact_values, dim=-2).float()

    remote_logits = (
        torch.einsum("bhajid,bhajrd->bhajir", query_h, remote_key_h) * scale
        - remote_bias_h.unsqueeze(-2)
    )
    exact_logits = torch.einsum(
        "bhajid,bhajld->bhajil", query_h, exact_key_h
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
        "bhajil,bhajld->bhajid",
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
            seed_rows = wide_local_c4_seed_rows(
                torch.arange(query_frame, query_frame + 1, device=q.device),
                torch.arange(frames, device=q.device),
                rows,
                width,
                rows,
            )
            seed_queries = gather_seed_queries(query, seed_rows)
            scale = dim**-0.5

            for candidate in CANDIDATES:
                windows = exact_windows(
                    candidate, query_frame, frames, width, query.device
                )
                merged_key, merged_value, routing_bias = masked_hard_route_merge(
                    seed_queries, k_blocks, v_blocks, scale, windows
                )
                output = mixed_output(
                    query,
                    k_blocks,
                    v_blocks,
                    merged_key,
                    merged_value,
                    routing_bias,
                    query_frame,
                    windows,
                    scale,
                )
                stats[candidate["name"]].add(output, reference)
                if (
                    call == args.validate_triton_call
                    and candidate["name"] == "bidirectional_center30"
                ):
                    triton_output = (
                        hard_c4_720p_exact_intra_frame_attention_triton_blocks(
                            q_blocks,
                            k_blocks,
                            v_blocks,
                            scale=scale,
                            exact_neighbor_window_columns=30,
                        )
                    )[:, query_frame : query_frame + 1]
                    triton_validation["vs_pytorch_definition"].add(
                        triton_output, output
                    )
                    triton_validation["vs_dense"].add(
                        triton_output, reference
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
    parser.add_argument("--validate_triton_call", type=int, default=0)
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
    baseline_mse = errors["local_only"]["mse_vs_dense"]
    for value in errors.values():
        value["mse_ratio_vs_local_only"] = value["mse_vs_dense"] / baseline_mse
    triton_summary = {
        name: value.summary()
        for name, value in sorted(triton_validation.items())
    }
    frames, rows, width = state["grid_size"]
    flops = {}
    for candidate in CANDIDATES:
        exact_columns = sum(window[1] for window in candidate["windows"])
        merge_ratio = 3 * 4 / (2 * rows)
        route_tokens = (frames - 1) * rows * 4 + rows * width
        route_tokens += rows * exact_columns
        route_ratio = route_tokens / (frames * rows * width)
        total_ratio = merge_ratio + route_ratio
        flops[candidate["name"]] = {
            "extra_exact_columns": exact_columns,
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
        "production_triton_validation": triton_summary,
        "flops": flops,
        "candidates": CANDIDATES,
        "invariants": {
            "dense_trajectory_preserved": True,
            "exact_and_compressed_key_sets_are_disjoint": True,
            "same_frame_is_fully_exact": True,
            "all_remote_frames_are_retained": True,
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
