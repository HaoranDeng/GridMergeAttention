"""Probe exact intra-frame plus compressed inter-frame C4 on StepDistill QKV.

For each sampled query frame, keys in the same latent frame remain exact while
all other frames use the ordinary entropy-corrected Hard-C4 representatives.
Both sets participate in one softmax, so remote attention is preserved rather
than replaced by a local window.
"""

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
    hard_c4_720p_attention_triton_blocks,
    hard_c4_720p_exact_intra_frame_attention_triton_blocks,
)
from attention.hard_route_pytorch import (
    gather_seed_queries,
    hard_route_merge_kv_pytorch,
    wide_local_c4_seed_rows,
)
from attention.token_merging import _hard_c4_video_to_blocks
from generate import load_prompts


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
            "mse_vs_dense": self.sqerr / self.numel if self.numel else 0.0,
            "rel_l2_vs_dense": math.sqrt(self.sqerr / max(self.ref_sq, 1e-20)),
        }


def parse_ints(value):
    return [int(item) for item in value.replace(",", " ").replace(":", " ").split()]


def routed_output(query, key, value, bias, scale):
    query_h = rearrange(query, "b a i j h d -> b h a j i d").float()
    key_h = rearrange(key, "b a f j k c h d -> b h a j (f k c) d").float()
    value_h = rearrange(value, "b a f j k c h d -> b h a j (f k c) d").float()
    bias_h = rearrange(bias, "b h a f k j c -> b h a j (f k c)").float()
    logits = torch.einsum("bhajid,bhajrd->bhajir", query_h, key_h) * scale
    probabilities = torch.softmax(logits - bias_h.unsqueeze(-2), dim=-1)
    output = torch.einsum("bhajir,bhajrd->bhajid", probabilities, value_h)
    return rearrange(output, "b h a j i d -> b a i j h d").to(query.dtype)


def exact_local_remote_output(
    query,
    key_blocks,
    value_blocks,
    merged_key,
    merged_value,
    routing_bias,
    query_frame,
    scale,
):
    remote_frames = torch.cat(
        (
            torch.arange(query_frame, device=query.device),
            torch.arange(query_frame + 1, key_blocks.size(1), device=query.device),
        )
    )
    remote_key = merged_key.index_select(2, remote_frames)
    remote_value = merged_value.index_select(2, remote_frames)
    remote_bias = routing_bias.index_select(3, remote_frames)

    query_h = rearrange(query, "b a i j h d -> b h a j i d").float()
    remote_key_h = rearrange(
        remote_key, "b a f j k c h d -> b h a j (f k c) d"
    ).float()
    remote_value_h = rearrange(
        remote_value, "b a f j k c h d -> b h a j (f k c) d"
    ).float()
    remote_bias_h = rearrange(
        remote_bias, "b h a f k j c -> b h a j (f k c)"
    ).float()
    local_key_h = rearrange(
        key_blocks[:, query_frame], "b k l h d -> b h (k l) d"
    ).float()
    local_value_h = rearrange(
        value_blocks[:, query_frame], "b k l h d -> b h (k l) d"
    ).float()

    remote_logits = (
        torch.einsum("bhajid,bhajrd->bhajir", query_h, remote_key_h) * scale
        - remote_bias_h.unsqueeze(-2)
    )
    local_logits = torch.einsum("bhajid,bhld->bhajil", query_h, local_key_h) * scale
    probabilities = torch.softmax(torch.cat((remote_logits, local_logits), dim=-1), dim=-1)
    remote_count = remote_logits.size(-1)
    remote_probability = probabilities[..., :remote_count]
    local_probability = probabilities[..., remote_count:]
    output = torch.einsum(
        "bhajir,bhajrd->bhajid", remote_probability, remote_value_h
    ) + torch.einsum("bhajil,bhld->bhajid", local_probability, local_value_h)
    return rearrange(output, "b h a j i d -> b a i j h d").to(query.dtype)


def install_probe(stats, triton_validation, args):
    import wan.modules.model as wan_model

    selected_heads = parse_ints(args.heads)
    record_calls = set(parse_ints(args.record_calls))
    state = {"calls_total": 0, "grid_size": None, "evaluated": []}

    def dense_attention(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

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
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        dense = dense_attention(q, k, v, self)

        state["calls_total"] += 1
        call = state["calls_total"]
        frames, rows, width = [int(item) for item in grid_size]
        state["grid_size"] = [frames, rows, width]
        if call in record_calls:
            for head in selected_heads:
                if not 0 <= head < heads:
                    raise ValueError(f"head index {head} is outside [0, {heads})")
            step = (call - 1) // 40
            layer = (call - 1) % 40
            query_frame = (step * 7 + layer * 3) % frames
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
            seed_rows = wide_local_c4_seed_rows(
                torch.arange(query_frame, query_frame + 1, device=query.device),
                torch.arange(frames, device=query.device),
                rows,
                width,
                rows,
            )
            seed_queries = gather_seed_queries(query, seed_rows)
            scale = dim**-0.5
            merged_key, merged_value, routing_bias = hard_route_merge_kv_pytorch(
                seed_queries, k_blocks, v_blocks, scale
            )
            hard = routed_output(
                query, merged_key, merged_value, routing_bias, scale
            )
            exact_local_remote = exact_local_remote_output(
                query,
                k_blocks,
                v_blocks,
                merged_key,
                merged_value,
                routing_bias,
                query_frame,
                scale,
            )
            if call == args.validate_triton_call:
                exact_triton_output = (
                    hard_c4_720p_exact_intra_frame_attention_triton_blocks(
                        q_blocks,
                        k_blocks,
                        v_blocks,
                        scale=scale,
                    )
                )
                hard_triton_output = hard_c4_720p_attention_triton_blocks(
                    q_blocks,
                    k_blocks,
                    v_blocks,
                    scale=scale,
                )
                exact_triton_frame = exact_triton_output[
                    :, query_frame : query_frame + 1
                ]
                hard_triton_frame = hard_triton_output[
                    :, query_frame : query_frame + 1
                ]
                triton_validation["exact_vs_pytorch_definition"].add(
                    exact_triton_frame,
                    exact_local_remote,
                )
                triton_validation["hard_vs_pytorch_definition"].add(
                    hard_triton_frame,
                    hard,
                )
                triton_validation["exact_vs_dense"].add(
                    exact_triton_frame,
                    reference,
                )
                triton_validation["hard_vs_dense"].add(
                    hard_triton_frame,
                    reference,
                )
            stats["hard_c4"].add(hard, reference)
            stats["exact_intra_frame_remote_c4"].add(exact_local_remote, reference)
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
    parser = argparse.ArgumentParser(
        description="Probe exact intra-frame plus compressed remote Hard C4."
    )
    parser.add_argument("--config", default="configs/dense_wan_14b_720p_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
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
        help="Compare the production Triton route with the PyTorch definition on this call.",
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
    start = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()

    errors = {name: value.summary() for name, value in sorted(stats.items())}
    hard_mse = errors["hard_c4"]["mse_vs_dense"]
    for result in errors.values():
        result["mse_ratio_vs_hard_c4"] = result["mse_vs_dense"] / hard_mse
    triton_summary = {}
    for name, value in sorted(triton_validation.items()):
        summary = value.summary()
        triton_summary[name] = {
            "calls": summary["calls"],
            "numel": summary["numel"],
            "mse": summary["mse_vs_dense"],
            "relative_l2": summary["rel_l2_vs_dense"],
        }
    frames, rows, width = state["grid_size"]
    merge_ratio = 3 * 4 / (2 * rows)
    route_ratio = ((frames - 1) / frames) * (4 / width) + 1 / frames
    total_ratio = merge_ratio + route_ratio
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
        "wall_seconds": time.perf_counter() - start,
        "errors_vs_dense": errors,
        "production_triton_validation": triton_summary,
        "flops": {
            "conservative_merge_ratio_vs_dense": merge_ratio,
            "exact_local_plus_remote_route_ratio_vs_dense": route_ratio,
            "total_ratio_vs_dense": total_ratio,
            "svg_ear_realized_density": args.svg_ear_density,
            "ratio_vs_svg_ear": total_ratio / args.svg_ear_density,
            "passes_70_percent_svg_ear_limit": total_ratio
            <= 0.7 * args.svg_ear_density,
        },
        "invariants": {
            "all_layers_replaced": True,
            "same_frame_keys_exact": True,
            "all_remote_frames_retained_as_c4_representatives": True,
            "single_joint_softmax": True,
        },
        "per_call": state["evaluated"],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
