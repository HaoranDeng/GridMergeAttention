"""Validate the production P11 centroid-seed Triton path on native Wan QKV."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from analysis.compare_hard_c4_backends_dense import load_pipeline, run_denoise_only, sync
from analysis.probe_query_centroid_gma_real_qkv import (
    ErrorStats,
    hard_c4_output,
    local_centroid_seed_queries,
)
from attention.hard_c4_wan21_14b_720p_triton import (
    hard_c4_720p_query_centroid_attention_triton_blocks,
)
from attention.token_merging import _hard_c4_video_to_blocks
from generate import load_prompts


def install_validation(stats, args):
    import wan.modules.model as wan_model

    state = {"calls_total": 0, "validated": None, "grid_size": None}

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
        if call == args.record_call:
            head_index = torch.tensor([args.head], device=q.device)
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
            scale = dim**-0.5
            production = hard_c4_720p_query_centroid_attention_triton_blocks(
                q_blocks, k_blocks, v_blocks, scale=scale
            )

            step = (call - 1) // 40
            layer = (call - 1) % 40
            query_frame = (step * 7 + layer * 3) % frames
            query = q_blocks[:, query_frame : query_frame + 1]
            reference_dense = dense_blocks[:, query_frame : query_frame + 1]
            seeds = local_centroid_seed_queries(
                query,
                frames,
                rows,
                candidate_count=11,
                steps=1,
                mode="euclidean",
                normalize=False,
            )
            reference = hard_c4_output(
                query, k_blocks, v_blocks, seeds, scale
            )
            production_slice = production[:, query_frame : query_frame + 1]
            stats["production_vs_reference"].add(production_slice, reference)
            stats["production_vs_dense"].add(production_slice, reference_dense)
            stats["reference_vs_dense"].add(reference, reference_dense)
            state["validated"] = {
                "call": call,
                "step": step,
                "layer": layer,
                "query_frame": query_frame,
                "head": args.head,
            }

        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dense_wan_14b_720p_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--record_call", type=int, default=1)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config, pipeline = load_pipeline(
        args.config, args.checkpoint, dtype, torch.device("cuda"), args.use_ema
    )
    prompt = load_prompts(args.prompt_path, args.prompt_index + 1)[args.prompt_index]
    stats = defaultdict(ErrorStats)
    state = install_validation(stats, args)
    sync()
    started = time.perf_counter()
    run_denoise_only(pipeline, config, prompt, args.seed, dtype)
    sync()
    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed": args.seed,
        "grid_size": state["grid_size"],
        "calls_total": state["calls_total"],
        "validated": state["validated"],
        "wall_seconds": time.perf_counter() - started,
        "errors": {
            name: value.summary() for name, value in sorted(stats.items())
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
