import argparse
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from analysis.attention_variant_metrics import (
    AttentionMetricCollector,
    dense_attention_for_column,
    denoise_steps,
    hard_route_distribution,
    load_generator_checkpoint,
    load_prompts,
    patch_self_attention_forward,
)
from attention.common import random_seed_rows
from attention.hard_route import assign_keys_to_seeds
from utils.misc import set_seed
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate row-local routing masks for hard-route C4 attention on real Wan QKV samples."
    )
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--sample_groups_per_call", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 2, 4, 8])
    return parser.parse_args()


def hard_route_row_local_distribution(
    q_grid,
    k_grid,
    v_grid,
    frame,
    col,
    head,
    num_merge_tokens,
    row_radius,
    scale,
    include_ith=True,
):
    frames, rows, width, _, _ = q_grid.shape
    sqrt_scale = math.sqrt(scale)

    query_outer = torch.tensor([frame], device=q_grid.device)
    key_outer = torch.arange(frames, device=q_grid.device)
    seed_rows = random_seed_rows(
        query_outer,
        key_outer,
        rows,
        width,
        rows,
        num_merge_tokens,
        seed=0,
        include_ith=include_ith,
    )[0, :, :, col]

    seed_queries = q_grid[frame, seed_rows, col, head].float() * sqrt_scale
    key_blocks = k_grid[:, :, :, head].float() * sqrt_scale
    value_blocks = v_grid[:, :, :, head].float()
    query_rows = q_grid[frame, :, col, head].float() * sqrt_scale

    seed_logits = torch.einsum("frcd,frld->frcl", seed_queries, key_blocks)
    assignment_mask = assign_keys_to_seeds(
        seed_logits[None, None, None, :, :, None, :, :]
    )[0, 0, 0, :, :, 0]

    masked_logits = seed_logits.masked_fill(~assignment_mask, torch.finfo(seed_logits.dtype).min)
    merge_logprob = torch.log_softmax(masked_logits, dim=-1)
    merge_weights = merge_logprob.exp().masked_fill(~assignment_mask, 0)
    merge_logprob = merge_logprob.masked_fill(~assignment_mask, 0)

    merged_keys = torch.einsum("frcl,frld->frcd", merge_weights, key_blocks)
    merged_values = torch.einsum("frcl,frld->frcd", merge_weights, value_blocks)
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)

    routing_logits = torch.einsum("frcd,id->ifrc", merged_keys, query_rows)
    routing_logits = routing_logits - routing_bias.unsqueeze(0)

    query_row_ids = torch.arange(rows, device=q_grid.device)
    key_row_ids = torch.arange(rows, device=q_grid.device)
    row_mask = (key_row_ids[None, :] - query_row_ids[:, None]).abs() <= int(row_radius)
    routing_logits = routing_logits.masked_fill(
        ~row_mask[:, None, :, None],
        torch.finfo(routing_logits.dtype).min,
    )
    routing_weights = torch.softmax(routing_logits.flatten(1), dim=-1).view(
        rows, frames, rows, num_merge_tokens
    )

    expanded = routing_weights[:, :, :, :, None] * merge_weights[None]
    approx_probs = expanded.sum(dim=3).reshape(rows, frames * rows * width)
    approx_outputs = torch.einsum("ifrc,frcd->id", routing_weights, merged_values)
    return approx_probs, approx_outputs


class RowLocalMetricCollector(AttentionMetricCollector):
    def __init__(self, sample_groups_per_call, radii):
        super().__init__(sample_groups_per_call)
        self.radii = [int(radius) for radius in radii]

    @torch.no_grad()
    def observe(self, q, k, v, grid_size, layer_idx):
        del layer_idx
        frames, rows, width = (int(x) for x in grid_size)
        heads, dim = q.size(2), q.size(3)
        self.grid = {
            "frames": frames,
            "rows": rows,
            "cols": width,
            "heads": heads,
            "head_dim": dim,
        }
        scale = dim ** -0.5

        q_grid = q[0].reshape(frames, rows, width, heads, dim)
        k_grid = k[0].reshape(frames, rows, width, heads, dim)
        v_grid = v[0].reshape(frames, rows, width, heads, dim)

        for frame, _row, col, head in self.sample_groups(frames, rows, width, heads):
            dense_probs, dense_outputs, _ = dense_attention_for_column(
                q_grid,
                k_grid,
                v_grid,
                frame,
                col,
                head,
                scale,
            )
            best_probs, best_outputs = hard_route_distribution(
                q_grid,
                k_grid,
                v_grid,
                frame,
                col,
                head,
                4,
                scale,
                include_ith=True,
            )
            self.add_metrics("hard_route_C4_w1_with_ith", dense_probs, dense_outputs, best_probs, best_outputs)

            for radius in self.radii:
                approx_probs, approx_outputs = hard_route_row_local_distribution(
                    q_grid,
                    k_grid,
                    v_grid,
                    frame,
                    col,
                    head,
                    4,
                    radius,
                    scale,
                    include_ith=True,
                )
                self.add_metrics(
                    f"hard_route_C4_row_local_r{radius}",
                    dense_probs,
                    dense_outputs,
                    approx_probs,
                    approx_outputs,
                )

        self.call_index += 1


def main():
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required.")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config = OmegaConf.load(args.config)
    config.token_merging_args.enable = False

    collector = RowLocalMetricCollector(args.sample_groups_per_call, args.radii)
    patch_self_attention_forward(collector)

    model_kwargs = dict(getattr(config, "model_kwargs", {}))
    shape = list(getattr(config, "image_or_video_shape", [1, 21, 16, 60, 104]))
    generator = WanDiffusionWrapper(**model_kwargs, image_or_video_shape=shape)
    for idx, block in enumerate(generator.model.blocks):
        block.self_attn._analysis_layer_idx = idx
    load_generator_checkpoint(generator, args.checkpoint, use_ema=True)
    generator.to(device=device, dtype=dtype).eval()
    generator.model.token_merging_args = {"enable": False}

    text_encoder = WanTextEncoder(
        model_name=model_kwargs.get("model_name", "Wan2.1-T2V-1.3B"),
        model_root=model_kwargs.get("model_root", None),
    ).to(device=device).eval()

    prompts = load_prompts(args.prompt_path)
    prompt = prompts[args.prompt_index]
    conditional_dict = text_encoder([prompt])
    conditional_dict["prompt_embeds"] = [
        embed.to(device=device, dtype=dtype)
        for embed in conditional_dict["prompt_embeds"]
    ]

    latent_shape = list(shape)
    noise = torch.randn(latent_shape, device=device, dtype=dtype)
    noisy = noise
    steps = denoise_steps(config, generator.scheduler)

    with torch.inference_mode():
        for denoise_idx, current_timestep in enumerate(steps[:-1]):
            collector.current_denoise = denoise_idx
            _, pred = generator(
                noisy_image_or_video=noisy,
                conditional_dict=conditional_dict,
                timestep=torch.ones(noise.shape[:2], dtype=torch.long, device=device) * current_timestep,
            )
            next_timestep = steps[denoise_idx + 1] * torch.ones(
                noise.shape[:2],
                dtype=torch.long,
                device=device,
            )
            noisy = generator.scheduler.add_noise(
                pred.flatten(0, 1),
                torch.randn_like(pred.flatten(0, 1)),
                next_timestep.flatten(0, 1),
            ).unflatten(0, noise.shape[:2])

    output = {
        "definition": (
            "Row-local hard-route C4 keeps the same per-block KV compression as hard_route_C4_w1_with_ith, "
            "but during routing it masks compressed tokens whose key row is farther than radius r from the query row."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "prompt_index": args.prompt_index,
        "sample_groups_per_attention_call": args.sample_groups_per_call,
        "radii": args.radii,
        "attention_calls_collected": collector.call_index,
        "expected_attention_calls": len(generator.model.blocks) * max(0, len(steps) - 1),
        "grid": collector.grid,
        "results": collector.summary(),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
