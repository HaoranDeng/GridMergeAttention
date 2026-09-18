"""Probe a fused nested multiresolution residual basis on native Wan QKV.

The four ordinary Hard-C4 measures are coarsened into C2 estimates under the
three perfect matchings of four elements.  Ridge coefficients are fitted on a
declared calibration prompt and frozen before a second-prompt evaluation.
The model itself always advances on the dense trajectory.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange

from analysis.compare_hard_c4_backends_dense import load_pipeline, run_denoise_only, sync
from analysis.probe_query_centroid_gma_real_qkv import (
    ErrorStats,
    fixed_seed_queries,
    parse_ints,
)
from attention.hard_route_pytorch import (
    hard_route_compressed_attention_pytorch,
    hard_route_merge_kv_pytorch,
)
from attention.token_merging import _hard_c4_video_to_blocks
from generate import load_prompts


PAIRINGS = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)


def _collapse_pairing(
    merged_key,
    merged_value,
    routing_bias,
    seeds,
    pairing,
    scale,
):
    """Moment-match two pairs of C4 measures at their mean seed anchors."""

    seed = rearrange(seeds, "b a f k j c h d -> b a f j k c h d")
    bias = rearrange(routing_bias, "b h a f k j c -> b a f j k c h")
    pair_keys = []
    pair_values = []
    pair_biases = []
    for children in pairing:
        index = torch.tensor(children, device=merged_key.device)
        child_key = merged_key.index_select(5, index).float()
        child_value = merged_value.index_select(5, index).float()
        child_bias = bias.index_select(5, index).float()
        child_seed = seed.index_select(5, index).float()
        anchor = child_seed.mean(dim=5)
        child_score = (
            (child_key * anchor.unsqueeze(5)).sum(dim=-1) * float(scale)
            - child_bias
        )
        valid = torch.isfinite(child_bias)
        any_valid = valid.any(dim=5, keepdim=True)
        safe_score = child_score.masked_fill(~valid, float("-inf"))
        safe_score = torch.where(any_valid, safe_score, torch.zeros_like(safe_score))
        weight = torch.softmax(safe_score, dim=5) * valid
        weight = weight / weight.sum(dim=5, keepdim=True).clamp_min(1e-20)
        pair_key = (weight.unsqueeze(-1) * child_key).sum(dim=5)
        pair_value = (weight.unsqueeze(-1) * child_value).sum(dim=5)
        log_mass = torch.logsumexp(safe_score, dim=5)
        collapsed_bias = (
            (anchor * pair_key).sum(dim=-1) * float(scale) - log_mass
        )
        collapsed_bias = collapsed_bias.masked_fill(
            ~any_valid.squeeze(5), float("inf")
        )
        pair_keys.append(pair_key.to(merged_key.dtype))
        pair_values.append(pair_value.to(merged_value.dtype))
        pair_biases.append(collapsed_bias)

    key = torch.stack(pair_keys, dim=5)
    value = torch.stack(pair_values, dim=5)
    collapsed_bias = torch.stack(pair_biases, dim=5)
    collapsed_bias = rearrange(
        collapsed_bias, "b a f j k c h -> b h a f k j c"
    )
    return key, value, collapsed_bias


def multiresolution_outputs(query, key, value, seeds, scale):
    merged_key, merged_value, routing_bias = hard_route_merge_kv_pytorch(
        seeds, key, value, scale
    )
    c4 = hard_route_compressed_attention_pytorch(
        query, merged_key, merged_value, routing_bias, scale
    )
    coarse = []
    for pairing in PAIRINGS:
        c2_key, c2_value, c2_bias = _collapse_pairing(
            merged_key,
            merged_value,
            routing_bias,
            seeds,
            pairing,
            scale,
        )
        coarse.append(
            hard_route_compressed_attention_pytorch(
                query, c2_key, c2_value, c2_bias, scale
            )
        )
    return c4, coarse


def _normal_equations(c4, coarse, dense):
    residual = torch.stack(
        [(c4.float() - value.float()) for value in coarse], dim=-1
    ).reshape(-1, len(coarse))
    target = (dense.float() - c4.float()).reshape(-1)
    return residual.T @ residual, residual.T @ target


def _correct(c4, coarse, coefficients):
    residual = torch.stack(
        [(c4.float() - value.float()) for value in coarse], dim=-1
    )
    coefficient = torch.as_tensor(
        coefficients, device=c4.device, dtype=residual.dtype
    )
    return (c4.float() + residual @ coefficient).to(c4.dtype)


def _select_heads(tensor, positions):
    index = torch.tensor(positions, device=tensor.device)
    return tensor.index_select(-2, index)


def _summarize(stats):
    result = {name: value.summary() for name, value in sorted(stats.items())}
    baseline = result["c4"]["mse_vs_dense"]
    for value in result.values():
        value["mse_ratio_vs_c4"] = value["mse_vs_dense"] / baseline
    return result


def _theoretical_flops(svg_density):
    rows, width, clusters = 45, 80, 4
    dense = 2 * rows * width
    c4 = 3 * clusters * width + 2 * rows * clusters
    extra_c2_qk = 3 * 2 * rows
    collapse = 3 * 3 * clusters
    fused = c4 + extra_c2_qk + collapse
    return {
        "definition": "QK, AV, routing, merge, and C4-to-C2 collapse multiply-adds",
        "dense_units_per_head_dim": dense,
        "c4_units_per_head_dim": c4,
        "extra_c2_qk_units_per_head_dim": extra_c2_qk,
        "collapse_units_per_head_dim": collapse,
        "fused_units_per_head_dim": fused,
        "ratio_vs_dense": fused / dense,
        "svg_ear_density": svg_density,
        "ratio_vs_svg_ear": (fused / dense) / svg_density,
        "passes_70pct_svg_limit": fused / dense <= 0.70 * svg_density,
        "fusion": (
            "Expand each C2 probability back to its four C4 parents, combine "
            "the probabilities, and execute one four-token AV."
        ),
    }


def install_probe(state, args):
    import wan.modules.model as wan_model

    selected_heads = sorted(set(parse_ints(args.fit_heads) + parse_ints(args.eval_heads)))
    head_position = {head: position for position, head in enumerate(selected_heads)}
    fit_positions = [head_position[head] for head in parse_ints(args.fit_heads)]
    eval_positions = [head_position[head] for head in parse_ints(args.eval_heads)]
    record_calls = set(parse_ints(args.record_calls))

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

        state["call"] += 1
        call = state["call"]
        if call in record_calls:
            frames, rows, width = [int(item) for item in grid_size]
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
            seeds = fixed_seed_queries(query, frames, rows)
            c4, coarse = multiresolution_outputs(
                query, k_blocks, v_blocks, seeds, dim**-0.5
            )

            positions = fit_positions if state["phase"] == "fit" else eval_positions
            c4_selected = _select_heads(c4, positions)
            coarse_selected = [_select_heads(item, positions) for item in coarse]
            reference_selected = _select_heads(reference, positions)

            if state["phase"] == "fit":
                xtx, xty = _normal_equations(
                    c4_selected, coarse_selected, reference_selected
                )
                state["xtx"] += xtx.double().cpu()
                state["xty"] += xty.double().cpu()
            else:
                candidates = {
                    "c4": c4_selected,
                    "c2_pairing_1": coarse_selected[0],
                    "c2_pairing_2": coarse_selected[1],
                    "c2_pairing_3": coarse_selected[2],
                    "mr_uniform_half": _correct(
                        c4_selected, coarse_selected, [1 / 6, 1 / 6, 1 / 6]
                    ),
                    "mr_uniform_full": _correct(
                        c4_selected, coarse_selected, [1 / 3, 1 / 3, 1 / 3]
                    ),
                    "mr_ridge_half": _correct(
                        c4_selected,
                        coarse_selected,
                        [0.5 * value for value in state["coefficients"]],
                    ),
                    "mr_ridge": _correct(
                        c4_selected, coarse_selected, state["coefficients"]
                    ),
                }
                for name, output in candidates.items():
                    state["stats"][name].add(output, reference_selected)
                    state["per_call_stats"][call][name].add(
                        output, reference_selected
                    )
            state["evaluated"].append(
                {
                    "phase": state["phase"],
                    "call": call,
                    "step": step,
                    "layer": layer,
                    "query_frame": query_frame,
                }
            )
        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    state["selected_heads"] = selected_heads


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dense_wan_14b_720p_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--fit_prompt", type=int, default=0)
    parser.add_argument("--eval_prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--fit_heads", default="0,20")
    parser.add_argument("--eval_heads", default="0,20")
    parser.add_argument(
        "--record_calls",
        default="1,20,40,41,60,80,81,100,120,121,140,160",
    )
    parser.add_argument("--ridge", type=float, default=1e-4)
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
    prompts = load_prompts(
        args.prompt_path, max(args.fit_prompt, args.eval_prompt) + 1
    )
    state = {
        "phase": "fit",
        "call": 0,
        "xtx": torch.zeros(3, 3, dtype=torch.float64),
        "xty": torch.zeros(3, dtype=torch.float64),
        "coefficients": None,
        "stats": defaultdict(ErrorStats),
        "per_call_stats": defaultdict(lambda: defaultdict(ErrorStats)),
        "evaluated": [],
    }
    install_probe(state, args)

    sync()
    started = time.perf_counter()
    run_denoise_only(pipeline, config, prompts[args.fit_prompt], args.seed, dtype)
    trace_scale = float(state["xtx"].trace()) / 3.0
    system = state["xtx"] + args.ridge * max(trace_scale, 1e-20) * torch.eye(
        3, dtype=torch.float64
    )
    state["coefficients"] = torch.linalg.solve(system, state["xty"]).tolist()

    state["phase"] = "eval"
    state["call"] = 0
    run_denoise_only(pipeline, config, prompts[args.eval_prompt], args.seed, dtype)
    sync()

    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "fit_prompt": args.fit_prompt,
        "eval_prompt": args.eval_prompt,
        "seed": args.seed,
        "fit_heads": parse_ints(args.fit_heads),
        "eval_heads": parse_ints(args.eval_heads),
        "record_calls": parse_ints(args.record_calls),
        "ridge": args.ridge,
        "coefficients": state["coefficients"],
        "wall_seconds": time.perf_counter() - started,
        "heldout_errors_vs_dense": _summarize(state["stats"]),
        "per_call_heldout_errors_vs_dense": {
            str(call): _summarize(values)
            for call, values in sorted(state["per_call_stats"].items())
        },
        "flops": _theoretical_flops(args.svg_ear_density),
        "evaluated": state["evaluated"],
        "trajectory": "dense for both calibration and heldout prompts",
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
