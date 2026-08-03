import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import wan.modules.model as wan_model
from attention.token_merging import _assign_keys_to_seeds, _random_seed_columns, _random_seed_rows
from utils.misc import set_seed
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample dense-vs-token-merging attention metrics across Wan self-attention calls."
    )
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--sample_groups_per_call", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    return parser.parse_args()


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_generator_checkpoint(generator, checkpoint_path, use_ema=True):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and ("generator" in state or "generator_ema" in state):
        key = "generator_ema" if use_ema and "generator_ema" in state else "generator"
        state = state[key]
    generator.load_state_dict(state)


def quantiles(values):
    if not values:
        return []
    x = torch.tensor(values, dtype=torch.float32)
    q = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=torch.float32)
    return [float(v) for v in torch.quantile(x, q)]


def summarize(values, include_bits=False):
    if not values:
        return {"mean": None, "quantiles_10_25_50_75_90_95": []}
    x = torch.tensor(values, dtype=torch.float32)
    out = {
        "mean": float(x.mean()),
        "quantiles_10_25_50_75_90_95": quantiles(values),
    }
    if include_bits:
        out["mean_bits"] = float(x.mean() / math.log(2.0))
    return out


def flatten_video(x):
    # [F, R, W, D] -> [F*R*W, D], matching Wan sequence order.
    return x.reshape(-1, x.size(-1))


def dense_attention_for_column(q_grid, k_grid, v_grid, frame, col, head, scale):
    queries = q_grid[frame, :, col, head].float()
    keys = flatten_video(k_grid[:, :, :, head].float())
    values = flatten_video(v_grid[:, :, :, head].float())
    logits = torch.matmul(queries, keys.transpose(0, 1)) * scale
    probs = torch.softmax(logits, dim=-1)
    outputs = torch.matmul(probs, values)
    return probs, outputs, values


def dense_attention_for_row(q_grid, k_grid, v_grid, frame, row, head, scale):
    queries = q_grid[frame, row, :, head].float()
    keys = flatten_video(k_grid[:, :, :, head].float())
    values = flatten_video(v_grid[:, :, :, head].float())
    logits = torch.matmul(queries, keys.transpose(0, 1)) * scale
    probs = torch.softmax(logits, dim=-1)
    outputs = torch.matmul(probs, values)
    return probs, outputs, values


def one_token_w_reduce_distribution(q_grid, k_grid, v_grid, frame, col, head, w_reduce, scale):
    frames, rows, width, _, dim = q_grid.shape
    if width % w_reduce:
        raise ValueError(f"width={width} is not divisible by w_reduce={w_reduce}.")

    group_width = width // w_reduce
    query_group = col // group_width
    local_col = col % group_width
    sqrt_scale = scale**0.5

    query_seed = q_grid[frame, :, query_group * group_width + local_col, head].float() * sqrt_scale
    query_rows = q_grid[frame, :, col, head].float() * sqrt_scale

    key_blocks = []
    value_blocks = []
    for key_frame in range(frames):
        for key_group in range(w_reduce):
            cols = slice(key_group * group_width, (key_group + 1) * group_width)
            key_blocks.append(k_grid[key_frame, :, cols, head].float() * sqrt_scale)
            value_blocks.append(v_grid[key_frame, :, cols, head].float())
    key_blocks = torch.stack(key_blocks, dim=0)
    value_blocks = torch.stack(value_blocks, dim=0)

    merge_logits = torch.einsum("rd,orld->orl", query_seed, key_blocks)
    merge_logprob = torch.log_softmax(merge_logits, dim=-1)
    merge_weights = merge_logprob.exp()
    merged_keys = torch.einsum("orl,orld->ord", merge_weights, key_blocks)
    merged_values = torch.einsum("orl,orld->ord", merge_weights, value_blocks)

    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)
    routing_logits = torch.einsum("ord,id->ior", merged_keys, query_rows)
    routing_logits = routing_logits - routing_bias.unsqueeze(0)
    routing_weights = torch.softmax(routing_logits.flatten(1), dim=-1).view(
        rows, frames * w_reduce, rows
    )

    expanded = routing_weights[:, :, :, None] * merge_weights[None]
    approx_probs = (
        expanded.view(rows, frames, w_reduce, rows, group_width)
        .permute(0, 1, 3, 2, 4)
        .reshape(rows, frames * rows * width)
    )
    approx_outputs = torch.einsum("ior,ord->id", routing_weights, merged_values)
    return approx_probs, approx_outputs


def multi_seed_distribution(
    q_grid,
    k_grid,
    v_grid,
    frame,
    col,
    head,
    num_merge_tokens,
    scale,
    include_ith=False,
):
    frames, rows, width, _, _ = q_grid.shape
    sqrt_scale = scale**0.5

    query_outer = torch.tensor([frame], device=q_grid.device)
    key_outer = torch.arange(frames, device=q_grid.device)
    seed_rows = _random_seed_rows(
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
    assignment_mask = _assign_keys_to_seeds(
        seed_logits[None, None, None, :, :, None, :, :]
    )[0, 0, 0, :, :, 0]

    masked_logits = seed_logits.masked_fill(~assignment_mask, torch.finfo(seed_logits.dtype).min)
    merge_logits = masked_logits - masked_logits.amax(dim=-1, keepdim=True)
    merge_logits = merge_logits.masked_fill(~assignment_mask, torch.finfo(seed_logits.dtype).min)
    merge_weights = torch.softmax(merge_logits, dim=-1).masked_fill(~assignment_mask, 0)

    merged_keys = torch.einsum("frcl,frld->frcd", merge_weights, key_blocks)
    merged_values = torch.einsum("frcl,frld->frcd", merge_weights, value_blocks)

    merge_logprob = merge_logits - torch.logsumexp(merge_logits, dim=-1, keepdim=True)
    merge_logprob = merge_logprob.masked_fill(~assignment_mask, 0)
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)

    routing_logits = torch.einsum("frcd,id->ifrc", merged_keys, query_rows)
    routing_logits = routing_logits - routing_bias.unsqueeze(0)
    routing_weights = torch.softmax(routing_logits.flatten(1), dim=-1).view(
        rows, frames, rows, num_merge_tokens
    )

    expanded = routing_weights[:, :, :, :, None] * merge_weights[None]
    approx_probs = expanded.sum(dim=3).reshape(rows, frames * rows * width)
    approx_outputs = torch.einsum("ifrc,frcd->id", routing_weights, merged_values)
    return approx_probs, approx_outputs


def select_seed_distribution(
    q_grid,
    k_grid,
    v_grid,
    frame,
    col,
    head,
    num_merge_tokens,
    scale,
    include_ith=False,
):
    frames, rows, width, _, _ = q_grid.shape
    sqrt_scale = scale**0.5

    query_outer = torch.tensor([frame], device=q_grid.device)
    key_outer = torch.arange(frames, device=q_grid.device)
    seed_rows = _random_seed_rows(
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

    merge_logits = torch.einsum("frcd,frld->frcl", seed_queries, key_blocks)
    merge_logits = merge_logits - merge_logits.amax(dim=-1, keepdim=True)
    merge_logprob = torch.log_softmax(merge_logits, dim=-1)
    merge_weights = merge_logprob.exp()

    merged_keys = torch.einsum("frcl,frld->frcd", merge_weights, key_blocks)
    merged_values = torch.einsum("frcl,frld->frcd", merge_weights, value_blocks)
    entropy = -(merge_weights * merge_logprob).sum(dim=-1)

    candidate_scores = torch.einsum("frcd,id->ifrc", merged_keys, query_rows)
    candidate_scores = candidate_scores + entropy.unsqueeze(0)
    selected_scores, selected_seed = candidate_scores.max(dim=-1)

    routing_weights = torch.softmax(selected_scores.flatten(1), dim=-1).view(
        rows, frames, rows
    )
    selected_mask = F.one_hot(selected_seed, num_classes=num_merge_tokens).float()
    selected_merge_weights = torch.einsum("ifrc,frcl->ifrl", selected_mask, merge_weights)
    selected_values = torch.einsum("ifrc,frcd->ifrd", selected_mask, merged_values)

    expanded = routing_weights[:, :, :, None] * selected_merge_weights
    approx_probs = expanded.reshape(rows, frames * rows * width)
    approx_outputs = torch.einsum("ifr,ifrd->id", routing_weights, selected_values)
    return approx_probs, approx_outputs


def transpose_multi_seed_distribution(
    q_grid,
    k_grid,
    v_grid,
    frame,
    row,
    head,
    num_merge_tokens,
    scale,
    include_ith=False,
):
    frames, rows, width, _, _ = q_grid.shape
    sqrt_scale = scale**0.5

    query_outer = torch.tensor([frame], device=q_grid.device)
    key_outer = torch.arange(frames, device=q_grid.device)
    seed_cols = _random_seed_columns(
        query_outer,
        key_outer,
        rows,
        width,
        width,
        num_merge_tokens,
        seed=0,
        include_ith=include_ith,
    )[0, :, row]

    seed_queries = q_grid[frame, row, seed_cols, head].float() * sqrt_scale
    key_blocks = k_grid[:, :, :, head].float() * sqrt_scale
    value_blocks = v_grid[:, :, :, head].float()
    query_cols = q_grid[frame, row, :, head].float() * sqrt_scale

    seed_logits = torch.einsum("flcd,fkld->flck", seed_queries, key_blocks)
    assignment_mask = _assign_keys_to_seeds(
        seed_logits[None, None, None, :, None, :, :, :]
    )[0, 0, 0, :, 0]
    cluster_has_keys = assignment_mask.any(dim=-1)

    masked_logits = seed_logits.masked_fill(~assignment_mask, torch.finfo(seed_logits.dtype).min)
    merge_logits = masked_logits - masked_logits.amax(dim=-1, keepdim=True)
    merge_logits = merge_logits.masked_fill(~assignment_mask, torch.finfo(seed_logits.dtype).min)
    merge_weights = torch.softmax(merge_logits, dim=-1).masked_fill(~assignment_mask, 0)

    merged_keys = torch.einsum("flck,fkld->flcd", merge_weights, key_blocks)
    merged_values = torch.einsum("flck,fkld->flcd", merge_weights, value_blocks)

    merge_logprob = merge_logits - torch.logsumexp(merge_logits, dim=-1, keepdim=True)
    merge_logprob = merge_logprob.masked_fill(~assignment_mask, 0)
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)

    routing_logits = torch.einsum("flcd,jd->jflc", merged_keys, query_cols)
    routing_logits = routing_logits - routing_bias.unsqueeze(0)
    routing_logits = routing_logits.masked_fill(
        ~cluster_has_keys.unsqueeze(0),
        torch.finfo(routing_logits.dtype).min,
    )
    routing_weights = torch.softmax(routing_logits.flatten(1), dim=-1).view(
        width, frames, width, num_merge_tokens
    )

    expanded = routing_weights[:, :, :, :, None] * merge_weights[None]
    approx_probs = expanded.sum(dim=3).permute(0, 1, 3, 2).reshape(width, frames * rows * width)
    approx_outputs = torch.einsum("jflc,flcd->jd", routing_weights, merged_values)
    return approx_probs, approx_outputs


def transpose_select_seed_distribution(
    q_grid,
    k_grid,
    v_grid,
    frame,
    row,
    head,
    num_merge_tokens,
    scale,
    include_ith=False,
):
    frames, rows, width, _, _ = q_grid.shape
    sqrt_scale = scale**0.5

    query_outer = torch.tensor([frame], device=q_grid.device)
    key_outer = torch.arange(frames, device=q_grid.device)
    seed_cols = _random_seed_columns(
        query_outer,
        key_outer,
        rows,
        width,
        width,
        num_merge_tokens,
        seed=0,
        include_ith=include_ith,
    )[0, :, row]

    seed_queries = q_grid[frame, row, seed_cols, head].float() * sqrt_scale
    key_blocks = k_grid[:, :, :, head].float() * sqrt_scale
    value_blocks = v_grid[:, :, :, head].float()
    query_cols = q_grid[frame, row, :, head].float() * sqrt_scale

    merge_logits = torch.einsum("flcd,fkld->flck", seed_queries, key_blocks)
    merge_logits = merge_logits - merge_logits.amax(dim=-1, keepdim=True)
    merge_logprob = torch.log_softmax(merge_logits, dim=-1)
    merge_weights = merge_logprob.exp()

    merged_keys = torch.einsum("flck,fkld->flcd", merge_weights, key_blocks)
    merged_values = torch.einsum("flck,fkld->flcd", merge_weights, value_blocks)
    entropy = -(merge_weights * merge_logprob).sum(dim=-1)

    candidate_scores = torch.einsum("flcd,jd->jflc", merged_keys, query_cols)
    candidate_scores = candidate_scores + entropy.unsqueeze(0)
    selected_scores, selected_seed = candidate_scores.max(dim=-1)

    routing_weights = torch.softmax(selected_scores.flatten(1), dim=-1).view(width, frames, width)
    selected_mask = F.one_hot(selected_seed, num_classes=num_merge_tokens).float()
    selected_merge_weights = torch.einsum("jflc,flck->jflk", selected_mask, merge_weights)
    selected_values = torch.einsum("jflc,flcd->jfld", selected_mask, merged_values)

    expanded = routing_weights[:, :, :, None] * selected_merge_weights
    approx_probs = expanded.permute(0, 1, 3, 2).reshape(width, frames * rows * width)
    approx_outputs = torch.einsum("jfl,jfld->jd", routing_weights, selected_values)
    return approx_probs, approx_outputs


class AttentionMetricCollector:
    def __init__(self, sample_groups_per_call):
        self.sample_groups_per_call = sample_groups_per_call
        self.current_denoise = 0
        self.call_index = 0
        self.grid = None
        self.results = defaultdict(lambda: defaultdict(list))
        self.invalid = {}

    def sample_groups(self, frames, rows, width, heads):
        groups = []
        base = self.call_index * 1009 + self.current_denoise * 9173
        for idx in range(self.sample_groups_per_call):
            frame = (base + idx * 7) % frames
            row = (base * 7 + idx * 17) % rows
            col = (base * 3 + idx * 11) % width
            head = (base * 5 + idx * 13) % heads
            groups.append((frame, row, col, head))
        return groups

    def add_metrics(self, name, dense_probs, dense_outputs, approx_probs, approx_outputs):
        eps = 1e-12
        approx_probs = approx_probs / approx_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        dense_probs = dense_probs / dense_probs.sum(dim=-1, keepdim=True).clamp_min(eps)

        log_dense = dense_probs.clamp_min(eps).log()
        log_approx = approx_probs.clamp_min(eps).log()
        kl = (dense_probs * (log_dense - log_approx)).sum(dim=-1)
        reverse_kl = (approx_probs * (log_approx - log_dense)).sum(dim=-1)
        tv = 0.5 * (dense_probs - approx_probs).abs().sum(dim=-1)

        diff = dense_outputs - approx_outputs
        rel_l2 = diff.norm(dim=-1) / dense_outputs.norm(dim=-1).clamp_min(eps)
        mse = diff.square().mean(dim=-1)
        cosine_error = 1 - F.cosine_similarity(dense_outputs, approx_outputs, dim=-1)

        bucket = self.results[name]
        bucket["kl_p_true_q_approx"].extend(float(x) for x in kl.detach().cpu())
        bucket["reverse_kl_q_approx_p_true"].extend(float(x) for x in reverse_kl.detach().cpu())
        bucket["attention_tv"].extend(float(x) for x in tv.detach().cpu())
        bucket["relative_l2_output_error"].extend(float(x) for x in rel_l2.detach().cpu())
        bucket["mse_output_error_per_head_dim"].extend(float(x) for x in mse.detach().cpu())
        bucket["cosine_error_1_minus_cosine"].extend(float(x) for x in cosine_error.detach().cpu())

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

        column_variants = [
            ("MonarchRT", "w", 1),
            ("C2_w1", "c", 2),
            ("C4_w1", "c", 4),
            ("C6_w1", "c", 6),
            ("C13_w1", "c", 13),
            ("C2_w1_with_ith", "c_ith", 2),
            ("C4_w1_with_ith", "c_ith", 4),
            ("C6_w1_with_ith", "c_ith", 6),
            ("select_C2_w1", "select", 2),
            ("select_C4_w1", "select", 4),
            ("select_C6_w1", "select", 6),
            ("select_C13_w1", "select", 13),
            ("select_C2_w1_with_ith", "select_ith", 2),
            ("select_C4_w1_with_ith", "select_ith", 4),
            ("select_C6_w1_with_ith", "select_ith", 6),
            ("w_reduce2", "w", 2),
            ("w_reduce4", "w", 4),
            ("w_reduce6", "w", 6),
            ("w_reduce13", "w", 13),
        ]

        transpose_variants = [
            ("transpose_C2_w1_with_ith", "transpose_c_ith", 2),
            ("transpose_C4_w1_with_ith", "transpose_c_ith", 4),
            ("transpose_C6_w1_with_ith", "transpose_c_ith", 6),
            ("transpose_select_C2_w1_with_ith", "transpose_select_ith", 2),
            ("transpose_select_C4_w1_with_ith", "transpose_select_ith", 4),
            ("transpose_select_C6_w1_with_ith", "transpose_select_ith", 6),
        ]

        for frame, row, col, head in self.sample_groups(frames, rows, width, heads):
            dense_probs, dense_outputs, _ = dense_attention_for_column(
                q_grid, k_grid, v_grid, frame, col, head, scale
            )
            for name, kind, value in column_variants:
                if kind == "w" and width % value:
                    self.invalid[name] = (
                        f"width token count {width} is not divisible by w_reduce={value}; "
                        "current token_merging_attn requires exact divisibility."
                    )
                    continue
                if kind == "c":
                    approx_probs, approx_outputs = multi_seed_distribution(
                        q_grid, k_grid, v_grid, frame, col, head, value, scale
                    )
                elif kind == "c_ith":
                    approx_probs, approx_outputs = multi_seed_distribution(
                        q_grid, k_grid, v_grid, frame, col, head, value, scale, include_ith=True
                    )
                elif kind == "select":
                    approx_probs, approx_outputs = select_seed_distribution(
                        q_grid, k_grid, v_grid, frame, col, head, value, scale
                    )
                elif kind == "select_ith":
                    approx_probs, approx_outputs = select_seed_distribution(
                        q_grid, k_grid, v_grid, frame, col, head, value, scale, include_ith=True
                    )
                else:
                    approx_probs, approx_outputs = one_token_w_reduce_distribution(
                        q_grid, k_grid, v_grid, frame, col, head, value, scale
                    )
                self.add_metrics(name, dense_probs, dense_outputs, approx_probs, approx_outputs)

            dense_probs, dense_outputs, _ = dense_attention_for_row(
                q_grid, k_grid, v_grid, frame, row, head, scale
            )
            for name, kind, value in transpose_variants:
                if kind == "transpose_c_ith":
                    approx_probs, approx_outputs = transpose_multi_seed_distribution(
                        q_grid, k_grid, v_grid, frame, row, head, value, scale, include_ith=True
                    )
                else:
                    approx_probs, approx_outputs = transpose_select_seed_distribution(
                        q_grid, k_grid, v_grid, frame, row, head, value, scale, include_ith=True
                    )
                self.add_metrics(name, dense_probs, dense_outputs, approx_probs, approx_outputs)

        self.call_index += 1

    def summary(self):
        results = {}
        for name, metrics in self.results.items():
            results[name] = {
                "sampled_query_outputs": len(metrics["kl_p_true_q_approx"]),
                "kl_p_true_q_approx": summarize(metrics["kl_p_true_q_approx"], include_bits=True),
                "reverse_kl_q_approx_p_true": summarize(
                    metrics["reverse_kl_q_approx_p_true"], include_bits=True
                ),
                "attention_tv": summarize(metrics["attention_tv"]),
                "relative_l2_output_error": summarize(metrics["relative_l2_output_error"]),
                "mse_output_error_per_head_dim": summarize(metrics["mse_output_error_per_head_dim"]),
                "cosine_error_1_minus_cosine": summarize(metrics["cosine_error_1_minus_cosine"]),
            }
        for name, reason in self.invalid.items():
            results[name] = {"valid": False, "reason": reason}
        return results


def patch_self_attention_forward(collector):
    def forward(self, x, grid_size, rope_cache):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        if torch.is_grad_enabled() or x.requires_grad:
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
        else:
            q = wan_model.rmsnorm(self.q(x), self.norm_q.weight, eps=self.norm_q.eps).view(b, s, n, d)
            k = wan_model.rmsnorm(self.k(x), self.norm_k.weight, eps=self.norm_k.eps).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        roped_query, roped_key = wan_model.apply_rope_video_tokens(q, k, rope_cache)
        collector.observe(
            roped_query.detach(),
            roped_key.detach(),
            v.detach(),
            grid_size,
            getattr(self, "_analysis_layer_idx", -1),
        )

        x = wan_model.flash_attention(
            q=roped_query,
            k=roped_key,
            v=v,
            window_size=self.window_size,
        )
        x = x.flatten(2)
        return self.o(x)

    wan_model.WanSelfAttention.forward = forward


def denoise_steps(config, scheduler):
    steps = torch.tensor(config.denoising_step_list, dtype=torch.long, device="cpu")
    if steps[-1] == 0:
        steps = steps[:-1]
    if getattr(config, "warp_denoising_step", False):
        timesteps = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        steps = timesteps[1000 - steps]
    return steps


def main():
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required.")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config = OmegaConf.load(args.config)
    config.token_merging_args.enable = False

    collector = AttentionMetricCollector(args.sample_groups_per_call)
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
    pred = None
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
            "Sampled dense-vs-token-merging attention metrics across all Wan self-attention "
            "layers and denoise forwards. Dense P is softmax(q k^T / sqrt(d)); approximate Q "
            "is expanded back to original key tokens before KL/output-error computation."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "prompt_index": args.prompt_index,
        "sample_groups_per_attention_call": args.sample_groups_per_call,
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
