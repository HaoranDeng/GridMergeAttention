"""Screen SAP-shuffled Hard C4 on one or more real Wan2.1-14B 720p QKV calls.

SVG-2 clusters and separately permutes Q and K/V. This reference applies the
existing 45-query x 80-key Hard C4 grid after that semantic permutation, then
runs full attention over all compressed tokens and restores query order.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from analysis.sweep_wan14b_720p_hard_c4_algorithm import (
    COLS,
    FRAMES,
    NEGATIVE_PROMPT,
    ROWS,
    ErrorStats,
    StopSweep,
    hard_c4_sample,
)


def install_rocm_import_stubs():
    if torch.version.hip is None:
        return
    sys.modules.setdefault("flashinfer", types.ModuleType("flashinfer"))
    if "cuvs.cluster.kmeans" not in sys.modules:
        cuvs = types.ModuleType("cuvs")
        cluster = types.ModuleType("cuvs.cluster")
        kmeans = types.ModuleType("cuvs.cluster.kmeans")
        kmeans.KMeansParams = object
        kmeans.fit = None
        cuvs.cluster = cluster
        cluster.kmeans = kmeans
        sys.modules.update(
            {
                "cuvs": cuvs,
                "cuvs.cluster": cluster,
                "cuvs.cluster.kmeans": kmeans,
            }
        )


def four_quantile_seeds(tokens):
    positions = torch.linspace(0, tokens.size(0) - 1, 4, device=tokens.device)
    return tokens[positions.round().long()]


def merge_key_clusters(seed_queries, key_clusters, value_clusters, key_mask, scale):
    """Compress every padded semantic key cluster into four K/V tokens."""
    logits = torch.einsum("cd,kld->kcl", seed_queries, key_clusters).float() * scale
    logits = logits.masked_fill(~key_mask[:, None, :], float("-inf"))
    assignment_scores = torch.log_softmax(logits, dim=-1)
    assigned_seed = assignment_scores.argmax(dim=-2)
    assignment = F.one_hot(assigned_seed, num_classes=4).permute(0, 2, 1).bool()
    assignment &= key_mask[:, None, :]

    cluster_counts = assignment.sum(dim=-1)
    selected_logits = logits.masked_fill(~assignment, float("-inf"))
    log_weights = torch.log_softmax(selected_logits, dim=-1)
    log_weights = torch.where(assignment, log_weights, 0.0)
    weights = torch.where(assignment, log_weights.exp(), 0.0)

    merged_key = torch.einsum("kcl,kld->kcd", weights.to(key_clusters.dtype), key_clusters)
    merged_value = torch.einsum(
        "kcl,kld->kcd", weights.to(value_clusters.dtype), value_clusters
    )
    routing_bias = (weights * log_weights).sum(dim=-1)
    routing_bias.masked_fill_(cluster_counts == 0, float("inf"))
    return merged_key, merged_value, routing_bias, cluster_counts


def sap_shuffle_hard_c4_frame(
    query,
    key,
    value,
    frame,
    batch_kmeans,
    q_clusters,
    k_clusters,
    kmeans_iters,
):
    """Return one query frame after SAP permutation and fixed-grid Hard C4."""
    batch, heads, _, _, _, dim = query.shape
    sequence = FRAMES * ROWS * COLS
    q_flat = query.reshape(batch * heads, sequence, dim)
    k_flat = key.reshape(batch * heads, sequence, dim)
    v_flat = value.reshape(batch * heads, sequence, dim)

    q_labels, _, _, _ = batch_kmeans(
        q_flat, n_clusters=q_clusters, max_iters=kmeans_iters
    )
    k_labels, _, _, _ = batch_kmeans(
        k_flat, n_clusters=k_clusters, max_iters=kmeans_iters
    )

    frame_start = frame * ROWS * COLS
    frame_indices = torch.arange(
        frame_start, frame_start + ROWS * COLS, device=query.device
    )
    output = torch.empty(
        batch * heads, ROWS * COLS, dim, device=query.device, dtype=query.dtype
    )
    all_cluster_counts = []
    scale = dim**-0.5

    for batch_head in range(batch * heads):
        q_order = q_labels[batch_head].argsort(stable=True)
        k_order = k_labels[batch_head].argsort(stable=True)
        q_blocks = q_flat[batch_head, q_order].reshape(-1, ROWS, dim)
        key_blocks = k_flat[batch_head, k_order].reshape(-1, COLS, dim)
        value_blocks = v_flat[batch_head, k_order].reshape(-1, COLS, dim)
        key_mask = torch.ones(
            key_blocks.shape[:2], device=query.device, dtype=torch.bool
        )

        inverse_q_order = torch.empty_like(q_order)
        inverse_q_order[q_order] = torch.arange(sequence, device=query.device)
        frame_sorted_positions = inverse_q_order[frame_indices]
        frame_query_blocks = frame_sorted_positions // ROWS
        frame_query_offsets = frame_sorted_positions % ROWS

        for query_block in frame_query_blocks.unique().tolist():
            output_positions = (frame_query_blocks == query_block).nonzero().squeeze(-1)
            seeds = four_quantile_seeds(q_blocks[query_block])
            merged_key, merged_value, bias, counts = merge_key_clusters(
                seeds, key_blocks, value_blocks, key_mask, scale
            )
            all_cluster_counts.append(counts)

            q_group = q_blocks[
                query_block, frame_query_offsets[output_positions]
            ][None, None]
            k_group = merged_key.flatten(0, 1)[None, None]
            v_group = merged_value.flatten(0, 1)[None, None]
            attention_bias = -bias.flatten()[None, None, None].to(q_group.dtype)
            group_output = F.scaled_dot_product_attention(
                q_group,
                k_group,
                v_group,
                attn_mask=attention_bias,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )
            output[batch_head, output_positions] = group_output[0, 0]

    output = output.reshape(batch, heads, ROWS, COLS, dim)
    output = output.permute(0, 2, 3, 1, 4).unsqueeze(1)
    return output, torch.cat([counts.flatten() for counts in all_cluster_counts]), k_labels


def sap_key_shuffle_hard_c4_frame(query, key, value, frame, k_labels):
    """Keep the spatial query grid, but form key grids after SAP sorting."""
    batch, heads, _, _, _, dim = query.shape
    sequence = FRAMES * ROWS * COLS
    flat_index = k_labels.argsort(dim=-1, stable=True).unsqueeze(-1).expand(-1, -1, dim)
    key_sorted = torch.gather(key.reshape(batch * heads, sequence, dim), 1, flat_index)
    value_sorted = torch.gather(value.reshape(batch * heads, sequence, dim), 1, flat_index)
    key_sorted = key_sorted.reshape(batch, heads, FRAMES, ROWS, COLS, dim)
    value_sorted = value_sorted.reshape_as(key_sorted)

    q_block = query[:, :, frame : frame + 1].permute(0, 2, 3, 4, 1, 5)
    k_block = key_sorted.permute(0, 2, 3, 4, 1, 5)
    v_block = value_sorted.permute(0, 2, 3, 4, 1, 5)
    output, counts = hard_c4_sample(
        q_block,
        k_block,
        v_block,
        dim**-0.5,
        {"seed": "wide", "score": "key_softmax"},
    )

    dense_control = F.scaled_dot_product_attention(
        query[:, :, frame].flatten(2, 3),
        key_sorted.flatten(2, 4),
        value_sorted.flatten(2, 4),
        dropout_p=0.0,
        is_causal=False,
    )
    dense_control = rearrange(
        dense_control, "b h (i j) d -> b 1 i j h d", i=ROWS, j=COLS
    )
    return output, counts, dense_control


def install_probe(pipe, args, stats, batch_kmeans):
    selected_layers = set(args.layers)
    total_calls = args.steps * len(pipe.transformer.blocks)
    state = {"call": 0, "evaluated": []}

    class Processor:
        def __init__(self, layer):
            self.layer = layer

        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            rotary_emb=None,
            timestep=None,
        ):
            del timestep
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            if rotary_emb is not None:
                q_complex = torch.view_as_complex(
                    query.to(torch.float64).unflatten(3, (-1, 2))
                )
                k_complex = torch.view_as_complex(
                    key.to(torch.float64).unflatten(3, (-1, 2))
                )
                query = torch.view_as_real(q_complex * rotary_emb).flatten(3, 4).type_as(query)
                key = torch.view_as_real(k_complex * rotary_emb).flatten(3, 4).type_as(key)

            step = state["call"] // len(pipe.transformer.blocks)
            if self.layer in selected_layers:
                heads = torch.tensor(args.heads, device=query.device)
                frame = (step * 7 + self.layer * 3) % FRAMES
                q_head = query.index_select(1, heads).reshape(
                    query.size(0), len(heads), FRAMES, ROWS, COLS, query.size(-1)
                )
                k_head = key.index_select(1, heads).reshape_as(q_head)
                v_head = value.index_select(1, heads).reshape_as(q_head)
                dense = F.scaled_dot_product_attention(
                    q_head[:, :, frame].flatten(2, 3),
                    k_head.flatten(2, 4),
                    v_head.flatten(2, 4),
                    dropout_p=0.0,
                    is_causal=False,
                )
                dense = rearrange(
                    dense, "b h (i j) d -> b 1 i j h d", i=ROWS, j=COLS
                )

                q_block = q_head[:, :, frame : frame + 1].permute(0, 2, 3, 4, 1, 5)
                k_block = k_head.permute(0, 2, 3, 4, 1, 5)
                v_block = v_head.permute(0, 2, 3, 4, 1, 5)
                current, current_counts = hard_c4_sample(
                    q_block,
                    k_block,
                    v_block,
                    query.size(-1) ** -0.5,
                    {"seed": "wide", "score": "key_softmax"},
                )
                stats["hard_c4"].add(current, dense, current_counts)

                hybrid, hybrid_counts, k_labels = sap_shuffle_hard_c4_frame(
                    q_head,
                    k_head,
                    v_head,
                    frame,
                    batch_kmeans,
                    args.q_clusters,
                    args.k_clusters,
                    args.kmeans_iters,
                )
                stats["sap_shuffle_hard_c4"].add(hybrid, dense, hybrid_counts)
                key_hybrid, key_hybrid_counts, dense_control = (
                    sap_key_shuffle_hard_c4_frame(
                        q_head, k_head, v_head, frame, k_labels
                    )
                )
                stats["sap_key_shuffle_hard_c4"].add(
                    key_hybrid, dense, key_hybrid_counts
                )
                stats["sap_shuffle_dense_control"].add(
                    dense_control, dense, torch.ones(1, device=query.device)
                )
                state["evaluated"].append(
                    {"call": state["call"], "step": step, "layer": self.layer, "frame": frame}
                )

            dense_full = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            state["call"] += 1
            if state["call"] == total_calls:
                raise StopSweep
            dense_full = dense_full.transpose(1, 2).flatten(2, 3).type_as(query)
            return attn.to_out[1](attn.to_out[0](dense_full))

    for layer, block in enumerate(pipe.transformer.blocks):
        block.attn1.set_processor(Processor(layer))
    return state


def parse_ints(value):
    return [int(item) for item in value.replace(",", " ").split()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sparse_video_gen_root", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--layers", type=parse_ints, default=[0])
    parser.add_argument("--heads", type=parse_ints, default=[0])
    parser.add_argument("--q_clusters", type=int, default=300)
    parser.add_argument("--k_clusters", type=int, default=1000)
    parser.add_argument("--kmeans_iters", type=int, default=10)
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sparse_root = Path(args.sparse_video_gen_root).resolve()
    sys.path.insert(0, str(sparse_root))
    install_rocm_import_stubs()
    from svg.kmeans_utils import batch_kmeans_Euclid

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    vae = AutoencoderKLWan.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    pipe = WanPipeline.from_pretrained(
        args.model, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipe.scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=5.0,
    )
    pipe.to("cuda")

    stats = defaultdict(ErrorStats)
    state = install_probe(pipe, args, stats, batch_kmeans_Euclid)
    prompts = [
        line.strip()
        for line in Path(args.prompt_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    try:
        with torch.inference_mode():
            pipe(
                prompt=prompts[args.prompt_index],
                negative_prompt=NEGATIVE_PROMPT,
                height=720,
                width=1280,
                num_frames=81,
                guidance_scale=5.0,
                num_inference_steps=args.steps,
                generator=generator,
            )
    except StopSweep:
        pass
    torch.cuda.synchronize()

    payload = {
        "device": torch.cuda.get_device_name(),
        "model": str(Path(args.model).resolve()),
        "steps": args.steps,
        "layers": args.layers,
        "heads": args.heads,
        "q_clusters": args.q_clusters,
        "k_clusters": args.k_clusters,
        "kmeans_iters": args.kmeans_iters,
        "evaluated_calls": state["evaluated"],
        "variants": {name: value.summary() for name, value in stats.items()},
        "algorithm": {
            "separate_q_kv_permutation": True,
            "grid_after_permutation": f"{ROWS} query x {COLS} key",
            "compressed_kv_per_grid": 4,
            "full_second_stage_attention": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
