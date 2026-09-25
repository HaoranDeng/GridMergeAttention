"""Measure how stable adaptive seed choices are across denoising steps and layers.

Adaptive seed selection is accurate but expensive, and two Triton attempts to
make it fast both came in slower than PyTorch (docs/negative_results.md). If the
chosen seeds barely move between denoising steps or between layers, the selector
can run once and be cached instead of kernelized, and the merge kernel only ever
reads a small integer table.

This script installs a probe that runs ordinary dense attention -- generation is
unaffected -- and on the side records which four query rows the selector picks
for each block and head. It then reports how often those picks agree.

Comparability note: the selector is head-specific and block-specific, so all
agreement figures are over (key frame, key row, query column, head) entries for
one fixed query frame per (step, layer).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

# Running this file by path puts analysis/ on sys.path, not the repo root, so
# the attention package would not import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FRAMES = int(os.environ.get("GRIDMERGE_FRAMES", "21"))
ROWS = int(os.environ.get("GRIDMERGE_ROWS", "30"))
COLS = int(os.environ.get("GRIDMERGE_COLS", "52"))


class StopAfterProbe(Exception):
    pass


def agreement(a, b):
    """Set agreement between two [..., C] index tensors.

    Returns (exact set match rate, mean number of shared seeds out of C).
    """
    a_sorted = a.sort(dim=-1).values
    b_sorted = b.sort(dim=-1).values
    exact = (a_sorted == b_sorted).all(dim=-1).float().mean().item()
    shared = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(dim=-1).float().sum(-1).mean().item()
    return exact, shared


def install(pipe, args, records):
    blocks = pipe.transformer.blocks
    total_calls = args.steps * len(blocks)
    selected_layers = [int(x) for x in args.layers.split(",")]
    selected_heads = [int(x) for x in args.heads.split(",")]
    state = {"call": 0}

    from attention.adaptive_seed_pytorch import adaptive_seed_row_indices

    class Processor:
        def __init__(self, layer):
            self.layer = layer

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, rotary_emb=None, timestep=None):
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
                q_complex = torch.view_as_complex(query.double().unflatten(3, (-1, 2)))
                k_complex = torch.view_as_complex(key.double().unflatten(3, (-1, 2)))
                query = torch.view_as_real(q_complex * rotary_emb).flatten(3, 4).type_as(query)
                key = torch.view_as_real(k_complex * rotary_emb).flatten(3, 4).type_as(key)

            step = state["call"] // len(blocks)
            if self.layer in selected_layers:
                heads = torch.tensor(selected_heads, device=query.device)
                # One fixed query frame so every (step, layer) sample is aligned
                # and differences reflect the selector, not the sampled block.
                frame = args.query_frame
                q_head = query.index_select(1, heads).reshape(
                    query.size(0), len(heads), FRAMES, ROWS, COLS, query.size(-1))
                k_head = key.index_select(1, heads).reshape_as(q_head)
                v_head = value.index_select(1, heads).reshape_as(q_head)
                q_block = q_head[:, :, frame:frame + 1].permute(0, 2, 3, 4, 1, 5)
                k_block = k_head.permute(0, 2, 3, 4, 1, 5)
                v_block = v_head.permute(0, 2, 3, 4, 1, 5)
                rows = adaptive_seed_row_indices(
                    q_block, k_block, v_block, query.size(-1) ** -0.5,
                    alpha=args.alpha, candidate_count=args.candidates,
                    probe_count=args.probes, value_dimensions=args.value_dims)
                records[(step, self.layer)] = rows.squeeze(0).squeeze(0).cpu()

            state["call"] += 1
            if state["call"] == total_calls:
                raise StopAfterProbe
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
            return attn.to_out[1](attn.to_out[0](out.transpose(1, 2).flatten(2, 3).type_as(query)))

    for layer, block in enumerate(blocks):
        block.attn1.set_processor(Processor(layer))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompt_index", type=int, default=0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--layers", default="0,10,20,29")
    p.add_argument("--heads", default="0,6")
    p.add_argument("--query_frame", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--candidates", type=int, default=30)
    p.add_argument("--probes", type=int, default=16)
    p.add_argument("--value_dims", type=int, default=32)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    args = p.parse_args()
    if args.height // 16 != ROWS or args.width // 16 != COLS:
        raise ValueError(f"{args.height}x{args.width} does not match ROWS={ROWS} COLS={COLS}")

    from diffusers import AutoencoderKLWan, WanPipeline, UniPCMultistepScheduler

    vae = AutoencoderKLWan.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(
        args.model, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True,
        num_train_timesteps=1000, flow_shift=5.0)
    pipe.to("cuda")

    prompt = Path(args.prompt_path).read_text().splitlines()[args.prompt_index].strip()
    records = {}
    install(pipe, args, records)
    try:
        pipe(prompt=prompt, height=args.height, width=args.width,
             num_frames=args.num_frames, guidance_scale=5.0,
             num_inference_steps=args.steps)
    except StopAfterProbe:
        pass

    layers = sorted({l for _, l in records})
    steps = sorted({s for s, _ in records})
    report = {"alpha": args.alpha, "candidates": args.candidates,
              "layers": layers, "steps": steps, "query_frame": args.query_frame}

    # Consecutive denoising steps, same layer: can seeds be reused across steps?
    consec, vs_first = defaultdict(list), defaultdict(list)
    for layer in layers:
        for i in range(1, len(steps)):
            a, b = records.get((steps[i - 1], layer)), records.get((steps[i], layer))
            if a is not None and b is not None:
                consec[layer].append(agreement(a, b))
        base = records.get((steps[0], layer))
        for s in steps[1:]:
            cur = records.get((s, layer))
            if base is not None and cur is not None:
                vs_first[layer].append(agreement(base, cur))
    report["step_consecutive"] = {
        str(l): {"exact": sum(x for x, _ in v) / len(v),
                 "shared_of_4": sum(y for _, y in v) / len(v)}
        for l, v in consec.items() if v}
    report["step_vs_first"] = {
        str(l): {"exact": sum(x for x, _ in v) / len(v),
                 "shared_of_4": sum(y for _, y in v) / len(v)}
        for l, v in vs_first.items() if v}

    # Same step, different layers: can one layer's table serve another?
    cross = []
    for s in steps:
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                a, b = records.get((s, layers[i])), records.get((s, layers[j]))
                if a is not None and b is not None:
                    cross.append(agreement(a, b))
    if cross:
        report["layer_cross"] = {"exact": sum(x for x, _ in cross) / len(cross),
                                 "shared_of_4": sum(y for _, y in cross) / len(cross)}

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
