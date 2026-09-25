"""Dump real post-RoPE Q/K/V from a Wan denoise trajectory, once.

Every Stage-1 architecture diagnostic is a statistic over real attention
inputs. Running the model once and saving the tensors means those diagnostics
become offline analysis: no pipeline, no model load, no GPU contention, and
every idea is scored on identical inputs.

Saved layout is the hard-C4 block layout [B, frames, rows, cols, heads, dim],
which is what attention/adaptive_seed_pytorch.py and the merge reference expect.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FRAMES = int(os.environ.get("GRIDMERGE_FRAMES", "21"))
ROWS = int(os.environ.get("GRIDMERGE_ROWS", "30"))
COLS = int(os.environ.get("GRIDMERGE_COLS", "52"))


class StopCapture(Exception):
    pass


def install(pipe, args, store):
    blocks = pipe.transformer.blocks
    total = args.steps * len(blocks)
    want_layers = {int(x) for x in args.layers.split(",")}
    want_steps = {int(x) for x in args.steps_kept.split(",")}
    heads = [int(x) for x in args.heads.split(",")]
    state = {"call": 0}

    class Processor:
        def __init__(self, layer):
            self.layer = layer

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, rotary_emb=None, timestep=None):
            del timestep
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            q = attn.to_q(hidden_states)
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)
            if attn.norm_q is not None:
                q = attn.norm_q(q)
            if attn.norm_k is not None:
                k = attn.norm_k(k)
            q = q.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            k = k.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            v = v.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            if rotary_emb is not None:
                qc = torch.view_as_complex(q.double().unflatten(3, (-1, 2)))
                kc = torch.view_as_complex(k.double().unflatten(3, (-1, 2)))
                q = torch.view_as_real(qc * rotary_emb).flatten(3, 4).type_as(q)
                k = torch.view_as_real(kc * rotary_emb).flatten(3, 4).type_as(k)

            step = state["call"] // len(blocks)
            if self.layer in want_layers and step in want_steps:
                sel = torch.tensor(heads, device=q.device)
                def block(x):
                    x = x.index_select(1, sel).reshape(
                        x.size(0), len(heads), FRAMES, ROWS, COLS, x.size(-1))
                    return x.permute(0, 2, 3, 4, 1, 5).contiguous().cpu()
                store[(step, self.layer)] = {
                    "q": block(q), "k": block(k), "v": block(v)}

            state["call"] += 1
            if state["call"] == total:
                raise StopCapture
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
            return attn.to_out[1](attn.to_out[0](out.transpose(1, 2).flatten(2, 3).type_as(q)))

    for layer, blk in enumerate(blocks):
        blk.attn1.set_processor(Processor(layer))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompt_index", type=int, default=0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--steps_kept", default="0,6,13,19")
    p.add_argument("--layers", default="0,10,20,29")
    p.add_argument("--heads", default="0,6")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    args = p.parse_args()

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
    store = {}
    install(pipe, args, store)
    try:
        pipe(prompt=prompt, height=args.height, width=args.width,
             num_frames=args.num_frames, guidance_scale=5.0,
             num_inference_steps=args.steps)
    except StopCapture:
        pass

    torch.save({"grid": (FRAMES, ROWS, COLS),
                "heads": [int(x) for x in args.heads.split(",")],
                "prompt_index": args.prompt_index,
                "tensors": {f"{s}_{l}": t for (s, l), t in store.items()}}, args.output)
    total_mb = sum(t[n].numel() * t[n].element_size() for t in store.values()
                   for n in ("q", "k", "v")) / 2**20
    print(f"captured {len(store)} (step, layer) points -> {args.output}  ({total_mb:.0f} MiB)")


if __name__ == "__main__":
    main()
