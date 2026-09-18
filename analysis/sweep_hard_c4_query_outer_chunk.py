import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.benchmark_h200_attention_backend_error_time import (
    Stats,
    flash_attention_triton,
    hard_c4_attention,
    load_pipeline,
    load_prompts,
    run_full_denoise,
    sync,
    timed,
)


def parse_int_list(text):
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def install_probe(stats, args, chunks):
    import wan.modules.model as wan_model

    state = {
        "calls_total": 0,
        "calls_recorded": 0,
        "calls_current_prompt": 0,
        "grid_size": None,
        "shape": None,
        "did_warmup": False,
    }

    def dense_attention(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

    def probed_self_attention(self, x, grid_size, rope_cache):
        batch, seq_len, heads, dim = *x.shape[:2], self.num_heads, self.head_dim
        q = wan_model.rmsnorm(self.q(x), self.norm_q.weight, eps=self.norm_q.eps).view(batch, seq_len, heads, dim)
        k = wan_model.rmsnorm(self.k(x), self.norm_k.weight, eps=self.norm_k.eps).view(batch, seq_len, heads, dim)
        v = self.v(x).view(batch, seq_len, heads, dim)
        q, k = wan_model.apply_rope_video_tokens(q, k, rope_cache)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        scale = dim**-0.5

        state["calls_total"] += 1
        state["calls_current_prompt"] += 1
        state["grid_size"] = [int(value) for value in grid_size]
        state["shape"] = {
            "batch": batch,
            "seq_len": seq_len,
            "heads": heads,
            "dim": dim,
        }
        should_record = args.max_calls <= 0 or state["calls_current_prompt"] <= args.max_calls

        if not should_record:
            return self.o(dense_attention(q, k, v, self).flatten(2))

        if not state["did_warmup"]:
            print("warmup sdpa", flush=True)
            dense_attention(q, k, v, self)
            print("warmup flash_attn_triton", flush=True)
            flash_attention_triton(q, k, v, scale)
            for chunk in chunks:
                print(f"warmup hard_c4 chunk={chunk}", flush=True)
                hard_c4_attention(q, k, v, grid_size, scale, chunk)
            sync()
            state["did_warmup"] = True

        print(f"record call={state['calls_recorded'] + 1} sdpa", flush=True)
        dense, seconds = timed(lambda: dense_attention(q, k, v, self))
        stats["sdpa"].add_time(seconds)

        print(f"record call={state['calls_recorded'] + 1} flash_attn_triton", flush=True)
        flash, seconds = timed(lambda: flash_attention_triton(q, k, v, scale))
        stats["flash_attn_triton"].add_time(seconds)
        stats["flash_attn_triton"].add_error(flash, dense)

        for chunk in chunks:
            name = f"hard_c4_chunk{chunk}"
            print(f"record call={state['calls_recorded'] + 1} {name}", flush=True)
            hard, seconds = timed(lambda chunk=chunk: hard_c4_attention(q, k, v, grid_size, scale, chunk))
            stats[name].add_time(seconds)
            stats[name].add_error(hard, dense)

        state["calls_recorded"] += 1
        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep hard-C4 query_outer_chunk on real Wan QKV.")
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=12, help="0 records all self-attention calls.")
    parser.add_argument("--chunks", default="1,3,7,10,21")
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    chunks = parse_int_list(args.chunks)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    config, pipeline = load_pipeline(args.config, args.checkpoint, dtype, device, args.use_ema)
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]

    stats = defaultdict(Stats)
    state = install_probe(stats, args, chunks)

    sync()
    start = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        state["calls_current_prompt"] = 0
        run_full_denoise(pipeline, config, prompt, args.seed + offset, dtype)
    sync()
    wall_seconds = time.perf_counter() - start

    dense_seconds = stats["sdpa"].seconds
    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "dtype": args.dtype,
        "prompt_path": args.prompt_path,
        "prompt_index": args.prompt_index,
        "num_prompts": args.num_prompts,
        "max_calls_per_prompt": args.max_calls,
        "chunks": chunks,
        "grid_size": state["grid_size"],
        "shape": state["shape"],
        "calls_total": state["calls_total"],
        "calls_recorded": state["calls_recorded"],
        "wall_seconds": wall_seconds,
        "variants_vs_dense": {
            name: value.summary(dense_seconds=dense_seconds if name != "sdpa" else None, videos=len(prompts))
            for name, value in sorted(stats.items())
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
