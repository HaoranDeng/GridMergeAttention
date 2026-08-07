import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from generate import load_generator_checkpoint, load_prompts
from pipeline import TokenMergingWanPipeline
from utils.misc import set_seed


MERGE_BACKEND = "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached"
R3_BACKEND = "triton_grid_outer_exact_r3_m128_n64_s1_exp2"
R4_BACKEND = "triton_grid_outer_exact_r4_m128_n64_s1_exp2"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


def error_stats(out, ref):
    diff = (out.float() - ref.float()).reshape(-1)
    ref_flat = ref.float().reshape(-1)
    out_flat = out.float().reshape(-1)
    sqerr = torch.dot(diff, diff)
    ref_sq = torch.dot(ref_flat, ref_flat).clamp_min(1e-20)
    out_sq = torch.dot(out_flat, out_flat).clamp_min(1e-20)
    dot = torch.dot(out_flat, ref_flat)
    return {
        "sqerr": float(sqerr.item()),
        "ref_sq": float(ref_sq.item()),
        "numel": int(diff.numel()),
        "mse": float((sqerr / diff.numel()).item()),
        "rel_l2": float(torch.sqrt(sqerr / ref_sq).item()),
        "cosine_error": float((1.0 - dot / torch.sqrt(out_sq * ref_sq)).item()),
        "max_abs": float(diff.abs().max().item()),
    }


def summarize_variant(records, variant):
    seconds = 0.0
    sqerr = 0.0
    ref_sq = 0.0
    numel = 0
    cosine = 0.0
    max_abs = 0.0
    for record in records:
        stats = record["variants"][variant]
        seconds += float(stats["seconds"])
        sqerr += float(stats["sqerr"])
        ref_sq += float(stats["ref_sq"])
        numel += int(stats["numel"])
        cosine += float(stats["cosine_error"])
        max_abs = max(max_abs, float(stats["max_abs"]))
    calls = len(records)
    return {
        "calls": calls,
        "seconds": seconds,
        "mean_ms": 1000.0 * seconds / calls if calls else 0.0,
        "mse": sqerr / max(numel, 1),
        "rel_l2": (sqerr / max(ref_sq, 1e-20)) ** 0.5,
        "mean_call_cosine_error": cosine / calls if calls else 0.0,
        "max_abs": max_abs,
    }


def summarize_policy(records, chooser):
    seconds = 0.0
    sqerr = 0.0
    ref_sq = 0.0
    numel = 0
    r3_calls = 0
    for record in records:
        variant = chooser(record)
        stats = record["variants"][variant]
        r3_calls += int(variant == "r3")
        seconds += float(stats["seconds"])
        sqerr += float(stats["sqerr"])
        ref_sq += float(stats["ref_sq"])
        numel += int(stats["numel"])
    calls = len(records)
    return {
        "calls": calls,
        "r3_calls": r3_calls,
        "r3_fraction": r3_calls / calls if calls else 0.0,
        "seconds": seconds,
        "mean_ms": 1000.0 * seconds / calls if calls else 0.0,
        "mse": sqerr / max(numel, 1),
        "rel_l2": (sqerr / max(ref_sq, 1e-20)) ** 0.5,
    }


def aggregate_by(records, key_fn):
    groups = defaultdict(list)
    for record in records:
        groups[str(key_fn(record))].append(record)
    return {
        name: {
            "r3": summarize_variant(group, "r3"),
            "r4": summarize_variant(group, "r4"),
        }
        for name, group in sorted(groups.items(), key=lambda item: item[0])
    }


def simulate_policies(records, num_layers, num_denoise_steps):
    policies = {
        "r4_all": lambda r: "r4",
        "r3_all": lambda r: "r3",
        "r3_first_step": lambda r: "r3" if r["denoise_idx"] == 0 else "r4",
        "r3_middle_step": lambda r: "r3" if r["denoise_idx"] == num_denoise_steps // 2 else "r4",
        "r3_last_step": lambda r: "r3" if r["denoise_idx"] == num_denoise_steps - 1 else "r4",
        "r3_first_half_layers": lambda r: "r3" if r["layer_idx"] < num_layers // 2 else "r4",
        "r3_second_half_layers": lambda r: "r3" if r["layer_idx"] >= num_layers // 2 else "r4",
        "r3_even_layers": lambda r: "r3" if r["layer_idx"] % 2 == 0 else "r4",
    }
    simulated = {name: summarize_policy(records, chooser) for name, chooser in policies.items()}

    ranked = []
    for record in records:
        r3 = record["variants"]["r3"]
        r4 = record["variants"]["r4"]
        delta_error = float(r3["sqerr"]) - float(r4["sqerr"])
        speed_gain = float(r4["seconds"]) - float(r3["seconds"])
        ranked.append((delta_error / max(float(r4["ref_sq"]), 1e-20), -speed_gain, record["call_idx"]))
    ranked.sort()

    record_by_call = {record["call_idx"]: record for record in records}
    for fraction in (0.1, 0.2, 0.33, 0.5):
        count = max(1, round(len(records) * fraction))
        selected = {call_idx for _, _, call_idx in ranked[:count]}
        simulated[f"r3_low_delta_{int(fraction * 100)}pct"] = summarize_policy(
            records,
            lambda record, selected=selected: "r3" if record["call_idx"] in selected else "r4",
        )
        simulated[f"r3_low_delta_{int(fraction * 100)}pct"]["selected_calls"] = [
            {
                "call_idx": int(call_idx),
                "denoise_idx": int(record_by_call[call_idx]["denoise_idx"]),
                "layer_idx": int(record_by_call[call_idx]["layer_idx"]),
            }
            for call_idx in sorted(selected)
        ]
    return simulated


def load_pipeline(config_path, checkpoint, dtype, device, use_ema):
    config = OmegaConf.load(config_path)
    pipeline = TokenMergingWanPipeline(config, device=device)
    load_generator_checkpoint(pipeline, checkpoint, use_ema)
    pipeline = pipeline.to(dtype=dtype)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval()
    for idx, block in enumerate(pipeline.generator.model.blocks):
        block.self_attn._analysis_layer_idx = idx
    return config, pipeline


def install_probe(records, max_calls, state, merge_backend):
    import wan.modules.model as wan_model

    call_index = {"value": 0}
    warmed = {"value": False}

    def run_dense(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

    def run_hard_route(q, k, v, grid_size, backend):
        _, h, w = grid_size
        return wan_model.token_merging_attn(
            q,
            k,
            v,
            1,
            1,
            1,
            h,
            w,
            q_init="ith",
            layout="hard_route_shared",
            num_merge_tokens=4,
            seed_mode="fast_random_with_ith",
            seed=0,
            hard_route_backend=merge_backend,
            hard_route_routing_backend=backend,
        )

    def forward(self, x, grid_size, rope_cache):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if torch.is_grad_enabled() or x.requires_grad:
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
        else:
            q = wan_model.rmsnorm(self.q(x), self.norm_q.weight, eps=self.norm_q.eps).view(b, s, n, d)
            k = wan_model.rmsnorm(self.k(x), self.norm_k.weight, eps=self.norm_k.eps).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        q, k = wan_model.apply_rope_video_tokens(q, k, rope_cache)

        should_record = max_calls <= 0 or call_index["value"] < max_calls
        current_call = call_index["value"]
        call_index["value"] += 1

        if should_record and not warmed["value"]:
            _ = run_dense(q, k, v, self)
            _ = run_hard_route(q, k, v, grid_size, R3_BACKEND)
            _ = run_hard_route(q, k, v, grid_size, R4_BACKEND)
            sync()
            warmed["value"] = True

        if should_record:
            dense, dense_seconds = timed(lambda: run_dense(q, k, v, self))
            r3, r3_seconds = timed(lambda: run_hard_route(q, k, v, grid_size, R3_BACKEND))
            r4, r4_seconds = timed(lambda: run_hard_route(q, k, v, grid_size, R4_BACKEND))
            r3_stats = error_stats(r3, dense)
            r4_stats = error_stats(r4, dense)
            r3_stats["seconds"] = r3_seconds
            r4_stats["seconds"] = r4_seconds
            records.append(
                {
                    "call_idx": current_call,
                    "denoise_idx": int(state["current_denoise_idx"]),
                    "layer_idx": int(getattr(self, "_analysis_layer_idx", 0)),
                    "dense_seconds": dense_seconds,
                    "variants": {
                        "r3": r3_stats,
                        "r4": r4_stats,
                    },
                }
            )
            x = dense
        else:
            x = run_dense(q, k, v, self)
        return self.o(x.flatten(2))

    wan_model.WanSelfAttention.forward = forward


@torch.inference_mode()
def run_denoise_only(pipeline, config, prompt, seed, dtype, state):
    device = torch.device("cuda")
    set_seed(seed)
    latent_shape = list(config.image_or_video_shape)
    noise = torch.randn(latent_shape, device=device, dtype=dtype)
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt] * latent_shape[0])
    timesteps = pipeline.denoising_step_list.to(device=device)
    timestep_shape = noise.shape[:2]

    noisy = noise
    for index, current_timestep in enumerate(timesteps[:-1]):
        state["current_denoise_idx"] = index
        _, pred = pipeline.generator(
            noisy_image_or_video=noisy,
            conditional_dict=conditional_dict,
            timestep=current_timestep.expand(timestep_shape),
        )
        next_timestep = timesteps[index + 1].expand(timestep_shape)
        noisy = pipeline.scheduler.add_noise(
            pred.flatten(0, 1),
            torch.randn_like(pred.flatten(0, 1)),
            next_timestep.flatten(0, 1),
        ).unflatten(0, noise.shape[:2])


def parse_args():
    parser = argparse.ArgumentParser(description="Record per-call hard C4 R3/R4 speed and error on real Wan QKV.")
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=90)
    parser.add_argument("--hard_route_backend", default=MERGE_BACKEND)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    config, pipeline = load_pipeline(args.config, args.checkpoint, dtype, device, args.use_ema)
    prompt = load_prompts(args.prompt_path, args.prompt_index + 1)[args.prompt_index]
    num_layers = len(pipeline.generator.model.blocks)
    num_denoise_steps = len(pipeline.denoising_step_list) - 1
    records = []
    state = {"current_denoise_idx": 0}
    install_probe(records, args.max_calls, state, args.hard_route_backend)

    sync()
    start = time.perf_counter()
    run_denoise_only(pipeline, config, prompt, args.seed, dtype, state)
    sync()

    dense_seconds = sum(record["dense_seconds"] for record in records)
    summaries = {
        "dense": {
            "calls": len(records),
            "seconds": dense_seconds,
            "mean_ms": 1000.0 * dense_seconds / len(records) if records else 0.0,
        },
        "r3": summarize_variant(records, "r3"),
        "r4": summarize_variant(records, "r4"),
    }
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "dtype": args.dtype,
        "max_calls": args.max_calls,
        "num_layers": num_layers,
        "num_denoise_steps": num_denoise_steps,
        "merge_backend": args.hard_route_backend,
        "r3_backend": R3_BACKEND,
        "r4_backend": R4_BACKEND,
        "wall_seconds": time.perf_counter() - start,
        "summaries": summaries,
        "by_denoise": aggregate_by(records, lambda record: record["denoise_idx"]),
        "by_layer": aggregate_by(records, lambda record: record["layer_idx"]),
        "simulated_policies": simulate_policies(records, num_layers, num_denoise_steps),
        "records": records,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
