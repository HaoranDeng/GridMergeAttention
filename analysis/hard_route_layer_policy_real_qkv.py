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


class Stats:
    def __init__(self):
        self.calls = 0
        self.seconds = 0.0
        self.sqerr = 0.0
        self.ref_sq = 0.0
        self.numel = 0
        self.rel_l2_sum = 0.0
        self.cosine_error_sum = 0.0
        self.max_abs = 0.0

    def add_time(self, seconds):
        self.calls += 1
        self.seconds += float(seconds)

    def add_error(self, out, ref):
        diff = (out.float() - ref.float()).reshape(-1)
        ref_flat = ref.float().reshape(-1)
        out_flat = out.float().reshape(-1)
        sqerr = torch.dot(diff, diff)
        ref_sq = torch.dot(ref_flat, ref_flat).clamp_min(1e-20)
        out_sq = torch.dot(out_flat, out_flat).clamp_min(1e-20)
        dot = torch.dot(out_flat, ref_flat)

        self.sqerr += float(sqerr.item())
        self.ref_sq += float(ref_sq.item())
        self.numel += diff.numel()
        self.rel_l2_sum += float(torch.sqrt(sqerr / ref_sq).item())
        self.cosine_error_sum += float((1.0 - dot / torch.sqrt(out_sq * ref_sq)).item())
        self.max_abs = max(self.max_abs, float(diff.abs().max().item()))

    def summary(self, dense_seconds=None):
        out = {
            "calls": self.calls,
            "seconds": self.seconds,
            "mean_ms": 1000.0 * self.seconds / self.calls if self.calls else 0.0,
        }
        if dense_seconds is not None and self.seconds:
            out["speedup_vs_dense"] = dense_seconds / self.seconds
        if self.numel:
            out.update(
                {
                    "mse": self.sqerr / self.numel,
                    "rel_l2": (self.sqerr / max(self.ref_sq, 1e-20)) ** 0.5,
                    "mean_call_rel_l2": self.rel_l2_sum / self.calls,
                    "cosine_error": self.cosine_error_sum / self.calls,
                    "max_abs": self.max_abs,
                }
            )
        return out


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


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


MERGE_BACKEND = "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached"
R3_BACKEND = "triton_grid_outer_exact_r3_m128_n64_s1_exp2"
R4_BACKEND = "triton_grid_outer_exact_r4_m128_n64_s1_exp2"


def policy_backend(policy, layer_idx, num_layers):
    if policy == "r3_all":
        return R3_BACKEND
    if policy == "r4_all":
        return R4_BACKEND
    if policy == "r3_first_half":
        return R3_BACKEND if layer_idx < num_layers // 2 else R4_BACKEND
    if policy == "r3_second_half":
        return R4_BACKEND if layer_idx < num_layers // 2 else R3_BACKEND
    if policy == "r3_middle_third":
        start = num_layers // 3
        end = 2 * num_layers // 3
        return R3_BACKEND if start <= layer_idx < end else R4_BACKEND
    if policy == "r3_even_layers":
        return R3_BACKEND if layer_idx % 2 == 0 else R4_BACKEND
    raise ValueError(f"unknown policy={policy!r}")


def install_probe(stats, max_calls, policies, num_layers):
    import wan.modules.model as wan_model

    call_index = {"value": 0}
    warmed = {"value": False}

    def run_dense(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

    def run_policy(q, k, v, grid_size, self, policy):
        f, h, w = grid_size
        del f
        layer_idx = int(getattr(self, "_analysis_layer_idx", 0))
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
            hard_route_backend=MERGE_BACKEND,
            hard_route_routing_backend=policy_backend(policy, layer_idx, num_layers),
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
        call_index["value"] += 1

        if should_record and not warmed["value"]:
            _ = run_dense(q, k, v, self)
            for policy in policies:
                _ = run_policy(q, k, v, grid_size, self, policy)
            sync()
            warmed["value"] = True

        if should_record:
            dense, seconds = timed(lambda: run_dense(q, k, v, self))
            stats["dense"].add_time(seconds)
            for policy in policies:
                out, seconds = timed(lambda policy=policy: run_policy(q, k, v, grid_size, self, policy))
                stats[policy].add_time(seconds)
                stats[policy].add_error(out, dense)
            x = dense
        else:
            x = run_dense(q, k, v, self)
        return self.o(x.flatten(2))

    wan_model.WanSelfAttention.forward = forward


@torch.inference_mode()
def run_denoise_only(pipeline, config, prompt, seed, dtype):
    device = torch.device("cuda")
    set_seed(seed)
    latent_shape = list(config.image_or_video_shape)
    noise = torch.randn(latent_shape, device=device, dtype=dtype)
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt] * latent_shape[0])

    noisy = noise
    for index, current_timestep in enumerate(pipeline.denoising_step_list[:-1]):
        _, pred = pipeline.generator(
            noisy_image_or_video=noisy,
            conditional_dict=conditional_dict,
            timestep=torch.ones(noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep,
        )
        next_timestep = pipeline.denoising_step_list[index + 1] * torch.ones(
            noise.shape[:2],
            dtype=torch.long,
            device=noise.device,
        )
        noisy = pipeline.scheduler.add_noise(
            pred.flatten(0, 1),
            torch.randn_like(pred.flatten(0, 1)),
            next_timestep.flatten(0, 1),
        ).unflatten(0, noise.shape[:2])


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark layer-dependent R3/R4 hard-route policies.")
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=90)
    parser.add_argument(
        "--policies",
        default="r4_all,r3_all,r3_first_half,r3_second_half,r3_middle_third,r3_even_layers",
    )
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required.")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    policies = [name.strip() for name in args.policies.split(",") if name.strip()]

    config, pipeline = load_pipeline(args.config, args.checkpoint, dtype, device, args.use_ema)
    prompt = load_prompts(args.prompt_path, args.prompt_index + 1)[args.prompt_index]
    num_layers = len(pipeline.generator.model.blocks)
    stats = defaultdict(Stats)
    install_probe(stats, args.max_calls, policies, num_layers)

    sync()
    start = time.perf_counter()
    run_denoise_only(pipeline, config, prompt, args.seed, dtype)
    sync()

    dense_seconds = stats["dense"].seconds
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "dtype": args.dtype,
        "num_layers": num_layers,
        "max_calls": args.max_calls,
        "policies": policies,
        "wall_seconds": time.perf_counter() - start,
        "merge_backend": MERGE_BACKEND,
        "r3_backend": R3_BACKEND,
        "r4_backend": R4_BACKEND,
        "variants": {
            name: value.summary(dense_seconds=dense_seconds if name != "dense" else None)
            for name, value in sorted(stats.items())
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
