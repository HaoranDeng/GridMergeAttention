import argparse
import contextlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from generate import load_generator_checkpoint, load_prompts
from pipeline import TokenMergingWanPipeline
from utils.misc import save_video, set_seed


class CudaTimer:
    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def section(self, name):
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.totals[name] += time.perf_counter() - start
            self.counts[name] += 1


def install_profiled_wan_forwards(timer):
    import wan.modules.model as wan_model
    import utils.wan_wrapper as wan_wrapper

    original_build_rope_cache = wan_model.build_cos_sin_cache_3d
    original_unpatchify = wan_model.WanModel.unpatchify
    original_convert_x0 = wan_wrapper.WanDiffusionWrapper._convert_flow_pred_to_x0

    def profiled_build_rope_cache(*args, **kwargs):
        with timer.section("transformer.rope_cache_build"):
            return original_build_rope_cache(*args, **kwargs)

    def profiled_unpatchify(self, x, grid_size, c=None):
        with timer.section("transformer.unpatchify"):
            return original_unpatchify(self, x, grid_size, c=c)

    def profiled_convert_x0(self, *args, **kwargs):
        with timer.section("diffusion.x0_conversion"):
            return original_convert_x0(self, *args, **kwargs)

    def profiled_self_attention(self, x, grid_size, rope_cache):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        with timer.section("transformer.self_attn.qkv_norm"):
            if torch.is_grad_enabled() or x.requires_grad:
                q = self.norm_q(self.q(x)).view(b, s, n, d)
                k = self.norm_k(self.k(x)).view(b, s, n, d)
            else:
                q = wan_model.rmsnorm(self.q(x), self.norm_q.weight, eps=self.norm_q.eps).view(b, s, n, d)
                k = wan_model.rmsnorm(self.k(x), self.norm_k.weight, eps=self.norm_k.eps).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)

        with timer.section("transformer.self_attn.rope"):
            roped_query, roped_key = wan_model.apply_rope_video_tokens(q, k, rope_cache)
            wan_model.maybe_save_first_attention_qkv(roped_query, roped_key, v, "wan_self_attn")

        with timer.section("transformer.self_attn.core"):
            if self.enable_token_merging:
                f, h, w = grid_size
                assert f % self.token_merging_f_tied == 0
                assert h % self.token_merging_h_reduce == 0
                assert w % self.token_merging_w_reduce == 0
                x = wan_model.token_merging_attn(
                    roped_query,
                    roped_key,
                    v,
                    self.token_merging_f_tied,
                    self.token_merging_h_reduce,
                    self.token_merging_w_reduce,
                    h,
                    w,
                    q_init=self.token_merging_q_init,
                    query_outer_chunk=self.token_merging_query_outer_chunk,
                    layout=self.token_merging_layout,
                    num_merge_tokens=self.token_merging_num_merge_tokens,
                    seed_mode=self.token_merging_seed_mode,
                    seed=self.token_merging_seed,
                    hard_route_backend=self.token_merging_hard_route_backend,
                    hard_route_routing_backend=self.token_merging_hard_route_routing_backend,
                )
            else:
                x = wan_model.flash_attention(
                    q=roped_query,
                    k=roped_key,
                    v=v,
                    window_size=self.window_size,
                )

        with timer.section("transformer.self_attn.output_proj"):
            x = self.o(x.flatten(2))
        return x

    def profiled_cross_attention(self, x, context, context_lens, crossattn_cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        with timer.section("transformer.cross_attn.qkv_norm"):
            q = self.norm_q(self.q(x)).view(b, -1, n, d)
            if crossattn_cache is not None:
                if not crossattn_cache["is_init"]:
                    crossattn_cache["is_init"] = True
                    k = self.norm_k(self.k(context)).view(b, -1, n, d)
                    v = self.v(context).view(b, -1, n, d)
                    crossattn_cache["k"] = k
                    crossattn_cache["v"] = v
                else:
                    k = crossattn_cache["k"]
                    v = crossattn_cache["v"]
            else:
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)

        with timer.section("transformer.cross_attn.core"):
            x = wan_model.flash_attention(q, k, v, k_lens=context_lens)

        with timer.section("transformer.cross_attn.output_proj"):
            x = self.o(x.flatten(2))
        return x

    def profiled_block(self, x, e, grid_size, context, context_lens, rope_cache):
        with timer.section("transformer.block.modulation"):
            e = (self.modulation + e).chunk(6, dim=1)

        with timer.section("transformer.block.self_input_norm_mod"):
            self_input = self.norm1(x) * (1 + e[1]) + e[0]
        y = self.self_attn(self_input, grid_size, rope_cache)
        with timer.section("transformer.block.self_residual"):
            x = x + y * e[2]

        with timer.section("transformer.block.cross_input_norm"):
            cross_input = self.norm3(x)
        y = self.cross_attn(cross_input, context, context_lens)
        with timer.section("transformer.block.cross_residual"):
            x = x + y

        with timer.section("transformer.block.ffn_input_norm_mod"):
            ffn_input = self.norm2(x) * (1 + e[4]) + e[3]
        with timer.section("transformer.block.ffn"):
            y = self.ffn(ffn_input)
        with timer.section("transformer.block.ffn_residual"):
            x = x + y * e[5]
        return x

    wan_model.build_cos_sin_cache_3d = profiled_build_rope_cache
    wan_model.WanModel.unpatchify = profiled_unpatchify
    wan_wrapper.WanDiffusionWrapper._convert_flow_pred_to_x0 = profiled_convert_x0
    wan_model.WanSelfAttention.forward = profiled_self_attention
    wan_model.WanT2VCrossAttention.forward = profiled_cross_attention
    wan_model.WanAttentionBlock.forward = profiled_block


def add_module_timers(pipeline, timer):
    handles = []

    def register(module, name):
        stack = []

        def pre_hook(_module, _inputs):
            timer.sync()
            stack.append(time.perf_counter())

        def post_hook(_module, _inputs, _output):
            timer.sync()
            timer.totals[name] += time.perf_counter() - stack.pop()
            timer.counts[name] += 1

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    model = pipeline.generator.model
    register(model.patch_embedding, "transformer.patch_embedding")
    register(model.time_embedding, "transformer.time_embedding")
    register(model.time_projection, "transformer.time_projection")
    register(model.text_embedding, "transformer.text_embedding")
    register(model.head, "transformer.head")
    return handles


def load_pipeline(config_path, checkpoint, use_ema, dtype, device):
    config = OmegaConf.load(config_path)
    pipeline = TokenMergingWanPipeline(config, device=device)
    load_generator_checkpoint(pipeline, checkpoint, use_ema)
    pipeline = pipeline.to(dtype=dtype)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval()
    return config, pipeline


@torch.inference_mode()
def run_one_video(pipeline, config, prompt, seed, dtype, fps, output_path, timer=None):
    device = torch.device("cuda")
    set_seed(seed)
    latent_shape = list(config.image_or_video_shape)
    noise = torch.randn(latent_shape, device=device, dtype=dtype)
    text_prompts = [prompt] * latent_shape[0]

    ctx = timer.section if timer is not None else contextlib.nullcontext

    with ctx("pipeline.text_encoder"):
        conditional_dict = pipeline.text_encoder(text_prompts=text_prompts)

    noisy_image_or_video = noise
    pred_image_or_video = None
    denoise_steps = int(len(pipeline.denoising_step_list) - 1)

    with ctx("pipeline.denoise_loop_total"):
        timesteps = pipeline.denoising_step_list.to(device=noise.device)
        timestep_shape = noise.shape[:2]
        for index, current_timestep in enumerate(timesteps[:-1]):
            with ctx("pipeline.generator_forward_total"):
                _, pred_image_or_video = pipeline.generator(
                    noisy_image_or_video=noisy_image_or_video,
                    conditional_dict=conditional_dict,
                    timestep=current_timestep.expand(timestep_shape),
                )
            next_timestep = timesteps[index + 1].expand(timestep_shape)
            with ctx("pipeline.scheduler_add_noise"):
                noisy_image_or_video = pipeline.scheduler.add_noise(
                    pred_image_or_video.flatten(0, 1),
                    torch.randn_like(pred_image_or_video.flatten(0, 1)),
                    next_timestep.flatten(0, 1),
                ).unflatten(0, noise.shape[:2])

    if pred_image_or_video is None:
        raise ValueError("denoising_step_list must contain at least two steps.")

    with ctx("pipeline.vae_decode"):
        video = pipeline.vae.decoder(pred_image_or_video)
        video.mul_(0.5).add_(0.5).clamp_(0, 1)

    with ctx("pipeline.postprocess_cpu_transfer"):
        video_cpu = video.permute(0, 1, 3, 4, 2).mul_(255.0).clamp_(0, 255).to(torch.uint8).cpu()

    if output_path is not None:
        with ctx("pipeline.save_video"):
            save_video(str(output_path), video_cpu[0], fps=fps)

    return {"denoise_steps": denoise_steps, "output_path": str(output_path) if output_path else None}


def group_totals(totals, total_wall):
    get = totals.get
    groups = {
        "text_encoder": get("pipeline.text_encoder", 0.0),
        "self_attention_core": get("transformer.self_attn.core", 0.0),
        "cross_attention_core": get("transformer.cross_attn.core", 0.0),
        "self_attention_qkv_rope_output": (
            get("transformer.block.self_input_norm_mod", 0.0)
            + get("transformer.self_attn.qkv_norm", 0.0)
            + get("transformer.self_attn.rope", 0.0)
            + get("transformer.self_attn.output_proj", 0.0)
            + get("transformer.block.self_residual", 0.0)
        ),
        "cross_attention_qkv_output": (
            get("transformer.block.cross_input_norm", 0.0)
            + get("transformer.cross_attn.qkv_norm", 0.0)
            + get("transformer.cross_attn.output_proj", 0.0)
            + get("transformer.block.cross_residual", 0.0)
        ),
        "ffn": (
            get("transformer.block.ffn_input_norm_mod", 0.0)
            + get("transformer.block.ffn", 0.0)
            + get("transformer.block.ffn_residual", 0.0)
        ),
        "embedding_head_rope_cache_unpatch": (
            get("transformer.patch_embedding", 0.0)
            + get("transformer.time_embedding", 0.0)
            + get("transformer.time_projection", 0.0)
            + get("transformer.text_embedding", 0.0)
            + get("transformer.rope_cache_build", 0.0)
            + get("transformer.head", 0.0)
            + get("transformer.unpatchify", 0.0)
            + get("transformer.block.modulation", 0.0)
        ),
        "x0_conversion": get("diffusion.x0_conversion", 0.0),
        "scheduler_add_noise": get("pipeline.scheduler_add_noise", 0.0),
        "vae_decode": get("pipeline.vae_decode", 0.0),
        "postprocess_cpu_transfer": get("pipeline.postprocess_cpu_transfer", 0.0),
        "save_video": get("pipeline.save_video", 0.0),
    }
    known = sum(groups.values())
    groups["unattributed_python_overhead"] = max(0.0, total_wall - known)
    return {
        name: {
            "seconds": seconds,
            "percent_of_total": 100.0 * seconds / total_wall if total_wall else 0.0,
        }
        for name, seconds in groups.items()
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Profile Wan video generation by pipeline and transformer component.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_video", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required for Wan2.1 generation profiling.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")

    config, pipeline = load_pipeline(args.config, args.checkpoint, args.use_ema, dtype, device)
    prompts = load_prompts(args.prompt_path, args.prompt_index + 1)
    prompt = prompts[args.prompt_index]

    for warmup_idx in range(args.warmup):
        run_one_video(
            pipeline,
            config,
            prompt,
            seed=args.seed + 1000 + warmup_idx,
            dtype=dtype,
            fps=args.fps,
            output_path=None,
            timer=None,
        )

    timer = CudaTimer()
    install_profiled_wan_forwards(timer)
    handles = add_module_timers(pipeline, timer)
    output_video = Path(args.output_video) if args.output_video else None
    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)

    timer.sync()
    start = time.perf_counter()
    info = run_one_video(
        pipeline,
        config,
        prompt,
        seed=args.seed,
        dtype=dtype,
        fps=args.fps,
        output_path=output_video,
        timer=timer,
    )
    timer.sync()
    total_wall = time.perf_counter() - start

    for handle in handles:
        handle.remove()

    raw_totals = dict(sorted(timer.totals.items()))
    payload = {
        "label": args.label or Path(args.config).stem,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "total_wall_seconds": total_wall,
        **info,
        "raw_sections": {
            name: {
                "seconds": seconds,
                "percent_of_total": 100.0 * seconds / total_wall if total_wall else 0.0,
                "count": timer.counts.get(name, 0),
            }
            for name, seconds in raw_totals.items()
        },
        "groups": group_totals(raw_totals, total_wall),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "label": payload["label"],
        "total_wall_seconds": total_wall,
        "groups": payload["groups"],
        "output_json": args.output_json,
        "output_video": str(output_video) if output_video else None,
    }, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKEN_MERGING_FORCE_TORCH_RMSNORM", "1")
    main()
