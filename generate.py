import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from pipeline import TokenMergingWanPipeline
from utils.misc import save_video, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos with Wan2.1 + token-merging attention.")
    parser.add_argument("--config", default="configs/token_merging_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--output_dir", default="videos/token_merging_wan")
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--vbench_filenames",
        action="store_true",
        help=(
            "Save videos as '<prompt>-<sample>.mp4' for VBench standard evaluation. "
            "Repeated prompts are generated only once."
        ),
    )
    parser.add_argument("--token_merging_w_reduce", type=int, default=None)
    parser.add_argument("--token_merging_query_outer_chunk", type=int, default=None)
    return parser.parse_args()


def load_prompts(path: str, limit: int):
    with open(path, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if limit >= 0:
        prompts = prompts[:limit]
    return prompts


def load_generator_checkpoint(pipeline: TokenMergingWanPipeline, checkpoint_path: str, use_ema: bool):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and ("generator" in state or "generator_ema" in state):
        key = "generator_ema" if use_ema and "generator_ema" in state else "generator"
        state = state[key]
    pipeline.generator.load_state_dict(state)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required for Wan2.1 generation.")

    set_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    config = OmegaConf.load(args.config)
    if args.token_merging_w_reduce is not None:
        config.token_merging_args.w_reduce = args.token_merging_w_reduce
    if args.token_merging_query_outer_chunk is not None:
        config.token_merging_args.query_outer_chunk = args.token_merging_query_outer_chunk

    prompts = load_prompts(args.prompt_path, args.num_videos)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_path}")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.batch_size > 1 and args.num_samples != 1:
        raise ValueError("--batch_size > 1 currently requires --num_samples 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_merging_enabled = bool(config.get("token_merging_args", {}).get("enable", False))
    attention_mode = "token-merging attention" if token_merging_enabled else "dense SDPA attention"
    print(f"Loading Wan2.1 + {attention_mode} on {device} with dtype={dtype}.")
    pipeline = TokenMergingWanPipeline(config, device=device)
    load_generator_checkpoint(pipeline, args.checkpoint, args.use_ema)
    pipeline = pipeline.to(dtype=dtype)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval()

    latent_shape = list(config.image_or_video_shape)
    latent_shape[0] = args.num_samples

    indexed_prompts = list(enumerate(prompts))
    if args.vbench_filenames:
        seen_prompts = set()
        unique_indexed_prompts = []
        for idx, prompt in indexed_prompts:
            if prompt in seen_prompts:
                continue
            if "/" in prompt or os.sep in prompt:
                raise ValueError(f"VBench prompt cannot contain a path separator: {prompt!r}")
            seen_prompts.add(prompt)
            unique_indexed_prompts.append((idx, prompt))
        indexed_prompts = unique_indexed_prompts

    for batch_start in tqdm(range(0, len(indexed_prompts), args.batch_size), desc="Generating"):
        batch = indexed_prompts[batch_start: batch_start + args.batch_size]
        batch_noises = []
        batch_text_prompts = []
        for idx, prompt in batch:
            set_seed(args.seed + idx)
            batch_noises.append(torch.randn(latent_shape, device=device, dtype=dtype))
            batch_text_prompts.extend([prompt] * args.num_samples)
        noise = torch.cat(batch_noises, dim=0)
        video, _ = pipeline.inference(noise=noise, text_prompts=batch_text_prompts, return_latents=True)
        video = (255.0 * video.permute(0, 1, 3, 4, 2)).cpu()
        pipeline.vae.model.clear_cache()

        for batch_idx, (idx, prompt) in enumerate(batch):
            for sample_idx in range(args.num_samples):
                video_idx = batch_idx * args.num_samples + sample_idx
                if args.vbench_filenames:
                    output_path = output_dir / f"{prompt}-{sample_idx}.mp4"
                else:
                    output_path = output_dir / f"{idx:03d}-{sample_idx:02d}.mp4"
                save_video(str(output_path), video[video_idx], fps=args.fps)
                print(f"Saved {output_path}")

    print(f"Done. Generated {len(indexed_prompts) * args.num_samples} videos in {output_dir}.")


if __name__ == "__main__":
    os.environ.setdefault("TOKEN_MERGING_FORCE_TORCH_RMSNORM", "1")
    main()
