import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from pipeline import CausalTokenMergingWanPipeline, TokenMergingWanPipeline
from utils.misc import save_video, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos with Wan2.1 + token-merging attention.")
    parser.add_argument("--config", default="configs/token_merging_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--output_dir", default="videos/token_merging_wan")
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=None,
        help="Generate this many samples at a time while keeping --num_samples total outputs per prompt.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip individual output videos that already exist and are non-empty.",
    )
    parser.add_argument(
        "--prompt_shard_id",
        type=int,
        default=0,
        help="Zero-based prompt shard assigned to this process.",
    )
    parser.add_argument(
        "--num_prompt_shards",
        type=int,
        default=1,
        help="Balance remaining prompt work across this many independent processes.",
    )
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


def build_pipeline(config, device):
    """Route causal configs to Self-Forcing and other configs to ordinary Wan."""
    if bool(config.get("causal", False)):
        return CausalTokenMergingWanPipeline(config, device=device)
    return TokenMergingWanPipeline(config, device=device)


def load_generator_checkpoint(pipeline: torch.nn.Module, checkpoint_path: str, use_ema: bool):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and ("generator" in state or "generator_ema" in state):
        key = "generator_ema" if use_ema and "generator_ema" in state else "generator"
        state = state[key]
    pipeline.generator.load_state_dict(state)


def get_output_path(
    output_dir: Path,
    prompt_idx: int,
    prompt: str,
    sample_idx: int,
    vbench_filenames: bool,
) -> Path:
    if vbench_filenames:
        return output_dir / f"{prompt}-{sample_idx}.mp4"
    return output_dir / f"{prompt_idx:03d}-{sample_idx:02d}.mp4"


def save_video_atomic(output_path: Path, video: torch.Tensor, fps: int) -> None:
    partial_path = output_path.with_name(f".{output_path.stem}.partial{output_path.suffix}")
    try:
        save_video(str(partial_path), video, fps=fps)
        os.replace(partial_path, output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required for Wan2.1 generation.")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if args.num_prompt_shards <= 0:
        raise ValueError("--num_prompt_shards must be positive.")
    if not 0 <= args.prompt_shard_id < args.num_prompt_shards:
        raise ValueError("--prompt_shard_id must be in [0, --num_prompt_shards).")

    sample_batch_size = args.num_samples if args.sample_batch_size is None else args.sample_batch_size
    if sample_batch_size <= 0:
        raise ValueError("--sample_batch_size must be positive.")
    sample_batch_size = min(sample_batch_size, args.num_samples)

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    skipped_count = 0
    all_prompt_work = []
    for idx, prompt in indexed_prompts:
        sample_indices = list(range(args.num_samples))
        if args.skip_existing:
            missing_sample_indices = []
            for sample_idx in sample_indices:
                output_path = get_output_path(
                    output_dir,
                    idx,
                    prompt,
                    sample_idx,
                    args.vbench_filenames,
                )
                if output_path.is_file() and output_path.stat().st_size > 0:
                    skipped_count += 1
                else:
                    missing_sample_indices.append(sample_idx)
            sample_indices = missing_sample_indices
        if sample_indices:
            all_prompt_work.append((idx, prompt, sample_indices))

    # Shard only the remaining work rather than the full prompt list. This
    # redistributes an incomplete shard across all GPUs when a run is resumed,
    # even if the new run uses the same number of GPUs as the old one.
    sharded_prompt_work = [[] for _ in range(args.num_prompt_shards)]
    shard_work_units = [0] * args.num_prompt_shards
    for work_item in all_prompt_work:
        target_shard = min(
            range(args.num_prompt_shards),
            key=lambda shard_id: (shard_work_units[shard_id], shard_id),
        )
        sharded_prompt_work[target_shard].append(work_item)
        sample_indices = work_item[2]
        shard_work_units[target_shard] += (
            len(sample_indices) + sample_batch_size - 1
        ) // sample_batch_size
    prompt_work = sharded_prompt_work[args.prompt_shard_id]
    pending_sample_count = sum(len(work_item[2]) for work_item in prompt_work)

    print(
        f"Remaining-work shard {args.prompt_shard_id + 1}/{args.num_prompt_shards}: "
        f"{len(prompt_work)} prompts and {pending_sample_count} samples assigned; "
        f"global scan found {len(all_prompt_work)} prompts with pending samples and "
        f"{skipped_count} existing videos."
    )
    if not prompt_work:
        print(f"Nothing to generate for prompt shard {args.prompt_shard_id + 1}.")
        return

    token_merging_enabled = bool(config.get("token_merging_args", {}).get("enable", False))
    attention_mode = "token-merging attention" if token_merging_enabled else "dense SDPA attention"
    generation_mode = (
        "causal Self-Forcing" if bool(config.get("causal", False)) else "bidirectional Wan"
    )
    print(
        f"Loading Wan2.1 + {attention_mode} with {generation_mode} "
        f"on {device} with dtype={dtype}."
    )
    pipeline = build_pipeline(config, device=device)
    load_generator_checkpoint(pipeline, args.checkpoint, args.use_ema)
    pipeline = pipeline.to(dtype=dtype)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval()

    latent_shape = list(config.image_or_video_shape)
    generated_count = 0
    for idx, prompt, sample_indices in tqdm(
        prompt_work,
        desc=f"Generating shard {args.prompt_shard_id + 1}/{args.num_prompt_shards}",
    ):

        for start in range(0, len(sample_indices), sample_batch_size):
            batch_sample_indices = sample_indices[start : start + sample_batch_size]
            batch_shape = list(latent_shape)
            batch_shape[0] = len(batch_sample_indices)

            # Include the first sample index so resumed partial prompts do not
            # reuse the same random stream for every missing chunk.
            set_seed(args.seed + idx * args.num_samples + batch_sample_indices[0])
            noise = torch.randn(batch_shape, device=device, dtype=dtype)
            text_prompts = [prompt] * len(batch_sample_indices)
            video = pipeline.inference(noise=noise, text_prompts=text_prompts)
            video = (255.0 * video.permute(0, 1, 3, 4, 2)).cpu()
            pipeline.vae.model.clear_cache()

            for batch_idx, sample_idx in enumerate(batch_sample_indices):
                output_path = get_output_path(
                    output_dir,
                    idx,
                    prompt,
                    sample_idx,
                    args.vbench_filenames,
                )
                save_video_atomic(output_path, video[batch_idx], fps=args.fps)
                generated_count += 1
                print(f"Saved {output_path}")
            del noise, video

    print(
        f"Done. Generated {generated_count} videos; the startup scan found "
        f"{skipped_count} existing videos in {output_dir}."
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKEN_MERGING_FORCE_TORCH_RMSNORM", "1")
    main()
