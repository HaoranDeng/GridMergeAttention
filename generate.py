import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from pipeline import CausalTokenMergingWanPipeline, TokenMergingWanPipeline
from utils.misc import save_video, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos with Wan2.1 + hard C4 GridMergeAttention.")
    parser.add_argument("--config", default="configs/token_merging_wan_hard_c4_triton_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--output_dir", default="videos/hard_c4_fused_wan")
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Batch this many prompt/sample chunks into one pipeline call. "
            "The resulting model batch can be as large as "
            "--batch_size * --sample_batch_size."
        ),
    )
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
    parser.add_argument("--video_preset", default="ultrafast")
    parser.add_argument("--video_crf", type=int, default=23)
    parser.add_argument("--video_quality", type=int, default=None)
    parser.add_argument(
        "--vae_batch_decode",
        action="store_true",
        help="Decode all videos in a batch together. Faster after kernel warmup, but uses more memory.",
    )
    parser.add_argument(
        "--vbench_filenames",
        action="store_true",
        help=(
            "Save videos as '<prompt>-<sample>.mp4' for VBench standard evaluation. "
            "Repeated prompts are generated only once."
        ),
    )
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


def save_video_atomic(
    output_path: Path,
    video: torch.Tensor,
    *,
    fps: int,
    preset: str,
    crf: int | None,
    quality: int | None,
) -> None:
    partial_path = output_path.with_name(f".{output_path.stem}.partial{output_path.suffix}")
    try:
        save_video(
            str(partial_path),
            video,
            fps=fps,
            preset=preset,
            crf=crf,
            quality=quality,
        )
        os.replace(partial_path, output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


def iter_sample_chunks(prompt_work, sample_batch_size):
    """Split each prompt's remaining samples into model-sized chunks."""

    for prompt_idx, prompt, sample_indices in prompt_work:
        for start in range(0, len(sample_indices), sample_batch_size):
            yield (
                prompt_idx,
                prompt,
                sample_indices[start : start + sample_batch_size],
            )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required for Wan2.1 generation.")
    if args.vae_batch_decode:
        os.environ["WAN_VAE_BATCH_DECODE"] = "1"
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if args.num_prompt_shards <= 0:
        raise ValueError("--num_prompt_shards must be positive.")
    if not 0 <= args.prompt_shard_id < args.num_prompt_shards:
        raise ValueError("--prompt_shard_id must be in [0, --num_prompt_shards).")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    sample_batch_size = args.num_samples if args.sample_batch_size is None else args.sample_batch_size
    if sample_batch_size <= 0:
        raise ValueError("--sample_batch_size must be positive.")
    sample_batch_size = min(sample_batch_size, args.num_samples)

    set_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    config = OmegaConf.load(args.config)
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

    # Balance only unfinished work. A resumed run can therefore redistribute
    # its remaining samples across every requested GPU/process.
    sharded_prompt_work = [[] for _ in range(args.num_prompt_shards)]
    shard_work_units = [0] * args.num_prompt_shards
    for work_item in all_prompt_work:
        target_shard = min(
            range(args.num_prompt_shards),
            key=lambda shard_id: (shard_work_units[shard_id], shard_id),
        )
        sharded_prompt_work[target_shard].append(work_item)
        shard_work_units[target_shard] += (
            len(work_item[2]) + sample_batch_size - 1
        ) // sample_batch_size

    prompt_work = sharded_prompt_work[args.prompt_shard_id]
    sample_chunks = list(iter_sample_chunks(prompt_work, sample_batch_size))
    pending_sample_count = sum(len(chunk[2]) for chunk in sample_chunks)
    print(
        f"Remaining-work shard {args.prompt_shard_id + 1}/{args.num_prompt_shards}: "
        f"{len(prompt_work)} prompts, {len(sample_chunks)} sample chunks, and "
        f"{pending_sample_count} samples assigned; global scan found "
        f"{len(all_prompt_work)} prompts with pending samples and "
        f"{skipped_count} existing videos."
    )
    if not sample_chunks:
        print(f"Nothing to generate for prompt shard {args.prompt_shard_id + 1}.")
        return

    token_merging_enabled = bool(
        config.get("token_merging_args", {}).get("enable", False)
    )
    attention_mode = (
        "token-merging attention"
        if token_merging_enabled
        else "dense SDPA attention"
    )
    generation_mode = (
        "causal Self-Forcing"
        if bool(config.get("causal", False))
        else "bidirectional Wan"
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
    for start in tqdm(
        range(0, len(sample_chunks), args.batch_size),
        desc=f"Generating shard {args.prompt_shard_id + 1}/{args.num_prompt_shards}",
    ):
        batch = sample_chunks[start : start + args.batch_size]
        noise_pieces = []
        text_prompts = []
        output_items = []
        for idx, prompt, sample_indices in batch:
            # Preserve the previous resume-safe stream choice for every sample
            # chunk before concatenating the chunks into the remote prompt
            # batching path.
            set_seed(args.seed + idx * args.num_samples + sample_indices[0])
            batch_shape = list(latent_shape)
            batch_shape[0] = len(sample_indices)
            noise_pieces.append(
                torch.randn(batch_shape, device=device, dtype=dtype)
            )
            text_prompts.extend([prompt] * len(sample_indices))
            output_items.extend(
                (idx, prompt, sample_idx) for sample_idx in sample_indices
            )

        noise = torch.cat(noise_pieces, dim=0)
        video = pipeline.inference(noise=noise, text_prompts=text_prompts)
        with torch.inference_mode():
            video = (
                video.permute(0, 1, 3, 4, 2)
                .mul_(255.0)
                .clamp_(0, 255)
                .to(torch.uint8)
                .cpu()
            )
        if video.size(0) != len(output_items):
            raise RuntimeError(
                "pipeline output batch does not match the requested sample batch."
            )

        for video_offset, (idx, prompt, sample_idx) in enumerate(output_items):
            output_path = get_output_path(
                output_dir,
                idx,
                prompt,
                sample_idx,
                args.vbench_filenames,
            )
            save_video_atomic(
                output_path,
                video[video_offset],
                fps=args.fps,
                preset=args.video_preset,
                crf=args.video_crf,
                quality=args.video_quality,
            )
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
