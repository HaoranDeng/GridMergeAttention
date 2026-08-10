#!/usr/bin/env python
"""Generate HunyuanVideo samples with joint Token-Merging attention."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml

from pipeline.token_merging_hunyuan import (
    load_hunyuan_token_merging_pipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate videos with Diffusers HunyuanVideo + Token Merging."
    )
    parser.add_argument(
        "--config", default="configs/token_merging_hunyuan_480p.yaml"
    )
    parser.add_argument(
        "--prompt_path", default="prompts/MovieGenVideoBench_extended.txt"
    )
    parser.add_argument(
        "--output_dir", default="videos/token_merging_hunyuan"
    )
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--token_merging_w_reduce", type=int, default=None)
    parser.add_argument(
        "--token_merging_query_outer_chunk", type=int, default=None
    )
    return parser.parse_args()


def load_yaml(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return config


def load_prompts(path, limit):
    with open(path, encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    return prompts if limit < 0 else prompts[:limit]


def output_path(output_dir, prompt_index, sample_index):
    return output_dir / f"{prompt_index:03d}-{sample_index:02d}.mp4"


def export_video_atomic(export_to_video, frames, destination, fps):
    partial = destination.with_name(
        f".{destination.stem}.partial{destination.suffix}"
    )
    try:
        export_to_video(frames, str(partial), fps=fps)
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("HunyuanVideo generation requires a CUDA GPU.")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")

    config = load_yaml(args.config)
    model = dict(config.get("model", {}))
    generation = dict(config.get("generation", {}))
    token_merging = dict(config.get("token_merging_args", {}))

    required_generation = ("height", "width", "num_frames")
    missing = [name for name in required_generation if name not in generation]
    if missing:
        raise ValueError(
            "generation config is missing: " + ", ".join(missing)
        )
    token_merging.update(
        {name: generation[name] for name in required_generation}
    )
    if args.token_merging_w_reduce is not None:
        token_merging["w_reduce"] = args.token_merging_w_reduce
    if args.token_merging_query_outer_chunk is not None:
        token_merging["query_outer_chunk"] = (
            args.token_merging_query_outer_chunk
        )

    model_id = args.model_id or model.get("id", "tencent/HunyuanVideo")
    dtype_name = str(model.get("dtype", "bf16")).lower()
    dtype_by_name = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError("model.dtype must be bf16 or fp16.")

    prompts = load_prompts(args.prompt_path, args.num_videos)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_path}")
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    work = []
    skipped = 0
    for prompt_index, prompt in enumerate(prompts):
        for sample_index in range(args.num_samples):
            path = output_path(destination, prompt_index, sample_index)
            if (
                args.skip_existing
                and path.is_file()
                and path.stat().st_size > 0
            ):
                skipped += 1
                continue
            work.append((prompt_index, prompt, sample_index, path))
    if not work:
        print(f"Nothing to generate; skipped {skipped} existing videos.")
        return

    pipe = load_hunyuan_token_merging_pipeline(
        model_id,
        token_merging_args=token_merging,
        torch_dtype=dtype_by_name[dtype_name],
        revision=model.get("revision", "refs/pr/18"),
        transformer_subfolder=model.get("transformer_subfolder", "transformer"),
        scheduler_shift=float(model.get("scheduler_shift", 7.0)),
        local_files_only=bool(
            args.local_files_only or model.get("local_files_only", False)
        ),
        device=model.get("device", "cuda"),
        model_cpu_offload=bool(model.get("model_cpu_offload", True)),
        sequential_cpu_offload=bool(
            model.get("sequential_cpu_offload", False)
        ),
        enable_vae_tiling=bool(model.get("enable_vae_tiling", True)),
    )
    pipe.set_progress_bar_config(disable=False)

    from diffusers.utils import export_to_video

    print(
        "Hunyuan Token Merging installed: "
        f"grid={pipe._hunyuan_token_merging_config['grid_f']}x"
        f"{pipe._hunyuan_token_merging_config['grid_h']}x"
        f"{pipe._hunyuan_token_merging_config['grid_w']}, "
        f"processors={len(pipe._hunyuan_token_merging_config['installed_processors'])}, "
        f"pending={len(work)}, skipped={skipped}."
    )

    fps = int(args.fps if args.fps is not None else generation.get("fps", 24))
    negative_prompt = generation.get("negative_prompt")
    true_cfg_scale = float(generation.get("true_cfg_scale", 1.0))
    for work_index, (prompt_index, prompt, sample_index, path) in enumerate(
        work, start=1
    ):
        sample_seed = args.seed + prompt_index * args.num_samples + sample_index
        generator = torch.Generator(device="cpu").manual_seed(sample_seed)
        call_kwargs = {
            "prompt": prompt,
            "height": int(generation["height"]),
            "width": int(generation["width"]),
            "num_frames": int(generation["num_frames"]),
            "num_inference_steps": int(
                generation.get("num_inference_steps", 50)
            ),
            "guidance_scale": float(generation.get("guidance_scale", 6.0)),
            "true_cfg_scale": true_cfg_scale,
            "max_sequence_length": int(
                generation.get("max_sequence_length", 256)
            ),
            "generator": generator,
            "output_type": "np",
        }
        if negative_prompt and true_cfg_scale > 1.0:
            call_kwargs["negative_prompt"] = negative_prompt

        print(
            f"[{work_index}/{len(work)}] prompt={prompt_index}, "
            f"sample={sample_index}, seed={sample_seed} -> {path}",
            flush=True,
        )
        frames = pipe(**call_kwargs).frames[0]
        export_video_atomic(export_to_video, frames, path, fps)


if __name__ == "__main__":
    main()
