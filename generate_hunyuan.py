#!/usr/bin/env python
"""Generate HunyuanVideo samples with joint Token-Merging attention."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

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
        help="Balance unfinished prompt work across this many workers.",
    )
    parser.add_argument(
        "--vbench_filenames",
        action="store_true",
        help="Save outputs as '<prompt>-<sample>.mp4' for VBench evaluation.",
    )
    parser.add_argument(
        "--timing_output",
        default=None,
        help=(
            "Optional JSON report for newly generated videos. Pipeline timing "
            "excludes model loading and MP4 export."
        ),
    )
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


def output_path(
    output_dir,
    prompt_index,
    prompt,
    sample_index,
    vbench_filenames=False,
):
    if vbench_filenames:
        return output_dir / f"{prompt}-{sample_index}.mp4"
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


def write_json_atomic(payload, destination):
    """Write a timing report without exposing a partially written JSON file."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    try:
        with partial.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def build_timing_report(
    *,
    args,
    model_id,
    skipped,
    process_started_unix,
    process_wall_seconds,
    model_load_seconds,
    work_started_unix,
    work_finished_unix,
    work_wall_seconds,
    samples,
):
    """Build one worker's machine-readable timing report."""

    generated_videos = len(samples)
    generation_total = sum(item["pipeline_generation_seconds"] for item in samples)
    export_total = sum(item["video_export_seconds"] for item in samples)
    return {
        "schema_version": 1,
        "config": str(args.config),
        "model_id": str(model_id),
        "prompt_shard_id": int(args.prompt_shard_id),
        "num_prompt_shards": int(args.num_prompt_shards),
        "generated_videos": generated_videos,
        "skipped_existing_videos": int(skipped),
        "pipeline_generation_seconds_total": generation_total,
        "pipeline_generation_seconds_mean": (
            generation_total / generated_videos if generated_videos else None
        ),
        "pipeline_generation_videos_per_second": (
            generated_videos / generation_total
            if generated_videos and generation_total > 0
            else None
        ),
        "video_export_seconds_total": export_total,
        "video_export_seconds_mean": (
            export_total / generated_videos if generated_videos else None
        ),
        "model_load_seconds": float(model_load_seconds),
        "work_wall_seconds": float(work_wall_seconds),
        "process_wall_seconds": float(process_wall_seconds),
        "process_started_unix": float(process_started_unix),
        "process_finished_unix": float(time.time()),
        "work_started_unix": work_started_unix,
        "work_finished_unix": work_finished_unix,
        "samples": samples,
    }


def print_timing_summary(report, timing_output=None):
    """Print the primary per-video metric and optional report location."""

    count = report["generated_videos"]
    if count:
        throughput = report["pipeline_generation_videos_per_second"]
        throughput_text = (
            f"{throughput:.6f} videos/s" if throughput is not None else "n/a"
        )
        print(
            "Timing summary: "
            f"generated={count}, "
            "average pipeline generation="
            f"{report['pipeline_generation_seconds_mean']:.3f} s/video "
            f"({throughput_text}), "
            f"pipeline total={report['pipeline_generation_seconds_total']:.3f} s, "
            f"export total={report['video_export_seconds_total']:.3f} s, "
            f"model load={report['model_load_seconds']:.3f} s, "
            f"worker loop wall={report['work_wall_seconds']:.3f} s.",
            flush=True,
        )
    else:
        print("Timing summary: no new videos were generated.", flush=True)
    if timing_output is not None:
        print(f"Timing report: {timing_output}", flush=True)


def main():
    process_started_unix = time.time()
    process_started_clock = time.perf_counter()
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("HunyuanVideo generation requires a CUDA GPU.")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if args.num_prompt_shards <= 0:
        raise ValueError("--num_prompt_shards must be positive.")
    if not 0 <= args.prompt_shard_id < args.num_prompt_shards:
        raise ValueError("--prompt_shard_id must be in [0, --num_prompt_shards).")

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

    indexed_prompts = list(enumerate(prompts))
    if args.vbench_filenames:
        unique_prompts = []
        seen_prompts = set()
        for prompt_index, prompt in indexed_prompts:
            if prompt in seen_prompts:
                continue
            if "/" in prompt or os.sep in prompt:
                raise ValueError(
                    f"VBench prompt cannot contain a path separator: {prompt!r}"
                )
            seen_prompts.add(prompt)
            unique_prompts.append((prompt_index, prompt))
        indexed_prompts = unique_prompts

    all_prompt_work = []
    skipped = 0
    for prompt_index, prompt in indexed_prompts:
        missing_samples = []
        for sample_index in range(args.num_samples):
            path = output_path(
                destination,
                prompt_index,
                prompt,
                sample_index,
                args.vbench_filenames,
            )
            if (
                args.skip_existing
                and path.is_file()
                and path.stat().st_size > 0
            ):
                skipped += 1
                continue
            missing_samples.append(sample_index)
        if missing_samples:
            all_prompt_work.append((prompt_index, prompt, missing_samples))

    sharded_prompt_work = [[] for _ in range(args.num_prompt_shards)]
    shard_work_units = [0] * args.num_prompt_shards
    for prompt_item in all_prompt_work:
        target_shard = min(
            range(args.num_prompt_shards),
            key=lambda shard_id: (shard_work_units[shard_id], shard_id),
        )
        sharded_prompt_work[target_shard].append(prompt_item)
        shard_work_units[target_shard] += len(prompt_item[2])

    work = []
    for prompt_index, prompt, sample_indices in sharded_prompt_work[
        args.prompt_shard_id
    ]:
        for sample_index in sample_indices:
            work.append(
                (
                    prompt_index,
                    prompt,
                    sample_index,
                    output_path(
                        destination,
                        prompt_index,
                        prompt,
                        sample_index,
                        args.vbench_filenames,
                    ),
                )
            )
    print(
        f"Remaining-work shard {args.prompt_shard_id + 1}/"
        f"{args.num_prompt_shards}: {len(work)} samples assigned; "
        f"global scan found {len(all_prompt_work)} prompts with pending "
        f"samples and {skipped} existing videos."
    )
    if not work:
        print(
            f"Nothing to generate for prompt shard "
            f"{args.prompt_shard_id + 1}."
        )
        report = build_timing_report(
            args=args,
            model_id=model_id,
            skipped=skipped,
            process_started_unix=process_started_unix,
            process_wall_seconds=time.perf_counter() - process_started_clock,
            model_load_seconds=0.0,
            work_started_unix=None,
            work_finished_unix=None,
            work_wall_seconds=0.0,
            samples=[],
        )
        if args.timing_output is not None:
            write_json_atomic(report, args.timing_output)
        print_timing_summary(report, args.timing_output)
        return

    model_load_started = time.perf_counter()
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
    model_load_seconds = time.perf_counter() - model_load_started
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
    timing_samples = []
    work_started_unix = time.time()
    work_started_clock = time.perf_counter()
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
        generation_started = time.perf_counter()
        frames = pipe(**call_kwargs).frames[0]
        generation_seconds = time.perf_counter() - generation_started
        export_started = time.perf_counter()
        export_video_atomic(export_to_video, frames, path, fps)
        export_seconds = time.perf_counter() - export_started
        timing_samples.append(
            {
                "prompt_index": int(prompt_index),
                "sample_index": int(sample_index),
                "seed": int(sample_seed),
                "pipeline_generation_seconds": generation_seconds,
                "video_export_seconds": export_seconds,
            }
        )
        print(
            f"Timing video {work_index}/{len(work)}: "
            f"pipeline={generation_seconds:.3f} s, "
            f"export={export_seconds:.3f} s.",
            flush=True,
        )

    work_finished_unix = time.time()
    work_wall_seconds = time.perf_counter() - work_started_clock
    report = build_timing_report(
        args=args,
        model_id=model_id,
        skipped=skipped,
        process_started_unix=process_started_unix,
        process_wall_seconds=time.perf_counter() - process_started_clock,
        model_load_seconds=model_load_seconds,
        work_started_unix=work_started_unix,
        work_finished_unix=work_finished_unix,
        work_wall_seconds=work_wall_seconds,
        samples=timing_samples,
    )
    if args.timing_output is not None:
        write_json_atomic(report, args.timing_output)
    print_timing_summary(report, args.timing_output)


if __name__ == "__main__":
    main()
