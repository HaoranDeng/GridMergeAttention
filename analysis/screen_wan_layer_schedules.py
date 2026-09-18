"""Generate matched videos for multiple Wan Dense--GMA layer schedules.

The 14B pipeline is loaded once. Every schedule then reuses the same model,
prompt, seed, and initial/random denoising stream, which makes broad schedule
search substantially cheaper than launching one process per schedule.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from omegaconf import OmegaConf

from attention.runtime_timing import ATTENTION_RUNTIME_RECORDER
from generate import (
    build_pipeline,
    load_generator_checkpoint,
    load_prompts,
    save_video_atomic,
)
from utils.misc import set_seed


SCHEDULE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_schedule(value: str) -> tuple[str, list[int]]:
    try:
        name, raw_layers = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "schedule must use NAME=LAYER0,LAYER1,... syntax"
        ) from error
    if not SCHEDULE_NAME.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "schedule name may contain only letters, digits, '.', '_', and '-'"
        )
    tokens = [token.strip() for token in raw_layers.split(",")]
    if not tokens or any(not token for token in tokens):
        raise argparse.ArgumentTypeError(
            "schedule must contain at least one layer and no empty entries"
        )
    try:
        layers = [int(token) for token in tokens]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "schedule layers must be zero-based integers"
        ) from error
    return name, layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--prompt_path", default="prompts/MovieGenVideoBench_extended.txt"
    )
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--schedule", action="append", type=parse_schedule, required=True)
    parser.add_argument(
        "--dense_name",
        help="Optionally append a dense SDPA reference under this output name.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--runtime_json", required=True)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--video_preset", default="ultrafast")
    parser.add_argument("--video_crf", type=int, default=23)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def write_runtime(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_runtime_records(path: Path, *, resume: bool) -> list[dict]:
    if not resume or not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ValueError(f"Invalid runtime records in {path}")
    return records


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required.")
    schedules: list[tuple[str, list[int] | None]] = list(args.schedule)
    if args.dense_name:
        if not SCHEDULE_NAME.fullmatch(args.dense_name):
            raise ValueError(
                "dense name may contain only letters, digits, '.', '_', and '-'"
            )
        schedules.append((args.dense_name, None))
    names = [name for name, _ in schedules]
    if len(names) != len(set(names)):
        raise ValueError("schedule names must be unique")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config = OmegaConf.load(args.config)
    pipeline = build_pipeline(config, device=device)
    load_generator_checkpoint(pipeline, args.checkpoint, args.use_ema)
    pipeline = pipeline.to(dtype=dtype)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval()

    prompts = load_prompts(args.prompt_path, args.prompt_index + 1)
    prompt = prompts[args.prompt_index]
    output_root = Path(args.output_dir)
    runtime_path = Path(args.runtime_json)
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    base_args = dict(config.token_merging_args)
    records = load_runtime_records(runtime_path, resume=args.skip_existing)

    for schedule_index, (name, layers) in enumerate(schedules):
        schedule_dir = output_root / name
        schedule_dir.mkdir(parents=True, exist_ok=True)
        output_path = schedule_dir / f"{args.prompt_index:03d}-00.mp4"
        if args.skip_existing and output_path.is_file() and output_path.stat().st_size:
            print(f"Skipping existing {output_path}", flush=True)
            continue

        token_args = dict(base_args)
        token_args["enable"] = layers is not None
        if layers is None:
            token_args.pop("enabled_layers", None)
        else:
            token_args["enabled_layers"] = layers
        pipeline.generator.model.token_merging_args = token_args

        set_seed(args.seed + args.prompt_index)
        noise = torch.randn(
            list(config.image_or_video_shape), device=device, dtype=dtype
        )
        ATTENTION_RUNTIME_RECORDER.begin_video()
        inference_start = time.perf_counter()
        video = pipeline.inference(noise=noise, text_prompts=[prompt])
        timings = ATTENTION_RUNTIME_RECORDER.finish_video()
        inference_seconds = time.perf_counter() - inference_start
        video = (
            video.permute(0, 1, 3, 4, 2)
            .mul_(255.0)
            .clamp_(0, 255)
            .to(torch.uint8)
            .cpu()
        )
        export_start = time.perf_counter()
        save_video_atomic(
            output_path,
            video[0],
            fps=args.fps,
            preset=args.video_preset,
            crf=args.video_crf,
            quality=None,
        )
        record = {
            "schedule_index": schedule_index,
            "name": name,
            "enabled_layers": layers,
            "output": str(output_path),
            "prompt_index": args.prompt_index,
            "attention_calls": timings["attention_calls"],
            "mean_attention_ms": (
                timings["attention_total_ms"] / timings["attention_calls"]
            ),
            "full_attention_calls": timings["full_attention_calls"],
            "mean_full_attention_ms": (
                timings["full_attention_total_ms"]
                / timings["full_attention_calls"]
            ),
            "inference_seconds": inference_seconds,
            "export_seconds": time.perf_counter() - export_start,
        }
        records.append(record)
        payload = {
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "prompt_path": str(Path(args.prompt_path).resolve()),
            "prompt_index": args.prompt_index,
            "seed": args.seed,
            "records": records,
        }
        write_runtime(runtime_path, payload)
        print(json.dumps(record), flush=True)
        del noise, video


if __name__ == "__main__":
    main()
