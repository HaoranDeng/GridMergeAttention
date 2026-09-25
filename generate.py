import argparse
import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from pipeline import CausalTokenMergingWanPipeline, TokenMergingWanPipeline
from attention.runtime_timing import ATTENTION_RUNTIME_RECORDER
from utils.misc import save_video, set_seed
from utils.prompt_contract import (
    build_prompt_manifest,
    ensure_output_prompt_manifest,
    load_nonempty_prompts,
    prompt_sha256,
    validate_canonical_prompt_contract,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos with Wan2.1 + Grid Compressed Attention (GCA).")
    parser.add_argument("--config", default="configs/token_merging_wan_720p_fisher_c4_exact_capacity_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--seed_query_proj_checkpoint",
        default=None,
        help="Optional checkpoint of layer-local full seed_query_proj modules.",
    )
    parser.add_argument(
        "--gma_output_calibration_checkpoint",
        default=None,
        help=(
            "Optional per-layer, per-head calibration transforms folded into "
            "the existing self-attention output projections."
        ),
    )
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument(
        "--allow_noncanonical_prompts",
        action="store_true",
        help=(
            "Explicitly allow a noncanonical prompt suite. Comparison runs must "
            "leave this disabled so video 000 is the Tokyo-woman prompt."
        ),
    )
    parser.add_argument("--output_dir", default="videos/hard_c4_wan")
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument("--prompt_start", type=int, default=0,
                        help="First canonical prompt index to generate; num_videos remains the exclusive upper bound.")
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
        "--runtime_json",
        default=None,
        help="Incrementally write per-video attention and inference timings.",
    )
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
    parser.add_argument("--token_merging_head_chunk", type=int, default=None)
    parser.add_argument("--token_merging_num_merge_tokens", type=int, default=None)
    parser.add_argument(
        "--token_merging_enabled_layers",
        default=None,
        help=(
            "Comma-separated zero-based transformer layers on which token "
            "merging is enabled. Overrides token_merging_args.enabled_layers."
        ),
    )
    parser.add_argument("--token_merging_full_attention_anchor_time_count", type=int)
    parser.add_argument("--token_merging_full_attention_anchor_row_count", type=int)
    parser.add_argument("--token_merging_full_attention_anchor_width_count", type=int)
    parser.add_argument("--token_merging_full_attention_anchor_scale", type=float)
    parser.add_argument(
        "--token_merging_exact_intra_frame_attention",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--token_merging_exact_neighbor_window_columns", type=int
    )
    return parser.parse_args()


def parse_enabled_layers(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",")]
    if any(not token for token in tokens):
        raise ValueError(
            "--token_merging_enabled_layers must be a comma-separated list "
            "of integers without empty entries."
        )
    return [int(token) for token in tokens]


def load_prompts(path: str, limit: int) -> list[str]:
    prompts = load_nonempty_prompts(path)
    return prompts if limit < 0 else prompts[:limit]


def build_pipeline(config, device):
    """Route causal configs to Self-Forcing and other configs to ordinary Wan."""
    if bool(config.get("causal", False)):
        return CausalTokenMergingWanPipeline(config, device=device)
    return TokenMergingWanPipeline(config, device=device)


def load_generator_checkpoint(pipeline: torch.nn.Module, checkpoint_path: str, use_ema: bool):
    state = torch.load(checkpoint_path, map_location="cpu")
    key = "raw_state_dict"
    if isinstance(state, dict) and ("generator" in state or "generator_ema" in state):
        key = "generator_ema" if use_ema and "generator_ema" in state else "generator"
        state = state[key]
    if next(iter(state)).startswith("model."):
        pipeline.generator.load_state_dict(state)
    else:
        pipeline.generator.model.load_state_dict(state)
    return key


def load_seed_query_projectors(model: torch.nn.Module, checkpoint_path: str):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    layer_states = payload.get("layer_state_dicts")
    if not isinstance(layer_states, dict) or not layer_states:
        raise ValueError("seed_query_proj checkpoint has no layer_state_dicts")
    expected = {str(index) for index in range(len(model.blocks))}
    present = set(layer_states)
    if present != expected:
        missing = sorted(expected - present, key=int)
        extra = sorted(present - expected, key=int)
        raise ValueError(
            f"seed_query_proj checkpoint must contain every layer; "
            f"missing={missing}, extra={extra}"
        )
    for layer_idx, block in enumerate(model.blocks):
        state = layer_states[str(layer_idx)]
        q_proj = block.self_attn.q
        projector = torch.nn.Linear(
            q_proj.in_features,
            q_proj.out_features,
            bias="bias" in state,
            device=q_proj.weight.device,
            dtype=q_proj.weight.dtype,
        )
        projector.load_state_dict(
            {
                name: tensor.to(
                    device=q_proj.weight.device,
                    dtype=q_proj.weight.dtype,
                )
                for name, tensor in state.items()
            },
            strict=True,
        )
        projector.requires_grad_(False)
        block.self_attn.seed_query_proj = projector
    print(
        f"Loaded {len(layer_states)} layer-local seed_query_proj modules "
        f"from {checkpoint_path}.",
        flush=True,
    )


@torch.no_grad()
def load_gma_output_calibrators(model: torch.nn.Module, checkpoint_path: str):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    timestep_transforms = payload.get("timestep_layer_transforms")
    timestep_query_transforms = payload.get(
        "timestep_layer_query_transforms"
    )
    timestep_biases = payload.get("timestep_layer_biases")
    if timestep_query_transforms is not None or timestep_biases is not None:
        if not timestep_transforms or not timestep_query_transforms or not timestep_biases:
            raise ValueError(
                "query-conditioned GMA checkpoint requires GMA transforms, "
                "query transforms, and biases"
            )
        if not (
            set(timestep_transforms)
            == set(timestep_query_transforms)
            == set(timestep_biases)
        ):
            raise ValueError(
                "query-conditioned GMA checkpoint timestep sets do not match"
            )
        expected = {str(index) for index in range(len(model.blocks))}
        for timestep in timestep_transforms:
            if not (
                set(timestep_transforms[timestep])
                == set(timestep_query_transforms[timestep])
                == set(timestep_biases[timestep])
                == expected
            ):
                raise ValueError(
                    f"timestep {timestep} query-conditioned GMA calibration "
                    "must contain every layer"
                )
        for layer_idx, block in enumerate(model.blocks):
            attention = block.self_attn
            heads, dim = attention.num_heads, attention.head_dim
            parameters = {}
            for timestep in timestep_transforms:
                tensors = (
                    timestep_transforms[timestep][str(layer_idx)],
                    timestep_query_transforms[timestep][str(layer_idx)],
                    timestep_biases[timestep][str(layer_idx)],
                )
                expected_shapes = (
                    (heads, dim, dim),
                    (heads, dim, dim),
                    (heads, dim),
                )
                for tensor, shape in zip(tensors, expected_shapes):
                    if tuple(tensor.shape) != shape:
                        raise ValueError(
                            f"timestep {timestep}, layer {layer_idx} query-aware "
                            f"parameter must have shape {shape}, got {tuple(tensor.shape)}"
                        )
                parameters[int(timestep)] = tuple(
                    tensor.to(
                        device=attention.o.weight.device,
                        dtype=attention.o.weight.dtype,
                    )
                    for tensor in tensors
                )
            attention.gma_query_calibration_parameters = parameters
        print(
            f"Loaded {len(timestep_transforms)} timestep-conditioned sets of "
            f"{len(model.blocks)} query-aware GMA calibrators from "
            f"{checkpoint_path}.",
            flush=True,
        )
        return
    if timestep_transforms:
        expected = {str(index) for index in range(len(model.blocks))}
        for timestep, transforms in timestep_transforms.items():
            present = set(transforms)
            if present != expected:
                raise ValueError(
                    f"timestep {timestep} GMA calibration must contain every layer; "
                    f"missing={sorted(expected - present, key=int)}, "
                    f"extra={sorted(present - expected, key=int)}"
                )

        for layer_idx, block in enumerate(model.blocks):
            attention = block.self_attn
            heads, dim = attention.num_heads, attention.head_dim
            weight = attention.o.weight.view(attention.dim, heads, dim)
            calibrated_weights = {}
            for timestep, transforms in timestep_transforms.items():
                transform = transforms[str(layer_idx)]
                if tuple(transform.shape) != (heads, dim, dim):
                    raise ValueError(
                        f"timestep {timestep}, layer {layer_idx} transform must "
                        f"have shape {(heads, dim, dim)}, got {tuple(transform.shape)}"
                    )
                transform = transform.to(
                    device=weight.device,
                    dtype=weight.dtype,
                )
                calibrated_weights[int(timestep)] = torch.einsum(
                    "ohd,hed->ohe", weight, transform
                ).reshape_as(attention.o.weight)
            attention.gma_output_calibration_weights = calibrated_weights
        print(
            f"Loaded {len(timestep_transforms)} timestep-conditioned sets of "
            f"{len(model.blocks)} projection-folded GMA calibrators from "
            f"{checkpoint_path}.",
            flush=True,
        )
        return

    transforms = payload.get("layer_transforms")
    if not isinstance(transforms, dict) or not transforms:
        raise ValueError("GMA output calibration checkpoint has no layer_transforms")
    expected = {str(index) for index in range(len(model.blocks))}
    present = set(transforms)
    if present != expected:
        raise ValueError(
            "GMA output calibration must contain every layer; "
            f"missing={sorted(expected - present, key=int)}, "
            f"extra={sorted(present - expected, key=int)}"
        )

    for layer_idx, block in enumerate(model.blocks):
        attention = block.self_attn
        heads, dim = attention.num_heads, attention.head_dim
        transform = transforms[str(layer_idx)]
        if tuple(transform.shape) != (heads, dim, dim):
            raise ValueError(
                f"layer {layer_idx} transform must have shape "
                f"{(heads, dim, dim)}, got {tuple(transform.shape)}"
            )
        transform = transform.to(
            device=attention.o.weight.device,
            dtype=attention.o.weight.dtype,
        )
        weight = attention.o.weight.view(attention.dim, heads, dim)
        folded = torch.einsum("ohd,hed->ohe", weight, transform)
        attention.o.weight.copy_(folded.reshape_as(attention.o.weight))
    print(
        f"Folded {len(transforms)} layer-local GMA output calibrators from "
        f"{checkpoint_path} into the existing output projections.",
        flush=True,
    )


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


def summarize_runtime(records):
    if not records:
        return {}
    steady = records[1:] if len(records) > 1 else records

    def aggregate(items, prefix=""):
        core_calls = sum(item["attention_calls"] for item in items)
        full_calls = sum(item["full_attention_calls"] for item in items)
        return {
            f"{prefix}videos": len(items),
            f"{prefix}mean_attention_ms": sum(
                item["attention_total_ms"] for item in items
            ) / core_calls,
            f"{prefix}mean_full_attention_ms": sum(
                item["full_attention_total_ms"] for item in items
            ) / full_calls,
            f"{prefix}mean_inference_seconds": sum(
                item["inference_seconds"] for item in items
            ) / len(items),
            f"{prefix}attention_calls_per_video": core_calls / len(items),
        }

    return {**aggregate(records), **aggregate(steady, "steady_state_")}


def write_runtime(path, records, args, prompt_manifest):
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "use_ema_requested": bool(args.use_ema),
        "checkpoint_weights": getattr(args, "loaded_checkpoint_weights", None),
        "prompt_manifest": prompt_manifest,
        "records": records,
        "summary": summarize_runtime(records),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


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
    if args.token_merging_head_chunk is not None:
        config.token_merging_args.head_chunk = args.token_merging_head_chunk
    if args.token_merging_num_merge_tokens is not None:
        config.token_merging_args.num_merge_tokens = args.token_merging_num_merge_tokens
    if args.token_merging_enabled_layers is not None:
        config.token_merging_args.enabled_layers = parse_enabled_layers(
            args.token_merging_enabled_layers
        )
    if args.token_merging_full_attention_anchor_time_count is not None:
        config.token_merging_args.full_attention_anchor_time_count = (
            args.token_merging_full_attention_anchor_time_count
        )
    if args.token_merging_full_attention_anchor_row_count is not None:
        config.token_merging_args.full_attention_anchor_row_count = (
            args.token_merging_full_attention_anchor_row_count
        )
    if args.token_merging_full_attention_anchor_width_count is not None:
        config.token_merging_args.full_attention_anchor_width_count = (
            args.token_merging_full_attention_anchor_width_count
        )
    if args.token_merging_full_attention_anchor_scale is not None:
        config.token_merging_args.full_attention_anchor_scale = (
            args.token_merging_full_attention_anchor_scale
        )
    if args.token_merging_exact_intra_frame_attention is not None:
        config.token_merging_args.exact_intra_frame_attention = (
            args.token_merging_exact_intra_frame_attention
        )
    if args.token_merging_exact_neighbor_window_columns is not None:
        config.token_merging_args.exact_neighbor_window_columns = (
            args.token_merging_exact_neighbor_window_columns
        )

    all_prompts = load_prompts(args.prompt_path, -1)
    if not all_prompts:
        raise ValueError(f"No prompts found in {args.prompt_path}")
    validate_canonical_prompt_contract(
        args.prompt_path,
        all_prompts,
        allow_noncanonical=args.allow_noncanonical_prompts,
    )
    prompts = all_prompts if args.num_videos < 0 else all_prompts[: args.num_videos]
    if not 0 <= args.prompt_start < len(prompts):
        raise ValueError("prompt_start must select a nonempty range within num_videos")
    prompt_manifest = build_prompt_manifest(
        args.prompt_path,
        all_prompts,
        selected_count=args.num_videos,
        seed=args.seed,
        num_samples=args.num_samples,
        vbench_filenames=args.vbench_filenames,
        allow_noncanonical=args.allow_noncanonical_prompts,
    )

    output_dir = Path(args.output_dir)
    manifest_path = ensure_output_prompt_manifest(output_dir, prompt_manifest)
    print(
        "Prompt contract verified: "
        f"index 0 is Tokyo woman; manifest={manifest_path}; "
        f"prompt_sha256={prompt_manifest['prompt_file_sha256']}",
        flush=True,
    )
    runtime_path = Path(args.runtime_json) if args.runtime_json else None
    runtime_records = []
    if runtime_path and runtime_path.is_file():
        runtime_records = json.loads(runtime_path.read_text(encoding="utf-8")).get(
            "records", []
        )

    indexed_prompts = list(enumerate(prompts))[args.prompt_start:]
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
    args.loaded_checkpoint_weights = load_generator_checkpoint(
        pipeline, args.checkpoint, args.use_ema
    )
    print(f"Loaded checkpoint weights: {args.loaded_checkpoint_weights}", flush=True)
    pipeline = pipeline.to(dtype=dtype)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    if args.seed_query_proj_checkpoint:
        load_seed_query_projectors(
            pipeline.generator.model, args.seed_query_proj_checkpoint
        )
    if args.gma_output_calibration_checkpoint:
        load_gma_output_calibrators(
            pipeline.generator.model, args.gma_output_calibration_checkpoint
        )
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
        if runtime_path and len(output_items) != 1:
            raise ValueError("--runtime_json requires one generated video per pipeline call.")
        if runtime_path:
            ATTENTION_RUNTIME_RECORDER.begin_video()
            inference_start = time.perf_counter()
        video = pipeline.inference(noise=noise, text_prompts=text_prompts)
        if runtime_path:
            attention_timings = ATTENTION_RUNTIME_RECORDER.finish_video()
            inference_seconds = time.perf_counter() - inference_start
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
            export_start = time.perf_counter()
            save_video_atomic(
                output_path,
                video[video_offset],
                fps=args.fps,
                preset=args.video_preset,
                crf=args.video_crf,
                quality=args.video_quality,
            )
            if runtime_path:
                runtime_records.append(
                    {
                        "output": output_path.name,
                        "prompt_index": idx,
                        "prompt_sha256": prompt_sha256(prompt),
                        "prompt_text": prompt,
                        "sample_index": sample_idx,
                        "attention_total_ms": attention_timings[
                            "attention_total_ms"
                        ],
                        "attention_calls": attention_timings["attention_calls"],
                        "mean_attention_ms": attention_timings[
                            "attention_total_ms"
                        ]
                        / attention_timings["attention_calls"],
                        "full_attention_total_ms": attention_timings[
                            "full_attention_total_ms"
                        ],
                        "full_attention_calls": attention_timings[
                            "full_attention_calls"
                        ],
                        "mean_full_attention_ms": attention_timings[
                            "full_attention_total_ms"
                        ]
                        / attention_timings["full_attention_calls"],
                        "inference_seconds": inference_seconds,
                        "export_seconds": time.perf_counter() - export_start,
                    }
                )
                write_runtime(runtime_path, runtime_records, args, prompt_manifest)
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
