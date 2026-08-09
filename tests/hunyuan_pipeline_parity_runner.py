#!/usr/bin/env python
"""Run one side of the real-checkpoint Hunyuan pipeline parity comparison.

The reference side uses Diffusers' untouched ``HunyuanVideoAttnProcessor2_0``.
The current side installs this repository's future Hunyuan Token-Merging
processor, but this short-lived test process replaces only the low-level
Token-Merging function with exact dense SDPA. Differences therefore identify
an integration error around attention--QKV projection, QK norm, video-only
RoPE, text masking, stream splitting, output projection, pipeline arguments,
or scheduler execution--rather than the intended Token-Merging approximation.

Both sides load the same real Hunyuan transformer checkpoint on CUDA, receive
byte-identical prompt embeddings and initial latents, execute the real
Diffusers denoising loop, and save latent output. Text encoders and VAE decoding
are excluded for the same reason they are excluded from the Self-Forcing
reference test: they are outside the generator parity question and would add
large unrelated dependencies and numerical noise.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        required=True,
        choices=("reference", "current"),
    )
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def configure_import_path() -> None:
    """Import current-repository modules alongside installed Diffusers."""

    repo_root = str(REPO_ROOT)
    sys.path[:] = [entry for entry in sys.path if entry != repo_root]
    sys.path.insert(0, repo_root)
    os.chdir(REPO_ROOT)


def load_real_pipeline(torch, args):
    """Load the real Hunyuan transformer while omitting unused components."""

    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        HunyuanVideoPipeline,
        HunyuanVideoTransformer3DModel,
    )

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }
    if args.transformer_subfolder:
        load_kwargs["subfolder"] = args.transformer_subfolder
    if args.revision:
        load_kwargs["revision"] = args.revision

    print(
        f"[hunyuan-parity:{args.implementation}] loading real transformer "
        f"from {args.model!r}",
        flush=True,
    )
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model,
        **load_kwargs,
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

    # The real Diffusers Hunyuan pipeline is still responsible for timestep
    # construction, latent model input, guidance, transformer calls, and
    # scheduler updates. Only components irrelevant to latent parity are absent.
    pipeline = HunyuanVideoPipeline(
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
        vae=None,
        scheduler=scheduler,
        text_encoder_2=None,
        tokenizer_2=None,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to("cuda")
    return pipeline


def checkpoint_fingerprint(torch, model):
    """Sample real checkpoint tensors without scanning/copying all 13B weights."""

    state = model.state_dict()
    keys = sorted(state)
    selected_indices = sorted({0, len(keys) // 2, len(keys) - 1})
    samples = {}
    for key_index in selected_indices:
        key = keys[key_index]
        value = state[key].detach().reshape(-1)
        element_indices = sorted(
            {0, value.numel() // 3, (2 * value.numel()) // 3, value.numel() - 1}
        )
        samples[key] = value[element_indices].float().cpu()
    return {
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "samples": samples,
    }


def install_current_dense_probe(torch, pipeline, fixture, attention_calls):
    """Install production Hunyuan integration with a test-only dense core.

    Production code must expose two ordinary implementation contracts:

    * ``pipeline.token_merging_hunyuan.install_hunyuan_token_merging_attention``
    * ``attention.hunyuan_token_merging.hunyuan_token_merging_attn``

    No production skip path is introduced. The second symbol is monkeypatched
    only in this subprocess, matching how the Self-Forcing differential test
    controls its lowest-level attention primitive.
    """

    import torch.nn.functional as functional

    try:
        integration = importlib.import_module("pipeline.token_merging_hunyuan")
        attention_core = importlib.import_module("attention.hunyuan_token_merging")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Hunyuan parity implementation is not present yet. Expected "
            "pipeline/token_merging_hunyuan.py and "
            "attention/hunyuan_token_merging.py."
        ) from error

    try:
        install = integration.install_hunyuan_token_merging_attention
    except AttributeError as error:
        raise RuntimeError(
            "pipeline.token_merging_hunyuan must define "
            "install_hunyuan_token_merging_attention(pipe, token_merging_args)."
        ) from error

    def dense_joint_attention(
        video_q,
        video_k,
        video_v,
        text_q,
        text_k,
        text_v,
        attention_mask=None,
        sm_scale=None,
        **unused_options,
    ):
        """Evaluate exact joint attention and record the stream boundary."""

        del unused_options
        tensors = (video_q, video_k, video_v, text_q, text_k, text_v)
        if any(value is None or value.ndim != 4 for value in tensors):
            raise AssertionError(
                "Hunyuan Token Merging must pass six [B, S, H, D] tensors."
            )
        if video_q.shape != video_k.shape or video_q.shape != video_v.shape:
            raise AssertionError("video Q/K/V shapes must match in dense parity")
        if text_q.shape != text_k.shape or text_q.shape != text_v.shape:
            raise AssertionError("text Q/K/V shapes must match in dense parity")
        if video_q.shape[0] != text_q.shape[0] or video_q.shape[2:] != text_q.shape[2:]:
            raise AssertionError("video and text batch/head dimensions must match")

        query = torch.cat((video_q, text_q), dim=1).transpose(1, 2)
        key = torch.cat((video_k, text_k), dim=1).transpose(1, 2)
        value = torch.cat((video_v, text_v), dim=1).transpose(1, 2)

        mask = attention_mask
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            mask = mask.to(device=query.device, dtype=torch.bool)
            valid_key_count = mask.reshape(mask.shape[0], -1).sum(dim=-1).tolist()
        else:
            valid_key_count = [key.shape[2]] * key.shape[0]

        attention_calls.append(
            {
                "video_tokens": int(video_q.shape[1]),
                "text_tokens": int(text_q.shape[1]),
                "heads": int(video_q.shape[2]),
                "head_dim": int(video_q.shape[3]),
                "valid_key_count": valid_key_count,
                "mask_shape": None if mask is None else tuple(mask.shape),
            }
        )

        output = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=sm_scale,
        ).transpose(1, 2)
        video_length = video_q.shape[1]
        return output[:, :video_length], output[:, video_length:]

    # Prefer module-qualified calls in production. Also patch a directly
    # imported alias if the processor module exposes one.
    attention_core.hunyuan_token_merging_attn = dense_joint_attention
    try:
        processor_module = importlib.import_module(
            "attention.hunyuan_attention_processor"
        )
    except ModuleNotFoundError:
        processor_module = None
    if processor_module is not None and hasattr(
        processor_module, "hunyuan_token_merging_attn"
    ):
        processor_module.hunyuan_token_merging_attn = dense_joint_attention

    install(
        pipeline,
        token_merging_args={
            "enable": True,
            "f_tied": 1,
            "h_reduce": 1,
            "w_reduce": 1,
            "layout": "hard_route",
            "num_merge_tokens": 1,
            "height": int(fixture["height"]),
            "width": int(fixture["width"]),
            "num_frames": int(fixture["num_frames"]),
        },
    )


def processor_inventory(pipeline):
    """Record processors installed in both Hunyuan block families."""

    inventory = []
    block_groups = (
        ("dual", pipeline.transformer.transformer_blocks),
        ("single", pipeline.transformer.single_transformer_blocks),
    )
    for block_kind, blocks in block_groups:
        for layer_index, block in enumerate(blocks):
            processor = block.attn.processor
            inventory.append(
                {
                    "block_kind": block_kind,
                    "layer_index": layer_index,
                    "module": type(processor).__module__,
                    "class": type(processor).__name__,
                }
            )
    return inventory


def selected_model_config(model):
    """Save the dimensions required to interpret the attention trace."""

    config = model.config
    return {
        "in_channels": int(config.in_channels),
        "out_channels": int(config.out_channels),
        "num_attention_heads": int(config.num_attention_heads),
        "attention_head_dim": int(config.attention_head_dim),
        "num_layers": int(config.num_layers),
        "num_single_layers": int(config.num_single_layers),
        "patch_size": int(config.patch_size),
        "patch_size_t": int(config.patch_size_t),
        "text_embed_dim": int(config.text_embed_dim),
        "pooled_projection_dim": int(config.pooled_projection_dim),
    }


def main() -> None:
    args = parse_args()
    configure_import_path()

    import torch
    import diffusers

    if not torch.cuda.is_available():
        raise RuntimeError("real Hunyuan pipeline parity requires a CUDA GPU")
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats()

    fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    pipeline = load_real_pipeline(torch, args)
    model_config = selected_model_config(pipeline.transformer)

    expected_config = {
        "in_channels": fixture["latents"].shape[1],
        "text_embed_dim": fixture["prompt_embeds"].shape[-1],
        "pooled_projection_dim": fixture["pooled_prompt_embeds"].shape[-1],
    }
    for key, expected in expected_config.items():
        if model_config[key] != expected:
            raise AssertionError(
                f"fixture/model {key} mismatch: fixture={expected}, "
                f"model={model_config[key]}"
            )

    attention_calls = []
    if args.implementation == "current":
        install_current_dense_probe(torch, pipeline, fixture, attention_calls)

    processors = processor_inventory(pipeline)
    model_fingerprint = checkpoint_fingerprint(torch, pipeline.transformer)
    step_latents = []

    def record_step_latents(unused_pipeline, step, timestep, callback_kwargs):
        del unused_pipeline, step, timestep
        step_latents.append(callback_kwargs["latents"].detach().cpu().clone())
        return {"latents": callback_kwargs["latents"]}

    device = torch.device("cuda")
    with torch.inference_mode():
        result = pipeline(
            prompt_embeds=fixture["prompt_embeds"].to(device),
            pooled_prompt_embeds=fixture["pooled_prompt_embeds"].to(device),
            prompt_attention_mask=fixture["prompt_attention_mask"].to(device),
            latents=fixture["latents"].to(device),
            height=int(fixture["height"]),
            width=int(fixture["width"]),
            num_frames=int(fixture["num_frames"]),
            num_inference_steps=int(fixture["num_inference_steps"]),
            guidance_scale=float(fixture["guidance_scale"]),
            true_cfg_scale=1.0,
            output_type="latent",
            return_dict=False,
            callback_on_step_end=record_step_latents,
            callback_on_step_end_tensor_inputs=["latents"],
        )[0]
    torch.cuda.synchronize()

    payload = {
        "implementation": args.implementation,
        "latents": result.detach().cpu(),
        "step_latents": torch.stack(step_latents),
        "timesteps": pipeline.scheduler.timesteps.detach().cpu(),
        "attention_calls": attention_calls,
        "processors": processors,
        "model_config": model_config,
        "model_fingerprint": model_fingerprint,
        "model_source": args.model,
        "revision": args.revision,
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "imported_svg_modules": sorted(
            name for name in sys.modules if name == "svg" or name.startswith("svg.")
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"[hunyuan-parity:{args.implementation}] saved "
        f"{tuple(result.shape)} after {len(step_latents)} steps; "
        f"peak CUDA allocation={payload['peak_memory_bytes'] / 2**30:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
