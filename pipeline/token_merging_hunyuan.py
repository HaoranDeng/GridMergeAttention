"""Diffusers HunyuanVideo pipeline integration for Token-Merging attention."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from attention.hunyuan_attention_processor import (
    HunyuanTokenMergingAttnProcessor2_0,
)


def _as_options(token_merging_args):
    if token_merging_args is None:
        return {}
    if isinstance(token_merging_args, Mapping):
        return dict(token_merging_args)
    try:
        return dict(token_merging_args)
    except (TypeError, ValueError) as error:
        raise TypeError("token_merging_args must be a mapping.") from error


def _positive_int(value, name):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def derive_hunyuan_video_grid(pipe, *, height, width, num_frames):
    """Derive the transformer's post-patch ``(frames, height, width)`` grid."""

    height = _positive_int(height, "height")
    width = _positive_int(width, "width")
    num_frames = _positive_int(num_frames, "num_frames")

    spatial_compression = _positive_int(
        getattr(pipe, "vae_scale_factor_spatial", 8),
        "vae_scale_factor_spatial",
    )
    temporal_compression = _positive_int(
        getattr(pipe, "vae_scale_factor_temporal", 4),
        "vae_scale_factor_temporal",
    )
    config = pipe.transformer.config
    patch_size = _positive_int(config.patch_size, "transformer.patch_size")
    patch_size_t = _positive_int(
        config.patch_size_t, "transformer.patch_size_t"
    )

    spatial_divisor = spatial_compression * patch_size
    if height % spatial_divisor or width % spatial_divisor:
        raise ValueError(
            "Hunyuan height and width must be divisible by VAE spatial "
            f"compression * transformer patch size ({spatial_divisor}); "
            f"got {height}x{width}."
        )

    latent_frames = 1 + (num_frames - 1) // temporal_compression
    if latent_frames % patch_size_t:
        raise ValueError(
            f"latent frame count {latent_frames} must be divisible by "
            f"transformer.patch_size_t={patch_size_t}."
        )

    return (
        latent_frames // patch_size_t,
        height // spatial_divisor,
        width // spatial_divisor,
    )


def install_hunyuan_token_merging_attention(pipe, token_merging_args):
    """Install Token-Merging processors in both main Hunyuan block families.

    The text-refiner attention is deliberately untouched: it is not joint
    video/text attention and has no video grid on which to apply GridMerge.
    """

    options = _as_options(token_merging_args)
    enabled = bool(options.pop("enable", True))
    if not enabled:
        pipe._hunyuan_token_merging_config = {"enable": False}
        return pipe

    missing_geometry = [
        name for name in ("height", "width", "num_frames") if name not in options
    ]
    if missing_geometry:
        raise ValueError(
            "installing Hunyuan Token Merging requires generation geometry: "
            + ", ".join(missing_geometry)
        )
    height = options.pop("height")
    width = options.pop("width")
    num_frames = options.pop("num_frames")
    grid_f, grid_h, grid_w = derive_hunyuan_video_grid(
        pipe,
        height=height,
        width=width,
        num_frames=num_frames,
    )

    processor_options = {
        "f_tied": _positive_int(options.pop("f_tied", 1), "f_tied"),
        "h_reduce": _positive_int(
            options.pop("h_reduce", 1), "h_reduce"
        ),
        "w_reduce": _positive_int(
            options.pop("w_reduce", 1), "w_reduce"
        ),
        "q_init": options.pop("q_init", "ith"),
        "query_outer_chunk": _positive_int(
            options.pop("query_outer_chunk", 1), "query_outer_chunk"
        ),
        "layout": options.pop("layout", "hard_route"),
        "num_merge_tokens": _positive_int(
            options.pop("num_merge_tokens", 1), "num_merge_tokens"
        ),
    }
    if options:
        raise ValueError(
            "unsupported Hunyuan Token-Merging option(s): "
            + ", ".join(sorted(options))
        )
    normalized_layout = {
        "query_column_key_row": "hard_route"
    }.get(processor_options["layout"], processor_options["layout"])
    if normalized_layout != "hard_route" or processor_options["num_merge_tokens"] != 1:
        raise ValueError(
            "Hunyuan currently supports only layout='hard_route' with "
            "num_merge_tokens=1."
        )
    processor_options["layout"] = normalized_layout
    if grid_f % processor_options["f_tied"]:
        raise ValueError(
            f"post-patch frame grid {grid_f} must be divisible by "
            f"f_tied={processor_options['f_tied']}."
        )

    block_groups = (
        ("dual", getattr(pipe.transformer, "transformer_blocks", None)),
        (
            "single",
            getattr(pipe.transformer, "single_transformer_blocks", None),
        ),
    )
    if any(blocks is None for _, blocks in block_groups):
        raise TypeError(
            "the transformer is not a Diffusers HunyuanVideo transformer with "
            "dual- and single-stream block lists."
        )

    installed = []
    for block_kind, blocks in block_groups:
        for layer_index, block in enumerate(blocks):
            if not hasattr(block, "attn") or not hasattr(
                block.attn, "set_processor"
            ):
                raise TypeError(
                    f"{block_kind} block {layer_index} has no replaceable attention."
                )
            processor = HunyuanTokenMergingAttnProcessor2_0(
                h=grid_h,
                w=grid_w,
                layer_index=layer_index,
                block_kind=block_kind,
                **processor_options,
            )
            block.attn.set_processor(processor)
            installed.append(
                {
                    "block_kind": block_kind,
                    "layer_index": layer_index,
                    "processor": type(processor).__name__,
                }
            )

    pipe._hunyuan_token_merging_config = {
        "enable": True,
        "pixel_height": int(height),
        "pixel_width": int(width),
        "num_frames": int(num_frames),
        "grid_f": grid_f,
        "grid_h": grid_h,
        "grid_w": grid_w,
        **processor_options,
        "installed_processors": installed,
    }
    return pipe


def load_hunyuan_token_merging_pipeline(
    model_id,
    *,
    token_merging_args,
    torch_dtype=torch.bfloat16,
    revision="refs/pr/18",
    transformer_subfolder="transformer",
    scheduler_shift=7.0,
    local_files_only=False,
    device="cuda",
    model_cpu_offload=True,
    sequential_cpu_offload=False,
    enable_vae_tiling=True,
):
    """Load Diffusers HunyuanVideo and install the production processors."""

    if model_cpu_offload and sequential_cpu_offload:
        raise ValueError(
            "model_cpu_offload and sequential_cpu_offload are mutually exclusive."
        )
    try:
        from diffusers import (
            FlowMatchEulerDiscreteScheduler,
            HunyuanVideoPipeline,
            HunyuanVideoTransformer3DModel,
        )
    except ImportError as error:
        raise ImportError(
            "Hunyuan generation needs Diffusers 0.34.0 and its compatible "
            "Transformers/Accelerate packages. See requirements-hunyuan.txt."
        ) from error

    component_load_kwargs = {
        "torch_dtype": torch_dtype,
        "local_files_only": bool(local_files_only),
        "low_cpu_mem_usage": True,
    }
    pipeline_load_kwargs = {
        "torch_dtype": torch_dtype,
        "local_files_only": bool(local_files_only),
    }
    if revision:
        component_load_kwargs["revision"] = revision
        pipeline_load_kwargs["revision"] = revision
    if transformer_subfolder:
        component_load_kwargs["subfolder"] = transformer_subfolder

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id, **component_load_kwargs
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=float(scheduler_shift))
    pipe = HunyuanVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        scheduler=scheduler,
        **pipeline_load_kwargs,
    )
    install_hunyuan_token_merging_attention(pipe, token_merging_args)

    if enable_vae_tiling:
        pipe.vae.enable_tiling()
    if sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(device=device)
    elif model_cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    return pipe


__all__ = [
    "derive_hunyuan_video_grid",
    "install_hunyuan_token_merging_attention",
    "load_hunyuan_token_merging_pipeline",
]
