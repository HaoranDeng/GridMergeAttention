#!/usr/bin/env python
"""Run one side of the dense Self-Forcing reference-parity test.

This helper is intentionally executed in a fresh subprocess. Both repositories
define top-level packages named ``wan``, ``pipeline``, and ``utils``; importing
them together could silently reuse modules from the wrong repository.

No production code is changed. The current implementation disables Token
Merging through its existing config, while MonarchRT disables Monarch through
its existing config. Test-only dependencies provide fixed prompt embeddings,
an identity decoder, and a replayed sequence of re-noising tensors.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
import types


CURRENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MONARCH_ROOT = Path("/xuanwu-tank/east/zrl/MonarchRT")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        required=True,
        choices=("current", "monarchrt"),
    )
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--monarch-root",
        type=Path,
        default=DEFAULT_MONARCH_ROOT,
    )
    return parser.parse_args()


def configure_source_tree(source_root: Path) -> None:
    """Make one repository authoritative for all colliding package names."""

    source_root = source_root.resolve()
    excluded = {CURRENT_ROOT.resolve(), DEFAULT_MONARCH_ROOT.resolve()}
    clean_path = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            clean_path.append(entry)
            continue
        if resolved not in excluded:
            clean_path.append(entry)
    sys.path[:] = [str(source_root), *clean_path]
    os.chdir(source_root)


def load_controlled_attention():
    """Load the current repository's SDPA helper under a collision-free name."""

    core_path = CURRENT_ROOT / "attention" / "core.py"
    spec = importlib.util.spec_from_file_location(
        "_self_forcing_parity_attention_core", core_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load controlled attention from {core_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.flash_attention


def install_reference_namespace_packages(source_root: Path) -> None:
    """Load only MonarchRT packages needed by causal inference.

    MonarchRT's package initializers eagerly import every training and
    inference pipeline plus all Wan configurations. None of those side imports
    participate in this comparison, and on the shared filesystem they can add
    several minutes of cold-page reads. Namespace shells skip only those
    initializers; imports below still execute MonarchRT's real
    ``causal_inference.py``, ``wan_wrapper.py``, ``causal_model.py``, and all
    modules they directly depend on.
    """

    package_paths = {
        "pipeline": source_root / "pipeline",
        "utils": source_root / "utils",
        "wan": source_root / "wan",
        "wan.modules": source_root / "wan" / "modules",
    }
    for package_name, package_path in package_paths.items():
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(package_path)]
        spec = importlib.machinery.ModuleSpec(
            package_name, loader=None, is_package=True
        )
        spec.submodule_search_locations = package.__path__
        package.__spec__ = spec
        sys.modules[package_name] = package


def main() -> None:
    args = parse_args()
    source_root = (
        CURRENT_ROOT if args.implementation == "current" else args.monarch_root
    )
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing source repository: {source_root}")

    # Both reference and current runs should use FlashInfer RMSNorm. Removing
    # this existing current-repo override keeps that numerical primitive equal.
    os.environ.pop("TOKEN_MERGING_FORCE_TORCH_RMSNORM", None)
    configure_source_tree(source_root)

    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("reference parity requires a CUDA GPU")

    # MonarchRT's real inference entry point sets this globally before it
    # constructs the pipeline. Besides avoiding autograd memory, this selects
    # its FlashInfer inference RoPE path rather than a training-only fallback.
    # Apply the same condition to both sides of the comparison.
    torch.set_grad_enabled(False)

    if args.implementation == "current":
        from pipeline.causal_token_merging_wan import (
            CausalTokenMergingWanPipeline as Pipeline,
        )

        pipeline_module = importlib.import_module(
            "pipeline.causal_token_merging_wan"
        )
        model_kwargs = {
            "model_name": "Wan2.1-T2V-1.3B",
            "model_root": str(CURRENT_ROOT / "wan_models"),
            "timestep_shift": 5.0,
            "local_attn_size": -1,
            "sink_size": 0,
        }
    else:
        install_reference_namespace_packages(source_root)
        from pipeline.causal_inference import CausalInferencePipeline as Pipeline

        pipeline_module = importlib.import_module("pipeline.causal_inference")

        # MonarchRT normally uses FlashAttention 2, whereas the current repo
        # uses PyTorch SDPA. For a tight code-path parity assertion, replace
        # only the reference process's attention primitive with the exact SDPA
        # helper used by the current implementation. The causal model and
        # pipeline remain MonarchRT's original source.
        controlled_attention = load_controlled_attention()
        reference_model = importlib.import_module("wan.modules.model")
        reference_causal_model = importlib.import_module(
            "wan.modules.causal_model"
        )
        reference_model.flash_attention = controlled_attention
        reference_causal_model.attention = controlled_attention
        model_kwargs = {
            "model_name": "Wan2.1-T2V-1.3B",
            "timestep_shift": 5.0,
            "local_attn_size": -1,
            "sink_size": 0,
        }

    fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    initial_noise = fixture["initial_noise"]
    prompt_embeds = fixture["prompt_embeds"]
    transition_noises = list(fixture["transition_noises"].unbind(0))
    num_frames = int(fixture["num_frames"])
    block_frames = int(fixture["num_frame_per_block"])

    class FixedTextEncoder(torch.nn.Module):
        """Return the exact same already-tokenized prompt context in both runs."""

        def __init__(self, embeddings, target_device, target_dtype):
            super().__init__()
            self.embeddings = embeddings
            self.target_device = target_device
            self.target_dtype = target_dtype

        def forward(self, text_prompts):
            if len(text_prompts) != self.embeddings.size(0):
                raise ValueError("fixture prompt batch does not match text prompts")
            return {
                "prompt_embeds": self.embeddings.to(
                    device=self.target_device,
                    dtype=self.target_dtype,
                )
            }

    class IdentityVAE(torch.nn.Module):
        """Keep the comparison at latent level and exclude VAE differences."""

        def decoder(self, latent, **kwargs):
            del kwargs
            return latent

    class ReplayRandnLike:
        """Supply identical per-transition noise without changing either pipeline."""

        def __init__(self, values):
            self.values = values
            self.index = 0

        def __call__(self, tensor, *unused_args, **unused_kwargs):
            del unused_args, unused_kwargs
            if self.index >= len(self.values):
                raise AssertionError("pipeline requested too many re-noising tensors")
            value = self.values[self.index]
            self.index += 1
            if tuple(value.shape) != tuple(tensor.shape):
                raise AssertionError(
                    f"re-noising shape mismatch: fixture={tuple(value.shape)}, "
                    f"request={tuple(tensor.shape)}"
                )
            return value.to(device=tensor.device, dtype=tensor.dtype)

    config = OmegaConf.create(
        {
            "causal": True,
            "image_or_video_shape": [1, num_frames, 16, 60, 104],
            "denoising_step_list": [1000, 750, 500, 250],
            "warp_denoising_step": True,
            "num_frame_per_block": block_frames,
            "independent_first_frame": False,
            "context_noise": 0,
            "model_kwargs": model_kwargs,
            # Each repository uses its normal public configuration switch.
            "token_merging_args": {"enable": False},
            "monarch_args": {"enable": False},
        }
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    text_encoder = FixedTextEncoder(prompt_embeds, device, dtype)
    pipeline = Pipeline(
        config,
        device=device,
        text_encoder=text_encoder,
        vae=IdentityVAE(),
    )

    print(f"[parity:{args.implementation}] loading EMA checkpoint strictly", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["generator_ema"]
    incompatible = pipeline.generator.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"strict checkpoint mismatch: {incompatible}")
    del checkpoint, state

    pipeline.generator.to(device=device, dtype=dtype).eval()
    noise = initial_noise.to(device=device, dtype=dtype)
    replay = ReplayRandnLike(transition_noises)
    original_randn_like = pipeline_module.torch.randn_like
    pipeline_module.torch.randn_like = replay
    try:
        print(
            f"[parity:{args.implementation}] running {num_frames} latent frames",
            flush=True,
        )
        _, latents = pipeline.inference(
            noise=noise,
            text_prompts=["fixed parity fixture"],
            return_latents=True,
        )
    finally:
        pipeline_module.torch.randn_like = original_randn_like

    expected_noise_calls = (num_frames // block_frames) * 3
    if replay.index != expected_noise_calls:
        raise AssertionError(
            f"expected {expected_noise_calls} re-noising calls, got {replay.index}"
        )

    kv_cache = getattr(pipeline, "kv_cache", None)
    if kv_cache is None:
        kv_cache = getattr(pipeline, "kv_cache1", None)
    if kv_cache is None:
        raise AssertionError("pipeline did not expose its causal KV cache")

    payload = {
        "implementation": args.implementation,
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "latents": latents.detach().cpu(),
        "denoising_steps": pipeline.denoising_step_list.detach().cpu(),
        "global_end_indices": [
            int(cache["global_end_index"]) for cache in kv_cache
        ],
        "local_end_indices": [
            int(cache["local_end_index"]) for cache in kv_cache
        ],
        "replay_noise_calls": replay.index,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"[parity:{args.implementation}] saved {tuple(latents.shape)} to "
        f"{args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
