"""Self-Forcing inference with causal Wan K/V caches and Token Merging."""

from __future__ import annotations

from typing import List, Optional

import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalTokenMergingWanPipeline(torch.nn.Module):
    """Generate one latent-frame block at a time using cached causal attention.

    Each block is denoised with the configured few-step schedule. All denoising
    evaluations overwrite the same K/V slots; after the final prediction, one
    additional call at ``context_noise`` commits a clean representation before
    generation advances to the next temporal block.
    """

    def __init__(
        self,
        args,
        device,
        generator=None,
        text_encoder=None,
        vae=None,
    ):
        super().__init__()
        self.args = args
        self.device = torch.device(device)

        if not bool(getattr(args, "causal", True)):
            raise ValueError(
                "CausalTokenMergingWanPipeline requires a config with causal: true."
            )

        model_kwargs = dict(getattr(args, "model_kwargs", {}))
        # The pipeline owns the causal choice, so a stale config value cannot
        # accidentally create the ordinary Wan model (or duplicate the kwarg).
        model_kwargs.pop("is_causal", None)
        image_or_video_shape = list(
            getattr(args, "image_or_video_shape", [1, 21, 16, 60, 104])
        )

        self.generator = (
            WanDiffusionWrapper(
                **model_kwargs,
                is_causal=True,
                image_or_video_shape=image_or_video_shape,
            )
            if generator is None
            else generator
        )
        model = self.generator.model

        model_name = model_kwargs.get("model_name", "Wan2.1-T2V-1.3B")
        model_root = model_kwargs.get("model_root")
        self.text_encoder = (
            WanTextEncoder(model_name=model_name, model_root=model_root)
            if text_encoder is None
            else text_encoder
        )
        self.vae = (
            WanVAEWrapper(model_name=model_name, model_root=model_root)
            if vae is None
            else vae
        )

        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            list(args.denoising_step_list), dtype=torch.long, device="cpu"
        )
        if self.denoising_step_list.numel() and self.denoising_step_list[-1] == 0:
            # The clean-context commit below is the explicit timestep-zero call.
            self.denoising_step_list = self.denoising_step_list[:-1]
        if not self.denoising_step_list.numel():
            raise ValueError("denoising_step_list must contain at least one nonzero step.")
        if bool(getattr(args, "warp_denoising_step", False)):
            scheduler_steps = torch.cat(
                (
                    self.scheduler.timesteps.detach().cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )
            )
            indices = 1000 - self.denoising_step_list
            if (indices < 0).any() or (indices >= scheduler_steps.numel()).any():
                raise ValueError("denoising steps must lie in the inclusive range [0, 1000].")
            self.denoising_step_list = scheduler_steps[indices]

        self.num_frame_per_block = int(getattr(args, "num_frame_per_block", 1))
        if self.num_frame_per_block <= 0:
            raise ValueError("num_frame_per_block must be positive.")
        self.independent_first_frame = bool(
            getattr(args, "independent_first_frame", False)
        )
        if self.independent_first_frame:
            raise NotImplementedError(
                "This T2V generation entry point currently supports uniform "
                "Self-Forcing blocks, not the I2V [1, N, N, ...] layout."
            )
        self.context_noise = int(getattr(args, "context_noise", 0))

        model.num_frame_per_block = self.num_frame_per_block
        model.token_merging_args = dict(getattr(args, "token_merging_args", {}))
        token_merging_args = model.token_merging_args
        if token_merging_args.get("enable", False):
            if token_merging_args.get("layout", "hard_route") != "hard_route":
                raise ValueError(
                    "causal cached Token Merging currently supports layout='hard_route'."
                )
            if int(token_merging_args.get("num_merge_tokens", 1)) != 1:
                raise ValueError(
                    "causal cached Token Merging currently supports num_merge_tokens=1."
                )

        self.num_transformer_blocks = int(model.num_layers)
        self.local_attn_size = int(getattr(model, "local_attn_size", -1))

        video_shape = [
            image_or_video_shape[1],
            image_or_video_shape[-2],
            image_or_video_shape[-1],
        ]
        for axis, patch in enumerate(model.patch_size):
            if video_shape[axis] % patch:
                raise ValueError(
                    "configured latent video dimensions must be divisible by "
                    "the Wan patch size."
                )
            video_shape[axis] //= patch
        self.num_latent_frames, frame_height, frame_width = video_shape
        self.frame_seq_length = frame_height * frame_width
        self.total_seq_len = self.num_latent_frames * self.frame_seq_length

        if self.num_latent_frames % self.num_frame_per_block:
            raise ValueError(
                "latent frame count must be divisible by num_frame_per_block."
            )
        if token_merging_args.get("enable", False):
            reductions = (
                int(token_merging_args.get("f_tied", 1)),
                int(token_merging_args.get("h_reduce", 1)),
                int(token_merging_args.get("w_reduce", 1)),
            )
            dimensions = (self.num_frame_per_block, frame_height, frame_width)
            if any(size % reduction for size, reduction in zip(dimensions, reductions)):
                raise ValueError(
                    "Self-Forcing block/spatial dimensions must be divisible by "
                    "the Token Merging reductions."
                )

        # Caches are allocated lazily using the real inference batch, dtype,
        # and device. They are not parameters and should not be checkpointed.
        self.kv_cache = None
        self.crossattn_cache = None

    def _kv_cache_capacity(self):
        if self.local_attn_size == -1:
            return self.total_seq_len
        return self.local_attn_size * self.frame_seq_length

    def _caches_match(self, batch_size, dtype, device):
        if self.kv_cache is None or self.crossattn_cache is None:
            return False
        if (
            len(self.kv_cache) != self.num_transformer_blocks
            or len(self.crossattn_cache) != self.num_transformer_blocks
        ):
            return False

        model = self.generator.model
        expected_kv = (
            batch_size,
            self._kv_cache_capacity(),
            int(model.num_heads),
            int(model.dim) // int(model.num_heads),
        )
        expected_cross = (
            batch_size,
            int(model.text_len),
            int(model.num_heads),
            int(model.dim) // int(model.num_heads),
        )
        for cache in self.kv_cache:
            if (
                tuple(cache["k"].shape) != expected_kv
                or tuple(cache["v"].shape) != expected_kv
                or cache["k"].dtype != dtype
                or cache["v"].dtype != dtype
                or cache["k"].device != device
                or cache["v"].device != device
            ):
                return False
        for cache in self.crossattn_cache:
            if (
                tuple(cache["k"].shape) != expected_cross
                or tuple(cache["v"].shape) != expected_cross
                or cache["k"].dtype != dtype
                or cache["v"].dtype != dtype
                or cache["k"].device != device
                or cache["v"].device != device
            ):
                return False
        return True

    def _prepare_caches(self, batch_size, dtype, device):
        device = torch.device(device)
        model = self.generator.model
        heads = int(model.num_heads)
        head_dim = int(model.dim) // heads

        if not self._caches_match(batch_size, dtype, device):
            kv_shape = (
                batch_size,
                self._kv_cache_capacity(),
                heads,
                head_dim,
            )
            cross_shape = (
                batch_size,
                int(model.text_len),
                heads,
                head_dim,
            )
            self.kv_cache = [
                {
                    "k": torch.zeros(kv_shape, dtype=dtype, device=device),
                    "v": torch.zeros(kv_shape, dtype=dtype, device=device),
                    "global_end_index": 0,
                    "local_end_index": 0,
                }
                for _ in range(self.num_transformer_blocks)
            ]
            self.crossattn_cache = [
                {
                    "k": torch.zeros(cross_shape, dtype=dtype, device=device),
                    "v": torch.zeros(cross_shape, dtype=dtype, device=device),
                    "is_init": False,
                }
                for _ in range(self.num_transformer_blocks)
            ]
            return

        # A new video must never see temporal or text context from the previous
        # sample. Cache contents need not be zeroed because visibility indices
        # and is_init control every read.
        for cache in self.kv_cache:
            cache["global_end_index"] = 0
            cache["local_end_index"] = 0
        for cache in self.crossattn_cache:
            cache["is_init"] = False

    @torch.inference_mode()
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ):
        if initial_latent is not None:
            raise NotImplementedError(
                "This generation script implements text-to-video Self-Forcing; "
                "initial-latent video extension is not wired here."
            )
        if noise.ndim != 5:
            raise ValueError("noise must have shape [batch, frames, channels, height, width].")
        batch_size, num_frames, num_channels, height, width = noise.shape
        if len(text_prompts) != batch_size:
            raise ValueError("text prompt count must match the inference batch size.")
        if num_frames != self.num_latent_frames:
            raise ValueError(
                f"noise contains {num_frames} latent frames; config expects "
                f"{self.num_latent_frames}."
            )
        if num_frames % self.num_frame_per_block:
            raise ValueError("noise frame count must contain complete causal blocks.")

        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        self._prepare_caches(batch_size, noise.dtype, noise.device)
        output = torch.zeros(
            (batch_size, num_frames, num_channels, height, width),
            dtype=noise.dtype,
            device=noise.device,
        )

        current_start_frame = 0
        while current_start_frame < num_frames:
            current_num_frames = min(
                self.num_frame_per_block, num_frames - current_start_frame
            )
            noisy_input = noise[
                :, current_start_frame : current_start_frame + current_num_frames
            ]
            denoised_pred = None
            timestep = None

            for step_index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    (batch_size, current_num_frames),
                    device=noise.device,
                    dtype=torch.long,
                ) * current_timestep.to(device=noise.device)
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=current_start_frame,
                )

                if step_index + 1 < len(self.denoising_step_list):
                    next_timestep = self.denoising_step_list[step_index + 1]
                    next_timestep = torch.ones(
                        batch_size * current_num_frames,
                        device=noise.device,
                        dtype=torch.long,
                    ) * next_timestep.to(device=noise.device)
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep,
                    ).unflatten(0, denoised_pred.shape[:2])

            if denoised_pred is None or timestep is None:
                raise RuntimeError("the denoising schedule produced no prediction.")
            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred

            # Commit clean context into exactly the same cache slots. The next
            # block will then see this block as part of its causal K/V prefix.
            context_timestep = torch.ones_like(timestep) * self.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start_frame=current_start_frame,
            )
            current_start_frame += current_num_frames

        video = self.vae.decoder(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if return_latents:
            return video, output
        return video


__all__ = ["CausalTokenMergingWanPipeline"]
