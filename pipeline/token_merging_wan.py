from typing import List

import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class TokenMergingWanPipeline(torch.nn.Module):
    """Few-step Wan2.1 text-to-video pipeline with optional token-merging attention."""

    def __init__(self, args, device):
        super().__init__()
        model_kwargs = dict(getattr(args, "model_kwargs", {}))
        image_or_video_shape = getattr(args, "image_or_video_shape", [1, 21, 16, 60, 104])

        self.generator = WanDiffusionWrapper(
            **model_kwargs,
            image_or_video_shape=image_or_video_shape,
        )
        self.generator.model.token_merging_args = dict(getattr(args, "token_merging_args", {}))
        self.text_encoder = WanTextEncoder(
            model_name=model_kwargs.get("model_name", "Wan2.1-T2V-1.3B"),
            model_root=model_kwargs.get("model_root", None),
        )
        self.vae = WanVAEWrapper(
            model_name=model_kwargs.get("model_name", "Wan2.1-T2V-1.3B"),
            model_root=model_kwargs.get("model_root", None),
        )

        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list,
            dtype=torch.long,
            device="cpu",
        )
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        if getattr(args, "warp_denoising_step", False):
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    @torch.inference_mode()
    def inference(self, noise: torch.Tensor, text_prompts: List[str], return_latents=False):
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        noisy_image_or_video = noise
        pred_image_or_video = None

        timesteps = self.denoising_step_list.to(device=noise.device)
        num_denoise_steps = len(timesteps)
        timestep_shape = noise.shape[:2]
        for index, current_timestep in enumerate(timesteps):
            _, pred_image_or_video = self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=current_timestep.expand(timestep_shape),
            )
            if index + 1 < num_denoise_steps:
                next_timestep = timesteps[index + 1].expand(timestep_shape)
                noisy_image_or_video = self.scheduler.add_noise(
                    pred_image_or_video.flatten(0, 1),
                    torch.randn_like(pred_image_or_video.flatten(0, 1)),
                    next_timestep.flatten(0, 1),
                ).unflatten(0, noise.shape[:2])

        if pred_image_or_video is None:
            raise ValueError("denoising_step_list must contain at least one step.")

        video = self.vae.decoder(pred_image_or_video)
        video.mul_(0.5).add_(0.5).clamp_(0, 1)
        if return_latents:
            return video, pred_image_or_video
        return video
