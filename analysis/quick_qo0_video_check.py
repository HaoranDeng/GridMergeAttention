"""Generate ONE video (prompt index 0, seed 0) with the qo=0-selector AxisCurrentC8
and score it against the same dense ground-truth reference and against the
already-generated original per-(qo,ko)-selector video, to get a fast read on
whether the qo=0 approximation causes a visible quality regression.
"""
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('AXIS_C8_HEAD_CHUNK', '4')

ROOT = Path(__file__).resolve().parents[1]
SVG = '/local2/denghaoran/GridMergeAttention_worktrees/svg_official_20260906ch'
MODEL = '/home/denghaoran/wan_models/Wan2.1-T2V-14B-Diffusers'
OUT_DIR = Path('/local2/denghaoran/GridMergeAttention_axis_c8_opt/qo0_check')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    sys.path.insert(0, SVG)
    # svg's compiled _kernels.so (fast fused RoPE) segfaults with the currently
    # installed diffusers>=0.36's cos/sin RoPE representation (a pre-existing
    # environment incompatibility, unrelated to the attention kernel under
    # test). Force the pure-PyTorch RoPE fallback for this quality check by
    # making `import _kernels` fail -- `sys.modules[name] = None` is the
    # documented way to make CPython raise ImportError for a given import.
    sys.modules['_kernels'] = None
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    from analysis.benchmark_wan14b_720p_methods import (AttentionTimer, AttentionErrorStats,
                                                          NEGATIVE_PROMPT, configure_processors)
    from attention.axis_current_c8_cuda import AxisCurrentC8
    from utils.prompt_contract import build_prompt_manifest, validate_canonical_prompt_contract

    prompt_path = ROOT / 'prompts/MovieGenVideoBench_extended.txt'
    prompts = [v.strip() for v in prompt_path.read_text().splitlines() if v.strip()]
    validate_canonical_prompt_contract(prompt_path, prompts)

    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    torch.backends.cuda.preferred_linalg_library(backend='magma')
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder='vae', torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True,
                                              num_train_timesteps=1000, flow_shift=5.0)
    pipe.to('cuda')
    timer, full_timer = AttentionTimer(), AttentionTimer()
    configure_processors(pipe, 'dense', timer, full_timer, AttentionErrorStats(0), SVG, 50, 720, 1280, 81,
                         gma_output_layout='head_major')
    core = AxisCurrentC8('wh')
    layer_counts = [0] * 40
    for layer, block in enumerate(pipe.transformer.blocks):
        processor = block.attn1.processor
        def invoke(q, k, v, layer=layer):
            layer_counts[layer] += 1
            return core(q, k, v)
        processor.use_gma = True
        processor.gma_attention = invoke

    video_path = OUT_DIR / '000-00.mp4'
    with torch.inference_mode():
        started = time.perf_counter()
        frames = pipe(prompt=prompts[0], negative_prompt=NEGATIVE_PROMPT, height=720, width=1280,
                      num_frames=81, guidance_scale=5., num_inference_steps=50,
                      generator=torch.Generator(device='cuda').manual_seed(0)).frames[0]
        seconds = time.perf_counter() - started
        ms, count = timer.finish()
        assert count == 4000 and len(frames) == 81
        export_to_video(frames, str(video_path), fps=16)
    print(json.dumps({'video': str(video_path), 'inference_seconds': seconds,
                       'mean_attention_ms': ms / count, 'attention_calls': count}), flush=True)


if __name__ == '__main__':
    main()
