"""Twenty matched videos, with immediate first-five quality reporting."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from analysis.quick_axis_c4_qkv import save, sha
from utils.prompt_contract import (build_prompt_manifest, ensure_output_prompt_manifest,
                                   validate_canonical_prompt_contract, assert_matching_prompt_manifests)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--axes', choices=['tw', 'wh', 'th'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--speed-gate', type=Path, required=True)
    p.add_argument('--allow-user-speed-exception',action='store_true')
    p.add_argument('--model', default='/home/denghaoran/wan_models/Wan2.1-T2V-14B-Diffusers')
    p.add_argument('--svg', default='/local2/denghaoran/GridMergeAttention_worktrees/svg_official_20260906ch')
    p.add_argument('--indices',nargs='+',type=int,required=True)
    p.add_argument('--aggregate-config',type=Path,required=True)
    a = p.parse_args()
    assert a.indices and len(set(a.indices))==len(a.indices)
    assert all(0<=i<20 for i in a.indices)
    from analysis.axis_video_shards import publish,evaluate_ready
    assert not (a.output/'runtime.json').exists(),'Fresh shard output required'

    validation = json.loads(a.validation.read_text())
    assert validation['passed'] and validation['axes'] == a.axes
    root = Path(__file__).resolve().parents[1]
    from analysis.axis_c4_speed_gate import verify_gate
    gate = verify_gate(a.speed_gate, root, a.axes,a.allow_user_speed_exception)
    prompt_path = root/'prompts/MovieGenVideoBench_extended.txt'
    prompts = [v.strip() for v in prompt_path.read_text().splitlines() if v.strip()]
    validate_canonical_prompt_contract(prompt_path, prompts)
    manifest = build_prompt_manifest(prompt_path, prompts, selected_count=20, seed=0,
                                     num_samples=1, vbench_filenames=False, allow_noncanonical=False)
    ensure_output_prompt_manifest(a.output, manifest)
    assert_matching_prompt_manifests(a.reference, a.output, prompt_indices=range(20))
    sys.path.insert(0, a.svg)
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    from analysis.benchmark_wan14b_720p_methods import (AttentionTimer, AttentionErrorStats,
                                                      NEGATIVE_PROMPT, configure_processors)
    import importlib
    AxisCurrentC4 = importlib.import_module(gate['methods'][a.axes]['module']).AxisCurrentC4
    os.environ.update(gate['methods'][a.axes]['environment'])
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    torch.backends.cuda.preferred_linalg_library(backend='magma')
    vae = AutoencoderKLWan.from_pretrained(a.model, subfolder='vae', torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(a.model, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True,
                                            num_train_timesteps=1000, flow_shift=5.0)
    pipe.to('cuda')
    timer, full_timer = AttentionTimer(), AttentionTimer()
    # Reuse audited Wan projections/norm/RoPE/output handling; replace the
    # self-attention callback in every block unconditionally before inference.
    configure_processors(pipe, 'dense', timer, full_timer, AttentionErrorStats(0), a.svg,
                         50, 720, 1280, 81, gma_output_layout='head_major')
    core = AxisCurrentC4(a.axes)
    layer_counts = [0] * 40
    assert len(pipe.transformer.blocks) == 40
    for layer, block in enumerate(pipe.transformer.blocks):
        processor = block.attn1.processor
        def invoke(q, k, v, layer=layer):
            layer_counts[layer] += 1
            return core(q, k, v)
        processor.use_gma = True
        processor.gma_attention = invoke
    protocol = {'model':a.model, 'steps':50, 'height':720, 'width':1280, 'frames':81, 'seed':0,
                'guidance_scale':5., 'scheduler':'UniPC flow_shift=5', 'axes':a.axes,
                'capacity':4, 'singleton_routes':0, 'seed_policy':'current-line anchor-Fisher Cmin32/g45 Uniform16',
                'all_layers':40, 'expected_calls_per_video':4000, 'dense_self_attention_calls':0,
                'wan_forward':'svg', 'qk_norm':'triton', 'rope':'svg', 'svg_source':a.svg,
                'block_j':int(os.environ.get('AXIS_C4_BLOCK_J','4')),
                'probe_precompute':os.environ.get('AXIS_C4_PRECOMPUTE','0') == '1',
                'fisher_precision':'float32',
                'source_sha256':gate['source_sha256'], 'speed_gate':gate,
                'validation':validation, 'fps':16, 'num_videos':20, 'first_quality_at':5,'shard_indices':a.indices}
    runtime_path = a.output/'runtime.json'
    runtime = {'protocol':protocol, 'videos':[], 'generation_complete':False,
               'first5_quality_complete':False, 'quality_complete':False}
    save(runtime_path, runtime)
    with torch.inference_mode():
        for index in a.indices:
            video_path = a.output/f'{index:03d}-00.mp4'
            if video_path.exists():
                raise RuntimeError('Fresh run only; refusing to overwrite an existing video')
            before = layer_counts.copy()
            timer.reset(); full_timer.reset(); torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); started = time.perf_counter()
            frames = pipe(prompt=prompts[index], negative_prompt=NEGATIVE_PROMPT,
                          height=720, width=1280, num_frames=81, guidance_scale=5.,
                          num_inference_steps=50, generator=torch.Generator(device='cuda').manual_seed(0)).frames[0]
            torch.cuda.synchronize(); seconds = time.perf_counter()-started
            ms, count = timer.finish(); full_ms, full_count = full_timer.finish()
            counts = [x-y for x,y in zip(layer_counts,before)]
            assert count == full_count == 4000 and counts == [100]*40
            assert len(frames) == 81
            started = time.perf_counter(); export_to_video(frames, str(video_path), fps=16)
            record = {'index':index, 'video':video_path.name, 'prompt':prompts[index], 'seed':0,
                      'attention_calls':count, 'mean_attention_ms':ms/count,
                      'mean_full_attention_ms':full_ms/full_count, 'inference_seconds':seconds,
                      'export_seconds':time.perf_counter()-started,
                      'max_memory_gib':torch.cuda.max_memory_allocated()/2**30,
                      'layer_calls':counts, 'dense_self_attention_calls':0}
            runtime['videos'].append(record); save(runtime_path,runtime)
            print(json.dumps(record),flush=True)
            del frames
            evaluate_ready(a.aggregate_config,a.reference,prompt_path,prompts,a.axes)
    runtime['shard_generation_complete'] = True; save(runtime_path,runtime)
    del pipe, vae, core, processor, invoke
    gc.collect(); torch.cuda.empty_cache()
    evaluate_ready(a.aggregate_config,a.reference,prompt_path,prompts,a.axes)


if __name__ == '__main__':
    main()
