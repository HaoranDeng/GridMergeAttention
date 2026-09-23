"""Native 50-step, all-layer Q80K90 C4/C6/C8 video shards with frozen gates."""
import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

BASE = Path('/local2/denghaoran/GridMergeAttention')
REPO = BASE/'axis_wh_c8_source_20260918'
KERNEL = BASE/'q80c4_c12_v4_20260923'
EVIDENCE = BASE/'experiments/q80c4_c12_borrow_gpu0_gpu4_20260923_v4/gpu0'
REFERENCE = BASE/'experiments/axis_video_c4_20260918/dense_first20'
MODEL = '/home/denghaoran/wan_models/Wan2.1-T2V-14B-Diffusers'
SVG = '/local2/denghaoran/GridMergeAttention_worktrees/svg_official_20260906ch'
LAYERS = [0, 5, 10, 15, 20, 25, 30, 35]
CAPACITIES = [4, 6, 8]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpu', type=int, choices=(0, 4), required=True)
    a = p.parse_args()
    indices = list(range(0, 10) if a.gpu == 0 else range(10, 20))
    os.environ['PATH'] = '/usr/local/cuda/bin:'+os.environ['PATH']
    os.environ.setdefault('TRITON_CACHE_DIR', str(BASE/'q80c4_c12_v4_triton'))
    sys.path[:0] = [str(KERNEL), str(REPO), SVG]
    from utils.prompt_contract import (build_prompt_manifest, ensure_output_prompt_manifest,
        validate_canonical_prompt_contract, assert_matching_prompt_manifests)
    prompt_path = REPO/'prompts/MovieGenVideoBench_extended.txt'
    prompts = [s.strip() for s in prompt_path.read_text().splitlines() if s.strip()]
    validate_canonical_prompt_contract(prompt_path, prompts)
    subprocess.run([sys.executable, str(REPO/'analysis/verify_prompt_contract.py'),
        '--prompt_path', str(prompt_path), '--num_videos', '20', '--seed', '0',
        '--num_samples', '1', '--manifest_out', str(a.root/'prompt_manifest.json')],
        check=True, capture_output=True)
    manifest = build_prompt_manifest(prompt_path, prompts, selected_count=20, seed=0,
        num_samples=1, vbench_filenames=False, allow_noncanonical=False)
    hashes = {n: sha(KERNEL/n) for n in ('q80k90_kernel.py', 'q80k90_merge.cu')}
    selected = read(KERNEL/'validated_h200_configs.json')
    for cap in CAPACITIES:
        evidence = read(EVIDENCE/f'C{cap}/result.json')
        numerical = read(EVIDENCE/f'C{cap}/numerical_gate.json')
        assert evidence['complete'] and evidence['numerical_passed']
        assert numerical['approved'] and numerical['summary']['cases'] == 384
        assert {n: evidence['source_sha256'][n] for n in hashes} == hashes
        assert selected[str(cap)]['implementation'] == 'v4'
        assert selected[str(cap)]['config'] == evidence['config']
        assert numerical['invariants']['exact_distinct_capacity'] == cap
        out = a.root/f'C{cap}/video20'
        assert not list(out.glob('*.mp4')), 'Fresh shard required'
        ensure_output_prompt_manifest(out, manifest)
        assert_matching_prompt_manifests(REFERENCE, out, prompt_indices=range(20))
        save(a.root/f'C{cap}/frozen_numerical_gate.json', numerical)
        save(a.root/f'C{cap}/selected_config.json', evidence['config'])
    def stage(name, **extra):
        row = dict(stage=name, time=time.time(), pid=os.getpid(), **extra)
        save(a.root/'stage.json', row)
        print(json.dumps(row), flush=True)
    stage('loading_native_model', gpu=a.gpu)
    import numpy as np
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    from analysis.benchmark_wan14b_720p_methods import (AttentionTimer, AttentionErrorStats,
        NEGATIVE_PROMPT, configure_processors)
    from recommended import make_q80k90
    from previous import create as make_previous
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.preferred_linalg_library(backend='magma')
    gpu = dict(index=a.gpu, name=torch.cuda.get_device_name(), uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    assert gpu['name'] == 'NVIDIA H200'
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder='vae', torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True,
        num_train_timesteps=1000, flow_shift=5.0)
    pipe.to('cuda')
    timer, full_timer = AttentionTimer(), AttentionTimer()
    configure_processors(pipe, 'dense', timer, full_timer, AttentionErrorStats(0), SVG,
        50, 720, 1280, 81, gma_output_layout='head_major')
    assert len(pipe.transformer.blocks) == 40
    layer_counts = [0]*40
    context = dict(probe=True, core=None, seen=[0]*40)
    gate = dict(complete=False, passed=False, gpu=gpu, source_sha256=hashes, layers=LAYERS,
        protocol='Native 50-step schedule first step, prompt000 seed0, both CFG branches; engineering integration/speed gate only',
        frozen_numerical_cases_per_capacity=384, rows=[])
    samples = torch.tensor([(t*45+h)*80+w for t, h, w in
        itertools.product((0, 10, 20), (0, 22, 44), (0, 40, 79))], device='cuda')
    def timed(fn):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end), out
    def invoke(q, k, v, layer):
        assert tuple(q.shape) == (1, 40, 75600, 128)
        if not context['probe']:
            layer_counts[layer] += 1
            return context['core'](q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        branch = context['seen'][layer]
        context['seen'][layer] += 1
        def dense():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)
        dense_output = dense()
        if layer not in LAYERS:
            return dense_output
        inputs = [x.transpose(1, 2) for x in (q, k, v)]
        for cap in CAPACITIES:
            previous = make_previous(cap)
            old = previous(*inputs).index_select(1, samples).float().clone()
            del previous
            gc.collect()
            torch.cuda.empty_cache()
            core = make_q80k90(cap)
            output = core(*inputs)
            sample = output.index_select(1, samples).float()
            relative = float((sample-old).norm()/old.norm())
            assert torch.isfinite(output).all() and relative < .02, (cap, layer, relative)
            del output, sample, old
            times = dict(method=[], dense=[])
            for rep in range(3):
                calls = [('method', lambda: core(*inputs)), ('dense', dense)]
                if rep % 2:
                    calls.reverse()
                for name, fn in calls:
                    ms, result = timed(fn)
                    times[name].append(ms)
                    del result
            row = dict(capacity=cap, layer=layer, branch=branch, relative_l2_vs_previous=relative,
                method_ms=times['method'], dense_ms=times['dense'],
                mean_attention_ms=statistics.mean(times['method']),
                dense_mean_attention_ms=statistics.mean(times['dense']))
            assert row['mean_attention_ms'] < row['dense_mean_attention_ms'], row
            gate['rows'].append(row)
            save(a.root/'native_gate.json', gate)
            print(json.dumps(dict(event='native_gate_case', **row)), flush=True)
            del core
            gc.collect()
            torch.cuda.empty_cache()
        return dense_output
    for layer, block in enumerate(pipe.transformer.blocks):
        processor = block.attn1.processor
        processor.use_gma = True
        processor.gma_attention = lambda q, k, v, layer=layer: invoke(q, k, v, layer)
    kwargs = dict(negative_prompt=NEGATIVE_PROMPT, height=720, width=1280, num_frames=81,
        guidance_scale=5., num_inference_steps=50)
    class ProbeComplete(Exception):
        pass
    def stop_probe(pipeline, step, timestep, callback_kwargs):
        assert step == 0 and context['seen'] == [2]*40
        raise ProbeComplete()
    stage('native_same_gpu_gate', gpu=a.gpu)
    try:
        with torch.inference_mode():
            pipe(prompt=prompts[0], generator=torch.Generator(device='cuda').manual_seed(0),
                callback_on_step_end=stop_probe, **kwargs)
    except ProbeComplete:
        pass
    else:
        raise AssertionError('Native probe did not stop at the first step')
    assert len(gate['rows']) == 48 and context['seen'] == [2]*40
    gate['capacities'] = {}
    for cap in CAPACITIES:
        rows = [r for r in gate['rows'] if r['capacity'] == cap]
        assert {(r['layer'], r['branch']) for r in rows} == set(itertools.product(LAYERS, (0, 1)))
        method = statistics.mean(r['mean_attention_ms'] for r in rows)
        dense = statistics.mean(r['dense_mean_attention_ms'] for r in rows)
        gate['capacities'][str(cap)] = dict(cases=16, mean_attention_ms=method,
            dense_mean_attention_ms=dense, speedup_vs_sdpa=dense/method,
            max_relative_l2_vs_previous=max(r['relative_l2_vs_previous'] for r in rows))
    gate.update(complete=True, passed=True)
    save(a.root/'native_gate.json', gate)
    context['probe'] = False
    timer.reset()
    full_timer.reset()
    gc.collect()
    torch.cuda.empty_cache()
    for cap in CAPACITIES:
        out = a.root/f'C{cap}/video20'
        runtime_path = out/'runtime.json'
        context['core'] = make_q80k90(cap)
        protocol = dict(model=MODEL, steps=50, height=720, width=1280, frames=81, seed=0,
            guidance_scale=5., scheduler='UniPC flow_shift=5', fps=16, q_group=80, key_group=90,
            capacity=cap, axes='wh', rf0=True, singleton_routes=0, all_layers=40,
            expected_calls_per_video=4000, dense_self_attention_calls=0, num_videos=20,
            seed_policy='RF0 two forced intersections then Fisher, balanced32 Uniform16',
            fisher_precision='FP32 selector', source_sha256=hashes,
            implementation='optimized_v4', kernel_config=selected[str(cap)]['config'],
            full_50_step_pflops=selected[str(cap)]['full_50_step_pflops'],
            planned_indices=indices, gpu=gpu, numerical_gate=str(a.root/f'C{cap}/frozen_numerical_gate.json'),
            native_same_gpu_gate=str(a.root/'native_gate.json'), wan_forward='svg', qk_norm='triton', rope='svg')
        runtime = dict(protocol=protocol, videos=[], generation_complete=False)
        save(runtime_path, runtime)
        for index in indices:
            stage('generating', capacity=cap, index=index, gpu=a.gpu)
            video = out/f'{index:03d}-00.mp4'
            assert not video.exists()
            before = layer_counts.copy()
            timer.reset()
            full_timer.reset()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                frames = pipe(prompt=prompts[index], generator=torch.Generator(device='cuda').manual_seed(0), **kwargs).frames[0]
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            ms, count = timer.finish()
            full_ms, full_count = full_timer.finish()
            counts = [x-y for x, y in zip(layer_counts, before)]
            assert count == full_count == 4000 and counts == [100]*40 and len(frames) == 81
            assert all(np.isfinite(np.asarray(frame)).all() for frame in frames)
            started = time.perf_counter()
            temporary = out/f'{index:03d}-00.partial.mp4'
            export_to_video(frames, str(temporary), fps=16)
            assert temporary.stat().st_size > 0
            os.replace(temporary, video)
            record = dict(index=index, video=video.name, prompt=prompts[index], seed=0,
                attention_calls=count, layer_calls=counts, dense_self_attention_calls=0,
                mean_attention_ms=ms/count, mean_full_attention_ms=full_ms/full_count,
                inference_seconds=seconds, export_seconds=time.perf_counter()-started,
                max_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                gpu=gpu, video_sha256=sha(video), saved_at=time.time())
            runtime['videos'].append(record)
            save(runtime_path, runtime)
            print(json.dumps(dict(event='video_saved', capacity=cap, **record)), flush=True)
            del frames
        runtime['generation_complete'] = True
        save(runtime_path, runtime)
        context['core'] = None
        gc.collect()
        torch.cuda.empty_cache()
    stage('generation_complete', videos=30, gpu=a.gpu)


if __name__ == '__main__':
    main()
