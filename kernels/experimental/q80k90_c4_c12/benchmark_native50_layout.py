"""Paired old/new C4 on native Wan QKV and complete 50-step generation.

Run only under the exclusive-GPU guard. No videos or frames are saved.
"""
import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import subprocess


LAYERS = [0, 5, 10, 15, 20, 25, 30, 35]
BASE = Path('/local2/denghaoran/GridMergeAttention')
REPO = BASE / 'axis_wh_c8_source_20260918'
SVG = Path('/local2/denghaoran/GridMergeAttention_worktrees/svg_official_20260906ch')
MODEL = '/home/denghaoran/wan_models/Wan2.1-T2V-14B-Diffusers'


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(temporary, path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def telemetry(root, stop):
    fields = 'uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used'
    with (root / 'telemetry.jsonl').open('w') as stream:
        while not stop.is_set():
            try:
                values = subprocess.check_output(
                    ['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
                     '--query-gpu=' + fields, '--format=csv,noheader,nounits'],
                    text=True, timeout=5).strip().split(',')
                stream.write(json.dumps(dict(time=time.time(), **dict(zip(
                    fields.split(','), [s.strip() for s in values])))) + '\n')
                stream.flush()
            except Exception as exc:
                stream.write(json.dumps(dict(time=time.time(), error=str(exc))) + '\n')
                stream.flush()
            stop.wait(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--full50', action='store_true')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    assert not (args.root / 'result.json').exists(), 'Do not overwrite a completed run'
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ['PATH']
    os.environ['TRITON_CACHE_DIR'] = str(args.root / 'triton')
    # Reuse the unchanged CUDA extension; all arithmetic kernels are frozen.
    os.environ['TORCH_EXTENSIONS_DIR'] = str(BASE / 'q80c4_c12_v4_extensions')
    sys.path[:0] = [str(args.source), str(REPO), str(SVG)]
    import numpy as np
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from analysis.benchmark_wan14b_720p_methods import (
        AttentionTimer, AttentionErrorStats, NEGATIVE_PROMPT, configure_processors)
    from utils.prompt_contract import validate_canonical_prompt_contract
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.preferred_linalg_library(backend='magma')
    torch.manual_seed(0)
    assert torch.cuda.get_device_name() == 'NVIDIA H200'
    prompts_path = REPO / 'prompts/MovieGenVideoBench_extended.txt'
    prompts = [s.strip() for s in prompts_path.read_text().splitlines() if s.strip()]
    validate_canonical_prompt_contract(prompts_path, prompts)
    config = json.loads((args.source / 'validated_h200_configs.json').read_text())['4']['config']
    old_module = load_module(args.source / 'reference_q80k90_kernel.py', 'c4_old_layout')
    new_module = load_module(args.source / 'q80k90_kernel.py', 'c4_direct_layout')
    cores = {'old': old_module.Q80K90(4, **config), 'new': new_module.Q80K90(4, **config)}
    hashes = {name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
              for name in ('reference_q80k90_kernel.py', 'q80k90_kernel.py', 'q80k90_merge.cu')}
    result = dict(protocol=dict(model=MODEL, steps=50, layers=LAYERS, all_layers=40,
        shape=[1, 40, 75600, 128], prompt_index=0, seed=0, frames=81,
        height=720, width=1280, input_layout='head-major native pipeline',
        full50_order=['old', 'new'], capacity=4, config=config, source_sha256=hashes,
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda, full50=args.full50),
        gate=[], videos=[], passed=False)
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(args.root, stop), daemon=True)
    monitor.start()
    def stage(name, **values):
        save(args.root / 'stage.json', dict(stage=name, time=time.time(), pid=os.getpid(), **values))
    try:
        stage('loading_model')
        vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder='vae',
            torch_dtype=torch.float32, local_files_only=True)
        pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16,
                                          local_files_only=True)
        pipe.scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction',
            use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=5.0)
        pipe.to('cuda')
        timer, full_timer = AttentionTimer(), AttentionTimer()
        configure_processors(pipe, 'dense', timer, full_timer, AttentionErrorStats(0),
            str(SVG), 50, 720, 1280, 81, gma_output_layout='head_major')
        # VAE fp32/384-wide heads need default SDPA dispatch, even when a
        # surrounding benchmark forces Flash for the transformer.
        vae_backends = [backend for backend, enabled in [
            (SDPBackend.FLASH_ATTENTION, torch.backends.cuda.flash_sdp_enabled()),
            (SDPBackend.EFFICIENT_ATTENTION, torch.backends.cuda.mem_efficient_sdp_enabled()),
            (SDPBackend.MATH, torch.backends.cuda.math_sdp_enabled()),
            (SDPBackend.CUDNN_ATTENTION, torch.backends.cuda.cudnn_sdp_enabled())] if enabled]
        original_decode = pipe.vae.decode
        def decode_with_default_backends(*values, **options):
            with sdpa_kernel(vae_backends):
                return original_decode(*values, **options)
        pipe.vae.decode = decode_with_default_backends
        result['protocol']['vae_sdpa_backends'] = [str(x) for x in vae_backends]
        state = dict(probe=True, seen=[0]*40, counts=[0]*40, method=None)
        samples = torch.tensor([(t*45+h)*80+w for t,h,w in
            itertools.product((0,10,20),(0,22,44),(0,40,79))], device='cuda')
        def timed(fn):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            output = fn()
            end.record()
            end.synchronize()
            return begin.elapsed_time(end), output
        def invoke(q, k, v, layer):
            inputs = [x.transpose(1,2) for x in (q,k,v)]
            assert q.shape == (1,40,75600,128)
            if not state['probe']:
                state['counts'][layer] += 1
                return cores[state['method']](*inputs).transpose(1,2)
            branch = state['seen'][layer]
            state['seen'][layer] += 1
            def dense_fn():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return torch.nn.functional.scaled_dot_product_attention(q,k,v)
            dense = dense_fn()
            if layer not in LAYERS:
                return dense
            reference = cores['old'](*inputs)
            updated = cores['new'](*inputs)
            exact = torch.equal(reference, updated)
            old_sample = reference.index_select(1,samples).float()
            new_sample = updated.index_select(1,samples).float()
            relative = float((new_sample-old_sample).norm()/old_sample.norm())
            finite = bool(torch.isfinite(updated).all())
            assert exact and finite, (layer, branch, relative)
            del reference, updated, old_sample, new_sample
            timings = {'old':[], 'new':[], 'dense':[]}
            for repeat in range(8):
                order = ('old','new') if repeat%2 == 0 else ('new','old')
                for name in order:
                    ms, output = timed(lambda: cores[name](*inputs))
                    timings[name].append(ms)
                    del output
            for repeat in range(3):
                ms, output = timed(dense_fn)
                timings['dense'].append(ms)
                del output
            assert statistics.mean(timings['new']) < statistics.mean(timings['dense'])
            buffers = [cores['new'].qp, cores['new'].kp, cores['new'].vp, cores['new'].packed]
            def old_pack():
                contiguous = [x.contiguous() for x in inputs]
                old_module._pack_inputs[((q.numel()+1023)//1024,)](
                    *contiguous, *buffers, 21, 40, 1024)
            def new_pack():
                new_module._pack_inputs_into(*inputs, *buffers, 21, 40)
            old_pack()
            new_pack()
            torch.cuda.synchronize()
            packing = {'old':[], 'new':[]}
            for repeat in range(6):
                for name in (('old','new') if repeat%2 == 0 else ('new','old')):
                    ms, _ = timed(old_pack if name == 'old' else new_pack)
                    packing[name].append(ms)
            row = dict(layer=layer, branch=branch, exact=exact, relative_l2_sample=relative,
                       timings_ms=timings, packing_ms=packing,
                       input_strides=[list(x.stride()) for x in inputs])
            result['gate'].append(row)
            save(args.root / 'progress.json', result)
            stage('native_gate', cases=len(result['gate']))
            return dense
        for layer, block in enumerate(pipe.transformer.blocks):
            block.attn1.processor.use_gma = True
            block.attn1.processor.gma_attention = lambda q,k,v,layer=layer: invoke(q,k,v,layer)
        kwargs = dict(prompt=prompts[0], negative_prompt=NEGATIVE_PROMPT, height=720,
            width=1280, num_frames=81, guidance_scale=5., num_inference_steps=50)
        class ProbeComplete(Exception):
            pass
        def first_step(pipeline, step, timestep, callback_kwargs):
            raise ProbeComplete()
        stage('native_gate', cases=0)
        try:
            with torch.inference_mode():
                pipe(**kwargs, generator=torch.Generator(device='cuda').manual_seed(0),
                     callback_on_step_end=first_step)
        except ProbeComplete:
            pass
        assert len(result['gate']) == 16
        assert sorted((r['layer'],r['branch']) for r in result['gate']) == list(itertools.product(LAYERS,(0,1)))
        result['gate_means_ms'] = {name: statistics.mean(v for row in result['gate']
            for v in row['timings_ms'][name]) for name in ('old','new','dense')}
        result['packing_means_ms'] = {name: statistics.mean(v for row in result['gate']
            for v in row['packing_ms'][name]) for name in ('old','new')}
        state['probe'] = False
        gc.collect()
        torch.cuda.empty_cache()
        if args.full50:
            for name in ('old','new'):
                state['method'] = name
                state['counts'] = [0]*40
                timer.reset()
                full_timer.reset()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                stage('full50', method=name, steps_done=0)
                def progress(pipeline, step, timestep, callback_kwargs):
                    if (step+1)%5 == 0:
                        stage('full50', method=name, steps_done=step+1)
                    return callback_kwargs
                started = time.perf_counter()
                with torch.inference_mode():
                    output = pipe(**kwargs, generator=torch.Generator(device='cuda').manual_seed(0),
                                  callback_on_step_end=progress, output_type='np').frames
                torch.cuda.synchronize()
                seconds = time.perf_counter()-started
                core_ms, calls = timer.finish()
                module_ms, module_calls = full_timer.finish()
                assert calls == module_calls == 4000 and state['counts'] == [100]*40
                row = dict(method=name, attention_calls=calls, mean_attention_ms=core_ms/calls,
                    mean_full_attention_ms=module_ms/module_calls, inference_seconds=seconds,
                    max_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                    output_sha256=hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest(),
                    note='Full pipeline through decoding; no export. One run/method; old then new.')
                result['videos'].append(row)
                save(args.root / 'progress.json', result)
                del output
                gc.collect()
        result['passed'] = True
        save(args.root / 'result.json', result)
        stage('complete')
        print(json.dumps(dict(passed=True, gate_means_ms=result['gate_means_ms'], videos=result['videos'])))
    finally:
        stop.set()
        monitor.join(timeout=8)


if __name__ == '__main__':
    main()
