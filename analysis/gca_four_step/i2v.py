"""Four-step distilled I2V, exact-capacity all-layer Q80K90Even.

Uses the frozen, previously deployed I2V conditioning/sampling implementation.
Workers atomically claim video indices; no saved or claimed video is overwritten.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from run import read, save, sha, module, LAYERS, TIMESTEPS


def cached_even(kdir, work, cache):
    import torch
    even = module('frozen_even', kdir / 'q80k90_even_kernel.py')
    original = even._native
    modules, hashes = {}, {}
    def native(stem, flags):
        full = ['-O3', '--expt-relaxed-constexpr', '-lineinfo'] + flags
        digest = hashlib.sha256((kdir / f'q80k90_even_{stem}.cu').read_bytes() + ' '.join(full).encode()).hexdigest()[:16]
        name = f'q80even_{stem}_{digest}'
        if name in modules:
            return modules[name]
        source = cache / name / (name + '.so')
        if source.exists():
            target = work / 'prebuilt' / (name + '.so')
            target.parent.mkdir(exist_ok=True)
            temporary = target.with_suffix('.so.tmp')
            shutil.copy2(source, temporary)
            temporary.replace(target)
            modules[name] = module(name, target)
            hashes[name] = dict(source=str(source), sha256=sha(target), torch=torch.__version__)
            save(work / 'prebuilt_modules.json', hashes)
        else:
            modules[name] = original(stem, flags)
        return modules[name]
    even._native = native
    return even


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    plan = read(a.plan)
    work = a.plan.parent
    root = Path(plan['campaign'])
    spec = read(root / 'experiment_plan.json')
    snap = Path(spec['snapshot'])
    assert os.uname().nodename.split('.')[0] == plan['host']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu']
    def stage(name, **fields):
        row = dict(stage=name, pid=os.getpid(), time=time.time(), **fields)
        save(work / 'stage.json', row)
        print(json.dumps(row), flush=True)
    stage('verifying')
    assert sha(root / 'dense.yaml') == spec['config_sha256']
    for name, digest in spec['source_sha256'].items():
        assert sha(snap / name) == digest, name
    for name, digest in spec['kernel_sha256'].items():
        assert sha(Path(spec['kernel_dir']) / name) == digest, name
    for item in spec['inputs']:
        assert sha(item['image']) == item['image_sha256']
        assert hashlib.sha256(item['prompt'].encode()).hexdigest() == item['prompt_sha256']
        assert item['seed'] == item['index']
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6', mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0, str(snap))
    os.chdir(snap)
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from generate import build_pipeline, load_generator_checkpoint, set_seed, save_video_atomic
    from attention.runtime_timing import ATTENTION_RUNTIME_RECORDER as timer
    from attention import q80k90_wan as adapter
    import wan.modules.model as wm
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.__version__ == '2.8.0+cu128'
    torch.empty(1, device='cuda')
    cap = plan['capacity']
    stage('compiling', capacity=cap)
    even = cached_even(Path(spec['kernel_dir']), work, Path(plan['native_cache']))
    even.native_layout(21, 40)
    even.native_select(21, 40, cap)
    even.native_merge(21, 40, 8, cap)
    even.native_route(21, 40, 8, cap)
    old = module('frozen_v4', snap / 'kernels/experimental/q80k90_c4_c12/recommended.py').make_q80k90(cap)
    old.native(21, 40, min(old.hc, 40))
    op = even.Q80K90Even(cap)
    config = OmegaConf.load(root / 'dense.yaml')
    assert list(config.denoising_step_list) == [1000, 750, 500, 250]
    stage('loading_model', capacity=cap)
    pipeline = build_pipeline(config, torch.device('cuda'))
    load_generator_checkpoint(pipeline, spec['checkpoint'], False)
    pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to('cuda')
    pipeline.generator.to('cuda')
    pipeline.vae.to('cuda')
    pipeline.eval()
    assert pipeline.is_i2v and pipeline.guidance_scale == 1.0 and pipeline.sampler == 'renoise'
    assert pipeline.denoising_step_list.tolist() == TIMESTEPS
    assert len(pipeline.generator.model.blocks) == 40
    state = dict(mode='gate', calls=0, layers=[0] * 40, layer=0, prompt=0)
    gate = []
    forward = wm.WanSelfAttention.forward
    def watched_forward(self, x, grid_size, rope_cache):
        assert tuple(x.shape) == (1, 75600, 5120)
        assert tuple(map(int, grid_size)) == (21, 45, 80)
        state['layer'] = self.layer_idx
        state['layers'][self.layer_idx] += 1
        return forward(self, x, grid_size, rope_cache)
    wm.WanSelfAttention.forward = watched_forward
    def timed(fn):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record(); end.synchronize()
        return result, start.elapsed_time(end)
    def attention(q, k, v, grid_size, capacity=8):
        assert capacity == cap and tuple(q.shape) == (1, 75600, 40, 128)
        layer = state['layer']
        step = state['calls'] // 40
        state['calls'] += 1
        if state['mode'] == 'gate':
            dense, dense_ms = timed(lambda: wm.flash_attention(q, k, v))
            if layer in LAYERS:
                new, new_ms = timed(lambda: op(q.contiguous(), k.contiguous(), v.contiguous()))
                previous = old(q.contiguous(), k.contiguous(), v.contiguous())
                rel = float((new.float() - previous.float()).norm() / previous.float().norm())
                dense_rel = float((new.float() - dense.float()).norm() / dense.float().norm())
                assert torch.isfinite(new).all() and math.isfinite(rel) and rel < .01
                gate.append(dict(prompt=state['prompt'], step=step, layer=layer, new_ms=new_ms,
                                 dense_ms=dense_ms, rel_l2_vs_v4=rel, rel_l2_vs_dense=dense_rel))
                del new, previous
            return dense
        return op(q.contiguous(), k.contiguous(), v.contiguous())
    adapter.q80k90_attention = attention
    def generate(index, mode):
        item = spec['inputs'][index]
        state.update(mode=mode, calls=0, layers=[0] * 40, prompt=index)
        pipeline.generator.model.token_merging_args = dict(enable=False, attention_backend=(
            'sdpa' if mode == 'dense' else f'q80k90_c{cap}_all'))
        set_seed(index)
        noise = torch.randn(list(config.image_or_video_shape), device='cuda', dtype=torch.bfloat16)
        with Image.open(item['image']) as source:
            image = source.convert('RGB')
        timer.begin_video()
        started = time.perf_counter()
        video = pipeline.inference(noise=noise, text_prompts=[item['prompt']], images=[image])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        timings = timer.finish_video()
        assert state['layers'] == [4] * 40
        assert timings['attention_calls'] == timings['full_attention_calls'] == 160
        assert state['calls'] == (0 if mode == 'dense' else 160)
        assert tuple(video.shape) == (1, 81, 3, 720, 1280) and torch.isfinite(video).all()
        return video, elapsed, timings
    stage('numerical_gate', capacity=cap)
    for index in range(3):
        video, _, _ = generate(index, 'gate')
        del video
        save(work / 'gate_progress.json', dict(prompts=index + 1, cases=len(gate), time=time.time()))
    assert len(gate) == 96
    means = {k: statistics.mean(r[k] for r in gate) for k in ('new_ms', 'dense_ms', 'rel_l2_vs_v4', 'rel_l2_vs_dense')}
    assert means['new_ms'] < means['dense_ms'], means
    save(work / 'gate.json', dict(passed=True, task='I2V', prompts=3, steps=4, layers=LAYERS,
         capacity=cap, rows=gate, summary=means, kernel_sha256=spec['kernel_sha256'],
         tolerance_scope='engineering gate, not a user-mandated universal threshold', relative_l2_limit=.01))
    del old
    gc.collect(); torch.cuda.empty_cache()
    for method, indices in [('Dense', plan['dense_indices']), (f'C{cap}', list(range(20)))]:
        out = root / method
        for index in indices:
            claim = out / 'claims' / f'{index:03d}'
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            save(claim / 'owner.json', dict(worker=str(work), host=plan['host'], gpu=plan['gpu'], pid=os.getpid()))
            assert not (out / f'{index:03d}-00.mp4').exists()
            stage('generating', method=method, index=index)
            video, elapsed, timings = generate(index, 'dense' if method == 'Dense' else 'gca')
            frames = (video[0].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu()
            path = out / f'{index:03d}-00.mp4'
            exported = time.perf_counter()
            save_video_atomic(path, frames, fps=16, preset='ultrafast', crf=23, quality=None)
            item = spec['inputs'][index]
            save(out / 'records' / f'{index:03d}.json', dict(index=index, seed=index, video=path.name,
                 video_sha256=sha(path), image_sha256=item['image_sha256'], prompt_sha256=item['prompt_sha256'],
                 inference_seconds=elapsed, export_seconds=time.perf_counter()-exported,
                 layer_calls=[4]*40, dense_self_attention_calls=160 if method == 'Dense' else 0,
                 host=plan['host'], gpu=plan['gpu'], worker=str(work), protocol=spec['protocol'], **timings))
            stage('video_saved', method=method, index=index, inference_seconds=elapsed)
            del video, frames
    stage('generation_complete', capacity=cap)


if __name__ == '__main__':
    main()
