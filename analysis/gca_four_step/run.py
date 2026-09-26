"""Frozen Q80K90Even 4-step T2V campaign, with isolated GPU supervision.

The numerical gate uses 3 prompts, all 4 timesteps and the fixed 8 layers.
Generation uses seed 0 independently for every prompt and all 40 layers.
"""
import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
LAYERS = [0, 5, 10, 15, 20, 25, 30, 35]
CAPS = {0: [4, 10], 1: [6, 8]}
TIMESTEPS = [1000.0, 937.5, 833.3333129882812, 625.0]
EVALUATOR_SHA = 'ea28e3c629165ef09a123f8145b8701996ea69daed7797bcfc14068910e717bb'


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def snapshot(gpu):
    def query(*args):
        return subprocess.check_output(['nvidia-smi', '-i', gpu, *args], text=True).strip()
    values = query('--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits').split(', ')
    processes = query('--query-compute-apps=pid', '--format=csv,noheader,nounits')
    return dict(index=int(values[0]), uuid=values[1], name=values[2], memory_mib=int(values[3]),
                utilization=int(values[4]), pids=[int(v) for v in processes.splitlines() if v.strip()], time=time.time())


def supervise(a):
    import fcntl
    root = a.root / f'worker{a.shard}'
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / 'completion.json').exists(), 'Use a fresh campaign after completion/failure'
    # A shared-filesystem lock avoids our two launches selecting the same UUID.
    gpu_lock = (a.root.parent / f'gca_gpu_{a.gpu}.lock').open('w')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for _ in range(3):
        preflight = snapshot(a.gpu)
        assert preflight['name'] == 'NVIDIA H200'
        assert not preflight['pids'] and preflight['memory_mib'] < 32 and preflight['utilization'] == 0, preflight
        time.sleep(1)
    save(root / 'preflight.json', preflight)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=preflight['uuid'], WAN_SDPA_BACKEND='flash',
               WAN_MODEL_ROOT=str(a.checkpoint.parent.parent), MAX_JOBS='2',
               TORCH_EXTENSIONS_DIR=str(root / 'torch_extensions'),
               TRITON_CACHE_DIR=str(root / 'triton_cache'), OMP_NUM_THREADS='4',
               PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    env['PATH'] = '/usr/local/cuda/bin:' + env.get('PATH', '')
    command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--root', str(a.root),
               '--shard', str(a.shard), '--gpu', a.gpu, '--checkpoint', str(a.checkpoint)]
    with (root / 'worker.log').open('a') as log:
        worker = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(root / 'launch.json', dict(pid=worker.pid, supervisor_pid=os.getpid(), host=os.uname().nodename,
             gpu=preflight, command=command, time=time.time()))
        valid = True
        while worker.poll() is None:
            sample = snapshot(a.gpu)
            foreign = []
            for pid in sample['pids']:
                try:
                    if os.getpgid(pid) != worker.pid:
                        foreign.append(pid)
                except ProcessLookupError:
                    pass
            sample['foreign_pids'] = foreign
            save(root / 'gpu_latest.json', sample)
            with (root / 'gpu_monitor.jsonl').open('a') as f:
                f.write(json.dumps(sample) + '\n')
            if foreign:
                valid = False
                save(root / 'contention.json', sample)
                os.killpg(worker.pid, signal.SIGTERM)
                break
            time.sleep(3)
        code = worker.wait()
    save(root / 'completion.json', dict(valid=valid and code == 0, exit_code=code, time=time.time()))
    return 0 if valid and code == 0 else 1


def worker(a):
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6', mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0, str(REPO))
    import torch
    from omegaconf import OmegaConf
    from unittest.mock import patch
    from generate import build_pipeline, load_generator_checkpoint, set_seed, summarize_runtime
    from utils.misc import save_video
    from utils.prompt_contract import load_nonempty_prompts, validate_canonical_prompt_contract
    from utils.prompt_contract import build_prompt_manifest, ensure_output_prompt_manifest
    from attention.runtime_timing import ATTENTION_RUNTIME_RECORDER as timer
    import wan.modules.model as wm
    from attention.core import _FORCED_SDPA_BACKEND
    assert _FORCED_SDPA_BACKEND == 'flash'
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = a.root / f'worker{a.shard}'
    def stage(name, **kw):
        row = dict(stage=name, time=time.time(), pid=os.getpid(), **kw)
        save(root / 'stage.json', row)
        print(json.dumps(row), flush=True)
    kdir = REPO / 'kernels/experimental/q80k90_even'
    even = module('campaign_even', kdir / 'q80k90_even_kernel.py')
    v4dir = REPO / 'kernels/experimental/q80k90_c4_c12'
    v4 = module('campaign_v4', v4dir / 'recommended.py')
    source_hashes = {p.name: sha(p) for p in kdir.glob('q80k90_even_*') if p.is_file()}
    source_hashes['runner'] = sha(__file__)
    caps = CAPS[a.shard]
    stage('compiling', capacities=caps)
    for cap in caps:
        even.native_layout(21, 40)
        even.native_select(21, 40, cap)
        even.native_merge(21, 40, 8, cap)
        even.native_route(21, 40, 8, cap)
        old = v4.make_q80k90(cap)
        old.native(21, 40, min(old.hc, 40))
        del old
    prompts_path = REPO / 'prompts/MovieGenVideoBench_extended.txt'
    prompts = load_nonempty_prompts(prompts_path)
    validate_canonical_prompt_contract(prompts_path, prompts)
    subprocess.run([sys.executable, str(REPO / 'analysis/verify_prompt_contract.py'),
                    '--prompt_path', str(prompts_path), '--num_videos', '20', '--seed', '0',
                    '--num_samples', '1', '--manifest_out', str(root / 'prompt_preflight.json')], check=True)
    manifest = build_prompt_manifest(prompts_path, prompts, selected_count=20, seed=0,
                                    num_samples=1, vbench_filenames=False, allow_noncanonical=False)
    manifest['seed_policy'] = 'constant_zero_per_prompt'
    manifest['steps'] = 4
    manifest['checkpoint'] = str(a.checkpoint)
    conf = OmegaConf.load(REPO / 'configs/dense_wan_14b_720p_fewstep.yaml')
    assert list(conf.denoising_step_list) == [1000, 750, 500, 250]
    assert list(conf.image_or_video_shape) == [1, 21, 16, 90, 160]
    stage('loading_model')
    pipeline = build_pipeline(conf, torch.device('cuda'))
    weights = load_generator_checkpoint(pipeline, str(a.checkpoint), True)
    assert weights == 'raw_state_dict', weights
    pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to('cuda')
    pipeline.generator.to('cuda')
    pipeline.eval()
    assert len(pipeline.generator.model.blocks) == 40
    assert pipeline.denoising_step_list.tolist() == TIMESTEPS
    protocol = dict(task='T2V', steps=4, timesteps=TIMESTEPS, cfg_branches=1, frames=81,
                    width=1280, height=720, q_group=80, key_group=90, capacities=caps,
                    seed=0, seed_policy='constant_zero_per_prompt', attention_calls_per_video=160,
                    all_layers=40, singleton_routes=0, source_sha256=source_hashes,
                    checkpoint=str(a.checkpoint), checkpoint_weights=weights,
                    gpu=read(root / 'preflight.json'), torch=torch.__version__)
    save(root / 'protocol.json', protocol)
    state = dict(layer=None, mode='gate', call=0, prompt=0, step=0, op=None)
    original_forward, original_attention = wm.WanSelfAttention.forward, wm.flash_attention
    gate_rows = []
    def forward(self, *args, **kwargs):
        assert state['layer'] is None and not self.enable_token_merging
        assert self.attention_backend == 'sdpa'
        state['layer'] = self.layer_idx
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            state['layer'] = None
    def measure(fn):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end), out
    def attention(q, k, v, *args, **kwargs):
        if state['layer'] is None:
            return original_attention(q, k, v, *args, **kwargs)
        layer = state['layer']
        assert layer == state['call'] % 40
        state['call'] += 1
        assert q.shape == (1, 75600, 40, 128)
        if state['mode'] == 'gca':
            return state['op'](q, k, v)
        dense = original_attention(q, k, v, *args, **kwargs)
        if state['mode'] != 'gate' or layer not in LAYERS:
            return dense
        for cap in caps:
            new, old = even.Q80K90Even(cap), v4.make_q80k90(cap)
            actual, expected = new(q, k, v), old(q, k, v)
            finite = bool(torch.isfinite(actual).all())
            rel = float((actual.float() - expected.float()).norm() / expected.float().norm())
            rel_dense = float((actual.float() - dense.float()).norm() / dense.float().norm())
            assert finite and rel < 0.01, (cap, layer, rel)
            del actual, expected, old
            # Both operators are already warm; this is a launch gate, not the headline benchmark.
            new_ms, output = measure(lambda: new(q, k, v))
            del output
            dense_ms, output = measure(lambda: original_attention(q, k, v, *args, **kwargs))
            del output, new
            gate_rows.append(dict(capacity=cap, prompt=state['prompt'], step=state['step'], layer=layer,
                                  rel_l2_vs_v4=rel, rel_l2_vs_dense=rel_dense, finite=finite,
                                  new_ms=new_ms, dense_ms=dense_ms))
        if layer == 35:
            stage('numerical_gate', prompt=state['prompt'], step=state['step'], cases=len(gate_rows))
        return dense
    with torch.inference_mode(), patch.object(wm.WanSelfAttention, 'forward', forward), patch.object(wm, 'flash_attention', attention):
        for p in range(3):
            state.update(prompt=p, call=0)
            set_seed(p)  # Frozen numerical-screen stream; generation below uses constant zero.
            noise = torch.randn(list(conf.image_or_video_shape), device='cuda', dtype=torch.bfloat16)
            current = noise
            conditional = pipeline.text_encoder(text_prompts=[prompts[p]])
            for step, timestep in enumerate(pipeline.denoising_step_list.to('cuda')):
                state['step'] = step
                _, predicted = pipeline.generator(noisy_image_or_video=current, conditional_dict=conditional,
                                                  timestep=timestep.expand(noise.shape[:2]))
                assert state['call'] == (step + 1) * 40
                if step < 3:
                    nxt = pipeline.denoising_step_list[step + 1].to('cuda').expand(noise.shape[:2])
                    current = pipeline.scheduler.add_noise(predicted.flatten(0, 1),
                              torch.randn_like(predicted.flatten(0, 1)), nxt.flatten(0, 1)).unflatten(0, noise.shape[:2])
            del noise, current, predicted, conditional
        gate_summary = {}
        for cap in caps:
            rows = [r for r in gate_rows if r['capacity'] == cap]
            assert len(rows) == 96
            avg = {k: statistics.mean(r[k] for r in rows) for k in ('new_ms', 'dense_ms', 'rel_l2_vs_v4', 'rel_l2_vs_dense')}
            assert avg['new_ms'] < avg['dense_ms'], avg
            gate_summary[str(cap)] = avg
        save(root / 'gate.json', dict(passed=True, layers=LAYERS, prompts=3, steps=4,
             numerical_seed_policy='prompt_index', relative_l2_limit=0.01,
             tolerance_reason='engineering gate; prior maximum approximately 0.0018, not a user-mandated tolerance',
             rows=gate_rows, summary=gate_summary, source_sha256=source_hashes))
        gc.collect()
        torch.cuda.empty_cache()
        pipeline.vae.to('cuda')
        dense_indices = list(range(a.shard * 10, (a.shard + 1) * 10))
        for cap in [0] + caps:
            state['mode'] = 'dense' if cap == 0 else 'gca'
            state['op'] = None if cap == 0 else even.Q80K90Even(cap)
            method = 'Dense' if cap == 0 else f'C{cap}'
            destination = a.root / method
            destination.mkdir(exist_ok=True)
            ensure_output_prompt_manifest(destination, manifest)
            runtime_path = destination / f'runtime_worker{a.shard}.json'
            assert not runtime_path.exists(), 'A fresh output is required; do not silently resume'
            records = []
            indices = dense_indices if cap == 0 else list(range(20))
            for index in indices:
                stage('generating', method=method, index=index, completed=len(records), total=len(indices))
                state['call'] = 0
                set_seed(0)
                noise = torch.randn(list(conf.image_or_video_shape), device='cuda', dtype=torch.bfloat16)
                torch.cuda.synchronize()
                started = time.perf_counter()
                timer.begin_video()
                video = pipeline.inference(noise=noise, text_prompts=[prompts[index]])
                timings = timer.finish_video()
                elapsed = time.perf_counter() - started
                assert state['call'] == timings['attention_calls'] == timings['full_attention_calls'] == 160
                assert tuple(video.shape) == (1, 81, 3, 720, 1280)
                assert torch.isfinite(video).all()
                exported = time.perf_counter()
                frames = (video[0].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                output = destination / f'{index:03d}-00.mp4'
                temporary = destination / f'{index:03d}-00.partial.mp4'
                assert not output.exists()
                save_video(str(temporary), frames, fps=16, preset='ultrafast', crf=23)
                temporary.replace(output)
                row = dict(index=index, prompt_index=index, prompt_sha256=manifest['prompts'][index]['sha256'],
                           seed=0, video=output.name, video_sha256=sha(output), layer_calls=[4] * 40,
                           dense_self_attention_calls=160 if cap == 0 else 0,
                           mean_attention_ms=timings['attention_total_ms'] / 160,
                           inference_seconds=elapsed, export_seconds=time.perf_counter() - exported,
                           warmup_video=(index == indices[0]), **timings)
                records.append(row)
                save(runtime_path, dict(protocol=protocol, method=method, records=records,
                     summary=summarize_runtime(records), complete=len(records) == len(indices)))
                stage('video_saved', method=method, index=index, inference_seconds=elapsed, completed=len(records), total=len(indices))
                del video, frames, noise
            state['op'] = None
            gc.collect()
            torch.cuda.empty_cache()
    stage('generation_complete', gca_videos=40, dense_videos=10)


def quality(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1')
    sys.path.insert(0, str(REPO))
    import torch
    import lpips
    source = REPO / 'analysis/evaluate_video_quality.py'
    assert sha(source) == EVALUATOR_SHA
    evaluator = module('frozen_quality', source)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = lpips.LPIPS(net='alex').to('cpu').eval()
    caps = CAPS[a.shard]
    results = {cap: [] for cap in caps}
    deadline = time.monotonic() + 12 * 3600
    root = a.root / f'worker{a.shard}'
    while sum(map(len, results.values())) < 40:
        progress = False
        assert time.monotonic() < deadline, 'Quality wait exceeded 12 hours'
        for worker_index in (0, 1):
            marker = a.root / f'worker{worker_index}/completion.json'
            if marker.exists():
                assert read(marker)['valid'], f'Generation failed: {marker}'
        for cap in caps:
            destination, reference = a.root / f'C{cap}', a.root / 'Dense'
            runtime = destination / f'runtime_worker{a.shard}.json'
            if not runtime.exists():
                continue
            done = {r['index'] for r in results[cap]}
            for record in read(runtime)['records']:
                index = record['index']
                if index in done:
                    continue
                dense_runtime = reference / f'runtime_worker{index // 10}.json'
                if not dense_runtime.exists():
                    continue
                paired = [r for r in read(dense_runtime)['records'] if r['index'] == index]
                if not paired:
                    continue
                dense_record = paired[0]
                evaluator.assert_matching_prompt_manifests(reference, destination, prompt_indices=[index])
                assert record['seed'] == dense_record['seed'] == 0
                assert record['prompt_sha256'] == dense_record['prompt_sha256']
                paths = [reference / record['video'], destination / record['video']]
                hashes = [sha(p) for p in paths]
                assert hashes == [dense_record['video_sha256'], record['video_sha256']]
                started = time.perf_counter()
                row = evaluator.evaluate_pair(*paths, model, 1, torch.device('cpu'), expected_frames=81)
                assert row['frames'] == 81 and all(math.isfinite(row[k]) for k in ('psnr', 'ssim', 'lpips'))
                assert hashes == [sha(p) for p in paths]
                row.update(index=index, evaluation_seconds=time.perf_counter() - started,
                           input_sha256=dict(reference=hashes[0], method=hashes[1]))
                results[cap].append(row)
                save(destination / 'quality' / f'{index:03d}.json', row)
                save(destination / 'quality.json', dict(complete=len(results[cap]) == 20,
                     evaluator_sha256=EVALUATOR_SHA, device='cpu', frame_subsampling=False,
                     image_resizing=False, records=results[cap], summary=evaluator.aggregate(results[cap])))
                save(root / 'quality_status.json', dict(stage='evaluating', capacity=cap,
                     completed={str(c): len(r) for c, r in results.items()}, time=time.time()))
                progress = True
        if not progress:
            time.sleep(5)
    save(root / 'quality_complete.json', dict(valid=True, videos=40, frames=3240, time=time.time()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['supervise', 'worker', 'quality'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shard', type=int, choices=[0, 1], required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--checkpoint', type=Path, default=Path.home() / 'wan_models/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt')
    a = p.parse_args()
    try:
        return globals()[a.mode](a) or 0
    except BaseException:
        save(a.root / f'worker{a.shard}' / f'{a.mode}_failure.json', dict(error=traceback.format_exc(), time=time.time()))
        raise


if __name__ == '__main__':
    sys.exit(main())
