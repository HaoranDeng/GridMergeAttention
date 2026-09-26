"""Shared, atomic per-video quality and per-dimension VBench work queue.

All video processing runs locally in the frozen metric implementations. No
video or extracted frame is sent to a language model. VBench is the existing
six-dimension custom-input protocol, not a full leaderboard score.
"""
import argparse
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback
from run import read, save, sha, module

DIMS = ['subject_consistency', 'background_consistency', 'motion_smoothness',
        'dynamic_degree', 'aesthetic_quality', 'imaging_quality']


def vbench_job(plan, job):
    vb = Path(plan['vbench_source'])
    os.chdir(vb)
    sys.path.insert(0, str(vb))
    import torch
    import vbench.distributed as distributed
    def single_gpu_init():
        # Each metric process has world size one. FileStore avoids collisions
        # with unrelated localhost TCPStore listeners on shared servers.
        store = Path(job['output']).parent / f'rendezvous_{os.getpid()}'
        torch.distributed.init_process_group('nccl', init_method=store.as_uri(), rank=0, world_size=1)
        torch.cuda.set_device(0)
    distributed.dist_init = single_gpu_init
    import clip
    original = clip.load
    def load(*args, **kwargs):
        kwargs.setdefault('download_root', plan['clip_cache'])
        return original(*args, **kwargs)
    clip.load = load
    sys.argv = ['evaluate.py', '--videos_path', job['videos'], '--output_path', job['output'],
                '--mode', 'custom_input', '--dimension', job['dimension']]
    runpy.run_path(str(vb / 'evaluate.py'), run_name='__main__')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--job', type=Path)
    a = parser.parse_args()
    plan = read(a.plan)
    work = a.plan.parent
    root = Path(plan['campaign'])
    queue = root / 'metrics'
    config = read(queue / 'plan.json')
    if a.job:
        vbench_job(config, read(a.job))
        return
    assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu']
    for path, digest in config['metric_source_sha256'].items():
        assert sha(path) == digest, path
    sys.path.insert(0, str(root / 'source/vendor'))
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    anchor = torch.ones(1, device='cuda')
    assert float(anchor.sum()) == 1
    spec = read(root / 'experiment_plan.json')
    sys.path.insert(0, spec['snapshot'])
    evaluator = module('frozen_i2v_quality', Path(spec['snapshot']) / 'analysis/score_videos_diffusers.py')
    model = None
    deadline = time.monotonic() + 6 * 3600
    completed = 0
    def stage(name, **fields):
        save(work / 'stage.json', dict(stage=name, time=time.time(), completed_jobs=completed, **fields))
    stage('ready', gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    while time.monotonic() < deadline:
        progress = False
        for cap in [4, 6, 8, 10]:
            out = root / f'C{cap}'
            for source in sorted((out / 'records').glob('*.json')):
                index = int(source.stem)
                reference = root / 'Dense/records' / source.name
                result = out / 'quality' / source.name
                if result.exists() or not reference.exists():
                    continue
                lock = queue / 'quality_claims' / f'C{cap}_{index:03d}'
                try:
                    lock.mkdir()
                except FileExistsError:
                    continue
                save(lock / 'owner.json', dict(plan=str(a.plan), time=time.time()))
                try:
                    method, dense = read(source), read(reference)
                    item = spec['inputs'][index]
                    for record in [method, dense]:
                        assert record['seed'] == item['seed']
                        assert record['prompt_sha256'] == item['prompt_sha256']
                        assert record['image_sha256'] == item['image_sha256']
                    paths = [root / 'Dense' / dense['video'], out / method['video']]
                    hashes = [sha(p) for p in paths]
                    assert hashes == [dense['video_sha256'], method['video_sha256']]
                    if model is None:
                        import lpips
                        model = lpips.LPIPS(net='alex').to('cuda').eval()
                    stage('quality', capacity=cap, index=index)
                    started = time.perf_counter()
                    row = evaluator.score_pair(*paths, model, torch.device('cuda'))
                    assert row['frames'] == 81 and all(math.isfinite(row[k]) for k in ['psnr', 'ssim', 'lpips'])
                    assert hashes == [sha(p) for p in paths]
                    row.update(index=index, device=torch.cuda.get_device_name(), gpu=plan['gpu'],
                               torch=torch.__version__, evaluation_seconds=time.perf_counter()-started,
                               input_sha256=dict(reference=hashes[0], method=hashes[1]),
                               evaluator_sha256=sha(Path(spec['snapshot']) / 'analysis/score_videos_diffusers.py'),
                               frame_subsampling=False, image_resizing=False)
                    save(result, row)
                    completed += 1
                except Exception:
                    save(lock / 'failure.json', dict(error=traceback.format_exc(), time=time.time()))
                progress = True
        for task in config['vbench_tasks']:
            if task['dimension'] not in plan['dimensions']:
                continue
            directory = queue / 'vbench' / task['name']
            if (directory / 'complete.json').exists() or (directory / 'claim').exists():
                continue
            videos = Path(task['videos'])
            expected = {f'{i:03d}-00.mp4' for i in range(20)}
            if {p.name for p in videos.glob('*.mp4') if '.partial' not in p.name} != expected:
                continue
            try:
                (directory / 'claim').mkdir()
            except FileExistsError:
                continue
            save(directory / 'claim/owner.json', dict(plan=str(a.plan), time=time.time()))
            before = {name: sha(videos / name) for name in sorted(expected)}
            job = dict(videos=str(videos), output=str(directory / 'out'), dimension=task['dimension'])
            save(directory / 'job.json', job)
            stage('vbench', task=task['name'])
            env = dict(os.environ, XDG_CACHE_HOME=config['xdg_cache'], VBENCH_CACHE_DIR=config['model_cache'],
                       MASTER_ADDR='127.0.0.1', MASTER_PORT=str(30000 + os.getpid() % 20000),
                       WORLD_SIZE='1', RANK='0', LOCAL_RANK='0')
            started = time.time()
            with (directory / 'log.txt').open('w') as log:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--plan', str(a.plan),
                                       '--job', str(directory / 'job.json')], env=env,
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            outputs = list((directory / 'out').glob('*_eval_results.json'))
            after = {name: sha(videos / name) for name in sorted(expected)}
            valid = proc.returncode == 0 and bool(outputs) and before == after
            save(directory / ('complete.json' if valid else 'failure.json'), dict(valid=valid,
                 exit_code=proc.returncode, videos=20, dimension=task['dimension'], host=plan['host'],
                 gpu=plan['gpu'], torch=torch.__version__, started=started, finished=time.time(),
                 video_sha256=before, outputs=[str(p) for p in outputs]))
            if valid:
                completed += 1
            progress = True
        quality_count = sum(len(list((root / f'C{c}/quality').glob('*.json'))) for c in [4, 6, 8, 10])
        benchmark_count = len(list((queue / 'vbench').glob('*/complete.json')))
        if quality_count == 80 and benchmark_count == len(config['vbench_tasks']):
            stage('complete', quality_videos=80, vbench_jobs=benchmark_count)
            return
        if not progress:
            stage('waiting_for_outputs', quality_videos=quality_count, vbench_jobs=benchmark_count)
            time.sleep(15)
    raise RuntimeError('Metric queue exceeded six hours; inspect unfinished claims')


if __name__ == '__main__':
    main()
