"""Run one authorized matched-EAR hybrid job, then evaluate all 20 videos."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.prompt_contract import assert_matching_prompt_manifests, load_output_prompt_manifest


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def gpu_snapshot():
    p = subprocess.run(['rocm-smi', '--showpids', '--showuse', '--showmeminfo', 'vram'],
                       capture_output=True, text=True, timeout=20, check=True)
    pids = [int(x) for x in re.findall(r'^\s*(\d+)\s+\S+', p.stdout, re.M)]
    return dict(time=time.time(), pids=pids, text=p.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['ear', 'gma_c8_mi300_mix'], required=True)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    output = a.root / a.method
    if (output / 'completion.json').exists():
        return
    output.mkdir(exist_ok=True)
    videos = output / 'videos'
    attempts = output / 'attempts'
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f'{len(list(attempts.iterdir())):03d}'
    attempt.mkdir()
    existing = json.loads((videos / 'runtime.json').read_text()) if (videos / 'runtime.json').exists() else None
    base_count = len(existing['videos']) if existing else 0
    if list(videos.glob('*.mp4')):
        load_output_prompt_manifest(videos)
        assert existing and existing['backend'] == a.method
    prompt = ROOT / 'prompts/MovieGenVideoBench_extended.txt'
    env = dict(os.environ, SVG_DENSITY_LOG=str(attempt / 'density.jsonl'),
               GMA_MI300_HYBRID_PREFLIGHT='1', GMA_MI300_SLICE_SECONDS='7200')
    subprocess.run([sys.executable, 'analysis/verify_prompt_contract.py', '--prompt_path', str(prompt),
                    '--num_videos', '20', '--seed', '0', '--num_samples', '1',
                    '--manifest_out', str(output / 'prompt_contract.json')], cwd=ROOT, check=True)
    reference = a.root / 'dense_reference'
    load_output_prompt_manifest(reference)
    initial = gpu_snapshot()
    if initial['pids']:
        raise RuntimeError('Allocated MI300 has active compute processes')
    util = re.findall(r'GPU use \(%\):\s*(\d+)', initial['text'])
    if not util or any(int(x) > 1 for x in util):
        raise RuntimeError('GPU not idle or utilization unavailable')
    write(attempt / 'startup.json', dict(method=a.method, job=os.environ.get('SLURM_JOB_ID'),
        base_count=base_count,
        node=os.environ.get('SLURMD_NODENAME'), initial=initial,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        model='/work1/jasoncong/denghaoran/models/Wan2.1-T2V-14B-Diffusers',
        svg_commit='f89aedaf169ac2ae5b186bda674e53c3dc08c476',
        reference=str(reference), reference_timing='historical NVIDIA; no timing speedup claimed',
        dense_calls_per_video=880, method_calls_per_video=3120))
    command = [sys.executable, '-u', 'analysis/benchmark_wan14b_720p_methods.py',
        '--backend', a.method, '--model', '/work1/jasoncong/denghaoran/models/Wan2.1-T2V-14B-Diffusers',
        '--sparse_video_gen_root', str(a.root / 'svg'), '--prompt_path', str(prompt),
        '--output_dir', str(videos), '--count', '20', '--start', '0', '--seed', '0',
        '--height', '720', '--width', '1280', '--num_frames', '81', '--num_steps', '50',
        '--svg_top_p', '0.9', '--svg_min_kc_ratio', '0.10', '--attention_error_calls', '0',
        '--canonical_names', '--original_script_seeding']
    write(output / 'command.json', command)
    started = time.time()
    foreign = []
    with (attempt / 'generation.log').open('x') as log, (attempt / 'gpu_monitor.jsonl').open('x') as monitor:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                snap = gpu_snapshot()
                # Native HIP compilation may launch CPU children, never extra GPU owners.
                unknown = [pid for pid in snap['pids'] if pid != child.pid]
                if unknown:
                    foreign.append(snap)
                monitor.write(json.dumps(snap) + '\n'); monitor.flush()
                time.sleep(20)
        finally:
            if child.poll() is None:
                child.terminate(); child.wait(timeout=30)
    write(attempt / 'generation_completion.json', dict(exit_code=child.returncode,
        wall_seconds=time.time()-started, timing_valid=not foreign, foreign=foreign))
    if child.returncode:
        raise RuntimeError(f'Generation failed: {child.returncode}')
    runtime = json.loads((videos / 'runtime.json').read_text())
    write(attempt / 'progress.json', dict(base_count=base_count, completed_count=len(runtime['videos']),
        new_videos=len(runtime['videos'])-base_count, timing_valid=not foreign))
    if len(runtime['videos']) < 20:
        sys.exit(75)  # Requeue the same Slurm job after a complete-video boundary.
    assert len(runtime['videos']) == 20
    for row in runtime['videos']:
        counts = row['attention_dispatch']
        assert row['attention_calls'] == row['full_attention_calls'] == 4000
        assert counts['dense'] == 880
        assert counts['sparse' if a.method == 'ear' else 'gca'] == 3120
    assert_matching_prompt_manifests(reference, videos, prompt_indices=range(20))
    if a.method == 'ear':
        from analysis.launch_official_hybrid_50step import summarize_density
        with (videos / 'density.jsonl').open('w') as combined:
            for prior in sorted(attempts.iterdir()):
                progress = json.loads((prior / 'progress.json').read_text())
                rows = (prior / 'density.jsonl').read_text().splitlines()
                assert len(rows) == progress['new_videos'] * 3120
                combined.write('\n'.join(rows) + '\n')
        write(output / 'density_summary.json', summarize_density(videos / 'density.jsonl', runtime, 'ear'))
    with (output / 'quality.log').open('x') as log:
        subprocess.run([sys.executable, '-u', 'analysis/evaluate_video_quality.py',
            '--reference_dir', str(reference), '--method_dir', str(videos), '--method_name', a.method,
            '--output', str(output / 'quality.json'), '--expected_videos', '20',
            '--expected_frames', '81', '--lpips_batch_size', '4'],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    timing_valid = all(json.loads((prior / 'progress.json').read_text())['timing_valid']
                       for prior in attempts.iterdir())
    write(output / 'completion.json', dict(complete=True, timing_valid=timing_valid,
        wall_seconds=time.time()-started, summary=runtime['summary']))


if __name__ == '__main__':
    main()
