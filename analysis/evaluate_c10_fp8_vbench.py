"""Compare the completed native BF16/FP8 video pairs on six VBench dimensions.

This is custom-input evaluation, not the official sixteen-dimension total.
Run only one child at a time on a freshly clean H200; preserve HOME.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

DIMENSIONS = ['subject_consistency', 'background_consistency', 'motion_smoothness',
              'dynamic_degree', 'aesthetic_quality', 'imaging_quality']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / 'vbench6'
    assert not out.exists(), 'Use the existing process/result; never restart into its directory'
    quality = read(run / 'quality_cpu.json')
    assert quality['valid'] and quality['summary']['videos'] == 20 and quality['summary']['frames'] == 1620
    for path, digest in quality['generation_receipts'].items():
        assert sha(run / path) == digest, path
    assert len(quality['inputs']) == 40
    for item in quality['inputs']:
        assert sha(item['path']) == item['sha256'], item['path']
    contracts = [read(run / variant / 'videos/prompt_manifest.json') for variant in ('control', 'candidate')]
    for contract in contracts:
        assert contract['seed'] == 0 and contract['selected_prompt_count'] == 20
        assert contract['prompt_file_sha256'] == '50293c2717c8f2df5773154d4680c74e636ecb08eb4fe13cdce7357d0ab9d121'
    assert contracts[0]['prompts'] == contracts[1]['prompts']
    source = Path('/home/denghaoran/workspace/VBench')
    python = Path('/local2/denghaoran/GridMergeAttention/envs/vbench/bin/python')
    assert python.is_file() and (source / 'evaluate.py').is_file()
    initial = require_clean(args.gpu)
    out.mkdir()
    result = {'valid': False, 'scope': 'Six custom-input VBench dimensions, paired native BF16/FP8, 20 videos each; no official VBench total',
              'initial_gpu': initial, 'parent_pid': os.getpid(), 'started_utc': now(), 'dimensions': DIMENSIONS,
              'quality_receipt_sha256': sha(run / 'quality_cpu.json'), 'driver_sha256': sha(__file__),
              'vbench_evaluate_sha256': sha(source / 'evaluate.py'), 'phases': {}, 'monitor_errors': [], 'monitor_samples': []}

    def save():
        temporary = out / 'progress.tmp'
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(out / 'progress.json')

    save()
    cache = Path('/local2/denghaoran/GridMergeAttention/vbench_home/.cache')
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=initial['gpu']['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
               VBENCH_CACHE_DIR=str(cache / 'vbench'), TORCH_HOME=str(cache / 'torch'),
               HF_HOME=str(cache / 'huggingface'), XDG_CACHE_HOME=str(cache),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONPYCACHEPREFIX=str(out / 'python_cache'), PYTHONUNBUFFERED='1')
    assert env.get('HOME') == os.environ.get('HOME')
    for variant, contract in zip(('control', 'candidate'), contracts):
        videos = run / variant / 'videos'
        paths = sorted(videos.glob('*.mp4'))
        assert [p.name for p in paths] == [f'{i:03d}-00.mp4' for i in range(20)]
        prompts = {p.name: contract['prompts'][i]['text'] for i, p in enumerate(paths)}
        prompt_file = out / f'{variant}_prompts.json'
        prompt_file.write_text(json.dumps(prompts, indent=2))
        require_clean(args.gpu, expected_uuid=initial['gpu']['uuid'])
        phase_dir = out / variant
        command = [str(python), str(source / 'evaluate.py'), '--videos_path', str(videos),
                   '--dimension', *DIMENSIONS, '--mode', 'custom_input', '--load_ckpt_from_local', 'True',
                   '--prompt_file', str(prompt_file), '--output_path', str(phase_dir)]
        with (out / f'{variant}.log').open('w') as log:
            child = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            phase = {'pid': child.pid, 'command': command, 'started_utc': now()}
            result['phases'][variant] = phase
            save()
            while child.poll() is None:
                processes = gpu_processes(initial['gpu']['uuid'])
                result['monitor_samples'].append({'utc': now(), 'phase': variant, 'processes': processes})
                for process in processes:
                    try:
                        own = os.getpgid(process['pid']) == child.pid
                    except ProcessLookupError:
                        continue
                    if not own:
                        result['monitor_errors'].append({'utc': now(), 'foreign_process': process})
                save()
                if result['monitor_errors']:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=30)
                    raise RuntimeError('Foreign GPU process detected; evaluation invalid')
                time.sleep(3)
            phase.update(exit_code=child.returncode, ended_utc=now())
            save()
            assert child.returncode == 0, f'{variant} failed; inspect its log'
        files = list(phase_dir.glob('*_eval_results.json'))
        assert len(files) == 1, files
        measured = read(files[0])
        assert set(measured) == set(DIMENSIONS)
        phase['scores'] = {}
        for dimension, values in measured.items():
            assert len(values[1]) == 20, dimension
            assert {Path(row['video_path']).name for row in values[1]} == set(prompts), dimension
            phase['scores'][dimension] = float(values[0])
        phase.update(result_path=str(files[0]), result_sha256=sha(files[0]))
        save()
        print('COMPLETE', variant, phase['scores'], flush=True)
    for item in quality['inputs']:
        assert sha(item['path']) == item['sha256'], item['path']
    assert sha(source / 'evaluate.py') == result['vbench_evaluate_sha256']
    assert sha(run / 'quality_cpu.json') == result['quality_receipt_sha256']
    result['deltas_candidate_minus_control'] = {d: result['phases']['candidate']['scores'][d] - result['phases']['control']['scores'][d] for d in DIMENSIONS}
    result.update(valid=not result['monitor_errors'], ended_utc=now())
    save()
    (out / 'result.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
