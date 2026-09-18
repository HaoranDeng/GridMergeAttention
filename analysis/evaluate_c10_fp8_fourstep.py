"""CPU fidelity for the completed native C10 BF16/FP8 four-step comparison.

Uses the generation run's frozen evaluator for every full-resolution frame.
Generation validity and fidelity measurements do not imply VBench acceptance.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'), 'CPU only'
    run = args.run.resolve()
    output = run / 'quality_cpu.json'
    progress = run / 'quality_cpu.progress.json'
    assert not output.exists() and not progress.exists(), 'Use fresh quality artifacts'
    metadata = read(run / 'manifest.json')
    completion = read(run / 'completion.json')
    assert metadata['num_videos'] == 20 and metadata['steps'] == 4 and metadata['seed'] == 0
    assert completion['generation_valid'] and completion['worker_exit_code'] == 0 and not completion['errors']
    assert not read(run / 'gpu_purity.json')['errors']
    audit = read(run / 'dispatch_audit.json')
    assert audit['complete'] and audit['dense_calls'] == 0 and len(audit['videos']) == 40
    receipts = {name: sha(run / name) for name in ('manifest.json', 'completion.json', 'dispatch_audit.json', 'gpu_purity.json')}
    source = run / 'source'
    for name, digest in metadata['sources'].items():
        assert sha(source / name) == digest, name
    for mode in ('control', 'candidate'):
        rows = [row for row in audit['videos'] if row['variant'] == mode]
        assert [row['prompt_index'] for row in rows] == list(range(20))
        for index, row in enumerate(rows):
            assert row['attention_calls'] == row['full_attention_calls'] == 160
            assert row['layer_calls'] == {str(layer): 4 * (index + 1) for layer in range(40)}
            calls = 160 * (index + 1)
            assert Counter(row['native_stage_calls']) == Counter(select=calls, pack=calls, run=8*calls, route=8*calls)
        runtime_path = run / mode / 'runtime.json'
        runtime = read(runtime_path)
        assert runtime['checkpoint_weights'] == 'raw_state_dict' and len(runtime['records']) == 20
        for index, row in enumerate(runtime['records']):
            assert row['prompt_index'] == index and row['output'] == f'{index:03d}-00.mp4'
            assert row['attention_calls'] == row['full_attention_calls'] == 160
        receipts[str(runtime_path.relative_to(run))] = sha(runtime_path)
    sys.path.insert(0, str(source))
    evaluator_path = source / 'analysis/evaluate_video_quality.py'
    assert sha(evaluator_path) == 'ea28e3c629165ef09a123f8145b8701996ea69daed7797bcfc14068910e717bb'
    spec = importlib.util.spec_from_file_location('frozen_c10_quality', evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    pairs, contract = evaluator.prepare_comparison(run / 'control/videos', run / 'candidate/videos', expected_videos=20)
    import torch
    import lpips
    assert not torch.cuda.is_available()
    assert (Path(torch.hub.get_dir()) / 'checkpoints/alexnet-owt-7be5be79.pth').is_file()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = lpips.LPIPS(net='alex').to('cpu').eval()
    result = {'valid': False, 'scope': 'FP8 fidelity against matched native BF16; VBench and visual acceptance pending',
              'protocol': {'videos': 20, 'frames_per_video': 81, 'resize': False, 'subsample': False, 'device': 'cpu'},
              'prompt_contract': contract, 'evaluator_sha256': sha(evaluator_path), 'driver_sha256': sha(__file__),
              'generation_receipts': receipts, 'inputs': [], 'videos': []}
    write(run / 'quality_cpu_process.json', {'pid': os.getpid(), 'command': sys.argv, 'driver_sha256': sha(__file__)})
    for reference, candidate in pairs:
        inputs = [{'path': str(path), 'sha256': sha(path)} for path in (reference, candidate)]
        for path in (reference, candidate):
            assert next(evaluator.frames(path)).shape == (720, 1280, 3)
        row = evaluator.evaluate_pair(reference, candidate, model, 1, torch.device('cpu'), expected_frames=81)
        assert all(math.isfinite(row[key]) for key in ('psnr', 'ssim', 'lpips')), row
        assert all(sha(item['path']) == item['sha256'] for item in inputs)
        result['inputs'].extend(inputs)
        result['videos'].append(row)
        result['summary'] = evaluator.aggregate(result['videos'])
        write(progress, result)
        print(json.dumps(row), flush=True)
    assert all(sha(item['path']) == item['sha256'] for item in result['inputs'])
    assert all(sha(run / name) == digest for name, digest in receipts.items())
    assert sha(evaluator_path) == result['evaluator_sha256']
    result['valid'] = True
    write(output, result)
    print(json.dumps({'valid': True, 'summary': result['summary']}), flush=True)


if __name__ == '__main__':
    main()
