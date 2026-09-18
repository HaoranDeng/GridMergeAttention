"""Publish a completed 20-video native phase without inventing quality/speedup."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--variant', choices=('control', 'candidate'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    run = args.run.resolve()
    meta = read(run / 'manifest.json')
    assert meta['num_videos'] == 20 and meta['steps'] == 4 and meta['seed'] == 0
    source = run / 'source'
    for name, digest in meta['sources'].items():
        assert sha(source / name) == digest, name
    sys.path.insert(0, str(source))
    from analysis.run_quality_first_videos import asset_preflight
    from utils.prompt_contract import load_output_prompt_manifest
    assert asset_preflight(meta['checkpoint'], {'WAN_MODEL_ROOT': meta['assets']['WAN_MODEL_ROOT']}) == meta['assets']
    directory = run / args.variant
    runtime = read(directory / 'runtime.json')
    rows = runtime['records']
    assert runtime['checkpoint_weights'] == 'raw_state_dict' and len(rows) == 20
    audit = read(run / 'dispatch_audit.json')
    audited = [row for row in audit['videos'] if row['variant'] == args.variant]
    assert len(audited) == 20 and audit['dense_calls'] == 0
    names = [f'{i:03d}-00.mp4' for i in range(20)]
    assert sorted(path.name for path in (directory / 'videos').glob('*.mp4')) == names
    prompt = load_output_prompt_manifest(directory / 'videos')
    assert prompt['seed'] == 0 and prompt['num_samples'] == 1 and prompt['selected_prompt_count'] == 20
    assert prompt['prompt_file_sha256'] == '50293c2717c8f2df5773154d4680c74e636ecb08eb4fe13cdce7357d0ab9d121'
    for index, (row, checked) in enumerate(zip(rows, audited, strict=True)):
        assert row['prompt_index'] == checked['prompt_index'] == index and row['output'] == names[index]
        assert row['attention_calls'] == row['full_attention_calls'] == checked['attention_calls'] == checked['full_attention_calls'] == 160
        assert row['attention_total_ms'] == checked['attention_total_ms']
        assert row['full_attention_total_ms'] == checked['full_attention_total_ms']
        assert checked['layer_calls'] == {str(layer): 4*(index+1) for layer in range(40)}
        calls = 160*(index+1)
        assert Counter(checked['native_stage_calls']) == Counter(select=calls, pack=calls, run=8*calls, route=8*calls)
        assert all(math.isfinite(row[key]) and row[key] > 0 for key in
                   ('attention_total_ms', 'full_attention_total_ms', 'inference_seconds', 'export_seconds'))
    purity = read(run / 'gpu_purity.json')
    assert not purity['errors']
    worker = read(run / 'process.json')['worker_pid']
    assert all(p['pid'] == worker for sample in purity['samples'] for p in sample['processes'])
    def summarize(records):
        return {'mean_attention_ms': sum(row['attention_total_ms'] for row in records)/(160*len(records)),
                'mean_full_attention_ms': sum(row['full_attention_total_ms'] for row in records)/(160*len(records)),
                'mean_e2e_seconds': statistics.fmean(row['inference_seconds']+row['export_seconds'] for row in records),
                'videos': len(records), 'attention_calls_per_video': 160}
    snapshot = directory / 'phase_audit_snapshot.json'
    assert not snapshot.exists()
    snapshot.write_text(json.dumps({'variant': args.variant, 'videos': audited, 'purity_observation': purity}, indent=2))
    result = {'valid': True, 'native_generation': True, 'variant': args.variant,
              'scope': 'Completed 20-video native generation phase; full comparison/quality acceptance remain separate',
              'generation': summarize(rows), 'steady_videos_1_to_19_diagnostic': summarize(rows[1:]),
              'speedup_vs_sdpa': None, 'quality_pending': True,
              'timing_scope': 'All 20 videos included, including first-video setup; inference plus export, model loading excluded. Instrumented native core includes finite checks. No matched SDPA generation control in this run.',
              'run': str(run), 'runtime': str(directory / 'runtime.json'),
              'runtime_sha256': sha(directory / 'runtime.json'), 'phase_audit_sha256': sha(snapshot),
              'manifest_sha256': sha(run / 'manifest.json'), 'frozen_source_commit': meta['source_commit'],
              'sources': meta['sources'], 'initial_gpu': purity['initial'], 'monitor_errors': [],
              'prompt_manifest_sha256': sha(directory / 'videos/prompt_manifest.json')}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'variant': args.variant, **result['generation']}), flush=True)


if __name__ == '__main__':
    main()
