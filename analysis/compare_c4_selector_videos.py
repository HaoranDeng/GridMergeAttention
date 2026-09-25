"""Aggregate the C4 selector video screen: does FP32 Gram-Schmidt move quality?

CPU only. Runs the established per-video PSNR screen three ways and pairs the
results by prompt:

  * control arm  vs the dense reference - the number the project already tracks;
  * candidate arm vs the dense reference - the same number for FP32 selection;
  * candidate arm vs control arm         - how far the two videos are from each
    other, which is the direct answer to "did the videos change at all".

The first two are paired per prompt and reported as a paired difference with a
t statistic, because the per-prompt spread of absolute PSNR is far larger than
any plausible effect and an unpaired comparison would drown it.

Both arms must come from the same host. `video-screen-node-confound` records
that a cross-node repeat of an identical configuration reproduces only to about
15 dB, which is worse than the effect being screened for, so a cross-node pair
cannot answer this question at all. The host of each arm is read back out of
its own run record and a mismatch refuses to report.
"""
import argparse, json, statistics, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DENSE = Path('/local2/denghaoran/GridMergeAttention/experiments/'
             'wan14b_720p_4step_dense_flash_matched_extended_ema_20260826/flash/videos')


def run_screen(reference, method, count, output):
    if output.exists():
        return json.loads(output.read_text())
    subprocess.run([sys.executable, '-u', str(ROOT / 'analysis/screen_video_psnr_cpu.py'),
                    '--reference-dir', str(reference), '--method-dir', str(method),
                    '--count', str(count), '--output', str(output)],
                   cwd=str(ROOT), check=True)
    return json.loads(output.read_text())


def head_to_head(control_videos, candidate_videos, count, output):
    """PSNR of the two arms against each other, tolerating identical videos.

    The tracked screen refuses a non-finite PSNR, which is the right gate when
    the reference is a different method but the wrong one here: two arms that
    agree bit for bit are the strongest possible result, not a failure.
    """
    if output.exists():
        return json.loads(output.read_text())
    sys.path.insert(0, str(ROOT / 'analysis'))
    import numpy as np
    from evaluate_video_quality import prepare_comparison, paired_frames
    from skimage.metrics import peak_signal_noise_ratio
    pairs, contract = prepare_comparison(control_videos, candidate_videos, max_videos=count)
    require_prefix = ['%03d-00.mp4' % index for index in range(count)]
    if [left.name for left, _ in pairs] != require_prefix:
        raise ValueError('not the predefined canonical prefix')
    rows = []
    for left, right in pairs:
        values, identical = [], True
        for a, b in paired_frames(left, right, expected_frames=81):
            if not np.array_equal(a, b):
                identical = False
            values.append(peak_signal_noise_ratio(a, b, data_range=255))
        finite = [v for v in values if np.isfinite(v)]
        rows.append({'video': left.name, 'frames': len(values),
                     'identical_frames': sum(1 for v in values if not np.isfinite(v)),
                     'all_frames_identical': identical,
                     'psnr': float(np.mean(finite)) if finite else float('inf'),
                     'min_frame_psnr': float(min(finite)) if finite else float('inf')})
    record = {'scope': 'arm-versus-arm PSNR; identical frames are reported, not rejected',
              'reference_dir': str(control_videos), 'method_dir': str(candidate_videos),
              'count': count, 'prompt_contract': contract, 'videos': rows}
    output.write_text(json.dumps(record, indent=2))
    return record


def per_video(screen):
    return {row['video']: row['psnr'] for row in screen['videos']}


def paired(a, b):
    """b minus a, prompt by prompt."""
    keys = sorted(set(a) & set(b))
    deltas = [b[k] - a[k] for k in keys]
    n = len(deltas)
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else float('nan')
    return {'n': n, 'videos': keys,
            'per_video_delta_db': dict(zip(keys, deltas)),
            'mean_delta_db': mean, 'sd_db': sd,
            't': mean / (sd / n ** .5) if sd and n > 1 else float('nan'),
            'max_abs_delta_db': max(abs(d) for d in deltas) if deltas else float('nan')}


def host_of(run_dir):
    for name in ('completion.json', 'startup.json', 'gpu_monitor.json'):
        path = run_dir / name
        if path.exists():
            record = json.loads(path.read_text())
            gpu = record.get('initial_gpu') or record.get('initial') or {}
            host = gpu.get('host') or record.get('host')
            if host:
                return host
    marker = run_dir / 'host'
    return marker.read_text().strip() if marker.exists() else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-run', type=Path, required=True)
    p.add_argument('--candidate-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-dir', type=Path, default=DENSE)
    p.add_argument('--count', type=int, action='append', default=[],
                   help='Repeatable prefix size, e.g. --count 5 --count 20.')
    p.add_argument('--allow-cross-host', action='store_true',
                   help='Report anyway. Only meaningful if the effect being screened '
                        'for is larger than the ~15 dB cross-node reproduction floor.')
    a = p.parse_args()
    counts = sorted(set(a.count or [5, 20]))
    assert not a.output.exists()
    a.output.parent.mkdir(parents=True, exist_ok=True)

    hosts = {'control': host_of(a.control_run), 'candidate': host_of(a.candidate_run)}
    same_host = hosts['control'] is not None and hosts['control'] == hosts['candidate']
    result = {
        'scope': 'CPU-only PSNR screen of a C4 selector change across the canonical '
                 'prompt prefix. Not the full metric suite.',
        'control_run': str(a.control_run), 'candidate_run': str(a.candidate_run),
        'reference_dir': str(a.reference_dir), 'hosts': hosts, 'same_host': same_host,
        'counts': counts, 'results': {}}
    if not same_host and not a.allow_cross_host:
        result['valid'] = False
        result['error'] = ('arms ran on different hosts (%s vs %s); the cross-node '
                           'reproduction floor is coarser than the effect' % (hosts['control'],
                                                                              hosts['candidate']))
        a.output.write_text(json.dumps(result, indent=2))
        raise SystemExit(result['error'])

    for count in counts:
        stem = a.output.parent / ('psnr_%d' % count)
        control = per_video(run_screen(a.reference_dir, a.control_run / 'videos', count,
                                       stem.with_name(stem.name + '_control_vs_dense.json')))
        candidate = per_video(run_screen(a.reference_dir, a.candidate_run / 'videos', count,
                                         stem.with_name(stem.name + '_candidate_vs_dense.json')))
        head = head_to_head(a.control_run / 'videos', a.candidate_run / 'videos', count,
                            stem.with_name(stem.name + '_candidate_vs_control.json'))
        head_psnr = per_video(head)
        finite = [value for value in head_psnr.values() if value != float('inf')]
        row = {
            'control_vs_dense_mean_db': statistics.fmean(control.values()),
            'candidate_vs_dense_mean_db': statistics.fmean(candidate.values()),
            'paired_candidate_minus_control_db': paired(control, candidate),
            'candidate_vs_control_direct': {
                'per_video_psnr_db': head_psnr,
                'mean_db': statistics.fmean(finite) if finite else float('inf'),
                'min_db': min(finite) if finite else float('inf'),
                'identical_videos': sum(1 for row_ in head['videos']
                                        if row_['all_frames_identical'])}}
        result['results'][str(count)] = row
        print('COUNT', count, json.dumps({
            'control_vs_dense_mean_db': round(row['control_vs_dense_mean_db'], 4),
            'candidate_vs_dense_mean_db': round(row['candidate_vs_dense_mean_db'], 4),
            'paired_mean_delta_db': round(row['paired_candidate_minus_control_db']['mean_delta_db'], 4),
            't': round(row['paired_candidate_minus_control_db']['t'], 3),
            'max_abs_delta_db': round(row['paired_candidate_minus_control_db']['max_abs_delta_db'], 4),
            'candidate_vs_control_mean_db': round(row['candidate_vs_control_direct']['mean_db'], 3),
            'candidate_vs_control_min_db': round(row['candidate_vs_control_direct']['min_db'], 3),
        }), flush=True)
        a.output.write_text(json.dumps(result, indent=2))

    result['valid'] = True
    print('SUMMARY_DONE', flush=True)
    a.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
