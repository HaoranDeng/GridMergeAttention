"""Stream frozen CPU metrics, preserving all81 frames and matching Dense hashes."""
import fcntl
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import traceback
from common import ROOT, read, save, sha


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    lock = (ROOT/'quality.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(ROOT/'experiment_plan.json')
    snap = Path(plan['snapshot'])
    sys.path.insert(0, str(snap))
    import torch
    import lpips
    assert not torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    path = snap/'analysis/score_videos_diffusers.py'
    assert sha(path) == plan['source_sha256']['analysis/score_videos_diffusers.py']
    spec = importlib.util.spec_from_file_location('frozen_metrics', path)
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    model = lpips.LPIPS(net='alex').to('cpu').eval()
    out = ROOT/'svg_ear_p04/quality_cpu'
    out.mkdir(parents=True, exist_ok=True)
    video = ROOT/'svg_ear_p04/video20'
    while True:
        rows = [read(p) for p in sorted(out.glob('[0-9][0-9][0-9].json'))]
        done = {r['index'] for r in rows}
        save(ROOT/'quality_status.json', dict(evaluated=len(rows), complete=len(rows)==20, time=time.time()))
        if len(rows) == 20:
            return
        records = read(video/'runtime.json')['records'] if (video/'runtime.json').exists() else []
        ready = [r for r in records if r['prompt_index'] not in done]
        if not ready:
            if (ROOT/'failure.json').exists():
                raise RuntimeError('Generation stopped; completed quality rows preserved')
            time.sleep(15)
            continue
        record = ready[0]
        i = record['prompt_index']
        item = plan['input_manifest'][i]
        assert record['output'] == f'{i:03d}-00.mp4'
        proof, = [r for r in read(video/'purity_records.json') if r['index'] == i]
        assert proof['attention_calls'] == 4000 and proof['layer_calls'] == [100]*40
        assert proof['prompt_sha256'] == item['prompt_sha256']
        assert sha(item['dense_video']) == item['dense_sha256']
        assert sha(item['image']) == item['image_sha256']
        method = video/record['output']
        assert sha(method) == proof['video_sha256']
        started = time.monotonic()
        row = metric.score_pair(Path(item['dense_video']), method, model, 'cpu')
        assert row['frames'] == 81 and all(math.isfinite(row[k]) for k in ('psnr', 'ssim', 'lpips'))
        assert sha(method) == proof['video_sha256']
        row.update(index=i, seed=i, evaluation_seconds=time.monotonic()-started,
                   method_sha256=proof['video_sha256'], dense_sha256=item['dense_sha256'],
                   image_sha256=item['image_sha256'], prompt_sha256=item['prompt_sha256'])
        save(out/f'{i:03d}.json', row)
        rows.append(row)
        save(out/'quality.json', dict(complete=len(rows)==20, device='cpu', source_sha256=sha(path),
                                     videos=sorted(rows, key=lambda r:r['index']), summary=metric.aggregate(rows)))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        save(ROOT/'quality_failure.json', dict(error=traceback.format_exc(), time=time.time()))
        raise
