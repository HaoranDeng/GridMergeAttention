"""CPU full81-frame metrics against the frozen historical Dense videos."""
import fcntl
import importlib.util
import math
import os
from pathlib import Path
import time
import traceback
from common import ROOT,read,save,sha

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    lock=(ROOT/'quality.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(ROOT/'experiment_plan.json');source=Path(plan['snapshot'])/'analysis/score_videos_diffusers.py'
    assert sha(source)==plan['source_sha256']['analysis/score_videos_diffusers.py']
    import torch,lpips
    torch.set_num_threads(4);torch.set_num_interop_threads(1);assert not torch.cuda.is_available()
    spec=importlib.util.spec_from_file_location('frozen_score',source);metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    network=lpips.LPIPS(net='alex').to('cpu').eval();out=ROOT/'quality_cpu';out.mkdir(exist_ok=True)
    started=time.monotonic()
    while time.monotonic()-started<3*86400:
        done={int(p.stem) for p in out.glob('[0-9][0-9][0-9].json')}
        if len(done)==20:return
        rt=ROOT/'video20/runtime.json';records=read(rt)['videos'] if rt.exists() else []
        ready=[r for r in records if r['index'] not in done]
        if not ready:
            latest=max(ROOT.glob('scai*_gpu*/launch.json'),key=lambda p:read(p)['time'],default=None)
            if latest and (latest.parent/'completion.json').exists():raise RuntimeError('Generation stopped before all20 videos')
            time.sleep(15);continue
        record=ready[0];i=record['index'];item=plan['inputs'][i]
        assert record['seed']==0 and record['prompt']==item['prompt']
        assert record['attention_calls']==record['full_attention_calls']==4000
        method=ROOT/'video20'/record['video'];digest=sha(method)
        assert sha(item['dense_video'])==item['dense_sha256']
        row=metrics.score_pair(Path(item['dense_video']),method,network,'cpu')
        assert row['frames']==81 and all(math.isfinite(row[k]) for k in ('psnr','ssim','lpips'))
        assert sha(method)==digest
        row.update(index=i,seed=0,prompt_sha256=item['prompt_sha256'],video_sha256=digest,dense_sha256=item['dense_sha256'])
        save(out/f'{i:03d}.json',row);rows=[read(p) for p in sorted(out.glob('[0-9][0-9][0-9].json'))]
        save(out/'quality.json',dict(complete=len(rows)==20,summary=metrics.aggregate(rows),videos=rows))
        print(f'121 T2V {i:03d}: PSNR={row["psnr"]:.4f}, SSIM={row["ssim"]:.4f}, LPIPS={row["lpips"]:.4f}',flush=True)
    raise TimeoutError('Bounded three-day evaluator')
if __name__=='__main__':
    try:main()
    except BaseException:save(ROOT/'quality_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
