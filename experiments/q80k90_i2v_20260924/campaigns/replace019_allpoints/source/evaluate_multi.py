"""Stream frozen all81-frame CPU quality as each new019 method commits."""
import fcntl
import importlib.util
import math
import os
from pathlib import Path
import time
import traceback
from worker import ROOT,read,save,sha

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    lock=(ROOT/'quality.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(ROOT/'experiment_plan.json');source=Path(plan['snapshot'])/'analysis/score_videos_diffusers.py'
    assert sha(source)==plan['source_sha256']['analysis/score_videos_diffusers.py']
    import torch,lpips
    assert not torch.cuda.is_available();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    spec=importlib.util.spec_from_file_location('frozen_score',source);score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
    metric=lpips.LPIPS(net='alex').to('cpu');metric.eval()
    ref=ROOT/'Dense/video20';refmanifest=read(ref/'input_manifest.json')['inputs'];dense_hash=sha(ref/'019-00.mp4')
    results=read(ROOT/'quality.json')['results'] if (ROOT/'quality.json').exists() else read(ROOT/'reused_quality.json');assert len(results)>=3
    for method,row in results.items():
        assert row['dense_sha256']==dense_hash and row['method_sha256']==sha(ROOT/method/'video20/019-00.mp4')
        assert row['source_sha256']==sha(source) and row['frames']==81 and row['seed']==19
    started=time.monotonic()
    while True:
        for method in plan['methods']:
            if method in results or not (ROOT/method/'generation_complete.json').exists():continue
            out=ROOT/method/'video20';assert read(out/'input_manifest.json')['inputs']==refmanifest
            proof,=read(out/'purity_records.json');assert proof['video_sha256']==sha(out/'019-00.mp4')
            row=score.score_pair(ref/'019-00.mp4',out/'019-00.mp4',metric,torch.device('cpu'))
            assert row['frames']==81 and all(math.isfinite(row[k]) for k in ['psnr','ssim','lpips'])
            results[method]=dict(index=19,seed=19,source_sha256=sha(source),dense_sha256=dense_hash,
                method_sha256=proof['video_sha256'],image_sha256=plan['input_manifest'][19]['image_sha256'],**row)
            print(method,{k:row[k] for k in ['psnr','ssim','lpips']},flush=True)
        save(ROOT/'quality.json',dict(complete=len(results)==18,device='cpu',frames='all81',results=results,time=time.time()))
        if len(results)==18:
            save(ROOT/'evaluation_complete.json',dict(complete=True,methods=list(results),time=time.time()));return
        controls=[ROOT/slot/'completion.json' for slot in plan['gpus']]
        if all(p.exists() for p in controls):
            save(ROOT/'evaluation_waiting.json',dict(reason='Generation stopped before all methods; partial quality preserved',time=time.time()));return
        assert time.monotonic()-started<13*3600,'Bounded evaluator deadline'
        time.sleep(15)

if __name__=='__main__':
    try:main()
    except BaseException:save(ROOT/'quality_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
