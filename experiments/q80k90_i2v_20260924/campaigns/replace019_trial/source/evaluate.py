"""Frozen full81-frame CPU evaluator; no frames are exported to a model."""
import importlib.util
import fcntl
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
    import torch
    import lpips
    assert not torch.cuda.is_available()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    spec=importlib.util.spec_from_file_location('frozen_score',source);score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
    metric=lpips.LPIPS(net='alex').to('cpu');metric.eval()
    ref=ROOT/'Dense/video20';refmanifest=read(ref/'input_manifest.json')
    results={}
    for method in ['C4','C8','EAR_p08']:
        if not (ROOT/method/'generation_complete.json').exists():continue
        out=ROOT/method/'video20';assert read(out/'input_manifest.json')==refmanifest
        for folder in [ref,out]:
            record=read(folder/'purity_records.json')[0]
            assert record['video_sha256']==sha(folder/'019-00.mp4')
        row=score.score_pair(ref/'019-00.mp4',out/'019-00.mp4',metric,torch.device('cpu'))
        assert row['frames']==81 and all(math.isfinite(row[k]) for k in ['psnr','ssim','lpips']),row
        results[method]=dict(index=19,seed=19,source_sha256=sha(source),
                            dense_sha256=sha(ref/'019-00.mp4'),method_sha256=sha(out/'019-00.mp4'),
                            image_sha256=plan['input_manifest'][19]['image_sha256'],**row)
        save(ROOT/'quality.json',dict(complete=len(results)==3,device='cpu',frames='all81',results=results,time=time.time()))
        print(method,row,flush=True)
    assert len(results)==3,'Partial results preserved; inspect generation failure'
    save(ROOT/'evaluation_complete.json',dict(complete=True,methods=list(results),time=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException:
        save(ROOT/'quality_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
