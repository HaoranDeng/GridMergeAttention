"""CPU stream using the exact metric code used by the other I2V experiments."""
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from worker import read,save,sha

ROOT=Path(__file__).resolve().parents[1]

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1')
    lock=(ROOT/'quality.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(ROOT/'experiment_plan.json');snap=Path(plan['snapshot']);sys.path.insert(0,str(snap))
    import torch
    import lpips
    assert not torch.cuda.is_available()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    path=snap/'analysis/score_videos_diffusers.py'
    assert sha(path)==plan['source_sha256']['analysis/score_videos_diffusers.py']
    spec=importlib.util.spec_from_file_location('frozen_scores',path)
    metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
    model=lpips.LPIPS(net='alex').to('cpu').eval()
    start=time.monotonic();complete=False
    try:
        while time.monotonic()-start<96*3600:
            ready=[];counts={}
            for cap in (4,6,8,10):
                output=ROOT/f'C{cap}/quality_cpu';output.mkdir(parents=True,exist_ok=True)
                results=[read(p) for p in sorted(output.glob('[0-9][0-9][0-9].json'))]
                counts[str(cap)]=len(results)
                done={r['index'] for r in results}
                method=Path(plan['existing_c8']['root']) if cap==8 else ROOT/f'C{cap}/video20'
                rt=Path(plan['existing_c8']['runtime']) if cap==8 else method/'runtime.json'
                if not rt.exists():continue
                try:records=read(rt)['records']
                except (json.JSONDecodeError,KeyError):continue
                for row in records:
                    i=row['prompt_index']
                    if i>=20 or i in done:continue
                    assert row['attention_calls']==row['full_attention_calls']==4000
                    name=f'{i:03d}-00.mp4';assert row['output']==name
                    if not (method/name).is_file():continue
                    if cap!=8:
                        purity=read(method/'purity_records.json')
                        proof,=[p for p in purity if p['video']==name]
                        assert proof['attention_calls']==4000 and proof['dense_self_attention_calls']==0
                    ready.append((cap,i,method/name))
            save(ROOT/'quality_status.json',dict(counts=counts,complete=all(v==20 for v in counts.values()),time=time.time()))
            if all(v==20 for v in counts.values()):complete=True;break
            if not ready:
                if (ROOT/'failure.json').exists():raise RuntimeError('Generation failure; preserve evaluated rows')
                time.sleep(10);continue
            # Evaluate one completed item per pass, independent of GPU method order.
            cap,i,method=sorted(ready,key=lambda x:(x[1],x[0]))[0]
            item=plan['input_manifest'][i];ref=Path(item['dense_video'])
            assert sha(ref)==item['dense_sha256'] and sha(Path(item['image']))==item['image_sha256']
            before=sha(method)
            save(ROOT/'quality_status.json',dict(counts=counts,capacity=cap,index=i,stage='evaluating',time=time.time()))
            t=time.perf_counter();row=metric.score_pair(ref,method,model,'cpu')
            assert row['frames']==81 and all(math.isfinite(row[k]) for k in ('psnr','ssim','lpips'))
            assert sha(method)==before
            row.update(index=i,seed=i,evaluation_seconds=time.perf_counter()-t,method_sha256=before,dense_sha256=item['dense_sha256'],image_sha256=item['image_sha256'])
            dest=ROOT/f'C{cap}/quality_cpu';save(dest/f'{i:03d}.json',row)
            rows=[read(p) for p in sorted(dest.glob('[0-9][0-9][0-9].json'))]
            save(dest/'quality.json',dict(complete=len(rows)==20,device='cpu',source_sha256=sha(path),videos=rows,summary=metric.aggregate(rows)))
            print(json.dumps({'capacity':cap,**row}),flush=True)
        assert complete,'96-hour evaluation deadline'
    except BaseException:
        save(ROOT/'quality_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
    save(ROOT/'quality_status.json',dict(complete=True,counts={str(c):20 for c in (4,6,8,10)},time=time.time()))

if __name__=='__main__':main()
