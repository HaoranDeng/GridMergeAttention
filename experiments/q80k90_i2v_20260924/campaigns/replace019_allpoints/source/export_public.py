"""Incremental whitelist export; no environments, caches or extracted frames."""
import argparse
import fcntl
import json
import tarfile
import time
from worker import ROOT,read,save,sha
from status import status

def main():
    p=argparse.ArgumentParser();p.add_argument('--known');a=p.parse_args()
    lock=(ROOT/'export.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    from pathlib import Path
    known=read(Path(a.known)) if a.known and Path(a.known).exists() else {}
    save(ROOT/'public_status.json',status());plan=read(ROOT/'experiment_plan.json');files={}
    for relative in ['experiment_plan.json','public_status.json','quality.json','evaluation_launch.json','generation_complete.json',
                     'evaluation_complete.json','evaluation_waiting.json','generation_failure.json','quality_failure.json','reused_quality.json',
                     'rollover_gpu2.json','rollover_gpu2_failure.json',
                     'gpu2/preflight.json','gpu2/launch.json','gpu2/stage.json','gpu2/completion.json','gpu2/native_gate.json']:
        p=ROOT/relative
        if p.exists():files[relative]=p
    for slot in plan.get('gpus',{}):
        for name in ['preflight','launch','stage','completion','native_gate','generation_failure','generation_complete',
                     'borrow_status','restoration','restoration_verified']:
            relative=f'{slot}/{name}.json';p=ROOT/relative
            if p.exists():files[relative]=p
    for method in ['Dense',*plan['methods']]:
        for relative in ['generation_complete.json','video20/input_manifest.json','video20/runtime.json','video20/purity_records.json','video20/flops50_019.json']:
            p=ROOT/method/relative
            if p.exists():files[method+'/'+relative]=p
        p=ROOT/method/'video20/019-00.mp4';proof=ROOT/method/'video20/purity_records.json';runtime=ROOT/method/'video20/runtime.json'
        if p.exists() and proof.exists() and runtime.exists():
            row,=read(proof);rt,=read(runtime)['records'];assert row['index']==rt['prompt_index']==19
            assert row['video_sha256']==sha(p)
            files[method+'/video20/019-00.mp4']=p
    for p in (ROOT/'claims').glob('*.json'):files[str(p.relative_to(ROOT))]=p
    for name,p in list(files.items()):
        if p.suffix=='.json':
            obj=read(p);target=ROOT/'public_snapshots/export'/name;save(target,obj);files[name]=target
        else:files[name]=p.resolve()
    hashes={k:sha(v) for k,v in files.items()};changed=[k for k in hashes if known.get(k)!=hashes[k]]
    save(ROOT/'public_hashes.json',hashes);archive=ROOT/'public_delta.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for name in changed:tar.add(files[name],arcname=name,recursive=False)
        tar.add(ROOT/'public_hashes.json',arcname='public_hashes.json',recursive=False)
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),changed=len(changed),files=len(files),time=time.time())))

if __name__=='__main__':main()
