"""Export only whitelisted public data and committed videos, with hashes."""
import argparse
import fcntl
import json
from pathlib import Path
import tarfile
import time
from worker import ROOT,read,save,sha
from status import status

def main():
    p=argparse.ArgumentParser();p.add_argument('--known',type=Path);a=p.parse_args()
    lock=(ROOT/'export.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    known=read(a.known) if a.known and a.known.exists() else {}
    save(ROOT/'public_status.json',status());files={}
    for name in ['experiment_plan','public_status','preflight','prelaunch_validation','native_gate','stage','launch','evaluation_launch','completion',
                 'quality','quality_failure','generation_failure','generation_complete','evaluation_complete','borrow_status','restoration','restoration_verified']:
        path=ROOT/(name+'.json')
        if path.exists():files[path.name]=path
    for method in ['Dense','C4','C8','EAR_p08']:
        for rel in ['generation_complete.json','video20/input_manifest.json','video20/runtime.json','video20/purity_records.json','video20/flops50_019.json']:
            p=ROOT/method/rel
            if p.exists():files[str(p.relative_to(ROOT))]=p
        done=ROOT/method/'generation_complete.json'
        if done.exists():
            v=ROOT/method/'video20/019-00.mp4'
            assert sha(v)==read(v.parent/'purity_records.json')[0]['video_sha256']
            files[str(v.relative_to(ROOT))]=v
    for name,path in list(files.items()):
        if path.suffix=='.json':
            tmp=ROOT/'public_snapshots'/name;save(tmp,read(path));files[name]=tmp
    manifest={name:sha(path) for name,path in files.items()}
    changed=[name for name,h in manifest.items() if known.get(name)!=h]
    save(ROOT/'public_hashes.json',manifest);archive=ROOT/'public_delta.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for name in changed:tar.add(files[name],arcname=name,recursive=False)
        tar.add(ROOT/'public_hashes.json',arcname='public_hashes.json',recursive=False)
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),changed=len(changed),files=len(manifest),time=time.time())))

if __name__=='__main__':main()
