"""Only whitelisted public metrics and committed videos; no environment dumps."""
import argparse
import errno
import fcntl
import hashlib
import json
from pathlib import Path
import tarfile
import time
from worker import read,save,sha
from status import status
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--known',type=Path);a=p.parse_args()
    lock=(ROOT/'export.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    plan=read(ROOT/'experiment_plan.json');known=read(a.known) if a.known and a.known.exists() else {}
    save(ROOT/'public_status.json',status());files={}
    for name in ('experiment_plan.json','public_status.json','preflight.json','native_gate.json','stage.json','launch.json','evaluation_launch.json','completion.json','failure.json','quality_status.json','quality_failure.json','generation_complete.json','borrow_status.json','restoration.json','restoration_verified.json'):
        path=ROOT/name
        if path.exists():files[name]=path
    for name in ('preflight','native_gate','stage','launch','completion','failure','batch_complete'):
        path=ROOT/f'gpu7/{name}.json'
        if path.exists():files[f'gpu7/{name}.json']=path
    for control in ('scai6','scai6_gpu2','scai6_gpu5'):
        for name in ('preflight','native_gate','stage','launch','completion','failure','supervisor_status','generation_complete'):
            path=ROOT/control/f'{name}.json'
            if path.exists():files[str(path.relative_to(ROOT))]=path
    for gpu in (2,5):
        for name in ('runtime','purity_records','input_manifest'):
            path=ROOT/f'C6/shards/scai6_gpu{gpu}/video20/{name}.json'
            if path.exists():files[str(path.relative_to(ROOT))]=path
    recovery=ROOT/'recovery_history/scai6_c6_cuda_cache_20260924'
    for relative in ('repair_receipt.json','C6_claim.json','scai6/completion.json',
                     'scai6_gpu2/failure.json','scai6_gpu2/completion.json',
                     'scai6_gpu5/failure.json','scai6_gpu5/completion.json'):
        path=recovery/relative
        if path.exists():files[str(path.relative_to(ROOT))]=path
    recovery=ROOT/'recovery_history/quality_nfs_20260924'
    for name in ('repair_receipt.json','quality_failure.json','evaluation_launch.json'):
        path=recovery/name
        if path.exists():files[str(path.relative_to(ROOT))]=path
    for path in (ROOT/'claims').glob('C*.json'):files[str(path.relative_to(ROOT))]=path
    vb=ROOT/'vbench20_20260924'
    for relative in ('plan.json','spawn_gpu2.json','repair_receipt.json','gpu2/launch.json','gpu2/stage.json','gpu2/completion.json','gpu2/failure.json',
                     'recovery_history/missing_unzip_20260924/gpu2/failure.json',
                     'recovery_history/missing_unzip_20260924/gpu2/completion.json'):
        path=vb/relative
        if path.exists():files[str(path.relative_to(ROOT))]=path
    for method in ('C4','C6','C8','SVG_EAR_p08'):
        path=vb/method/'summary.json'
        if path.exists():files[str(path.relative_to(ROOT))]=path
        for pattern in ('*_eval_results.json','*_full_info.json'):
            for path in (vb/method/'out').glob(pattern):
                # VBench timestamps contain colons, which Windows cannot store.
                files[str(path.relative_to(ROOT)).replace(':','-')]=path
    for c in (4,6,8,10):
        method=Path(plan['existing_c8']['root']) if c==8 else ROOT/f'C{c}/video20'
        rt=Path(plan['existing_c8']['runtime']) if c==8 else method/'runtime.json'
        if rt.exists():
            # Snapshot external C8's changing runtime before archiving.
            save(ROOT/f'public_snapshots/C{c}_runtime.json',read(rt))
            snap=ROOT/f'public_snapshots/C{c}_runtime.json';files[f'C{c}/video20/runtime.json']=snap
            for row in read(snap)['records']:
                if row['prompt_index']>=20:continue
                name=row['output'];assert name==f"{row['prompt_index']:03d}-00.mp4"
                video=method/name
                if video.is_file():files[f'C{c}/video20/{name}']=video
        for relative in ('input_manifest.json','purity_records.json'):
            p=method/relative
            if p.exists():files[f'C{c}/video20/{relative}']=p
        for p in (ROOT/f'C{c}/quality_cpu').glob('*.json'):files[str(p.relative_to(ROOT))]=p
        p=ROOT/f'C{c}/generation_complete.json'
        if p.exists():files[str(p.relative_to(ROOT))]=p
    # Freeze changing JSON before hashing/archiving; videos are committed atomically.
    for name,path in list(files.items()):
        if path.suffix!='.json':continue
        for attempt in range(20):
            try:
                data=path.read_bytes()
                json.loads(data)
                break
            except json.JSONDecodeError:
                if attempt==19:raise
                time.sleep(.2)
            except OSError as e:
                # Shared NFS files are atomically replaced by live supervisors.
                # Retry transient lookup races without hiding persistent errors.
                if e.errno not in (errno.EEXIST,errno.ESTALE,errno.ENOENT,errno.EAGAIN) or attempt==19:raise
                time.sleep(.2)
        target=ROOT/'public_snapshots/export'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data);files[name]=target
    manifest={name:sha(path) for name,path in files.items()}
    changed=[name for name,h in manifest.items() if known.get(name)!=h]
    save(ROOT/'public_hashes.json',manifest)
    archive=ROOT/'public_delta.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for name in changed:tar.add(files[name],arcname=name,recursive=False)
        tar.add(ROOT/'public_hashes.json',arcname='public_hashes.json',recursive=False)
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),changed=len(changed),files=len(manifest),time=time.time())))
if __name__=='__main__':main()
