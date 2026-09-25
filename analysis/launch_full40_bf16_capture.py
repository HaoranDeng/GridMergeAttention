"""One-clean-H200 guard for a fresh original-BF16 external dense capture."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from analysis import launch_quality_first_videos as guard

WORKER='analysis/capture_full40_bf16_diagnostic.py'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def child_environment(output,uuid):
    env=guard.child_environment(output,uuid)
    # The external reference uses canonical native RMSNorm, not the
    # quality-first candidates' Torch RMSNorm numerical implementation.
    env.pop('TOKEN_MERGING_FORCE_TORCH_RMSNORM',None)
    env['FULL40_CAPTURE_PARENT']=str(os.getpid())
    return env


def verify_complete(output):
    root=Path(output)/'captures'
    report=json.loads((root/'capture_manifest.json').read_text())
    if not (report.get('complete') and report.get('total_self_attention_calls')==480
            and report.get('total_captures')==24 and len(report['records'])==480
            and len(report['captures'])==24 and (root/'exit_code').read_text().strip()=='0'
            and report.get('checkpoint_weights')=='raw_state_dict'
            and report.get('actual_forced_sdpa_backend')=='flash'
            and report.get('actual_use_torch_rmsnorm') is False):
        raise ValueError('require all480 dense calls and24 original full40 captures')
    expected=[f'prompt{p:03d}_call{c:03d}_layer{c%40}.pt'
              for p in range(3) for c in (0,33,40,73,80,113,120,153)]
    if sorted(p.name for p in root.glob('*.pt'))!=sorted(expected):
        raise ValueError('capture filename coverage differs')
    if sorted(row['capture_file'] for row in report['captures'])!=sorted(expected):
        raise ValueError('capture metadata coverage differs')
    ordered=[(row['prompt_index'],row['call_index'],row['layer_index'],row['step_index'])
             for row in report['records']]
    if ordered!=[(p,c,c%40,c//40) for p in range(3) for c in range(160)]:
        raise ValueError('dense call order differs')
    for row in report['captures']:
        name=row['capture_file']
        if name not in expected: raise ValueError('unexpected capture file')
        path=root/name
        if (row.get('dtype')!='torch.bfloat16' or row.get('shape')!=[1,75600,40,128]
                or row.get('heads')!=list(range(40)) or path.stat().st_size!=row['bytes']
                or sha(path)!=row['sha256']):
            raise ValueError('original-BF16 capture shape/dtype/identity mismatch')
    return {'manifest_sha256':sha(root/'capture_manifest.json'),
            'captures':[{k:row[k] for k in ('capture_file','bytes','sha256')}
                        for row in report['captures']]}


def launch(args):
    output=args.output.resolve(); source=args.source_root.resolve()
    if output.exists(): raise ValueError('fresh output required; no overwrite/resume/queue')
    initial=guard.require_clean(args.gpu)
    worker=source/WORKER
    spec=importlib.util.spec_from_file_location('full40_capture_definition',worker)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    paths=[source/name for name in module.SOURCES]
    paths += [Path(__file__).resolve(),args.config.resolve(),args.prompt_path.resolve()]
    source_hashes={str(path):sha(path) for path in paths}
    output.mkdir(parents=True)
    write(output/'preflight.json',{'initial_gpu':initial,'source_hashes':source_hashes,
        'worker':str(worker),'checkpoint':str(args.checkpoint.resolve()),
        'scope':'External dense reference capture only, no candidate or video-quality result'})
    env=child_environment(output,initial['gpu']['uuid'])
    process=None;failure=None;foreign=[];verified=None
    def interrupted(*unused): raise KeyboardInterrupt('capture guard interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        immediate=guard.require_clean(args.gpu,expected_uuid=initial['gpu']['uuid'])
        with (output/'worker.log').open('x') as log,(output/'gpu_monitor.jsonl').open('x') as monitor:
            command=[sys.executable,'-u',str(worker),'--config',str(args.config.resolve()),
                '--checkpoint',str(args.checkpoint.resolve()),'--prompt_path',str(args.prompt_path.resolve()),
                '--output-root',str(output/'captures')]
            process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(output/'process.json',{'parent_pid':os.getpid(),'worker_pid':process.pid,
                'immediate_gpu':immediate,'command':command,'spawned_utc':guard.now()})
            while True:
                active=guard.gpu_processes(initial['gpu']['uuid'])
                foreign=[row for row in active if row['pid']!=process.pid]
                monitor.write(json.dumps({'utc':guard.now(),'processes':active,'foreign':foreign})+'\n')
                monitor.flush()
                if foreign: raise RuntimeError('foreign GPU process; capture invalid')
                if process.poll() is not None: break
                time.sleep(5)
            if process.returncode!=0: raise RuntimeError(f'capture worker exit {process.returncode}')
        if any(sha(path)!=digest for path,digest in source_hashes.items()):
            raise ValueError('capture source/config/prompt changed')
        verified=verify_complete(output)
    except BaseException as error: failure=f'{type(error).__name__}: {error}'
    finally: guard.stop_own_child(process)
    receipt={'valid':failure is None,'failure':failure,'foreign':foreign,
        'worker_exit_code':process.returncode if process else None,'completed_utc':guard.now(),
        'verified_artifacts':verified,'timing_claim':None}
    write(output/'completion.json',receipt)
    if failure: raise RuntimeError(failure)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,required=True)
    for key in ('source-root','output','config','checkpoint','prompt_path'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(launch(args),indent=2))


if __name__=='__main__':main()
