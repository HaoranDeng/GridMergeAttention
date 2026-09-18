"""Guarded full-resolution PSNR/SSIM/LPIPS for one completed canonical run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REFERENCE='/local2/denghaoran/GridMergeAttention/experiments/wan14b_720p_4step_dense_flash_matched_extended_ema_20260826/flash/videos'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def inventory(args):
    from analysis.quality_first_mode_contract import method_contract
    from analysis.evaluate_video_quality import prepare_comparison
    completion=json.loads((args.run_dir/'completion.json').read_text())
    audit=json.loads((args.run_dir/'dispatch_audit.json').read_text())
    runtime=json.loads((args.run_dir/'runtime.json').read_text())
    if not (completion['valid'] and not completion['foreign'] and completion['worker_exit_code']==0
            and audit['complete'] and audit['core_calls']==3200
            and audit['per_layer_calls']==[80]*40 and audit['declaration']==method_contract(args.method)
            and len(runtime['records'])==20):
        raise ValueError('not a valid complete20 all-layer generation')
    pairs,contract=prepare_comparison(REFERENCE,args.run_dir/'videos',expected_videos=20)
    if [p.name for p,_ in pairs]!=[f'{i:03d}-00.mp4' for i in range(20)]:
        raise ValueError('noncanonical filenames')
    paths=[p for pair in pairs for p in pair]
    paths += [args.run_dir/name for name in ('completion.json','dispatch_audit.json','runtime.json',
                                           'source_provenance.json','videos/prompt_manifest.json')]
    paths += [Path(REFERENCE)/'prompt_manifest.json']
    return pairs,contract,{str(p):sha(p) for p in paths}


def worker(args):
    if os.environ.get('SEED_EVAL_PARENT')!=str(os.getppid()):
        raise ValueError('requires monitoring parent')
    import torch
    import lpips
    from analysis import evaluate_video_quality as quality
    from analysis.run_quality_first_evaluation import full_resolution_decoder,validate_video_metric
    torch.set_num_threads(4)
    pairs,contract,inputs=inventory(args)
    prepared=json.loads((args.output/'preflight.json').read_text())
    if inputs!=prepared['input_hashes']: raise ValueError('input changed before evaluation')
    model=lpips.LPIPS(net='alex').to('cuda').eval()
    result={'complete':False,'method':args.method,'prompt_contract':contract,'videos':[],
            'reference':REFERENCE,'input_hashes':inputs,'vbench':None,
            'mean_attention_ms':None,'speedup_vs_sdpa':None,
            'protocol':{'frames':81,'resolution':[720,1280],'resize':False,'crop':False,
                        'psnr':'framewise RGB, data_range255','ssim':'skimage channel_axis2, data_range255',
                        'lpips':'AlexNet, full resolution, RGB /127.5 - 1','lpips_batch_size':8},
            'versions':{x:importlib.metadata.version(x) for x in ('torch','lpips','scikit-image')}}
    with torch.inference_mode(),full_resolution_decoder(quality):
        for index,(reference,method) in enumerate(pairs):
            row=quality.evaluate_pair(reference,method,model,8,'cuda',expected_frames=81)
            validate_video_metric(row,index)
            result['videos'].append(row)
            result['summary']=quality.aggregate(result['videos'])
            write(args.output/'results.json',result)
            print(json.dumps(row),flush=True)
    if inventory(args)[2]!=inputs: raise ValueError('input changed during evaluation')
    result['complete']=True
    write(args.output/'results.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--method',required=True)
    for name in ('source-root','run-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.source_root))
    from analysis import launch_quality_first_videos as guard
    if args.worker:return worker(args)
    initial=guard.require_clean(args.gpu)
    if args.output.exists():raise ValueError('fresh output required')
    _,contract,inputs=inventory(args)
    paths=[Path(__file__),*(args.source_root/x for x in (
        'analysis/evaluate_video_quality.py','analysis/run_quality_first_evaluation.py',
        'analysis/launch_quality_first_videos.py','analysis/quality_first_mode_contract.py',
        'utils/prompt_contract.py'))]
    sources={str(path):sha(path) for path in paths}
    if sources[str(args.source_root/'analysis/evaluate_video_quality.py')]!='ea28e3c629165ef09a123f8145b8701996ea69daed7797bcfc14068910e717bb':
        raise ValueError('metric source differs from established protocol')
    args.output.mkdir(parents=True)
    write(args.output/'preflight.json',{'initial_gpu':initial,'input_hashes':inputs,
                                      'source_hashes':sources,'prompt_contract':contract})
    process=None;failure=None;foreign=[]
    env=guard.child_environment(args.output,initial['gpu']['uuid'])
    env['SEED_EVAL_PARENT']=str(os.getpid())
    def interrupted(*unused):raise KeyboardInterrupt('parent interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        guard.require_clean(args.gpu,expected_uuid=initial['gpu']['uuid'])
        with (args.output/'worker.log').open('x') as log,(args.output/'gpu_monitor.jsonl').open('x') as monitor:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                                      '--worker',*sys.argv[1:]],env=env,stdout=log,stderr=subprocess.STDOUT)
            write(args.output/'process.json',{'parent_pid':os.getpid(),'worker_pid':process.pid,
                                            'gpu':initial['gpu']})
            while True:
                active=guard.gpu_processes(initial['gpu']['uuid'])
                foreign=[x for x in active if x['pid']!=process.pid]
                monitor.write(json.dumps({'utc':guard.now(),'processes':active,'foreign':foreign})+'\n');monitor.flush()
                if foreign:raise RuntimeError('foreign process on evaluation GPU')
                if process.poll() is not None:break
                time.sleep(5)
            if process.returncode!=0:raise RuntimeError(f'worker exit {process.returncode}')
        result=json.loads((args.output/'results.json').read_text())
        if not result['complete'] or len(result['videos'])!=20:raise ValueError('incomplete metrics')
        if any(sha(path)!=h for path,h in sources.items()):raise ValueError('source changed')
    except BaseException as error:failure=f'{type(error).__name__}: {error}'
    finally:guard.stop_own_child(process)
    write(args.output/'completion.json',{'valid':failure is None,'failure':failure,'foreign':foreign,
                                        'worker_exit_code':process.returncode if process else None,
                                        'completed_utc':guard.now(),'mean_attention_ms':None,'speedup_vs_sdpa':None})
    if failure:raise RuntimeError(failure)


if __name__=='__main__':main()
