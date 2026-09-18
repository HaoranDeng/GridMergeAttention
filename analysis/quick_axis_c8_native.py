"""Matched native C4/C8-r2 replay supplement to the exact-C4 axis screen."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

from quick_axis_c4_qkv import gpu_state, save, sha, GRID, SAMPLES

FULL = Path('/local2/denghaoran/GridMergeAttention/experiments/full40_bf16_dense_capture_20260905bb/captures')
SLICE = Path('/local2/denghaoran/GridMergeAttention/experiments/rf0_layer33_dense_capture_20260904n/captures')


def worker(args):
    if os.environ.get('AXIS_C8_PARENT') != str(os.getppid()):
        raise RuntimeError('requires monitored parent')
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0,str(args.repo))
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from kernels.best import create_c4, create_c8
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    report={'scope':'Native production H40 C4 and default C8 r2 on the same QKV as the axis screen',
            'grid':GRID,'samples':SAMPLES,'rows':[],'timings':[],'captures':[],
            'shape':[1,21,45,80,40,128],'capacity':{'c4_native':4,'c8_r2':8},
            'singleton_count':0,'source_hashes':{},'full_call_timing_repeats':3,
            'selector_note':'C4 production Fisher RF0; C8 production default Hellinger RF0. Not a selector-controlled capacity ablation.'}
    for package in ('c4','c8_r2'):
        manifest=json.loads((args.repo/'kernels/best'/package/'manifest.json').read_text())
        for name in manifest['files']:
            path=args.repo/'kernels/best'/package/name
            report['source_hashes'][str(path.relative_to(args.repo))]=sha(path)
    print('Preparing native C8 r2',flush=True)
    c8=create_c8()
    print('Preparing native C4',flush=True)
    c4=create_c4()
    coordinates=list(itertools.product(*SAMPLES))
    token_indices=torch.tensor([(t*45+h)*80+w for t,h,w in coordinates],device='cuda')
    heads=(0,10,20,30)
    head_indices=torch.tensor(heads,device='cuda')
    receipt=json.loads((FULL.parent/'completion.json').read_text())
    assert receipt['valid']
    pins={x['capture_file']:x for x in receipt['verified_artifacts']['captures']}
    def timed(fn):
        begin,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        begin.record(); value=fn(); end.record(); end.synchronize()
        return value,begin.elapsed_time(end)
    for prompt in range(3):
        for step in range(4):
            for layer in (0,33):
                call=40*step+layer
                path=FULL/f'prompt{prompt:03d}_call{call:03d}_layer{layer}.pt'
                assert path.stat().st_size==pins[path.name]['bytes']
                payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                meta=payload['metadata']
                assert tuple(payload['q'].shape)==(1,75600,40,128)
                assert meta['prompt_index']==prompt and meta['call_index']==call and meta['heads']==list(range(40))
                # The prior layer33 axis screen used an independent four-head
                # capture. Verify its actual Q/K/V bytes before pairing it.
                equal=True
                if layer==33:
                    sliced=torch.load(SLICE/path.name,map_location='cpu',weights_only=True,mmap=True)
                    for name in ('q','k','v'):
                        for slot,head in enumerate(heads):
                            equal &= torch.equal(payload[name][:,:,head],sliced[name][:,:,slot])
                    if not equal:raise RuntimeError('four-head and full40 QKV differ; cannot pair')
                    del sliced
                q,k,v=[payload[name].to('cuda') for name in ('q','k','v')]
                selected=[x[0,:,head_indices].float() for x in (q,k,v)]
                reference=[]
                for slot in range(4):
                    qh,kh,vh=[x[:,slot] for x in selected]
                    reference.append((qh[token_indices]@kh.T*(128**-.5)).softmax(-1)@vh)
                dense=torch.stack(reference,dim=1)
                funcs={'c8_r2':lambda:c8(q,k,v),
                       'c4_native':lambda:c4(q.view(1,*GRID,40,128),k.view(1,*GRID,40,128),v.view(1,*GRID,40,128)).view_as(q)}
                outputs={}
                for name,fn in funcs.items():
                    full=fn()
                    outputs[name]=full[0,token_indices][:,head_indices].float().clone()
                    del full
                flat=[x.permute(0,2,1,3) for x in (q,k,v)]
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    funcs['sdpa']=lambda:torch.nn.functional.scaled_dot_product_attention(*flat,scale=128**-.5)
                    funcs['sdpa']()
                    timings={name:[] for name in funcs}
                    for repeat in range(3):
                        order=list(funcs) if repeat%2==0 else list(reversed(funcs))
                        for name in order:
                            full,ms=timed(funcs[name]);timings[name].append(ms);del full
                trow={'prompt':prompt,'step':step,'layer':layer,
                      'samples_ms':timings,'mean_attention_ms':{name:statistics.mean(ms) for name,ms in timings.items()}}
                trow['speedup_vs_sdpa']={name:trow['mean_attention_ms']['sdpa']/trow['mean_attention_ms'][name] for name in ('c4_native','c8_r2')}
                report['timings'].append(trow)
                report['captures'].append({'path':str(path),'metadata':meta,'manifest_pin':pins[path.name],
                    'whole_file_hash_recomputed':False,'four_head_qkv_bitwise_equal':equal if layer==33 else None})
                for name,out in outputs.items():
                    for slot,head in enumerate(heads):
                        error=out[:,slot]-dense[:,slot]
                        report['rows'].append({'method':name,'prompt':prompt,'step':step,'call':call,'layer':layer,'head':head,
                            'relative_l2':float(error.norm()/dense[:,slot].norm()),'squared_error':float(error.square().sum()),
                            'reference_squared':float(dense[:,slot].square().sum()),
                            'per_query_relative_l2':(error.norm(dim=-1)/dense[:,slot].norm(dim=-1)).cpu().tolist()})
                save(args.output/'results.json',report)
                print(json.dumps({'prompt':prompt,'step':step,'layer':layer,'qkv_matched':equal,
                    'errors':{name:statistics.mean(r['relative_l2'] for r in report['rows'][-8:] if r['method']==name) for name in outputs},
                    'mean_attention_ms':trow['mean_attention_ms']}),flush=True)
                del payload,q,k,v,selected,outputs,reference,dense,flat,funcs
    for rel,expected in report['source_hashes'].items():
        assert sha(args.repo/rel)==expected
    report['summary']=[]
    for name in ('c4_native','c8_r2'):
        rows=[r for r in report['rows'] if r['method']==name]
        ms=statistics.mean(t['mean_attention_ms'][name] for t in report['timings'])
        sdpa=statistics.mean(t['mean_attention_ms']['sdpa'] for t in report['timings'])
        report['summary'].append({'method':name,'cases':len(rows),'mean_relative_l2':statistics.mean(r['relative_l2'] for r in rows),
            'layer_means':{str(layer):statistics.mean(r['relative_l2'] for r in rows if r['layer']==layer) for layer in (0,33)},
            'mean_attention_ms':ms,'mean_sdpa_attention_ms':sdpa,'speedup_vs_sdpa':sdpa/ms})
    report['complete']=True
    save(args.output/'results.json',report)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args);return
    state=gpu_state(args.gpu)
    if 'H200' not in state['name'] or state['pids'] or state['memory_mib']>100 or state['utilization']:
        raise RuntimeError('No clean H200; no launch or queue')
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'preflight.json',state)
    for name in ('quick_axis_c8_native.py','quick_axis_c4_qkv.py'):
        shutil.copyfile(Path(__file__).with_name(name),args.output/name)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),AXIS_C8_PARENT=str(os.getpid()),
             TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             TORCH_EXTENSIONS_DIR=str(args.output/'torch_extensions'),TRITON_CACHE_DIR=str(args.output/'triton_cache'),
             TMPDIR=str(args.output/'tmp'))
    (args.output/'tmp').mkdir()
    command=[sys.executable,str(args.output/'quick_axis_c8_native.py'),'--worker','--repo',str(args.repo),'--output',str(args.output)]
    state=gpu_state(args.gpu)
    if state['pids'] or state['memory_mib']>100 or state['utilization']:raise RuntimeError('GPU changed; no launch')
    child=None;failure=None
    try:
        with (args.output/'worker.log').open('w') as log,(args.output/'gpu_monitor.jsonl').open('w') as monitor:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(args.output/'process.json',{'parent_pid':os.getpid(),'child_pid':child.pid,'command':command})
            while child.poll() is None:
                state=gpu_state(args.gpu)
                monitor.write(json.dumps({'time':time.time(),**state})+'\n');monitor.flush()
                if set(state['pids'])-{child.pid}:raise RuntimeError('Foreign GPU process; timing invalid')
                time.sleep(2)
            if child.returncode:raise RuntimeError(f'Worker exit {child.returncode}')
    except BaseException as error:
        failure=str(error)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20)
        raise
    finally:
        save(args.output/'completion.json',{'valid':failure is None,'failure':failure,
            'worker_exit_code':None if child is None else child.returncode})


if __name__=='__main__':main()
