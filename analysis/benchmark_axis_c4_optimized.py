"""Matched full-call speed gate; short staged trials before any video launch."""
import argparse
import importlib
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from analysis.quick_axis_c4_qkv import save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--axes',choices=['tw','wh','th'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--configs',nargs='+',default=['4:4:1','4:4:4','8:4:1','8:4:4','16:8:1'])
    p.add_argument('--module',default='attention.axis_current_c4_optimized')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--capture',type=Path)
    p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args(); torch.set_num_threads(1);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    module=importlib.import_module(a.module)
    source=Path(module.__file__).read_bytes()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix('.source.py').write_bytes(source)
    gen=torch.Generator(device='cuda').manual_seed(714)
    if a.capture:
        payload=torch.load(a.capture,map_location='cpu',weights_only=True,mmap=True)
        tensors=[payload[name].permute(0,2,1,3).to('cuda').contiguous() for name in ('q','k','v')]
        del payload
    else:
        tensors=[torch.randn((1,40,75600,128),dtype=torch.bfloat16,device='cuda',generator=gen) for _ in range(3)]
    assert all(tuple(x.shape)==(1,40,75600,128) and x.dtype==torch.bfloat16 for x in tensors)
    def measure(fn):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record();out=fn();e.record();e.synchronize()
        return s.elapsed_time(e),out
    report={'axes':a.axes,'shape':[1,40,75600,128],'dtype':'bfloat16','trials':[],
            'source_sha256':hashlib.sha256(source).hexdigest(),
            'scope':'full attention including layouts and all40 heads',
            'input_kind':'captured_real_qkv' if a.capture else 'synthetic_random',
            'capture':str(a.capture) if a.capture else None,'module':a.module,
            'gpu':torch.cuda.get_device_name(), 'torch_version':torch.__version__,
            'environment':{key:value for key,value in os.environ.items() if key.startswith('AXIS_')}}
    root=Path(__file__).resolve().parents[1]
    files=[module.__file__,'attention/axis_current_c4_selector.cu','attention/axis_current_c4_merge.cu',
           'kernels/best/c4/warp_mma.cuh','kernels/best/c4/route_reference.py','analysis/quick_axis_c4_qkv.py']
    report['source_sha256']={str((root/filename).relative_to(root)):hashlib.sha256((root/filename).read_bytes()).hexdigest()
                           for filename in files}
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        torch.nn.functional.scaled_dot_product_attention(*tensors)
        report['sdpa_samples_ms']=[measure(lambda:torch.nn.functional.scaled_dot_product_attention(*tensors))[0] for _ in range(a.repeats)]
        report['sdpa_ms']=statistics.median(report['sdpa_samples_ms'])
    print(json.dumps({'sdpa_ms':report['sdpa_ms']}),flush=True)
    for config in a.configs:
        pieces=list(map(int,config.split(':')))
        bj,warps,qog=pieces[:3]
        os.environ.update(AXIS_C4_BLOCK_J=str(bj),AXIS_C4_MERGE_WARPS=str(warps),AXIS_C4_OUTER_GROUP=str(qog),AXIS_C4_PRECOMPUTE='0')
        if a.module.endswith('_cuda'):
            os.environ['AXIS_C4_CUDA_TILES']=str(qog)
        if len(pieces)>3:
            assert len(pieces)==7
            os.environ.update(dict(zip(('AXIS_C4_ROUTE_TOKENS','AXIS_C4_ROUTE_WARPS',
                                        'AXIS_C4_ROUTE_STAGES','AXIS_C4_HEAD_CHUNK'),map(str,pieces[3:]))))
        core=module.AxisCurrentC4(a.axes)
        started=time.monotonic()
        try:
            os.environ['AXIS_C4_KERNEL_METADATA']='1'
            result=core(*tensors);torch.cuda.synchronize()
            os.environ['AXIS_C4_KERNEL_METADATA']='0'
            assert bool(torch.isfinite(result).all())
            samples=[measure(lambda:core(*tensors))[0] for _ in range(a.repeats)]
            row={'config':config,'samples_ms':samples,'median_ms':statistics.median(samples),
                 'speedup_vs_sdpa':report['sdpa_ms']/statistics.median(samples),
                 'compile_seconds':time.monotonic()-started,
                 'environment':{key:value for key,value in os.environ.items() if key.startswith('AXIS_')}}
            if a.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    core(*tensors);torch.cuda.synchronize()
                row['kernels']=[{'name':e.key,'ms':e.device_time_total/1000,'count':e.count}
                                for e in prof.key_averages() if e.device_time_total>1000 and 'elementwise_kernel' not in e.key]
        except Exception as e:
            row={'config':config,'error':repr(e)}
        report['trials'].append(row);save(a.output,report);print(json.dumps(row),flush=True)
    report['complete']=True;save(a.output,report)


if __name__=='__main__':main()
