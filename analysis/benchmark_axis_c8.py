"""Interleaved real-QKV full-call GMA-C8 and dense SDPA measurements."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from attention.axis_current_c8_cuda import AxisCurrentC8,_workspace
from analysis.axis_c8_speed_gate import source_hashes,environment
from analysis.quick_axis_c4_qkv import save

CAPTURES=Path('/local2/denghaoran/GridMergeAttention/experiments/full40_bf16_dense_capture_20260905bb/captures')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--repeats',type=int,default=7);p.add_argument('--profile',action='store_true')
    p.add_argument('--single',action='store_true',help='Tuning only: cannot satisfy video launch gate')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    report={'passed':False,'complete':False,'axes':'wh','capacity':8,'shape':[1,40,75600,128],
            'dtype':'bfloat16','scope':'full attention including layouts and all40 heads',
            'source_sha256':source_hashes(),'environment':environment(),'samples':[],
            'gpu':torch.cuda.get_device_name(),'torch_version':torch.__version__}
    captures=['prompt000_call000_layer0.pt','prompt000_call033_layer33.pt',
              'prompt001_call073_layer33.pt','prompt002_call120_layer0.pt']
    def measure(fn):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record();value=fn();e.record();e.synchronize()
        return s.elapsed_time(e),value
    try:
        for capture in captures[:1 if a.single else 4]:
            path=CAPTURES/capture
            payload=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
            tensors=[payload[name].permute(0,2,1,3).to('cuda').contiguous() for name in ('q','k','v')]
            del payload
            assert all(x.shape==(1,40,75600,128) and x.dtype==torch.bfloat16 for x in tensors)
            core=AxisCurrentC8();method=lambda:core(*tensors)
            dense=lambda:torch.nn.functional.scaled_dot_product_attention(*tensors)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for _ in range(2):method();dense()
                times={'method':[],'dense':[]}
                for repeat in range(a.repeats):
                    order=(('method',method),('dense',dense)) if repeat%2 else (('dense',dense),('method',method))
                    for name,fn in order:
                        elapsed,value=measure(fn);times[name].append(elapsed)
                        assert torch.isfinite(value).all()
                row={'capture':str(path),'input_kind':'captured_real_qkv',
                     'method_samples_ms':times['method'],'dense_samples_ms':times['dense'],
                     'method_median_ms':statistics.median(times['method']),
                     'dense_median_ms':statistics.median(times['dense'])}
                row['speedup_vs_sdpa']=row['dense_median_ms']/row['method_median_ms']
            if a.profile and not report['samples']:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    method();torch.cuda.synchronize()
                report['kernel_profile']=[{'name':e.key,'ms':e.device_time_total/1000,'count':e.count}
                                          for e in prof.key_averages() if e.device_time_total>1000]
            report['samples'].append(row);save(a.output,report);print(json.dumps(row),flush=True)
            del tensors,core,value,method,dense
            _workspace.cache_clear();gc.collect();torch.cuda.empty_cache()
        mt=[t for s in report['samples'] for t in s['method_samples_ms']]
        dt=[t for s in report['samples'] for t in s['dense_samples_ms']]
        report.update(mean_attention_ms=statistics.mean(mt),dense_mean_attention_ms=statistics.mean(dt),
                      speedup_vs_sdpa=statistics.mean(dt)/statistics.mean(mt),complete=True,
                      passed=len(report['samples'])>=4 and all(s['method_median_ms']<s['dense_median_ms'] for s in report['samples']))
    finally:
        save(a.output,report)
    print(json.dumps({key:report.get(key) for key in ('passed','mean_attention_ms','speedup_vs_sdpa')}),flush=True)


if __name__=='__main__':main()
