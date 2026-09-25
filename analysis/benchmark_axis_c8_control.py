"""Complete head-major C8 r2 control for the directed-axis C4 screen.

Run under guard_axis_job on an exclusively idle H200. This only replays
captured QKV; it does not regenerate existing baseline videos.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from analysis.quick_axis_c4_qkv import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6', mode=ctypes.RTLD_GLOBAL)
    os.environ.setdefault('TORCH_EXTENSIONS_DIR',
        '/local2/denghaoran/GridMergeAttention/axis_cuda_extensions_20260918/'+socket.gethostname())
    os.environ.setdefault('MAX_JOBS', '1')
    from kernels.best import create_c8
    payload = torch.load(args.capture, map_location='cpu', weights_only=True, mmap=True)
    tensors = [payload[name].permute(0, 2, 1, 3).to('cuda').contiguous()
               for name in ('q', 'k', 'v')]
    del payload
    assert all(x.shape == (1,40,75600,128) and x.dtype == torch.bfloat16 for x in tensors)
    report = {'implementation':'C8 r2', 'shape':list(tensors[0].shape),
              'dtype':'bfloat16', 'input_kind':'captured_real_qkv',
              'capture':str(args.capture), 'gpu':torch.cuda.get_device_name(),
              'scope':'complete attention including input/output layout conversions',
              'status':'compiling'}
    root = Path(__file__).resolve().parents[1]
    report['source_sha256'] = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (root/'kernels/best/c8_r2').iterdir() if p.is_file()}
    save(args.output,report)
    op = create_c8()

    def control():
        token_major = [x.permute(0,2,1,3).contiguous() for x in tensors]
        return op(*token_major).permute(0,2,1,3).contiguous()

    def measure(fn):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); result = fn(); end.record(); end.synchronize()
        return start.elapsed_time(end), result

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        dense = lambda: torch.nn.functional.scaled_dot_product_attention(*tensors)
        dense()
        output = control()
        assert bool(torch.isfinite(output).all())
        paired = []
        for i in range(args.repeats):
            row = {}
            for name, fn in [('dense',dense),('c8',control)][::1 if i%2==0 else -1]:
                row[name] = measure(fn)[0]
            paired.append(row)
    report.update(status='timed', paired_samples_ms=paired,
                  dense_mean_ms=statistics.fmean(x['dense'] for x in paired),
                  c8_mean_ms=statistics.fmean(x['c8'] for x in paired),
                  dense_median_ms=statistics.median(x['dense'] for x in paired),
                  c8_median_ms=statistics.median(x['c8'] for x in paired),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    report['speedup_vs_sdpa']=report['dense_mean_ms']/report['c8_mean_ms']
    save(args.output,report)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        control(); torch.cuda.synchronize()
    report['kernels'] = [{'name':e.key,'ms':e.device_time_total/1000,'count':e.count}
                         for e in prof.key_averages() if e.device_time_total>1000]
    report['status']='complete'
    save(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','kernels')}),flush=True)


if __name__=='__main__':
    main()
