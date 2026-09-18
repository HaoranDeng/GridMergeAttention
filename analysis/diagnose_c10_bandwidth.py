"""Destructive workspace-only roofline diagnostics, never attention candidates.

Compare retained merge against its write schedule with synthetic data, and
against a compute-only ablation that consumes every accumulator in opaque asm.
All variants use identical H5 workspace sizes. Timings are not additive.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now


def sources(directory):
    original = (ROOT / 'native_fisher/fisher_c10_half_owner.cu').read_text()
    start = original.index('  for(int vec=tid;vec<80*16;vec+=128)')
    end = original.index('  int total=Q*', start)
    io = original[:start] + original[end:]
    start = io.index('    int row0=warp*16')
    end = io.index('      if(tma_store==2 || cb==13)', start)
    io = io[:start] + '''    #pragma unroll
    for(int kind=0;kind<2;++kind) {
      float acc[64];
      #pragma unroll
      for(int i=0;i<64;++i) {
        uint32_t h=(uint32_t(blockIdx.x)*747796405u+uint32_t(blockIdx.y)*2891336453u+
                    uint32_t(tile)*277803737u+uint32_t(tid)*1597334677u+i*3812015801u+kind);
        h=(h^(h>>16))*2246822519u;h^=h>>13;
        acc[i]=__uint_as_float((h&0x807fffffu)|((120u+((h>>24)&15u))<<23));
      }
''' + io[end:]
    # Synthetic output is deliberately varied, not a compressible zero fill.
    lean = io.replace('bf16* temp=sv+80*128;', 'bf16* temp=reinterpret_cast<bf16*>(raw);')
    lean = lean.replace('(80*128*2)*2+64*128*2', '64*128*2')
    lean = lean.replace('__launch_bounds__(128,4)', '__launch_bounds__(128,8)')
    no_store = original.replace(
        '      if(tma_store==2 || cb==13) {',
        '''      #pragma unroll
      for(int i=0;i<64;++i) asm volatile(""::"f"(acc[i]):"memory");
      continue; // Diagnostic: omit BF16 staging and global K/V output.
      if(tma_store==2 || cb==13) {''', 1)
    # A contiguous nonzero vector-store control and a device-to-device copy.
    extra = r'''
__global__ void linear_write(uint4* a,uint4* b,int64_t n) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x){
    uint32_t h=uint32_t(i)*747796405u+2891336453u;
    h=(h^(h>>16))*2246822519u;
    uint4 x={h,h*277803737u,h*1597334677u,h*3812015801u};
    a[i]=x;b[i]=x;
  }
}
void linear(torch::Tensor a,torch::Tensor b) {
  linear_write<<<4096,256,0,at::cuda::getCurrentCUDAStream()>>>(
    (uint4*)a.data_ptr(),(uint4*)b.data_ptr(),a.numel()*2/16);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
    io = io.replace('PYBIND11_MODULE', extra + '\nPYBIND11_MODULE')
    io = io.replace('m.def("run",&run);', 'm.def("linear",&linear);m.def("run",&run);')
    result = {}
    for name, source in [('write_same_shared', io), ('write_lean_shared', lean), ('compute_no_kv_store', no_store)]:
        path = directory / (name + '.cu')
        path.write_text(source)
        result[name] = path
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    initial = require_clean(args.gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    folder = args.output.parent / 'bandwidth_sources'
    folder.mkdir(exist_ok=True)
    paths = sources(folder)
    os.environ.update(CUDA_VISIBLE_DEVICES=initial['gpu']['uuid'], OMP_NUM_THREADS='4', MAX_JOBS='2',
                      TORCH_EXTENSIONS_DIR=str(args.output.parent/'torch_extensions'))
    errors, samples = [], []
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                ps = gpu_processes(initial['gpu']['uuid'])
                samples.append({'utc': now(), 'processes': ps})
                if any(x['pid'] != os.getpid() for x in ps): errors.append('foreign GPU process')
            except Exception as e: errors.append(repr(e))
            stop.wait(2)
    thread = threading.Thread(target=monitor)
    thread.start()
    result = {'initial_gpu': initial, 'scope': 'H5 merge bandwidth diagnostic only; no algorithm candidate or full-call speedup', 'variants': {}}
    def save():
        result.update(monitor_errors=errors, monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2))
    try:
        import torch
        from native_fisher.operator import NativeFisherAttention, extension
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        op = NativeFisherAttention(capacity=10, selector_source=ROOT/'native_fisher/fisher_gs_register_distribute.cu', merge_source=ROOT/'native_fisher/fisher_c10_half_owner.cu')
        exts = {name: extension(10, path) for name, path in paths.items()}
        exts['retained_merge'] = op.merge
        for name, ext in exts.items():
            result['variants'][name] = {'resources_regs_local_shared_active_ctas': ext.resources(), 'source_sha256': ext.source_sha256}
        payload = torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt', map_location='cpu', weights_only=True)
        q,k,v = [payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')]
        del payload
        seeds = op.select(q,k,128**-.5)
        op.merge.pack(q,op.packed)
        def run(ext):
            ext.run(op.packed,k,v,seeds,op.mk,op.mv,op.bias,0,5,21,128**-.5,op.empty,0,1)
        calls = {name: (lambda ext=ext: run(ext)) for name,ext in exts.items()}
        calls['linear_write'] = lambda: exts['write_same_shared'].linear(op.mk,op.mv)
        calls['contiguous_copy'] = lambda: op.mv.copy_(op.mk)
        nbytes = op.mk.numel()*op.mk.element_size()
        result['kv_write_bytes_per_h5'] = 2*nbytes
        result['bias_write_bytes_per_h5'] = op.bias.numel()*op.bias.element_size()
        result['retained_input_shapes'] = list(q.shape)
        for name, fn in calls.items():
            fn();fn()
        torch.cuda.synchronize()
        # Alternating order limits drift; all arrays greatly exceed L2 capacity.
        timings = {name: [] for name in calls}
        for rep in range(8):
            for name in (list(calls) if rep%2==0 else list(calls)[::-1]):
                assert not errors, errors
                a,b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record();calls[name]();b.record();b.synchronize()
                timings[name].append(a.elapsed_time(b))
        for name, values in timings.items():
            row = result['variants'].setdefault(name,{})
            row.update(samples_ms=values, mean_ms=statistics.fmean(values))
            if name != 'compute_no_kv_store':
                row['effective_kv_TB_per_s'] = 2*nbytes/row['mean_ms']/1e9
            print('RESULT', name, json.dumps(row), flush=True)
        result['valid'] = not errors
        result['caveats'] = ['Synthetic write and no-store kernels are invalid attention operators, diagnostics only.',
                             'No-store still performs original query, normalization, owner, bias and MK/MV matrix arithmetic; opaque asm retains accumulators. Compiler resource scheduling may differ.',
                             'Byte rates count useful KV bytes, not hardware DRAM counters; pure copy counts both read and write.',
                             'Ablation times cannot be subtracted into additive phase costs. H5 figures are not complete H40 call timings.']
        save()
    except BaseException as e:
        result.update(valid=False,error=repr(e));save();raise
    finally:
        stop.set();thread.join(5);save()


if __name__ == '__main__':
    main()
