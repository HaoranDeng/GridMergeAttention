"""Fragment-map gate for CUDA fusion; no complete-attention speed claim."""
import argparse, hashlib, json, os, threading
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--merge-capacity', type=int, choices=[4,10])
    parser.add_argument('--source-dir', type=Path, default=ROOT/'native_fisher')
    args = parser.parse_args()
    assert not args.output.exists()
    initial = require_clean(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = initial['gpu']['uuid']
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
    os.environ['MAX_JOBS'] = '2'
    os.environ['TORCH_EXTENSIONS_DIR'] = str(args.output.parent/'torch_extensions')
    stop, errors, samples = threading.Event(), [], []
    def monitor():
        while not stop.is_set():
            try:
                ps = gpu_processes(initial['gpu']['uuid'])
                samples.append({'utc': now(), 'processes': ps})
                if any(p['pid'] != os.getpid() for p in ps): errors.append('foreign GPU process')
            except Exception as e: errors.append(repr(e))
            stop.wait(3)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    result = {'scope': 'Warp BF16 GEMM fragment gate only; not an attention implementation or timing',
              'initial_gpu': initial, 'cases': []}
    try:
        import torch
        from torch.utils.cpp_extension import load
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        sources = [args.source_dir.resolve()/n for n in ('warp_mma.cuh', 'warp_mma_diagnostic.cu')]
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        key = hashlib.sha256(''.join(hashes.values()).encode()).hexdigest()[:16]
        print('BUILD', key, flush=True)
        ext = load(name='fisher_warp_mma_'+key, sources=[str(sources[1])],
                   extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'], verbose=False)
        assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        result['sources'] = hashes
        torch.manual_seed(0)
        def check(name, a, raw, score, exact):
            actual = ext.run(a, raw, score, False)
            expected = a.float() @ (raw.float().transpose(-1,-2) if score else raw.float())
            delta = actual-expected
            row = {'name': name, 'bitwise_equal': torch.equal(actual, expected),
                   'max_abs': float(delta.abs().max()), 'finite': bool(torch.isfinite(actual).all()),
                   'relative_l2': float((delta.square().sum()/expected.square().sum().clamp_min(1e-30)).sqrt())}
            if score:
                # Exactly the existing native query pack permutation, expressed
                # here as a view/transpose so its map can be tested independently.
                packed = a.reshape(-1,16,4,4,4,2).transpose(-3,-2).contiguous().reshape_as(a)
                row['packed_equal'] = torch.equal(ext.run(packed,raw,True,True),actual)
                assert row['packed_equal'], row
            result['cases'].append(row)
            print(json.dumps(row), flush=True)
            assert row['finite'] and (row['bitwise_equal'] if exact else row['relative_l2'] < 2e-6), row
        for score in (True,False):
            shape = (7,16,128 if score else 80)
            raw = torch.randint(-3,4,(7,80,128),device='cuda').to(torch.bfloat16)
            a = torch.randint(-3,4,shape,device='cuda').to(torch.bfloat16)
            check(('score' if score else 'merge')+'_integer',a,raw,score,True)
            check(('score' if score else 'merge')+'_zero',torch.zeros_like(a),raw,score,True)
            a = torch.randn(shape,device='cuda',dtype=torch.bfloat16)
            raw = torch.randn((7,80,128),device='cuda',dtype=torch.bfloat16)
            check(('score' if score else 'merge')+'_random',a,raw,score,False)
        capture = args.output.parent.parent/'gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt'
        payload = torch.load(capture,map_location='cpu',weights_only=True)
        q,k,v = [payload[n].reshape(21,45,80,40,128) for n in ('q','k','v')]
        qs=[];ks=[];vs=[]
        for frame,head,column,keyframe,keyrow in [(0,0,0,0,0),(10,19,37,7,23),(20,39,79,20,44)]:
            qs.append(q[frame,:16,column,head,:]);ks.append(k[keyframe,keyrow,:,head,:]);vs.append(v[keyframe,keyrow,:,head,:])
        qa,ka,va = [torch.stack(x).contiguous().cuda() for x in (qs,ks,vs)]
        check('real_query_score',qa,ka,True,False)
        weights = torch.softmax((qa.float() @ ka.float().transpose(-1,-2))*128**-.5,dim=-1).to(torch.bfloat16)
        check('real_weight_merge',weights,va,False,False)
        if args.merge_capacity:
            from analysis.validate_fisher_warp_merge import validate_merge
            result['merge_gate'] = validate_merge(args.merge_capacity, torch, q, k, v,args.source_dir)
        assert not errors, errors
        result['valid'] = result.get('merge_gate', {}).get('exact_gate_passed', True)
    except Exception as e:
        result.update(valid=False, error=repr(e))
        raise
    finally:
        stop.set();thread.join()
        result.update(monitor_errors=errors, monitor_samples=samples)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2))

if __name__ == '__main__': main()
