"""Single-H200, exact-output screen of C10 register scheduling variants."""
import argparse, hashlib, json, os, statistics, sys, threading
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

def generate():
    base = (ROOT/'native_fisher/fisher_c10_merge.cu').read_text()
    paths = []
    for resident in (3, 2, 1):
        p = ROOT/f'native_fisher/fisher_c10_regs{resident}.cu'
        p.write_text(base.replace('__launch_bounds__(128,4)', f'__launch_bounds__(128,{resident})'))
        paths.append(p)
    # Reuse the no-longer-live normalized scores for exponentials. This
    # keeps the shared allocation and occupancy identical to the control.
    s = base.replace('selected_winners[index]=best;selected_owners[index]=owner;',
                     'selected_winners[index]=best;selected_owners[index]=owner;')
    needle = '    uint32_t packed_weights[20];'
    s = s.replace(needle, '    for(int index=tid;index<6*80;index+=128) normalized[index]=ex2(selected_winners[index]);\n    __syncthreads();\n' + needle)
    s = s.replace('winner==owner?ex2(best):0.f', 'winner==owner?normalized[index]:0.f')
    p = ROOT/'native_fisher/fisher_c10_exp_reuse.cu';p.write_text(s);paths.append(p)
    return paths

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int);p.add_argument('--output',type=Path)
    p.add_argument('--generate-only',action='store_true');p.add_argument('--layers',nargs='+',default=['00'])
    p.add_argument('--sources',nargs='+')
    p.add_argument('--tiles',type=int,choices=[1,3,7,21],default=21)
    p.add_argument('--tile-counts',type=int,nargs='+',help='Explicit runtime tile counts for source variants that support them')
    p.add_argument('--control-source',type=Path,help='Use a retained merge source as the paired control')
    p.add_argument('--control-tiles',type=int,default=21)
    args=p.parse_args();paths=[ROOT/'native_fisher'/s for s in args.sources] if args.sources else generate()
    if args.generate_only:return
    configs=[(path,tile_count) for path in paths for tile_count in (args.tile_counts or [args.tiles])]
    assert not args.output.exists()
    initial=require_clean(args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=initial['gpu']['uuid'],OMP_NUM_THREADS='4',MAX_JOBS='2',
        TORCH_EXTENSIONS_DIR=str(args.output.parent/'torch_extensions'),TRITON_CACHE_DIR=str(args.output.parent/'triton_cache'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    errors=[];samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                ps=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':ps})
                if any(x['pid']!=os.getpid() for x in ps):errors.append('foreign process')
            except Exception as e:errors.append(repr(e))
            stop.wait(2)
    thread=threading.Thread(target=monitor);thread.start()
    result={'initial_gpu':initial,'scope':'C10 complete H5 merge only; not full attention','variants':[]}
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples)
        args.output.write_text(json.dumps(result,indent=2))
    try:
        import torch
        from native_fisher.operator import NativeFisherAttention,extension
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        op=NativeFisherAttention(capacity=10,merge_source=args.control_source);control=op.merge
        def timed(fn):
            assert not errors,errors
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();fn();b.record();b.synchronize();return a.elapsed_time(b)
        for layer in args.layers:
            payload=torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer'+layer+'.pt',map_location='cpu',weights_only=True)
            q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
            seeds=op.select(q,k,128**-.5);control.pack(q,op.packed)
            def run(ext):ext.run(op.packed,k,v,seeds,op.mk,op.mv,op.bias,0,5,args.control_tiles if ext is control else tile_count,128**-.5,op.empty,0,1)
            run(control);expected=[x.clone() for x in (op.mk,op.mv,op.bias)]
            for path,tile_count in configs:
                print('BUILD',path.name,flush=True);ext=extension(10,path)
                row={'layer':layer,'source':path.name,'tiles':tile_count,'control_tiles':args.control_tiles,'sha256':ext.source_sha256,'resources':ext.resources(),'control_source_sha256':control.source_sha256}
                result['variants'].append(row);save()
                run(ext);row['exact']=[torch.equal(a,b) for a,b in zip((op.mk,op.mv,op.bias),expected)]
                if not all(row['exact']):print('REJECT',row,flush=True);save();continue
                run(control);run(ext)
                pair={'candidate':[],'control':[]}
                for block in range(4):
                    for name in (('candidate','control','control','candidate') if block%2==0 else ('control','candidate','candidate','control')):
                        pair[name].append(timed(lambda:run(ext if name=='candidate' else control)))
                row['timing']={n:{'mean_ms':statistics.fmean(v),'samples_ms':v} for n,v in pair.items()}
                print('RESULT',json.dumps(row),flush=True);save()
            del q,k,v,seeds,expected
        result['valid']=not errors;save()
    except BaseException as e:
        result.update(valid=False,error=repr(e));save();raise
    finally:stop.set();thread.join(5);save()

if __name__=='__main__':main()
