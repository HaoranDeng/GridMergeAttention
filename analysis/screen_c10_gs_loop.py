"""Test identical FP64 Fisher GS with bounded instruction footprint."""
import argparse,hashlib,json,os,sys,threading,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',default='fisher_gs_loop.cu')
    p.add_argument('--probe-lanes',type=int,choices=[1,2,4,8,16],default=4)
    args=p.parse_args();assert not args.output.exists()
    initial=require_clean(args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=initial['gpu']['uuid'],OMP_NUM_THREADS='4',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='9.0a',
        TORCH_EXTENSIONS_DIR=str(args.output.parent/'torch_extensions'),TRITON_CACHE_DIR=str(args.output.parent/'triton_cache'))
    errors=[];samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                ps=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':ps})
                if any(x['pid']!=os.getpid() for x in ps):errors.append('foreign GPU process')
            except Exception as e:errors.append(repr(e))
            stop.wait(2)
    thread=threading.Thread(target=monitor);thread.start()
    result={'initial_gpu':initial,'scope':'FP64 GS kernel only, all40 heads; excludes probe logits and complete attention','variants':[]}
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples);args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2))
    try:
        import torch
        from torch.utils.cpp_extension import load
        from native_fisher.selector import NativeFisherSelector,_logits
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        selector=NativeFisherSelector(10,layout='probe4');control=selector.extension
        base=(ROOT/'native_fisher/fisher_gs_distributed.cu').read_text()
        source=ROOT/'native_fisher'/args.source
        needle='#pragma unroll\n  for(int step=1;step<SEEDS;++step)'
        assert needle in base
        if args.source=='fisher_gs_loop.cu':
            source.write_text(base.replace(needle,'#pragma unroll 1\n  for(int step=1;step<SEEDS;++step)'))
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        print('BUILD_GS',flush=True)
        candidate=load(name='fisher_gs_loop_'+digest[:16],sources=[str(source)],extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=False)
        result['source_sha256']=digest
        result['probe_lanes']=args.probe_lanes
        def run(ext):ext.run(selector.logits,selector.candidates,selector.anchors,selector.seeds,128**-.5,args.probe_lanes if ext is candidate else 4)
        def timed(ext):
            assert not errors,errors
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();run(ext);b.record();b.synchronize();return a.elapsed_time(b)
        for layer in ('00','19','39'):
            payload=torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer'+layer+'.pt',map_location='cpu',weights_only=True)
            q,k=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k')];del payload
            _logits[(900,40)](q[:,:1],k,selector.candidates,selector.logits,num_warps=4,num_stages=1)
            run(control);expected=selector.seeds.clone();run(candidate)
            row={'layer':layer,'mismatches':int((selector.seeds!=expected).sum())};result['variants'].append(row)
            assert row['mismatches']==0,row
            times={'candidate':[],'control':[]};run(control);run(candidate)
            for block in range(4):
                for name in (('candidate','control','control','candidate') if block%2==0 else ('control','candidate','candidate','control')):
                    times[name].append(timed(candidate if name=='candidate' else control))
            row['timing']={n:{'mean_ms':statistics.fmean(v),'samples_ms':v} for n,v in times.items()}
            print('RESULT',json.dumps(row),flush=True);save();del q,k,expected
        selector.logits.zero_();run(control);expected=selector.seeds.clone();run(candidate)
        result['zero_logits_ties_exact']=torch.equal(expected,selector.seeds)
        assert result['zero_logits_ties_exact']
        result['valid']=not errors;save()
    except BaseException as e:result.update(valid=False,error=repr(e));save();raise
    finally:stop.set();thread.join(5);save()

if __name__=='__main__':main()
