"""Exhaust every positive FP32 bit pattern in the proposed fast-path domain."""
import argparse,json,os,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    initial=require_clean(a.gpu)
    os.environ['CUDA_VISIBLE_DEVICES']=initial['gpu']['uuid']
    os.environ['TORCH_EXTENSIONS_DIR']=str(a.output.parent/'torch_extensions')
    os.environ['OMP_NUM_THREADS']='4'
    stop=threading.Event();errors=[];samples=[]
    def monitor():
        while not stop.is_set():
            try:
                ps=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':ps})
                if any(row['pid']!=os.getpid() for row in ps):errors.append('foreign GPU process')
            except Exception as error:errors.append(repr(error))
            stop.wait(3)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    result={'initial_gpu':initial,'scope':'Exhaustive scalar instruction comparison, not an attention benchmark',
            'domain':'Every positive FP32 x with 2^-24 <= x < 2^7',
            'input_count':0x43000000-0x33800000}
    try:
        import torch
        from native_fisher.operator import extension
        module=extension(10,ROOT/'native_fisher/reciprocal_gate.cu')
        stats=torch.tensor([0,2**63-1,0],device='cuda',dtype=torch.int64)
        module.run(stats);torch.cuda.synchronize()
        mismatches,first,max_ulp=stats.tolist()
        result.update(source_sha256=module.source_sha256,mismatches=mismatches,
                      first_mismatch_bits=hex(first) if mismatches else None,max_ulp=max_ulp,
                      valid=not errors,bitwise_equal=mismatches==0)
    except BaseException as error:
        result.update(valid=False,error=repr(error));raise
    finally:
        stop.set();thread.join(10)
        result.update(monitor_errors=errors,monitor_samples=samples)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
