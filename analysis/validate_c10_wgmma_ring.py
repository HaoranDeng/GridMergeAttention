"""Validate shared-ring WGMMA producer, not full fused attention performance."""
import argparse,hashlib,itertools,json,os,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists();initial=require_clean(args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=initial['gpu']['uuid'],OMP_NUM_THREADS='4',MAX_JOBS='2',TORCH_EXTENSIONS_DIR=str(args.output.parent/'torch_extensions'))
    errors=[];samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                ps=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':ps})
                if any(x['pid']!=os.getpid() for x in ps):errors.append('foreign GPU process')
            except Exception as e:errors.append(repr(e))
            stop.wait(2)
    thread=threading.Thread(target=monitor);thread.start()
    result={'initial_gpu':initial,'scope':'Five-column native WGMMA shared-ring producer component only; no attention timing or speedup','layers':[]}
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    try:
        import torch
        from native_fisher.operator import NativeFisherAttention,extension
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        header=ROOT/'native_fisher/c10_wgmma_ring.cuh'
        h=hashlib.sha256(header.read_bytes()).hexdigest();result['header_sha256']=h
        ext=extension(10,ROOT/'native_fisher/c10_wgmma_ring_gate.cu',name_suffix='_h'+h[:12])
        assert hashlib.sha256(header.read_bytes()).hexdigest()==h
        result.update(gate_sha256=ext.source_sha256,resources=ext.resources());save();print('RESOURCES',result['resources'],flush=True)
        op=NativeFisherAttention(capacity=10,tiles=84,merge_source=ROOT/'native_fisher/fisher_c10_long_tiles_tree.cu',selector_source=ROOT/'native_fisher/fisher_gs_register_distribute.cu')
        checks=list(itertools.product((0,10,20),(0,35,75),(0,448,944),(0,2,4)))
        check_tensor=torch.tensor(checks,device='cuda',dtype=torch.int32)
        mk=torch.empty((len(checks),5,16,128),device='cuda',dtype=torch.bfloat16);mv=torch.empty_like(mk)
        bias=torch.empty((len(checks),5,16),device='cuda')
        for layer in ('00','19','39'):
            payload=torch.load(f'/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{layer}.pt',map_location='cpu',weights_only=True)
            q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
            seeds=op.select(q,k,128**-.5);op.merge.pack(q,op.packed)
            op.merge.run(op.packed,k,v,seeds,op.mk,op.mv,op.bias,0,5,84,128**-.5,op.empty,0,1)
            ext.run(op.packed,k,v,seeds,check_tensor,mk,mv,bias);torch.cuda.synchronize()
            row={'layer':layer,'checks':[]};result['layers'].append(row)
            for index,(frame,col,kb,head) in enumerate(checks):
                slots=torch.tensor([(frame*80+c)*5+head for c in range(col,col+5)],device='cuda')
                record={'frame':frame,'column':col,'key_block':kb,'head':head}
                for name,a,ref in [('k',mk,op.mk),('v',mv,op.mv),('bias',bias,op.bias)]:
                    record[name+'_exact']=torch.equal(a[index,:,:10],ref[slots,kb*10:kb*10+10])
                rawk=k.reshape(945,80,40,128)[kb,col:col+5,head]
                rawv=v.reshape(945,80,40,128)[kb,col:col+5,head]
                record['sq1_exact']=torch.equal(mk[index,:,10],rawk) and torch.equal(mv[index,:,10],rawv) and bool((bias[index,:,10]==0).all())
                record['padding_valid']=bool((mk[index,:,11:]==0).all()) and bool((mv[index,:,11:]==0).all()) and bool(torch.isposinf(bias[index,:,11:]).all())
                row['checks'].append(record)
                assert all(record[x] for x in ('k_exact','v_exact','bias_exact','sq1_exact','padding_valid')),record
            print('PASS',layer,len(checks),'ring slices',flush=True);save();del q,k,v,seeds
        result['valid']=not errors;save()
    except BaseException as e:result.update(valid=False,error=repr(e));save();raise
    finally:stop.set();thread.join(5);save()

if __name__=='__main__':main()
