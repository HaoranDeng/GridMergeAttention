"""Confirm a small complete-H40 improvement with matched retained FP8 timing."""
import argparse, hashlib, json, os, statistics, sys, threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference-source',type=Path,default=ROOT/'native_fisher/fisher_c10_fp8_shared_store.cu')
    p.add_argument('--native-route-source',type=Path)
    p.add_argument('--paired-kv',action='store_true')
    p.add_argument('--candidate-route-warps',type=int,choices=(4,8),default=4)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists()
    initial=require_clean(args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=initial['gpu']['uuid'],OMP_NUM_THREADS='4',
        TORCH_EXTENSIONS_DIR=str(args.output.parent/'torch_extensions'),TRITON_CACHE_DIR=str(args.output.parent/'triton_cache'))
    result={'valid':False,'initial_gpu':initial,'mode':'direct','fp8_storage':True,'control_is_fp8':True,
        'control_label':'Retained FP8 '+args.reference_source.stem+' / explicit FP32 conversion, route64 stages1',
        'scope':'Matched complete H40 candidate versus retained FP8, no BF16 timing in this receipt',
        'generation_validation_pending':True,'layers':[],'source_files':{}}
    stop=threading.Event();errors=[];samples=[]
    def monitor():
        while not stop.is_set():
            try:
                processes=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':processes})
                if any(x['pid']!=os.getpid() for x in processes):errors.append('foreign GPU process')
            except Exception as e:errors.append(repr(e))
            stop.wait(3)
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples)
        args.output.write_text(json.dumps(result,indent=2))
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    try:
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from native_fisher.fp8_c10 import retained_control,FP8C10Attention
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        base=retained_control()
        old=FP8C10Attention(base,'direct',source=args.reference_source)
        if args.paired_kv:
            assert not args.native_route_source
            from native_fisher.fp8_paired_kv import PairedFP8C10Attention
            new=PairedFP8C10Attention(base,args.source)
            result['paired_byte_storage']=True
            for path in ('native_fisher/fp8_paired_kv.py','native_fisher/fp8_paired_route.py'):
                result['source_files'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            import triton
            import triton.language as tl
            from native_fisher.fp8_paired_route import decode_pair
            @triton.jit
            def decode_combinations(src,kout,vout):
                idx=tl.program_id(0)*256+tl.arange(0,256)
                kvalue,vvalue=decode_pair(tl.load(src+idx))
                tl.store(kout+idx,kvalue);tl.store(vout+idx,vvalue)
            raw=torch.arange(65536,device='cuda',dtype=torch.int32)
            encoded=raw.to(torch.int16)
            decoded=[torch.empty(65536,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
            decode_combinations[(256,)](encoded,*decoded)
            checks=[]
            for shift,actual in zip((0,8),decoded):
                expected=((raw>>shift)&255).to(torch.uint8).view(torch.float8_e4m3fn).float().bfloat16()
                finite=torch.isfinite(expected)
                checks.append({'finite_bits_equal':bool(torch.equal(expected.view(torch.int16)[finite],actual.view(torch.int16)[finite])),
                    'nan_mask_equal':bool(torch.equal(torch.isnan(expected),torch.isnan(actual)))})
            result['paired_decoder']={'combinations':65536,'k_v_checks':checks}
            assert all(check[k] for check in checks for k in ('finite_bits_equal','nan_mask_equal'))
            del raw,encoded,decoded,actual,expected,finite
        else:new=FP8C10Attention(base,'direct',source=args.source)
        for op in (old,new):op.via_float_route=True;op.route_block=64;op.route_stages=1
        if args.candidate_route_warps!=4:
            assert not args.native_route_source and not args.paired_kv
            new.route_warps=args.candidate_route_warps
            result.update(candidate_route_warps=new.route_warps,reference_route_warps=old.route_warps,
                allow_roundoff=True,numerical_screen='relative L2 <=0.002 and max abs <=0.25; screening only, generation required before promotion')
        if args.native_route_source:
            from native_fisher.operator import extension
            native_route=extension(10,args.native_route_source)
            if hasattr(native_route,'run_packed'):
                new.flash_route=lambda op,q,k,v,out,head,scale:native_route.run_packed(op.packed,k,v,op.mk,op.mv,op.bias,out,head,5,scale)
            else:
                new.flash_route=lambda op,q,k,v,out,head,scale:native_route.run(q,k,v,op.mk,op.mv,op.bias,out,head,5,scale)
            result.update(native_route=True,allow_roundoff=True,route_resources=native_route.resources(),
                          numerical_screen='relative L2 <=0.002 and max abs <=0.25; screening only, generation required before promotion')
            encoded=torch.arange(256,device='cuda',dtype=torch.int32).to(torch.uint8)
            decoded=torch.empty(256,device='cuda',dtype=torch.bfloat16)
            native_route.decode(encoded,decoded)
            expected=encoded.view(torch.float8_e4m3fn).float().bfloat16()
            finite=torch.isfinite(expected)
            result['decoder_all_encodings']={'encodings':256,
                'finite_bits_equal':bool(torch.equal(decoded.view(torch.int16)[finite],expected.view(torch.int16)[finite])),
                'nan_mask_equal':bool(torch.equal(torch.isnan(decoded),torch.isnan(expected)))}
            assert result['decoder_all_encodings']['finite_bits_equal'] and result['decoder_all_encodings']['nan_mask_equal']
            result['source_files'][str(args.native_route_source.resolve().relative_to(ROOT))]=native_route.source_sha256
        result.update(source_sha256=new.merge.source_sha256,native_control_source_sha256=old.merge.source_sha256,resources=new.merge.resources())
        for source in (Path(__file__),ROOT/'native_fisher/fp8_c10.py',ROOT/'native_fisher/fp8_via_float_route.py',args.source.resolve(),args.reference_source.resolve()):
            result['source_files'][str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
        for layer in ('00','19','39'):
            payload=torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer'+layer+'.pt',map_location='cpu',weights_only=True)
            q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
            a=new(q,k,v);b=old(q,k,v)
            if args.paired_kv:
                result['paired_route_resources']={'registers':new._route.n_regs,'spills':new._route.n_spills,'shared_bytes':new._route.metadata.shared}
                result['reference_route_resources']={'registers':old._route.n_regs,'spills':old._route.n_spills,'shared_bytes':old._route.metadata.shared}
            if args.candidate_route_warps!=4:
                result['candidate_route_resources']={'registers':new._route.n_regs,'spills':new._route.n_spills,'shared_bytes':new._route.metadata.shared}
                result['reference_route_resources']={'registers':old._route.n_regs,'spills':old._route.n_spills,'shared_bytes':old._route.metadata.shared}
            delta=a.float()-b.float()
            correctness={'bitwise_equal':bool(torch.equal(a.view(torch.uint16),b.view(torch.uint16))),
                'finite':bool(torch.isfinite(a).all()),'relative_l2':float(delta.norm()/b.float().norm()),
                'max_abs':float(delta.abs().max()),'mean_abs':float(delta.abs().mean())}
            result['pending_correctness']={'layer':layer,**correctness};save()
            assert correctness['finite'],correctness
            if args.native_route_source or args.candidate_route_warps!=4:
                assert correctness['relative_l2']<=0.002 and correctness['max_abs']<=0.25,correctness
            else:assert correctness['bitwise_equal'],correctness
            assert torch.equal(new.seeds,old.seeds) and torch.equal(new.bias.view(torch.uint8),old.bias.view(torch.uint8))
            del a,b,delta
            row={'layer':layer,'native_control_correctness':correctness,
                 'seed_mismatch_count':0,'bias_bitwise_equal':True,'interleaved_blocks':[]}
            for _ in range(2):new(q,k,v);old(q,k,v)
            measured={'candidate':[],'native_control':[]}
            def timing(fn):
                assert not errors,errors
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();output=fn();b.record();b.synchronize()
                return a.elapsed_time(b)
            for block in range(8):
                order=('candidate','native_control','native_control','candidate') if block%2==0 else ('native_control','candidate','candidate','native_control')
                entries=[]
                for name in order:
                    ms=timing(lambda:(new if name=='candidate' else old)(q,k,v))
                    measured[name].append(ms);entries.append({'method':name,'ms':ms})
                row['interleaved_blocks'].append(entries)
            for name,values in measured.items():row[name]={'mean_ms':statistics.fmean(values),'samples_ms':values}
            flat=[x.reshape(1,75600,40,128).transpose(1,2) for x in (q,k,v)]
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for _ in range(2):torch.nn.functional.scaled_dot_product_attention(*flat)
                values=[timing(lambda:torch.nn.functional.scaled_dot_product_attention(*flat)) for _ in range(5)]
            row['sdpa']={'mean_ms':statistics.fmean(values),'samples_ms':values}
            row['speedup_vs_sdpa']=row['sdpa']['mean_ms']/row['candidate']['mean_ms']
            row['valid']=not errors;result['layers'].append(row);save()
            print('PAIRED',layer,row['candidate']['mean_ms'],row['native_control']['mean_ms'],flush=True)
            del q,k,v,flat
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in result['source_files'].items())
        result['valid']=not errors
    except BaseException as e:
        result.update(valid=False,error=repr(e));raise
    finally:
        stop.set();worker.join(5);result['valid']=result['valid'] and not errors;save()

if __name__=='__main__':main()
