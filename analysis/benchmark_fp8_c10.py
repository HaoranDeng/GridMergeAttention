"""C10 FP8-storage experiment: real layers, paired complete H40, raw errors.

Quantization errors are reported, not accepted using the BF16 roundoff gate.
No generation quality or production eligibility is inferred from this test.
"""
import argparse, hashlib, json, os, statistics, sys, threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layers',nargs='+',default=['00','19','39'])
    p.add_argument('--merge-source',type=Path)
    p.add_argument('--route-sweep',action='store_true')
    p.add_argument('--mode',choices=['scaled','direct'],default='scaled')
    p.add_argument('--fused-selector',action='store_true')
    p.add_argument('--route-stages',type=int,default=3)
    p.add_argument('--route-block',type=int,default=64)
    p.add_argument('--flashinfer-route',action='store_true')
    p.add_argument('--pipeline',action='store_true')
    p.add_argument('--unit-scale-route',action='store_true')
    p.add_argument('--via-float-route',action='store_true')
    p.add_argument('--integer-route',action='store_true')
    p.add_argument('--same-fp8-reference',action='store_true',help='Require exact complete output against retained single-round FP8 merge and route')
    p.add_argument('--head-chunk',type=int,choices=[1,5],default=5)
    p.add_argument('--single-round',action='store_true',help='FP32 directly to FP8; gate against combined BF16/FP8 rounding envelope')
    args=p.parse_args()
    if args.single_round:assert args.mode=='direct' and args.merge_source
    assert not args.output.exists()
    initial=require_clean(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES']=initial['gpu']['uuid']
    os.environ['TORCH_EXTENSIONS_DIR']=str(args.output.parent/'torch_extensions')
    os.environ['TRITON_CACHE_DIR']=str(args.output.parent/'triton_cache')
    os.environ['OMP_NUM_THREADS']='4'
    stop=threading.Event();errors=[];samples=[]
    def monitor():
        while not stop.is_set():
            try:
                ps=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':ps})
                if any(r['pid']!=os.getpid() for r in ps):errors.append('foreign GPU process')
            except Exception as e:errors.append(repr(e))
            stop.wait(3)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    result={'initial_gpu':initial,'scope':'Explicit user C10 FP8 compressed KV storage experiment; complete H40 with unchanged BF16 QKV/weights/dot products, Fisher C10 RF0 SQ1 full route',
            'generation_validation_pending':True,'production_default_changed':False,'layers':[]}
    result.update(target_ms=150,mode=args.mode,fused_selector=args.fused_selector,single_round=args.single_round)
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples)
        args.output.write_text(json.dumps(result,indent=2))
    try:
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from native_fisher.fp8_c10 import retained_control,FP8C10Attention
        torch.set_num_threads(4);torch.set_grad_enabled(False)
        if args.integer_route:
            assert args.mode=='direct' and args.same_fp8_reference and not args.via_float_route and not args.pipeline
            import triton
            import triton.language as tl
            from native_fisher.fp8_integer_route import _integer_decode
            @triton.jit
            def decode_all(src,dst):
                idx=tl.arange(0,256)
                tl.store(dst+idx,_integer_decode(tl.load(src+idx)))
            encoded=torch.arange(256,device='cuda',dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn)
            decoded=torch.empty(256,device='cuda',dtype=torch.bfloat16)
            decode_all[(1,)](encoded,decoded)
            expected=encoded.float().bfloat16()
            finite=torch.isfinite(expected)
            result['integer_decoder_all_encodings']={'finite_bits_equal':bool(torch.equal(decoded.view(torch.int16)[finite],expected.view(torch.int16)[finite])),
                                                    'nan_mask_equal':bool(torch.equal(torch.isnan(decoded),torch.isnan(expected))),'encodings':256}
            print('INTEGER_DECODER',result['integer_decoder_all_encodings'],flush=True);save()
            assert all(result['integer_decoder_all_encodings'][k] for k in ('finite_bits_equal','nan_mask_equal'))
        control=retained_control();op=FP8C10Attention(control,args.mode,args.fused_selector,args.merge_source,head_chunk=args.head_chunk)
        op.integer_route=args.integer_route
        result['integer_route']=args.integer_route
        result['head_chunk']=args.head_chunk
        op.route_stages=args.route_stages;op.route_block=args.route_block
        if args.flashinfer_route:
            assert args.mode=='direct' and not args.route_sweep and args.head_chunk==5
            from native_fisher.fp8_flashinfer_full import FlashinferFullRoute
            op.flash_route=FlashinferFullRoute()
        result['flashinfer_route']=args.flashinfer_route
        if args.unit_scale_route:
            assert args.mode=='direct' and not args.flashinfer_route
            op.ks.fill_(1);op.vs.fill_(1);op.unit_scale_route=True
        result['unit_scale_route']=args.unit_scale_route
        if args.via_float_route:
            assert args.mode=='direct' and not args.unit_scale_route and not args.flashinfer_route
            if args.pipeline:
                assert args.same_fp8_reference, 'Pipeline conversion changes require the complete retained FP8 reference'
            op.via_float_route=True
        result['via_float_route']=args.via_float_route
        if args.pipeline:
            assert not args.flashinfer_route and not args.route_sweep
            op.enable_pipeline()
        result['pipeline']=args.pipeline
        reference_op=None
        if args.same_fp8_reference:
            assert args.mode=='direct' and args.single_round
            reference_op=FP8C10Attention(control,'direct',False,ROOT/'native_fisher/fisher_c10_fp8_shared_store.cu')
            reference_op.via_float_route=True
            reference_op.route_block=64;reference_op.route_stages=1;reference_op.route_warps=4
            result['reference_fp8_source_sha256']=reference_op.merge.source_sha256
        result.update(source_sha256=op.merge.source_sha256,native_control_source_sha256=control.merge.source_sha256,
                      selector_source_sha256=op.fused.source_sha256 if op.fused else hashlib.sha256(control.native_selector.source_path.read_bytes()).hexdigest(),
                      control_selector_source_sha256=hashlib.sha256(control.native_selector.source_path.read_bytes()).hexdigest(),
                      resources=op.merge.resources(),control_resources=control.merge.resources())
        result['source_files']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                               [Path(__file__),ROOT/'native_fisher/fp8_c10.py',ROOT/'native_fisher/fp8_direct_route.py',ROOT/'attention/hard_c4_wan21_14b_720p_triton.py']}
        if args.via_float_route:
            f=ROOT/'native_fisher/fp8_via_float_route.py'
            result['source_files'][str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
        if args.integer_route:
            for name in ('fp8_integer_route.py','fp8_via_float_route.py'):
                f=ROOT/'native_fisher'/name
                result['source_files'][str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
        if args.flashinfer_route:
            import flashinfer
            result['flashinfer_version']=flashinfer.__version__
            for name in ('fp8_flashinfer_route.py','fp8_flashinfer_full.py','fp8_route_states.py'):
                f=ROOT/'native_fisher'/name
                result['source_files'][str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
        result['workspace_bytes']={'bf16_kv':2*control.mk.numel()*2,'fp8_kv':2*op.mk.numel(),
                                  'fp8_scales':2*op.ks.numel()*4 if args.mode=='scaled' else 0,'bias_each':op.bias.numel()*4}
        print('READY',result['resources'],flush=True);save()
        def check(a,b):
            diff=a.float()-b.float()
            return {'bitwise_equal':bool(torch.equal(a,b)),'max_abs':float(diff.abs().max()),
                    'mean_abs':float(diff.abs().mean()),'relative_l2':float((diff.square().sum()/b.float().square().sum().clamp_min(1e-30)).sqrt()),
                    'finite':bool(torch.isfinite(a).all())}
        def timed(fn,repeats=5):
            for _ in range(2):fn()
            values=[]
            for _ in range(repeats):
                assert not errors,errors
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();y=fn();b.record();b.synchronize();values.append(a.elapsed_time(b));del y
            return {'mean_ms':statistics.fmean(values),'samples_ms':values}
        for layer in args.layers:
            print('LOAD_LAYER',layer,flush=True)
            payload=torch.load(f'/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{layer}.pt',map_location='cpu',weights_only=True)
            q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
            row={'layer':layer};result['layers'].append(row)
            if reference_op is not None:
                reference_fp8=reference_op(q,k,v);candidate_fp8=op(q,k,v)
                row['error_to_retained_fp8']=check(candidate_fp8,reference_fp8)
                print('RETAINED_FP8_ERROR',row['error_to_retained_fp8'],flush=True);save()
                assert row['error_to_retained_fp8']['bitwise_equal'],row
                del reference_fp8,candidate_fp8
            if args.via_float_route and reference_op is None:
                op.via_float_route=False;reference_fp8=op(q,k,v)
                op.via_float_route=True;candidate_fp8=op(q,k,v)
                row['via_float_route_error']=check(candidate_fp8,reference_fp8)
                assert row['via_float_route_error']['bitwise_equal'],row
                del reference_fp8,candidate_fp8
            if args.unit_scale_route:
                assert not args.pipeline
                op.unit_scale_route=False;reference_fp8=op(q,k,v)
                op.unit_scale_route=True;candidate_fp8=op(q,k,v)
                row['unit_scale_route_error']=check(candidate_fp8,reference_fp8)
                assert row['unit_scale_route_error']['bitwise_equal'],row
                del reference_fp8,candidate_fp8
            if args.pipeline:
                op.pipeline_enabled=False;reference_fp8=op(q,k,v)
                op.pipeline_enabled=True;candidate_fp8=op(q,k,v)
                row['pipeline_error_to_serial']=check(candidate_fp8,reference_fp8)
                assert row['pipeline_error_to_serial']['bitwise_equal'],row
                del reference_fp8,candidate_fp8
            if args.flashinfer_route:
                fi=op.flash_route
                op.flash_route=None;reference_fp8=op(q,k,v)
                op.flash_route=fi;candidate_fp8=op(q,k,v)
                row['flashinfer_error_to_same_fp8_operator']=check(candidate_fp8,reference_fp8)
                print('FLASHINFER_ERROR',row['flashinfer_error_to_same_fp8_operator'],flush=True);save()
                assert row['flashinfer_error_to_same_fp8_operator']['finite'] and row['flashinfer_error_to_same_fp8_operator']['relative_l2']<.002
                del candidate_fp8,reference_fp8
            a=op(q,k,v);candidate_seeds=op.seeds.clone();b=control(q,k,v)
            row['native_control_correctness']=check(a,b)
            row['route_resources']={'registers':op._route.n_regs,'spills':op._route.n_spills,'shared_bytes':op._route.metadata.shared}
            row['seed_mismatch_count']=int((candidate_seeds!=control.seeds).sum());del candidate_seeds
            assert row['native_control_correctness']['finite'] and row['seed_mismatch_count']==0
            del a,b
            # Stratified samples spanning every query frame/column/head and
            # compressed key region catch layout/scale mistakes. Both current
            # workspaces contain the last H5 chunk of the complete call.
            indices=torch.arange(65536,device='cuda',dtype=torch.int64)*(op.ks.numel()-1)//65535
            storage={}
            expected_kv=[t if args.head_chunk==5 else t.reshape(21,80,5,9450,128)[:,:,4:5].reshape(-1,9450,128) for t in (control.mk,control.mv)]
            for name,actual,scales,expected in [('k',op.mk,op.ks,expected_kv[0]),('v',op.mv,op.vs,expected_kv[1])]:
                values=actual.reshape(-1,128)[indices].float()
                sc=scales.flatten()[indices] if args.mode=='scaled' else torch.ones(indices.numel(),device='cuda')
                ref=expected.reshape(-1,128)[indices].float()
                refscale=(ref.abs().amax(-1)/448).clamp_min(1e-8) if args.mode=='scaled' else torch.ones_like(sc)
                reference=(ref/sc[:,None]).to(torch.float8_e4m3fn).float()
                item=check(values*sc[:,None],ref)
                item['scale_relative_l2']=check(sc,refscale)['relative_l2']
                item['quantized_reference_mismatch_fraction']=float((values!=reference).float().mean())
                item['max_normalized_quantization_error']=float((values-ref/sc[:,None]).abs().max())
                item['positive_finite_scales']=bool(((sc>0)&torch.isfinite(sc)).all())
                item['reference_absmax']=float(ref.abs().max())
                item['sample_values_outside_e4m3_range']=int((ref.abs()>448).sum())
                if args.mode=='direct':assert item['sample_values_outside_e4m3_range']==0,item
                assert item['finite'] and item['positive_finite_scales'] and item['scale_relative_l2']<1e-6,item
                if args.single_round:
                    # Reference is rounded BF16, while this variant replaces
                    # that boundary. Bound the sum of the two rounding errors;
                    # this is a storage/layout screen, not a quality acceptance.
                    envelope=ref.abs()*(1/16+1/256)+1/1024
                    item['combined_rounding_envelope_violations']=int(((values-ref).abs()>envelope).sum())
                    assert item['combined_rounding_envelope_violations']==0,item
                else:
                    assert item['quantized_reference_mismatch_fraction']<1e-4 and item['max_normalized_quantization_error']<16.01,item
                storage[name]=item
                del values,sc,ref,refscale,reference
            row['storage_samples_65536_tokens_last_chunk']=storage;del indices,expected_kv
            expected_bias=control.bias if args.head_chunk==5 else control.bias.reshape(21,80,5,9450)[:,:,4:5].reshape(-1,9450)
            row['bias_bitwise_equal']=bool(torch.equal(op.bias,expected_bias));assert row['bias_bitwise_equal']
            del expected_bias
            print('ERROR',layer,row['native_control_correctness'],storage,flush=True);save()
            if args.route_sweep and layer==args.layers[0]:
                assert args.head_chunk==5
                # Sweep only the FP8 route. Compare every configuration to the
                # same dequantized-BF16 operator, separate from quantization error.
                expected=torch.empty_like(q);op.route(q,k,v,expected,35,128**-.5)
                actual=torch.empty_like(q);sweep=[]
                for block,stages in ((32,1),(32,2),(32,3),(64,1),(64,2),(64,3),(128,1),(128,2),(128,3),(256,1)):
                    op.route_block,op.route_stages=block,stages
                    item={'block':block,'stages':stages,'warps':4}
                    from triton.runtime.errors import OutOfResources
                    try:
                        op.route(q,k,v,actual,35,128**-.5)
                    except OutOfResources as e:
                        item['excluded_compile_resource_error']=str(e);sweep.append(item);continue
                    item['error_to_default_fp8_route']=check(actual[...,35:40,:],expected[...,35:40,:])
                    item['resources']={'registers':op._route.n_regs,'spills':op._route.n_spills,'shared_bytes':op._route.metadata.shared}
                    assert item['error_to_default_fp8_route']['finite'] and item['error_to_default_fp8_route']['relative_l2']<.002
                    item['timing']=timed(lambda:op.route(q,k,v,actual,35,128**-.5))
                    sweep.append(item);print('ROUTE',item,flush=True)
                chosen=min((x for x in sweep if 'timing' in x),key=lambda x:x['timing']['mean_ms'])
                op.route_block,op.route_stages=chosen['block'],chosen['stages']
                result['route_sweep_h5']=sweep;result['chosen_route']=chosen
                del expected,actual
                a=op(q,k,v);b=control(q,k,v)
                row['native_control_correctness']=check(a,b);assert row['native_control_correctness']['finite']
                del a,b;save()
            for _ in range(2):op(q,k,v);control(q,k,v)
            torch.cuda.synchronize()
            paired={'candidate':[],'control':[]};row['interleaved_blocks']=[]
            for block in range(8):
                order=('candidate','control','control','candidate') if block%2==0 else ('control','candidate','candidate','control')
                blockrows=[]
                for name in order:
                    assert not errors,errors
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();output=(op if name=='candidate' else control)(q,k,v);end.record();end.synchronize()
                    ms=start.elapsed_time(end);del output
                    paired[name].append(ms);blockrows.append({'method':name,'ms':ms})
                row['interleaved_blocks'].append(blockrows)
            row['candidate']={'mean_ms':statistics.fmean(paired['candidate']),'samples_ms':paired['candidate']}
            row['native_control']={'mean_ms':statistics.fmean(paired['control']),'samples_ms':paired['control']}
            row['timing_protocol']='Eight alternating ABBA/BAAB blocks; 16 complete H40 calls per method after warmup; all intermediate IO included'
            flat=[x.reshape(1,75600,40,128).transpose(1,2) for x in (q,k,v)]
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):row['sdpa']=timed(lambda:torch.nn.functional.scaled_dot_product_attention(*flat))
            row['speedup_vs_sdpa']=row['sdpa']['mean_ms']/row['candidate']['mean_ms'];del flat
            if layer==args.layers[0]:
                seeds=op.select(q,k,128**-.5);out=torch.empty_like(q);op.merge.pack(q,op.packed)
                result['stages_diagnostic_h5']={
                    'selector':timed(lambda:op.select(q,k,128**-.5)),
                    'pack':timed(lambda:op.merge.pack(q,op.packed)),
                    'bf16_merge':timed(lambda:control.merge.run(control.packed,k,v,seeds,control.mk,control.mv,control.bias,0,5,control.tiles,128**-.5,control.empty,0,1)),
                    'bf16_route':timed(lambda:control.route(q,k,v,out,0,128**-.5)),
                    'fp8_merge':timed(lambda:op.merge_chunk(k,v,seeds,0,128**-.5)),
                    'fp8_route':timed(lambda:op.route(q,k,v,out,0,128**-.5))}
                result['stage_probe_head_counts']={'bf16':5,'fp8':args.head_chunk}
                del seeds,out
            print('TIMING',layer,row['candidate']['mean_ms'],row['native_control']['mean_ms'],row['speedup_vs_sdpa'],flush=True)
            row['valid']=not errors;save();del q,k,v
        result['valid']=not errors;save()
    except BaseException as e:
        result.update(valid=False,error=repr(e));save();raise
    finally:
        stop.set();thread.join(10);save()


if __name__=='__main__':main()
