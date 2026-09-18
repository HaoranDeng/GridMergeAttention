"""Validate and time complete Fisher C4 calls on freshly exclusive H200."""
import argparse,json,os,sys,threading,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layer',default='00')
    p.add_argument('--capacity',type=int,choices=[4,10],default=4)
    p.add_argument('--tiles-sweep',action='store_true')
    p.add_argument('--route-sweep',action='store_true')
    p.add_argument('--route-block',type=int,default=64)
    p.add_argument('--route-warps',type=int,default=4)
    p.add_argument('--route-stages',type=int,default=3)
    p.add_argument('--profile-only',action='store_true')
    p.add_argument('--io-diagnostic',action='store_true')
    p.add_argument('--activity-trace',action='store_true')
    p.add_argument('--cuda-graph',action='store_true')
    p.add_argument('--native-control-source',type=Path)
    p.add_argument('--interleaved-control',action='store_true')
    p.add_argument('--merge-source',type=Path)
    p.add_argument('--merge-tiles',type=int,choices=[1,3,7,21,42,84,294],default=21)
    p.add_argument('--native-control-tiles',type=int,default=21)
    p.add_argument('--workspace-columns',type=int,choices=[1,16],default=1)
    p.add_argument('--selector-backend',choices=['triton','probe4','probe8','probe16'],default='probe4')
    p.add_argument('--selector-source',type=Path)
    p.add_argument('--selector-control',action='store_true')
    p.add_argument('--joint-control',action='store_true',help='Compare old selector plus --native-control-source together')
    p.add_argument('--allow-rounding',action='store_true',help='Permit ordinary kernel rounding; retain finite/L2 gates and report bitwise equality')
    p.add_argument('--max-relative-l2',type=float,default=.002)
    args=p.parse_args()
    if args.joint_control and not (args.selector_control and args.native_control_source):
        p.error('--joint-control requires --selector-control and --native-control-source')
    if args.joint_control and args.merge_tiles!=21:
        p.error('Nondefault merge tile counts require the separately configured native control')
    if args.interleaved_control and not args.native_control_source:
        p.error('--interleaved-control requires --native-control-source')
    print('PREFLIGHT',flush=True)
    if args.cuda_graph and (args.profile_only or args.tiles_sweep or args.route_sweep):
        p.error('Graph validation must run separately from profile-only or configuration sweeps')
    assert not args.output.exists()
    initial=require_clean(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=initial['gpu']['uuid']
    os.environ['TORCH_EXTENSIONS_DIR']=str(args.output.parent/'torch_extensions');os.environ['TRITON_CACHE_DIR']=str(args.output.parent/'triton_cache')
    os.environ['OMP_NUM_THREADS']='4'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    stop=threading.Event();errors=[];samples=[]
    def monitor():
        while not stop.is_set():
            try:
                ps=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':ps})
                if any(r['pid']!=os.getpid() for r in ps):errors.append('foreign GPU process')
            except Exception as e:errors.append(repr(e))
            stop.wait(3)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    result={'initial_gpu':initial,'method':f'fisher_geometric_c{args.capacity}_rf0_native_merge','scope':'Complete all40 includes current Fisher Uniform16 selector, packing, SQ1 merge, original full route','target_ms':80 if args.capacity==4 else 150}
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples);args.output.write_text(json.dumps(result,indent=2))
    try:
        print('IMPORT_TORCH',flush=True)
        import torch
        from native_fisher.operator import NativeFisherAttention
        from attention.affine_hellinger_seed_triton import affine_hellinger_seed_rows_triton
        from attention.singleton_c4_wan21_14b_720p_triton import _hard_c4_720p_temporal_shared_global_mass_attention_triton_blocks as ref
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        print('LOAD_CAPTURE',flush=True)
        payload=torch.load(f'/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{args.layer}.pt',map_location='cpu',weights_only=True)
        q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
        print('BUILD_EXTENSION',flush=True)
        op=NativeFisherAttention(capacity=args.capacity,tiles=args.merge_tiles,route_block=args.route_block,route_warps=args.route_warps,route_stages=args.route_stages,selector_backend=args.selector_backend,merge_source=args.merge_source,workspace_columns=args.workspace_columns,selector_source=args.selector_source);print('EXTENSION_READY',op.merge.resources(),flush=True)
        result['layer']=args.layer
        result['operator_config']={'selector_backend':op.selector_backend,'selector_block':op.selector_block,'tiles':op.tiles,'route_block':op.route_block,'route_warps':op.route_warps,'route_stages':op.route_stages}
        result['operator_config']['workspace_columns']=op.workspace_columns
        if op.workspace_columns != 1:
            result['workspace_route_source_sha256']=hashlib.sha256((ROOT/'native_fisher/column_workspace_route.py').read_bytes()).hexdigest()
        if op.native_selector is not None:
            result['selector_source_sha256']=hashlib.sha256(op.native_selector.source_path.read_bytes()).hexdigest()
        if args.profile_only:
            op(q,k,v,head_count=5);torch.cuda.synchronize()
            result.update(profile_only=True,scope='Profiler-only H5 call; no usable full-call timing or correctness claim',valid=not errors)
            save();return
        def reference_seeds():
            return affine_hellinger_seed_rows_triton(q[:,:1],k,scale=128**-.5,feature_mode='fisher_geometric',num_seeds=args.capacity,block_j=4)
        a=op.select(q,k,128**-.5);b=reference_seeds()
        result['seed_mismatch_count']=int((a!=b).sum());del a,b
        assert result['seed_mismatch_count']==0
        result['resources']=op.merge.resources();result['source_sha256']=op.merge.source_sha256
        if args.capacity==10:
            from analysis.fisher_c10_video_adapter import FisherC10MergeAdapter
            ref=FisherC10MergeAdapter()
        def reference(head_count=40):
            seeds=reference_seeds();out=torch.empty_like(q)
            chunk=(4 if args.capacity==10 else 8) if head_count==40 else 5
            for head in range(0,head_count,chunk):
                ref(q,k,v,seeds,scale=128**-.5,top_r=1,head_start=head,head_count=chunk,
                    **({'selector':'query_column'} if args.capacity==10 else {'query_column_selector':True}),
                    seed_rows_native_layout=True,query_column_major=True,merge_loop_stages=1 if args.capacity==10 else 2,merge_disable_licm=False,
                    split_kv_accum=True,block_j=4,output=out)
            return out
        def check(a,b):
            diff=a.float()-b.float();return {'bitwise_equal':torch.equal(a,b),'max_abs':float(diff.abs().max()),
                'relative_l2':float((diff.square().sum()/b.float().square().sum()).sqrt()),'finite':bool(torch.isfinite(a).all())}
        a=op(q,k,v,head_count=5)[...,:5,:].clone();b=reference(5)[...,:5,:].clone()
        result['h5_correctness']=check(a,b);save();print('H5',result['h5_correctness'],flush=True)
        assert result['h5_correctness']['finite'] and result['h5_correctness']['relative_l2']<.002,result['h5_correctness']
        del a,b
        a=op(q,k,v);b=reference();result['full40_correctness']=check(a,b);save();print('FULL40',result['full40_correctness'],flush=True)
        assert result['full40_correctness']['finite'] and result['full40_correctness']['relative_l2']<.002
        del a,b
        def timed(fn,repeats=5):
            for _ in range(2):fn()
            values=[]
            for _ in range(repeats):
                assert not errors,errors
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();y=fn();b.record();b.synchronize();values.append(a.elapsed_time(b));del y
            return {'mean_ms':statistics.fmean(values),'samples_ms':values}
        result['candidate']=timed(lambda:op(q,k,v));result['reference']=timed(reference)
        if args.selector_control:
            from native_fisher.selector import NativeFisherSelector
            selected=op.native_selector
            control_selector=NativeFisherSelector(args.capacity,layout='probe4')
            selected_merge=op.merge
            from native_fisher.operator import extension
            joint_merge=extension(args.capacity,args.native_control_source) if args.joint_control else selected_merge
            result['joint_control']=args.joint_control
            result['joint_control_merge_sha256']=joint_merge.source_sha256
            def with_selector(selector):
                try:
                    op.native_selector=selector
                    op.merge=joint_merge if selector is control_selector else selected_merge
                    return op(q,k,v)
                finally:op.native_selector=selected;op.merge=selected_merge
            a=with_selector(selected);b=with_selector(control_selector)
            result['selector_control_correctness']=check(a,b);del a,b
            gate=result['selector_control_correctness']
            assert gate['bitwise_equal'] or (args.allow_rounding and gate['finite'] and gate['relative_l2']<args.max_relative_l2)
            result['numerical_policy']={'bitwise_required':not args.allow_rounding,'max_relative_l2':args.max_relative_l2,'generation_validation_required':args.allow_rounding}
            times={'candidate':[],'control':[]}
            for block in range(6):
                for name in (('candidate','control','control','candidate') if block%2==0 else ('control','candidate','candidate','control')):
                    assert not errors,errors
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();y=with_selector(selected if name=='candidate' else control_selector);b.record();b.synchronize()
                    times[name].append(a.elapsed_time(b));del y
            result['candidate_unpaired']=result['candidate']
            result['candidate']={'mean_ms':statistics.fmean(times['candidate']),'samples_ms':times['candidate']}
            result['selector_control']={'mean_ms':statistics.fmean(times['control']),'samples_ms':times['control']}
            result['timing_protocol']='Six alternating ABBA/BAAB blocks, 12 complete calls per implementation; '+('joint old GS/merge control' if args.joint_control else 'same merge and route')
            print('SELECTOR_FULL40',result['candidate'],result['selector_control'],flush=True);save()
        if args.native_control_source and not args.joint_control:
            from native_fisher.operator import extension
            control=extension(args.capacity,args.native_control_source,'_control')
            candidate_merge=op.merge
            candidate_columns=op.workspace_columns
            candidate_tiles=op.tiles
            result['native_control_tiles']=args.native_control_tiles
            def native_control():
                try:
                    op.merge=control
                    op.workspace_columns=1
                    op.tiles=args.native_control_tiles
                    return op(q,k,v)
                finally:
                    op.merge=candidate_merge
                    op.workspace_columns=candidate_columns
                    op.tiles=candidate_tiles
            a=op(q,k,v);b=native_control()
            result['native_control_correctness']=check(a,b);del a,b
            gate=result['native_control_correctness']
            assert gate['bitwise_equal'] or (args.allow_rounding and gate['finite'] and gate['relative_l2']<args.max_relative_l2),'Native output failed numerical gate'
            result['numerical_policy']={'bitwise_required':not args.allow_rounding,'max_relative_l2':args.max_relative_l2,'generation_validation_required':args.allow_rounding}
            result['native_control']=timed(native_control)
            if args.interleaved_control:
                result['candidate_unpaired']=result['candidate']
                result['native_control_unpaired']=result['native_control']
                paired={'candidate':[],'control':[]}
                result['interleaved_blocks']=[]
                for block in range(8):
                    order=('candidate','control','control','candidate') if block%2==0 else ('control','candidate','candidate','control')
                    rows=[]
                    for name in order:
                        assert not errors,errors
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        output=op(q,k,v) if name=='candidate' else native_control()
                        end.record();end.synchronize()
                        ms=start.elapsed_time(end);del output
                        paired[name].append(ms);rows.append({'method':name,'ms':ms})
                    result['interleaved_blocks'].append(rows)
                result['candidate']={'mean_ms':statistics.fmean(paired['candidate']),'samples_ms':paired['candidate']}
                result['native_control']={'mean_ms':statistics.fmean(paired['control']),'samples_ms':paired['control']}
                result['timing_protocol']='Eight alternating ABBA/BAAB blocks; 16 complete calls per method after correctness/warmup'
            result['native_control_source_sha256']=control.source_sha256
            print('NATIVE_CONTROL',result['native_control'],result['native_control_correctness'],flush=True);save()
        measured_call=lambda:op(q,k,v)
        if args.cuda_graph:
            from native_fisher.graph import GraphFisherAttention
            graph=GraphFisherAttention(op,q,k,v)
            result['uncaptured_control']=result['candidate']
            result['graph_bitwise_checks']=[]
            for changed in (False,True,False):
                test_q=q*.875 if changed else q
                expected=op(test_q,k,v);actual=graph(test_q,k,v)
                equal=torch.equal(actual,expected)
                result['graph_bitwise_checks'].append({'changed_query':changed,'equal':equal})
                assert equal,'Graph output differs from uncaptured native output'
                del test_q,expected,actual
            measured_call=lambda:graph(q,k,v)
            result['candidate']=timed(measured_call)
            result['graph_scope']='Full operator replay including Q/K/V input copies and independent output clone; capture/warmup excluded'
            result['operator_config']['cuda_graph']=True
            print('GRAPH',result['candidate'],result['uncaptured_control'],flush=True);save()
        if args.tiles_sweep:
            native21=op(q,k,v)
            result['candidate_tiles21']=result['candidate']
            result['tile_sweep']=[]
            for tiles in [1,3,7,21]:
                op.tiles=tiles;actual=op(q,k,v)
                equal=torch.equal(actual,native21);del actual
                assert equal,f'tile {tiles} changed native output'
                row={'tiles':tiles,'bitwise_equal_to_tiles21':equal,'timing':timed(lambda:op(q,k,v))}
                result['tile_sweep'].append(row);save();print('TILE',json.dumps(row),flush=True)
            best=min(result['tile_sweep'],key=lambda row:row['timing']['mean_ms'])
            op.tiles=best['tiles'];result['selected_tiles']=op.tiles;result['candidate']=best['timing'];del native21
        if args.route_sweep:
            op(q,k,v,head_count=5)
            route_ref=torch.empty_like(q);op.route(q,k,v,route_ref,0,128**-.5)
            route_ref=route_ref[...,:5,:].clone();route_out=torch.empty_like(q)
            configs=[(b,4,s) for b in (64,128,256) for s in (1,3)]+[(b,8,3) for b in (128,256)]
            result['route_sweep']=[]
            for block,warps,stages in configs:
                op.route_block,op.route_warps,op.route_stages=block,warps,stages
                row={'block':block,'warps':warps,'stages':stages}
                try:
                    op.route(q,k,v,route_out,0,128**-.5)
                    row['correctness']=check(route_out[...,:5,:],route_ref)
                    assert row['correctness']['finite'] and row['correctness']['relative_l2']<.002
                    row['timing']=timed(lambda:op.route(q,k,v,route_out,0,128**-.5))
                except Exception as e:row['error']=repr(e)
                result['route_sweep'].append(row);save();print('ROUTE',json.dumps(row),flush=True)
            best=min((x for x in result['route_sweep'] if 'error' not in x),key=lambda x:x['timing']['mean_ms'])
            op.route_block,op.route_warps,op.route_stages=best['block'],best['warps'],best['stages']
            result['selected_route']={k:best[k] for k in ('block','warps','stages')}
            result['candidate_default_route']=result['candidate'];result['candidate']=timed(lambda:op(q,k,v))
            a=op(q,k,v);b=reference();result['full40_selected_route_correctness']=check(a,b)
            assert result['full40_selected_route_correctness']['finite'] and result['full40_selected_route_correctness']['relative_l2']<.002
            del a,b,route_out,route_ref
        seeds=op.select(q,k,128**-.5);out=torch.empty_like(q)
        result['stages']={'selector':timed(lambda:op.select(q,k,128**-.5)),
            'pack':timed(lambda:op.merge.pack(q,op.packed)),
            'merge_h5':timed(lambda:op.merge.run(op.packed,k,v,seeds,op.mk,op.mv,op.bias,0,5,op.tiles,128**-.5,op.empty,0,1)),
            'route_h5':timed(lambda:op.route(q,k,v,out,0,128**-.5))}
        from torch.nn.attention import sdpa_kernel,SDPBackend
        flat=[x.reshape(1,75600,40,128).transpose(1,2) for x in (q,k,v)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):result['sdpa']=timed(lambda:torch.nn.functional.scaled_dot_product_attention(*flat))
        result['speedup_vs_sdpa']=result['sdpa']['mean_ms']/result['candidate']['mean_ms'];result['valid']=not errors
        if args.io_diagnostic:
            byte_count=2*op.mk.numel()*op.mk.element_size()
            fill=timed(lambda:(op.mk.zero_(),op.mv.zero_()))
            copy=timed(lambda:op.mk.copy_(op.mv))
            result['io_diagnostic']={'h5_kv_bytes':byte_count,'full40_logical_kv_write_plus_read_bytes':byte_count*8*2,
                'h5_contiguous_write':fill,'h5_contiguous_copy_read_plus_write':copy,
                'write_TB_per_s':byte_count/(fill['mean_ms']*1.e9),'copy_TB_per_s':byte_count/(copy['mean_ms']*1.e9),
                'scope':'Bandwidth controls only, not measured merge DRAM traffic or attention results'}
            result['valid']=not errors
        if args.activity_trace:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function('native_fisher_full40'):
                    traced=measured_call();torch.cuda.synchronize();del traced
            trace_path=args.output.with_suffix('.trace.json')
            prof.export_chrome_trace(str(trace_path))
            result['activity_trace']=str(trace_path)
            result['valid']=not errors
        save();print(json.dumps({k:v for k,v in result.items() if k!='monitor_samples'}),flush=True)
    except BaseException as e:
        result.update(valid=False,error=repr(e));save();raise
    finally:stop.set();thread.join(10);save()

if __name__=='__main__':main()
