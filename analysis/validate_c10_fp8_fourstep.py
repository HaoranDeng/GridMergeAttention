"""Frozen native BF16/FP8 comparison: 20 canonical videos, all 40 layers, 4 steps."""
import argparse,gc,hashlib,json,os,runpy,shutil,subprocess,sys,time
from collections import Counter
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
METHOD='fisher_geometric_c10_rf0'
GRID=(1,21,45,80,40,128)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    temp=Path(path).with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def worker(run):
    import torch
    from analysis.run_quality_first_videos import asset_preflight,HYBRID_FIELDS
    meta=json.loads((run/'manifest.json').read_text())
    assert asset_preflight(meta['checkpoint'])==meta['assets']
    for name,digest in meta['sources'].items():assert sha(ROOT/name)==digest,name
    from native_fisher.fp8_c10 import retained_control,FP8C10Attention
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    control=retained_control()
    candidate=FP8C10Attention(control,'direct',False,ROOT/'native_fisher/fisher_c10_fp8_shared_store.cu')
    candidate.via_float_route=True;candidate.route_block=64;candidate.route_stages=1;candidate.route_warps=4
    ops={'control':control,'candidate':candidate}
    # Compile and warm the exact complete operators before model loading and
    # attention-event collection. Capture provenance remains separate.
    payload=torch.load(meta['warmup_capture'],map_location='cpu',weights_only=True)
    q,k,v=[payload[n].reshape(GRID).cuda() for n in ('q','k','v')];del payload
    a=control(q,k,v);b=candidate(q,k,v)
    diff=b.float()-a.float()
    proof={'finite':bool(torch.isfinite(b).all()),'relative_l2':float((diff.square().sum()/a.float().square().sum()).sqrt()),
           'seed_equal':bool(torch.equal(control.seeds,candidate.seeds)),
           'bf16_source_sha256':control.merge.source_sha256,'fp8_source_sha256':candidate.merge.source_sha256}
    assert proof['finite'] and proof['seed_equal'] and proof['relative_l2']<.08,proof
    write(run/'warmup.json',proof)
    del q,k,v,a,b,diff;gc.collect();torch.cuda.empty_cache()
    namespace=runpy.run_path(str(ROOT/'generate.py'),run_name='native_c10_fp8_generate')
    import wan.modules.model as wan
    from attention.runtime_timing import ATTENTION_RUNTIME_RECORDER as recorder
    from utils.prompt_contract import assert_matching_prompt_manifests
    original_forward=wan.WanSelfAttention.forward;original_finish=recorder.finish_video
    counters={name:Counter() for name in ops};layers={name:Counter() for name in ops}
    active={'mode':None,'layer':None,'core':0};records=[]
    for name,op in ops.items():
        for attribute in ('select','route'):
            original=getattr(op,attribute)
            def wrapped(*args,_name=name,_attribute=attribute,_original=original,**kwargs):
                assert active['mode']==_name and active['layer'] is not None
                counters[_name][_attribute]+=1
                return _original(*args,**kwargs)
            setattr(op,attribute,wrapped)
        for attribute in ('run','pack'):
            original=getattr(op.merge,attribute)
            def wrapped(*args,_name=name,_attribute=attribute,_original=original,**kwargs):
                assert active['mode']==_name and active['layer'] is not None
                counters[_name][_attribute]+=1
                return _original(*args,**kwargs)
            setattr(op.merge,attribute,wrapped)
    def forward(module,x,grid_size,rope_cache):
        mode=active['mode'];assert mode in ops and active['layer'] is None
        assert tuple(grid_size)==(21,45,80) and module.layer_idx==sum(layers[mode].values())%40
        assert module.enable_token_merging and module.token_merging_dense_heads==()
        assert all(not getattr(module,'token_merging_'+key) for key in HYBRID_FIELDS)
        assert all(getattr(module,key) is None for key in ('seed_query_proj','gma_output_calibration_weights','gma_query_calibration_parameters'))
        before=counters[mode].copy();active.update(layer=module.layer_idx,core=0)
        output=original_forward(module,x,grid_size,rope_cache)
        assert active['core']==1
        assert counters[mode]-before==Counter(select=1,route=8,run=8,pack=1)
        layers[mode][module.layer_idx]+=1;active['layer']=None
        return output
    def core(*args,**kwargs):
        assert active['layer'] is not None and active['core']==0
        assert tuple(args[3:8])==(1,1,1,45,80)
        assert kwargs.get('seed_query') is None and kwargs.get('q_init') is None
        assert all(x.shape==(1,75600,40,128) and x.dtype==torch.bfloat16 for x in args[:3])
        active['core']=1
        output=ops[active['mode']](*(x.contiguous().reshape(GRID) for x in args[:3]))
        assert bool(torch.isfinite(output).all()),'nonfinite native attention output'
        return output.reshape(1,75600,40,128)
    def finish_video():
        timing=original_finish();mode=active['mode']
        assert timing['attention_calls']==timing['full_attention_calls']==160
        index=sum(r['variant']==mode for r in records)
        assert layers[mode]==Counter({i:(index+1)*4 for i in range(40)})
        records.append({'variant':mode,'prompt_index':index,**timing,'native_stage_calls':dict(counters[mode]),'layer_calls':dict(layers[mode])})
        write(run/'dispatch_audit.json',{'complete':False,'videos':records,'dense_calls':0})
        return timing
    with patch.object(wan.WanSelfAttention,'forward',forward),patch.object(wan,'token_merging_attn',core),patch.object(recorder,'finish_video',finish_video):
        for name in ops:
            active['mode']=name;directory=run/name;directory.mkdir()
            argv=[str(ROOT/'generate.py'),'--config',str(ROOT/'configs/native_c10_fp8_4step.json'),
                  '--checkpoint',meta['checkpoint'],'--prompt_path',str(ROOT/'prompts/MovieGenVideoBench_extended.txt'),
                  '--output_dir',str(directory/'videos'),'--runtime_json',str(directory/'runtime.json'),
                  '--num_videos',str(meta['num_videos']),'--num_samples','1','--batch_size','1','--seed','0',
                  '--dtype','bf16','--fps','16','--video_preset','ultrafast','--video_crf','23']
            write(run/'phase.json',{'phase':'generation','variant':name,'completed_videos':len(records)})
            with patch.object(sys,'argv',argv):namespace['main']()
            runtime=json.loads((directory/'runtime.json').read_text())
            assert runtime['checkpoint_weights']=='raw_state_dict'
            assert [r['prompt_index'] for r in runtime['records']]==list(range(meta['num_videos']))
            assert all(r['attention_calls']==r['full_attention_calls']==160 for r in runtime['records'])
            assert sorted(f.name for f in (directory/'videos').glob('*.mp4'))==[f'{i:03d}-00.mp4' for i in range(meta['num_videos'])]
            gc.collect();torch.cuda.empty_cache()
    assert len(records)==2*meta['num_videos']
    assert_matching_prompt_manifests(run/'control/videos',run/'candidate/videos')
    for name,digest in meta['sources'].items():assert sha(ROOT/name)==digest,name
    assert asset_preflight(meta['checkpoint'])==meta['assets']
    write(run/'dispatch_audit.json',{'complete':True,'videos':records,'dense_calls':0})
    write(run/'phase.json',{'phase':'generation_complete','quality_metrics_pending':True,'completed_videos':len(records)})

def launch(args):
    from analysis.launch_quality_first_videos import require_clean,gpu_processes,child_environment
    from analysis.run_quality_first_videos import asset_preflight,config_document
    os.environ['WAN_MODEL_ROOT']=str(args.model_root.resolve())
    initial=require_clean(args.gpu);assets=asset_preflight(args.checkpoint)
    run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=False);frozen=run/'source'
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    names+=['analysis/validate_c10_fp8_fourstep.py','prompts/MovieGenVideoBench_extended.txt']
    extensions={'.py','.yaml','.yml','.json','.jinja','.jinja2','.cu','.cuh','.h','.hpp'}
    names=sorted({n for n in names if n and (Path(n).suffix in extensions or n.startswith('prompts/'))
                  and not n.startswith(('experiments/','outputs/','videos/'))})
    sources={}
    for name in names:
        src=(ROOT/name).resolve();assert src.is_relative_to(ROOT) and src.is_file() and src.stat().st_size<=32*1024*1024,name
        dst=frozen/name;dst.parent.mkdir(parents=True,exist_ok=True);before=sha(src);shutil.copyfile(src,dst)
        assert before==sha(src)==sha(dst);sources[name]=before
    config=frozen/'configs/native_c10_fp8_4step.json';write(config,config_document(METHOD));sources[str(config.relative_to(frozen))]=sha(config)
    meta={'num_videos':args.num_videos,'steps':4,'expected_calls_per_video':160,'seed':0,
          'checkpoint':str(args.checkpoint.resolve()),'assets':assets,'sources':sources,
          'actual_operator':'native Fisher Uniform16 C10 RF0 SQ1 full route in every layer and call; BF16 control vs direct E4M3 intermediate KV',
          'config_scope':'Existing token-merging projection/setup configuration; complete attention core replaced by the audited native operator.',
          'timing_scope':'Instrumented generation; includes finite checks; no same-stream SDPA ratio inferred.',
          'warmup_capture':'/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt',
          'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    write(run/'manifest.json',meta)
    subprocess.run([sys.executable,str(frozen/'analysis/verify_prompt_contract.py'),'--prompt_path',str(frozen/'prompts/MovieGenVideoBench_extended.txt'),
                    '--num_videos',str(args.num_videos),'--seed','0','--manifest_out',str(run/'prompt_preflight.json')],check=True,stdout=subprocess.DEVNULL)
    initial=require_clean(args.gpu,expected_uuid=initial['gpu']['uuid']);env=child_environment(run,initial['gpu']['uuid'])
    env['TORCH_EXTENSIONS_DIR']=str(run.parent/'torch_extensions');env['TRITON_CACHE_DIR']=str(run.parent/'triton_cache')
    samples=[];errors=[];child=None;code=None;verified=False
    try:
        with (run/'generation.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(frozen/'analysis/validate_c10_fp8_fourstep.py'),'--worker','--run-dir',str(run)],cwd=frozen,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(run/'process.json',{'parent_pid':os.getpid(),'worker_pid':child.pid,'run':str(run),'gpu':initial})
            while child.poll() is None:
                ps=gpu_processes(initial['gpu']['uuid']);samples.append({'time':time.time(),'processes':ps})
                if any(p['pid']!=child.pid for p in ps):errors.append('foreign GPU process');child.terminate();break
                write(run/'gpu_purity.json',{'initial':initial,'samples':samples,'errors':errors})
                time.sleep(5)
            code=child.wait(timeout=30)
        assert code==0 and not errors,(code,errors)
        assert json.loads((run/'dispatch_audit.json').read_text())['complete']
        verified=True
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
        write(run/'gpu_purity.json',{'initial':initial,'samples':samples,'errors':errors})
        write(run/'completion.json',{'generation_valid':verified,'worker_exit_code':code,'errors':errors,'quality_metrics_pending':True})
        (run/'generation.exitcode').write_text('0\n' if verified else '1\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--gpu',type=int)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--model-root',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--num-videos',type=int,default=20);args=p.parse_args()
    assert 1<=args.num_videos<=20
    worker(args.run_dir.resolve()) if args.worker else launch(args)
