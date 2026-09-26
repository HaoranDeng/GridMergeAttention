"""Bounded p=.4-only three-step diagnostic; never saves or inspects video."""
import argparse, csv, hashlib, importlib.util, json, os, signal, socket, subprocess, sys, time, traceback
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parent
OUT = CAMPAIGN / 'quick_diagnostic'
def read(p): return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker():
    plan=read(CAMPAIGN/'experiment_plan.json');snap=Path(plan['snapshot'])
    for root,hs in [(snap,plan['source_sha256']),(Path(plan['source_svg_root']),plan['svg_sha256'])]:
        for name,digest in hs.items(): assert sha(root/name)==digest,name
    assert sha(plan['config'])==plan['config_sha256']
    assert os.environ['SVG_TOP_P']=='0.4'
    os.chdir(snap);sys.path.insert(0,str(snap))
    import torch
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    from attention import svg_family_wan as family
    modules=family._load_svg_modules();cls=modules['ear'];am=modules['attention']
    import flashinfer
    assert not __import__('svg.timer',fromlist=['ENABLE_LOGGING']).ENABLE_LOGGING
    state=dict(calls=0,events=[],audit=True,iterations=[],rows=[],replays=[],current={})
    class Finished(Exception): pass
    def wrap(obj,name,label):
        original=getattr(obj,name)
        def timed(*args,**kwargs):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();result=original(*args,**kwargs);b.record()
            state['events'].append((label,a,b));return result
        setattr(obj,name,timed)
    old_cluster=cls.kmeans_clustering
    def cluster(self,*args,**kwargs):
        result=old_cluster(self,*args,**kwargs)
        if state['audit']:state['iterations'].append((int(result[3]),int(result[7])))
        return result
    cls.kmeans_clustering=cluster
    wrap(cls,'kmeans_clustering','kmeans')
    wrap(cls,'semantic_aware_permutation','semantic_total')
    for name,label in [('identify_dynamic_map_estimated','map'),('dynamic_block_sparse_prune_fwd_flashinfer','sparse_total'),('apply_inverse_permutation_triton','inverse'),('density_calculation','density')]:wrap(am,name,label)
    wrap(flashinfer.sparse.VariableBlockSparseAttentionWrapper,'plan','flashinfer_plan')
    wrap(flashinfer.sparse.VariableBlockSparseAttentionWrapper,'run','flashinfer_run')
    original=cls.attention_core_logic
    def measure(self,args,kwargs):
        state['events']=[];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();wall=time.perf_counter();a.record()
        y=original(self,*args,**kwargs);b.record();b.synchronize()
        stages={}
        for label,x,z in state['events']:stages[label]=stages.get(label,0)+x.elapsed_time(z)
        return y,dict(core_ms=a.elapsed_time(b),wall_ms=1000*(time.perf_counter()-wall),stages=stages)
    def core(self,*args,**kwargs):
        n=state['calls'];step=n//80;branch=(n%80)//40;layer=self.layer_idx
        if n==240:raise Finished()
        assert layer==n%40 and args[0].shape==(1,40,75600,128)
        selected=step in (1,2) and layer in (0,5,10,15,20,25,30,35)
        prior=(self.q_centroids.clone(),self.k_centroids.clone()) if selected else None
        y,record=measure(self,args,kwargs)
        state['rows'].append(dict(call=n,step=step,branch=branch,layer=layer,**record))
        if selected:
            after=(self.q_centroids,self.k_centroids);logging=self.logging_file
            modes=[('no_density_log',None,True),('no_log_no_audit',None,False),('log_to_dev_null','/dev/null',True),('original_log',logging,True)]
            if (n//5)%2:modes.reverse()
            for mode,path,audit in modes:
                self.q_centroids=prior[0].clone();self.k_centroids=prior[1].clone();self.logging_file=path;state['audit']=audit
                yy,rr=measure(self,args,kwargs)
                # Compare scalar tensor diagnostics only; never decode images/video.
                diff=(yy.float()-y.float()).abs().max().item();assert diff==0,(mode,diff)
                state['replays'].append(dict(call=n,step=step,branch=branch,layer=layer,mode=mode,max_abs_difference=diff,**rr))
                del yy
            self.q_centroids,self.k_centroids=after;self.logging_file=logging;state['audit']=True
        state['calls']+=1
        if n%40==39:
            save(OUT/'progress.json',dict(calls=state['calls'],step=step,time=time.time()))
            print(json.dumps(dict(calls=state['calls'],last_core_ms=record['core_ms'])),flush=True)
        return y
    cls.attention_core_logic=core
    spec=importlib.util.spec_from_file_location('frozen_generate',snap/'generate.py');gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    sys.argv=['generate.py','--config',plan['config'],'--checkpoint','pretrained','--prompt_path',plan['prompt_path'],'--image_list',plan['image_list'],'--image_dir',plan['image_dir'],'--output_dir',str(OUT/'unused_video_output'),'--runtime_json',str(OUT/'unused_runtime.json'),'--num_videos','1','--num_samples','1','--seed','0']
    try:gen.main()
    except Finished:pass
    else:raise RuntimeError('Three-step stop was not reached')
    assert state['calls']==240 and len(state['replays'])==128
    save(OUT/'diagnostic.json',dict(valid=True,top_p=.4,steps=3,calls=240,sampled_layers=[0,5,10,15,20,25,30,35],input=plan['input_manifest'][0],rows=state['rows'],replays=state['replays'],scope='First three steps of unchanged 50-step trajectory; init separately excluded. Same real QKV and restored centroids for logging/audit ablations. Core excludes native layout conversions. Not a full-video runtime or quality estimate.',time=time.time()))

def guard():
    assert socket.gethostname().split('.')[0]=='scai6'
    OUT.mkdir(exist_ok=False)
    sys.path.insert(0,str(CAMPAIGN/'source'))
    from gpu_checks import inventory,clean
    plan=read(CAMPAIGN/'experiment_plan.json');gpu=plan['gpus']['2']
    for _ in range(2):
        g,=[g for g in inventory() if g['uuid']==gpu['uuid']];assert clean(g),g;time.sleep(2)
    save(OUT/'preflight.json',dict(time=time.time(),gpu=g))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],SVG_TOP_P='0.4',TIME_BENCH='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',MAX_JOBS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',WAN_MODEL_ROOT=plan['protocol']['model_root'],GRIDMERGE_SDPA_BACKEND='flash_attn2',SPARSE_VIDEO_GEN_ROOT=plan['source_svg_root'],SVG_NUM_Q_CENTROIDS='300',SVG_NUM_K_CENTROIDS='1000',SVG_MIN_KC_RATIO='0.10',SVG_KMEANS_INIT='50',SVG_KMEANS_STEP='2',FLASHINFER_WORKSPACE_BASE='/xuanwu-tank/east/zrl/exp-gridmergeattn/.cache/environments/monarch-rt/flashinfer_workspace',TORCH_EXTENSIONS_DIR=str(CAMPAIGN/'gpu2/torch_extensions'),TRITON_CACHE_DIR=str(CAMPAIGN/'gpu2/triton_cache'),SVG_DENSITY_LOG=str(OUT/'density.jsonl'),PATH='/usr/local/cuda/bin:'+os.environ['PATH'])
    with (OUT/'worker.log').open('x') as log:child=subprocess.Popen([plan['python'],str(Path(__file__).resolve()),'--worker'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,guard=os.getpid(),gpu=gpu,time=time.time()))
    failure=None;start=time.monotonic()
    try:
        with (OUT/'monitor.jsonl').open('x') as monitor:
            while child.poll() is None:
                g,=[g for g in inventory() if g['uuid']==gpu['uuid']]
                foreign=[]
                for pid in g['pids']:
                    try:
                        if os.getsid(pid)!=child.pid:foreign.append(pid)
                    except ProcessLookupError:pass
                monitor.write(json.dumps(dict(time=time.time(),gpu=g,foreign=foreign))+'\n');monitor.flush()
                assert not foreign,foreign
                assert time.monotonic()-start<1200,'20-minute bounded diagnostic timeout'
                time.sleep(5)
        assert child.returncode==0,child.returncode
        assert read(OUT/'diagnostic.json')['valid']
    except BaseException:failure=traceback.format_exc()
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save(OUT/'completion.json',dict(valid=failure is None,failure=failure,time=time.time(),returncode=child.returncode))
    if failure:raise RuntimeError(failure)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
    worker() if a.worker else guard()
