"""Timed I2V generation with the Q80K90 even-capacity native kernels.

Runs the frozen I2V snapshot's unchanged generate.py (Wan2.1-I2V-14B-720P,
720x1280, 81 frames, 50 FlowUniPC steps, shift 5, CFG 5, base seed 0 + prompt
index) with its q80k90_c{C}_all backend, replacing only the adapter's kernel
factory by analysis/q80k90_even_kernel.Q80K90Even(C). Nothing in the snapshot
is modified. Before the model loads, the native kernels are built and checked
against the snapshot's production v4 operator on random QKV; during generation
every self-attention call is counted (4000 per video, all 40 layers, both CFG
branches) and checked finite on the first call. generate.py writes the usual
runtime.json (inference_seconds, attention ms per call, ...); summary.json
adds the means over the generated videos.

python q80k90_even_i2v_timing.py --plan .../experiment_plan.json --kernels ~/workspace/gma_q80k90_even_opt \
    --out /local2/.../run --capacity 8 --num-videos 5 [--num-shards 1 --shard 0]
"""
import argparse,hashlib,importlib.util,json,os,statistics,sys,time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',type=Path,required=True,help='an I2V experiment_plan.json with snapshot, configs, inputs and hashes')
    ap.add_argument('--kernels',type=Path,required=True,help='directory containing analysis/q80k90_even_*.{py,cu}')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--capacity',type=int,choices=[4,6,8,10],default=8)
    ap.add_argument('--num-videos',type=int,default=5);ap.add_argument('--num-shards',type=int,default=1);ap.add_argument('--shard',type=int,default=0)
    a=ap.parse_args()
    assert 'TORCH_EXTENSIONS_DIR' in os.environ,'set a private TORCH_EXTENSIONS_DIR for this Python environment'
    plan=json.loads(a.plan.read_text());snap=Path(plan['snapshot'])
    for rel,h in plan['source_sha256'].items():assert sha(snap/rel)==h,('snapshot changed',rel)
    inputs=[row for row in plan['input_manifest'] if row['index']<a.num_videos]
    assert [row['index'] for row in inputs]==list(range(a.num_videos))
    prompts=Path(plan['prompt_path']).read_text().splitlines()
    for row in inputs:
        assert sha(Path(row['image']))==row['image_sha256'],row['image']
        assert hashlib.sha256(prompts[row['index']].encode()).hexdigest()==row['prompt_sha256'],row['index']
        assert row['seed']==row['index']
    a.out.mkdir(parents=True,exist_ok=True)
    assert not list(a.out.glob('*.mp4')),'fresh output directory required'
    kdir=a.kernels/'analysis'
    kernel_sources={n:sha(kdir/n) for n in ('q80k90_even_kernel.py','q80k90_even_layout.cu','q80k90_even_select.cu','q80k90_even_merge.cu','q80k90_even_route.cu')}
    record=dict(started_at=time.time(),host=os.uname().nodename,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),capacity=a.capacity,
                plan=str(a.plan),snapshot=str(snap),protocol=plan['protocol'],inputs=inputs,kernel_sha256=kernel_sources,
                num_shards=a.num_shards,shard=a.shard,python=sys.executable)
    save(a.out/'timing_run.json',record)

    import torch
    torch.set_grad_enabled(False)
    even=module('q80k90_even_kernel',kdir/'q80k90_even_kernel.py')
    cap=a.capacity
    t0=time.time()
    even.native_layout(21,40);even.native_select(21,40,cap);even.native_merge(21,40,8,cap);even.native_route(21,40,8,cap)
    record['build_seconds']=time.time()-t0
    # Build check against the snapshot's production operator on random QKV.
    rec=module('q80k90_recommended_v4',snap/'kernels/experimental/q80k90_c4_c12/recommended.py')
    g=torch.Generator(device='cuda').manual_seed(0)
    q,k,v=(torch.randn(1,75600,40,128,device='cuda',dtype=torch.bfloat16,generator=g) for _ in range(3))
    new=even.Q80K90Even(cap)(q,k,v);v4=rec.make_q80k90(cap);old=v4(q,k,v);torch.cuda.synchronize()
    rel=float((new.float()-old.float()).norm()/old.float().norm())
    record['build_check']=dict(rel_l2_vs_v4=rel,finite=bool(torch.isfinite(new).all()))
    print(json.dumps({'event':'build_check',**record['build_check'],'build_seconds':record['build_seconds']}),flush=True)
    assert record['build_check']['finite'] and rel<1e-2,record['build_check']
    del new,old,v4,q,k,v;torch.cuda.empty_cache()
    save(a.out/'timing_run.json',record)

    sys.path[:0]=[str(snap)]
    os.chdir(snap)
    from attention import q80k90_wan as adapter
    calls=dict(count=0,checked=False)
    make0=adapter._make
    adapter._make=lambda capacity:even.Q80K90Even(capacity)
    original=adapter.q80k90_attention
    def attention(q,k,v,grid_size,capacity=8):
        assert capacity==cap and tuple(q.shape)==(1,75600,40,128) and tuple(map(int,grid_size))==(21,45,80)
        out=original(q,k,v,grid_size,capacity)
        if not calls['checked']:
            assert type(adapter._INST[capacity]).__name__=='Q80K90Even' and bool(torch.isfinite(out).all())
            calls['checked']=True
        calls['count']+=1
        return out
    adapter.q80k90_attention=attention
    gen=module('frozen_i2v_generate',snap/'generate.py')
    old_save=gen.save_video_atomic
    videos=[]
    def save_video(path,video,**kwargs):
        assert calls['count']==4000,calls
        assert tuple(video.shape)==(81,720,1280,3),tuple(video.shape)
        old_save(path,video,**kwargs)
        videos.append(dict(video=Path(path).name,attention_calls=calls['count'],saved_at=time.time()))
        record['videos']=videos;save(a.out/'timing_run.json',record)
        calls['count']=0
    gen.save_video_atomic=save_video
    sys.argv=['generate.py','--config',plan['configs'][str(cap)],'--checkpoint','pretrained',
              '--prompt_path',plan['prompt_path'],'--image_list',plan['image_list'],'--image_dir',plan['image_dir'],
              '--output_dir',str(a.out),'--runtime_json',str(a.out/'runtime.json'),'--num_videos',str(a.num_videos),
              '--num_samples','1','--seed','0','--num_prompt_shards',str(a.num_shards),'--prompt_shard_id',str(a.shard)]
    gen.main()
    runtime=json.loads((a.out/'runtime.json').read_text())
    rows=runtime['records'];assert rows and len(rows)==len(videos)
    assert all(r['attention_calls']==r['full_attention_calls']==4000 for r in rows)
    keys=('inference_seconds','mean_attention_ms','mean_full_attention_ms','mean_watts')
    record['summary']={f'mean_{k}':statistics.mean(r[k] for r in rows) for k in keys}
    record['summary'].update(videos=len(rows),per_video=[{k:r.get(k) for k in ('output','prompt_index')+keys} for r in rows])
    record['finished_at']=time.time();save(a.out/'timing_run.json',record)
    print(json.dumps({'event':'summary',**record['summary']}),flush=True)
    adapter._make=make0

if __name__=='__main__':main()
