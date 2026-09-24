"""Unchanged I2V generator, with native gates and per-video pure-call audits."""
import argparse
import errno
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def read(p):
    # Live public records are replaced atomically across scai6/scai7's shared
    # NFS mount. Retry transient lookups; retain a bounded failure for real errors.
    for attempt in range(20):
        try:return json.loads(p.read_text())
        except json.JSONDecodeError:
            if attempt==19:raise
        except OSError as e:
            if e.errno not in (errno.EEXIST,errno.ESTALE,errno.ENOENT,errno.EAGAIN) or attempt==19:raise
        time.sleep(.2)
def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(d,indent=2)+'\n'); os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capacity',type=int,choices=[6,10],default=6)
    parser.add_argument('--gate',action='store_true');parser.add_argument('--gpu',type=int,choices=[2],default=2);a=parser.parse_args()
    plan=read(ROOT/'experiment_plan.json'); snap=Path(plan['snapshot']);kernel=snap/'kernels/experimental/q80k90_c4_c12'
    gpu=dict(plan['gpu'])
    control=ROOT/'gpu2';control.mkdir(exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES']==gpu['uuid']
    for rel,h in plan['source_sha256'].items():assert sha(snap/rel)==h,rel
    for rel,h in plan['svg_sha256'].items():assert sha(Path(plan['source_svg_root'])/rel)==h,rel
    for row in plan['input_manifest']:
        assert sha(Path(row['image']))==row['image_sha256']
    out=control/'native_gate_probe' if a.gate else ROOT/f'C{a.capacity}/video20'
    out.mkdir(parents=True,exist_ok=True)
    assert not list(out.glob('*.mp4')), 'Fresh method output required; diagnose before resume'
    save(out/'input_manifest.json',dict(protocol=plan['protocol'],inputs=plan['input_manifest']))
    sys.path[:0]=[str(snap),str(kernel)]
    os.chdir(snap)
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from attention import q80k90_wan as adapter
    from recommended import make_q80k90
    from previous import create as previous
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    original=adapter.q80k90_attention
    gate=dict(complete=False,passed=False,scope='I2V first denoising step, both CFG branches, native integration/speed check',
              layers=[0,5,10,15,20,25,30,35],capacities=[6,10],rows=[],gpu=gpu)
    count=0;layers=[0]*40;purity=[]
    samples=torch.tensor([(t*45+h)*80+w for t in (0,10,20) for h in (0,22,44) for w in (0,40,79)],device='cuda')
    class GateComplete(Exception):pass
    def timed(fn):
        b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        b.record();result=fn();e.record();e.synchronize();return b.elapsed_time(e),result
    def invoke(q,k,v,grid_size,capacity=8):
        nonlocal count
        assert tuple(q.shape)==(1,75600,40,128) and tuple(map(int,grid_size))==(21,45,80)
        layer=count%40;branch=count//40;count+=1;layers[layer]+=1
        if not a.gate:
            assert capacity==a.capacity
            return original(q,k,v,grid_size,capacity)
        q,k,v=(x.contiguous() for x in (q,k,v))
        def dense():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
        dense_out=dense()
        if layer in gate['layers']:
            rng=torch.cuda.get_rng_state().clone()
            for cap in gate['capacities']:
                old=previous(cap);ref=old(q,k,v).index_select(1,samples).float().clone()
                del old;gc.collect();torch.cuda.empty_cache()
                core=make_q80k90(cap);o=core(q,k,v)
                assert bool(torch.isfinite(o).all())
                rel=float((o.index_select(1,samples).float()-ref).norm()/ref.norm())
                dense_rel=float((o.index_select(1,samples).float()-dense_out.index_select(1,samples).float()).norm()/dense_out.index_select(1,samples).float().norm())
                assert rel<.02,(cap,layer,branch,rel)
                del o,ref
                times={'method':[],'dense':[]}
                for rep in range(3):
                    calls=[('method',lambda:core(q,k,v)),('dense',dense)]
                    for name,fn in (calls if rep%2==0 else calls[::-1]):
                        ms,result=timed(fn);times[name].append(ms);del result
                row=dict(capacity=cap,layer=layer,branch=branch,relative_l2_vs_previous=rel,relative_l2_vs_dense=dense_rel,
                         method_ms=times['method'],dense_ms=times['dense'])
                assert statistics.mean(times['method'])<statistics.mean(times['dense']),row
                gate['rows'].append(row);save(control/'native_gate.json',gate)
                print(json.dumps({'event':'native_gate_case',**row}),flush=True)
                del core;gc.collect();torch.cuda.empty_cache()
            assert torch.equal(rng,torch.cuda.get_rng_state()),'Gate kernel consumed RNG'
        if count==80:raise GateComplete()
        return dense_out
    adapter.q80k90_attention=invoke
    spec=importlib.util.spec_from_file_location('frozen_i2v_generate',snap/'generate.py')
    gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    old_save=gen.save_video_atomic
    def save_video(path,video,**kwargs):
        nonlocal count,layers
        assert count==4000 and layers==[100]*40,(count,layers)
        assert tuple(video.shape)==(81,720,1280,3),tuple(video.shape)
        old_save(path,video,**kwargs)
        assert Path(path).name=='019-00.mp4'
        row=dict(video=Path(path).name,index=19,seed=19,attention_calls=count,layer_calls=layers,dense_self_attention_calls=0,
                 video_sha256=sha(Path(path)),full_50_step_pflops=plan['selected_kernels'][str(a.capacity)]['full_50_step_pflops'],
                 gpu=gpu,saved_at=time.time())
        purity.append(row);save(out/'purity_records.json',purity)
        count=0;layers=[0]*40
    gen.save_video_atomic=save_video
    sys.argv=['generate.py','--config',plan['configs'][str(a.capacity)],'--checkpoint','pretrained',
              '--prompt_path',plan['prompt_path'],'--image_list',plan['image_list'],'--image_dir',plan['image_dir'],
              '--output_dir',str(out),'--runtime_json',str(out/'runtime.json'),'--num_videos','20',
              '--num_samples','1','--seed','0','--num_prompt_shards','20','--prompt_shard_id','19']
    if a.gate:
        try:gen.main()
        except GateComplete:
            assert count==80 and len(gate['rows'])==32 and layers==[2]*40
            gate.update(complete=True,passed=True,finished_at=time.time());save(control/'native_gate.json',gate)
            print('NATIVE I2V GATE PASS',flush=True)
        else:raise AssertionError('Native gate did not intercept first complete CFG step')
    else:
        assert read(control/'native_gate.json')['passed']
        gen.main()
        runtime=read(out/'runtime.json');assert len(runtime['records'])==len(purity)==1
        assert all(r['attention_calls']==r['full_attention_calls']==4000 for r in runtime['records'])
        save(ROOT/f'C{a.capacity}/generation_complete.json',dict(complete=True,videos=1,index=19,seed=19,time=time.time()))

if __name__=='__main__':main()
