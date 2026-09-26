"""Frozen I2V generation, native C4 gate and CPU paired-quality evaluation."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_text())


def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    os.replace(tmp, p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(plan):
    snap = Path(plan['snapshot'])
    for rel, h in plan['source_sha256'].items():
        assert sha(snap/rel) == h, rel
    prompts = Path(plan['prompt_path']).read_text().splitlines()
    names = Path(plan['image_list']).read_text().splitlines()
    assert len(prompts) == len(names) == len(plan['input_manifest']) == 50
    for i, row in enumerate(plan['input_manifest']):
        assert row['index'] == row['seed'] == i
        assert prompts[i] == row['prompt'] and names[i] == Path(row['image']).name
        assert hashlib.sha256(prompts[i].encode()).hexdigest() == row['prompt_sha256']
        assert sha(Path(row['image'])) == row['image_sha256']


def quality(plan):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    import torch
    import lpips
    assert not torch.cuda.is_available()
    torch.set_num_threads(4)
    metric = load_module('frozen_scores', Path(plan['snapshot'])/'analysis/score_videos_diffusers.py')
    model = lpips.LPIPS(net='alex').to('cpu').eval()
    out = ROOT/'C4/quality_cpu'
    out.mkdir(parents=True, exist_ok=True)
    for i in range(50):
        name = f'{i:03d}-00.mp4'
        dst = out/f'{i:03d}.json'
        assert not dst.exists(), 'Fresh evaluation only'
        while True:
            runtime = ROOT/'C4/videos/runtime.json'
            if runtime.exists() and any(r['output']==name for r in read(runtime)['records']):
                break
            if (ROOT/'failure.json').exists():
                raise RuntimeError('Generation failed')
            time.sleep(15)
        reference, method = ROOT/'Dense/videos'/name, ROOT/'C4/videos'/name
        for arm in ('Dense','C4'):
            assert read(ROOT/arm/'videos/prompt_manifest.json')['inputs'] == plan['input_manifest']
        proofs = read(ROOT/'C4/videos/purity_records.json')
        proof, = [p for p in proofs if p['video']==name]
        assert proof['dense_self_attention_calls'] == 0 and proof['attention_calls'] == 4000
        dense_hash, method_hash = sha(reference), sha(method)
        assert proof['video_sha256'] == method_hash
        started = time.perf_counter()
        row = metric.score_pair(reference, method, model, 'cpu')
        assert row['frames']==81 and all(math.isfinite(row[k]) for k in ('psnr','ssim','lpips'))
        assert sha(reference)==dense_hash and sha(method)==method_hash
        row.update(index=i, seed=i, dense_sha256=dense_hash, method_sha256=method_hash,
                   image_sha256=plan['input_manifest'][i]['image_sha256'],
                   evaluation_seconds=time.perf_counter()-started)
        save(dst,row)
        rows = [read(p) for p in sorted(out.glob('[0-9][0-9][0-9].json'))]
        save(out/'quality.json',dict(complete=len(rows)==50,device='cpu',resize=False,frames=81,
                                   videos=rows,summary=metric.aggregate(rows)))
        print(json.dumps(dict(event='quality',**row)),flush=True)


def generate(plan, mode):
    import torch
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu']['uuid']
    assert torch.cuda.device_count() == 1
    snap = Path(plan['snapshot'])
    kernel = snap/'kernels/experimental/q80k90_c4_c12'
    sys.path.insert(0,str(kernel))
    gate_mode = mode == 'gate'
    out = ROOT/'native_gate_probe' if gate_mode else ROOT/mode/'videos'
    out.mkdir(parents=True, exist_ok=True)
    assert not list(out.glob('*.mp4')), 'Fresh generation only; inspect before any resume'
    save(out/'prompt_manifest.json',dict(protocol=plan['protocol'],inputs=plan['input_manifest']))
    count = 0
    layers = [0]*40
    purity = []
    gate = dict(complete=False,passed=False,layers=[0,5,10,15,20,25,30,35],capacities=[4],rows=[],gpu=plan['gpu'],
                tolerance_note='Retain the existing engineering gate <0.02 vs previous implementation; not a user-defined universal tolerance.',
                scope='First I2V denoising step, both CFG branches; diagnostic, no saved videos')
    class GateComplete(Exception):
        pass
    if mode != 'Dense':
        from attention import q80k90_wan as adapter
        from recommended import make_q80k90
        from previous import create as previous
        from torch.nn.attention import sdpa_kernel, SDPBackend
        original = adapter.q80k90_attention
        samples = torch.tensor([(t*45+h)*80+w for t in (0,10,20) for h in (0,22,44) for w in (0,40,79)],device='cuda')
        def timed(fn):
            b,e = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            b.record(); result=fn(); e.record(); e.synchronize()
            return b.elapsed_time(e),result
        def invoke(q,k,v,grid_size,capacity=8):
            nonlocal count
            assert capacity==4 and tuple(q.shape)==(1,75600,40,128)
            assert tuple(map(int,grid_size))==(21,45,80)
            layer,branch=count%40,count//40
            count+=1; layers[layer]+=1
            if not gate_mode:
                return original(q,k,v,grid_size,capacity)
            q,k,v=(x.contiguous() for x in (q,k,v))
            def dense():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
            dense_out=dense()
            if layer in gate['layers']:
                rng=torch.cuda.get_rng_state().clone()
                old=previous(4); ref=old(q,k,v).index_select(1,samples).float().clone()
                del old; gc.collect(); torch.cuda.empty_cache()
                core=make_q80k90(4); result=core(q,k,v)
                assert bool(torch.isfinite(result).all())
                rel=float((result.index_select(1,samples).float()-ref).norm()/ref.norm())
                dense_ref=dense_out.index_select(1,samples).float()
                dense_rel=float((result.index_select(1,samples).float()-dense_ref).norm()/dense_ref.norm())
                assert rel<.02,(layer,branch,rel)
                del result,ref,dense_ref
                times={'method':[],'dense':[]}
                for rep in range(3):
                    calls=[('method',lambda:core(q,k,v)),('dense',dense)]
                    for name,fn in (calls if rep%2==0 else calls[::-1]):
                        ms,result=timed(fn); times[name].append(ms); del result
                row=dict(layer=layer,branch=branch,relative_l2_vs_previous=rel,relative_l2_vs_dense=dense_rel,
                         method_ms=times['method'],dense_ms=times['dense'])
                assert statistics.mean(times['method'])<statistics.mean(times['dense']),row
                gate['rows'].append(row); save(ROOT/'native_gate.json',gate)
                print(json.dumps(dict(event='gate_case',**row)),flush=True)
                del core; gc.collect(); torch.cuda.empty_cache()
                assert torch.equal(rng,torch.cuda.get_rng_state())
            if count==80:
                raise GateComplete()
            return dense_out
        adapter.q80k90_attention=invoke
    gen=load_module('frozen_generate',snap/'generate.py')
    old_save=gen.save_video_atomic
    def save_video(path,video,**kwargs):
        nonlocal count,layers
        assert tuple(video.shape)==(81,720,1280,3)
        if mode=='C4':
            assert count==4000 and layers==[100]*40
        old_save(path,video,**kwargs)
        row=dict(video=Path(path).name,video_sha256=sha(Path(path)),gpu=plan['gpu'],saved_at=time.time())
        if mode=='C4':
            row.update(attention_calls=count,layer_calls=layers,dense_self_attention_calls=0,
                       full_50_step_pflops=plan['selected_kernels']['4']['full_50_step_pflops'])
        purity.append(row); save(out/'purity_records.json',purity)
        count=0; layers=[0]*40
    gen.save_video_atomic=save_video
    sys.argv=['generate.py','--config',plan['configs']['C4' if gate_mode else mode],'--checkpoint','pretrained',
              '--prompt_path',plan['prompt_path'],'--image_list',plan['image_list'],'--image_dir',plan['image_dir'],
              '--output_dir',str(out),'--runtime_json',str(out/'runtime.json'),'--num_videos','1' if gate_mode else '50',
              '--num_samples','1','--seed','0']
    if gate_mode:
        try:
            gen.main()
        except GateComplete:
            assert count==80 and layers==[2]*40 and len(gate['rows'])==16
            gate.update(complete=True,passed=True,finished_at=time.time());save(ROOT/'native_gate.json',gate)
        else:
            raise AssertionError('Gate did not stop at first complete CFG step')
    else:
        if mode=='C4':
            assert read(ROOT/'native_gate.json')['passed']
        gen.main()
        records=read(out/'runtime.json')['records']
        assert len(records)==len(purity)==50
        assert [r['prompt_index'] for r in records]==list(range(50))
        assert all(r['attention_calls']==r['full_attention_calls']==4000 for r in records)
        save(ROOT/mode/'generation_complete.json',dict(complete=True,videos=50,time=time.time()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['Dense','gate','C4','quality'])
    ap.add_argument('--parallel',action='store_true',help='Use the explicitly authorized C4 GPU assignment')
    a=ap.parse_args()
    plan=read(ROOT/'experiment_plan.json')
    assignment_path=ROOT/'parallel_plan.json'
    if a.parallel or (assignment_path.exists() and a.mode in ('gate','C4')):
        assignment=read(assignment_path)
        assert assignment['original_plan_sha256']==sha(ROOT/'experiment_plan.json')
        assert assignment['methods']==['gate','C4'] and assignment['max_gpus']==2
        if not a.parallel:
            # The already-running Dense supervisor later reaches these stages.
            # Await their new owner without loading a model or duplicating C4.
            assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu']['uuid']
            marker=ROOT/'native_gate.json' if a.mode=='gate' else ROOT/'C4/generation_complete.json'
            started=time.monotonic()
            while not marker.exists():
                failure=ROOT/'parallel/failure.json'
                if failure.exists():
                    raise RuntimeError(f'Parallel C4 failed: {read(failure)}')
                assert time.monotonic()-started<120*3600,'Parallel stage wait exceeded 120 hours'
                time.sleep(10)
            result=read(marker)
            # native_gate.json exists while the gate is still in progress.
            while not result.get('complete'):
                failure=ROOT/'parallel/failure.json'
                if failure.exists():
                    raise RuntimeError(f'Parallel C4 failed: {read(failure)}')
                assert time.monotonic()-started<120*3600,'Parallel stage wait exceeded 120 hours'
                time.sleep(10);result=read(marker)
            if a.mode=='gate':
                assert result['passed'] and result['gpu']==assignment['gpu']
            else:
                assert result['videos']==50
                records=read(ROOT/'C4/videos/purity_records.json')
                assert len(records)==50 and all(r['gpu']==assignment['gpu'] for r in records)
            print(json.dumps(dict(event='parallel_stage_already_owned',stage=a.mode,gpu=assignment['gpu'])),flush=True)
            return
        assert a.mode in ('gate','C4'), 'Dense and CPU evaluation retain their original owner'
        plan=dict(plan,gpu=assignment['gpu'])
    verify(plan)
    sys.path.insert(0,plan['snapshot']);os.chdir(plan['snapshot'])
    quality(plan) if a.mode=='quality' else generate(plan,a.mode)


if __name__=='__main__':
    main()
