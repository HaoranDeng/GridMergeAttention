"""Paired H40 timing and fixed-eight-layer QKV accuracy for current-frame C8.

Run under analysis.guard_axis_job on one exclusively idle H200.
"""
import argparse
import ctypes
import gc
import itertools
import json
import os
from pathlib import Path
import statistics
import time

LAYERS = (0, 5, 10, 15, 20, 25, 30, 35)
HEADS = (0, 10, 20, 30)
SAMPLES = ((0, 10, 20), (0, 22, 44), (0, 40, 79))


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--timing-only', action='store_true')
    p.add_argument('--quality-only', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite results')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(a.output.parent/'extensions'))
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6', mode=ctypes.RTLD_GLOBAL)
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from kernels.experimental.c8_current import create
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {'complete':False, 'layers':list(LAYERS), 'heads':list(HEADS), 'samples':SAMPLES,
              'capacity':8, 'selector':'shipped r2 Hellinger C32/g45 A8U8 alpha=.2',
              'changed':'independent selection per query frame, including adaptive probes',
              'prompts':3, 'steps':4, 'rows':[], 'psnr':None,'ssim':None,'lpips':None,'vbench':None}
    save(a.output, report)

    def measure(fn):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); value = fn(); end.record(); end.synchronize()
        return start.elapsed_time(end), value

    if not a.quality_only:
        payload = torch.load(a.capture, map_location='cpu', mmap=True, weights_only=True)
        q,k,v = [payload[n].cuda().contiguous() for n in ('q','k','v')]
        del payload
        assert q.shape == (1,75600,40,128)
        print('compile current-frame C8 H40', flush=True)
        op = create()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            def dense():
                return torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)).transpose(1,2).contiguous()
            def current(): return op(q,k,v)
            def rf0(): return op(q,k,v,policy='rf0')
            for fn in (dense,rf0,current):
                value=fn(); assert bool(value.isfinite().all()); del value
            current_seeds = op.select(q,k).clone()
            report['different_seed_fraction_after_frame0'] = float(
                (current_seeds[:,1:] != current_seeds[:,:1]).float().mean())
            # Broadcast seeds must reproduce the RF0 control's output exactly.
            rf0_seeds = current_seeds[:,:1].clone()
            broadcast = rf0_seeds.expand_as(current_seeds).contiguous()
            y0=op.run_with_seeds(q,k,v,rf0_seeds)
            y1=op.run_with_seeds(q,k,v,broadcast)
            report['broadcast_matches_rf0'] = bool(torch.equal(y0,y1))
            assert report['broadcast_matches_rf0']
            del current_seeds, broadcast, y1
            print('compile shipped r2 and slow_reference controls', flush=True)
            from kernels.best import create_c8, create_c8_slow_reference
            baseline=create_c8()
            yr2=baseline(q,k,v)
            gap=(y0.float()-yr2.float()).norm()/yr2.float().norm()
            report['rf0_vs_shipped_r2_relative_l2']=float(gap)
            assert float(gap) < .002, 'RF0 address/layout regression'
            report['rf0_seeds_equal_shipped_r2']=bool(torch.equal(rf0_seeds,baseline.seeds))
            assert report['rf0_seeds_equal_shipped_r2']
            slow=create_c8_slow_reference()
            ys=slow(q,k,v)
            report['r2_vs_slow_reference_relative_l2']=float((yr2.float()-ys.float()).norm()/ys.float().norm())
            del slow,ys,yr2,y0,rf0_seeds;gc.collect();torch.cuda.empty_cache()
            rows=[]
            for i in range(7):
                row={}
                functions=[('dense',dense),('r2_rf0',lambda:baseline(q,k,v)),('current',current)]
                for name,fn in functions[::1 if i%2==0 else -1]:
                    row[name]=measure(fn)[0]
                rows.append(row)
                print(json.dumps({'timing_repeat':i,**row}),flush=True)
            report['timing']={'capture':str(a.capture),'heads':40,'paired_ms':rows,
                'scope':'complete token-major attention including selector, pack, merge, route; dense output contiguous',
                'mean_attention_ms':{n:statistics.mean(r[n] for r in rows) for n in rows[0]}}
            means=report['timing']['mean_attention_ms']
            report['timing']['speedup_vs_sdpa']={n:means['dense']/ms for n,ms in means.items()}
            report['timing']['selector_current_ms']=measure(lambda:op.select(q,k))[0]
            report['timing']['selector_rf0_ms']=measure(lambda:op.select(q,k,policy='rf0'))[0]
            report['timing']['speed_gate_passed']=means['current'] <= means['dense']*1.05
            save(a.output,report)
            print(json.dumps({'timing':report['timing']}),flush=True)
        del baseline,op,q,k,v;gc.collect();torch.cuda.empty_cache()
    if a.timing_only:
        report['complete']=True;save(a.output,report);return

    expected=[a.captures/f'p{p:03d}_s{s}_l{l:02d}.pt' for p in range(3) for s in range(4) for l in LAYERS]
    missing=[str(x) for x in expected if not x.is_file()]
    if missing:
        raise ValueError(f'Missing fixed-eight-layer captures; no fallback: {missing[:3]} ({len(missing)} total)')
    receipt=json.loads((a.captures.parent/'results.json').read_text())
    assert receipt['complete'], 'Capture run must be complete before comparison'
    assert receipt['layers']==list(LAYERS) and receipt['heads']==list(HEADS)
    assert receipt['base_seed']==0 and receipt['samples']==[list(s) for s in SAMPLES]
    manifest=json.loads((a.captures.parent/'prompt_manifest.json').read_text())
    report['capture_protocol']={k:receipt[k] for k in ('layers','heads','samples','base_seed','per_prompt_seed_rule','checkpoint_identity')}
    report['capture_receipt']=str(a.captures.parent/'results.json')
    report['capture_hash_pins']=receipt['captures']
    report['prompt_manifest']=manifest
    report['quality_scope']='27 sampled queries/head; 3 prompts x 4 steps x 8 layers x 4 heads; external dense trajectory'
    op=create(heads=1)
    coords=list(itertools.product(*SAMPLES))
    indices=torch.tensor([(t*45+h)*80+w for t,h,w in coords],device='cuda')
    for prompt in range(3):
        for step in range(4):
            for layer in LAYERS:
                path=a.captures/f'p{prompt:03d}_s{step}_l{layer:02d}.pt'
                payload=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
                meta=payload['metadata']
                assert (meta['prompt'],meta['step'],meta['layer'],meta['heads'])==(prompt,step,layer,list(HEADS))
                for slot,head in enumerate(HEADS):
                    q,k,v=[payload[n][:,:,slot:slot+1].cuda().contiguous() for n in ('q','k','v')]
                    if not report['rows']:
                        from analysis.validate_c8_current import validate
                        report['semantic_validation']=validate(op,q,k,v)
                        save(a.output,report)
                    target=payload['dense_output'][0,indices.cpu(),slot].cuda().float()
                    cases={}
                    for policy in ('rf0','current'):
                        value=op(q,k,v,policy=policy)[0,indices,0].float()
                        error=value-target
                        assert bool(value.isfinite().all())
                        cases[policy]={'relative_l2':float(error.norm()/target.norm()),
                            'squared_error':float(error.square().sum()),'reference_squared':float(target.square().sum()),
                            'per_frame_l2':{str(t):float(error[i*9:(i+1)*9].norm()/target[i*9:(i+1)*9].norm())
                                            for i,t in enumerate(SAMPLES[0])}}
                    report['rows'].append({'prompt':prompt,'step':step,'layer':layer,'head':head,'cases':cases})
                    del q,k,v,target,value,error
                summary={}
                for policy in ('rf0','current'):
                    rr=[r['cases'][policy] for r in report['rows']]
                    summary[policy]={'cases':len(rr),'mean_relative_l2':statistics.mean(r['relative_l2'] for r in rr),
                        'pooled_relative_l2':(sum(r['squared_error'] for r in rr)/sum(r['reference_squared'] for r in rr))**.5}
                summary['current_wins']=sum(r['cases']['current']['relative_l2']<r['cases']['rf0']['relative_l2'] for r in report['rows'])
                report['summary']=summary;save(a.output,report)
                print(json.dumps({'prompt':prompt,'step':step,'layer':layer,'summary':summary}),flush=True)
                del payload
    expected_cases={(p,s,l,h) for p in range(3) for s in range(4) for l in LAYERS for h in HEADS}
    actual_cases={(r['prompt'],r['step'],r['layer'],r['head']) for r in report['rows']}
    assert actual_cases==expected_cases and len(report['rows'])==384
    report['complete']=True;save(a.output,report)


if __name__=='__main__':
    main()
