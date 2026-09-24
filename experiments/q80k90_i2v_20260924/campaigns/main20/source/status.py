"""Compact public status, using structured records only."""
import json
from pathlib import Path
import time
from worker import read
ROOT=Path(__file__).resolve().parents[1]
def status():
    plan=read(ROOT/'experiment_plan.json');counts={};timings={}
    for c in (4,6,8,10):
        p=Path(plan['existing_c8']['runtime']) if c==8 else ROOT/f'C{c}/video20/runtime.json'
        try:r=read(p) if p.exists() else {}
        except json.JSONDecodeError:r={}
        q=ROOT/f'C{c}/quality_cpu/quality.json'
        quality=read(q) if q.exists() else {}
        counts[str(c)]=dict(generated=len([x for x in r.get('records',[]) if x['prompt_index']<20]),evaluated=len(quality.get('videos',[])),expected=20)
        claim=ROOT/f'claims/C{c}.json';assignment=read(claim) if claim.exists() else {}
        timings[str(c)]={'host':'scai6' if c==8 else assignment.get('host','scai7'),'gpu':4 if c==8 else assignment.get('gpu'),'summary':r.get('summary'),
                         'full_50_step_pflops':plan['selected_kernels'][str(c)]['full_50_step_pflops']}
    result=dict(time=time.time(),progress=counts,timing_by_capacity=timings,complete=all(v['generated']==v['evaluated']==20 for v in counts.values()))
    for name in ('stage','completion','failure','quality_status','quality_failure','launch','evaluation_launch','borrow_status','restoration','restoration_verified'):
        p=ROOT/f'{name}.json'
        if p.exists():result[name]=read(p)
    p=ROOT/'native_gate.json'
    if p.exists():
        g=read(p);result['native_gate']={k:g.get(k) for k in ('complete','passed','scope','layers')};result['native_gate']['cases']=len(g.get('rows',[]))
    result['gpu7']={}
    for name in ('stage','completion','failure','launch'):
        p=ROOT/f'gpu7/{name}.json'
        if p.exists():result['gpu7'][name]=read(p)
    result['scai6_c6']={}
    for control in ('scai6','scai6_gpu2','scai6_gpu5'):
        values={}
        for name in ('stage','completion','failure','launch','supervisor_status','generation_complete'):
            p=ROOT/control/f'{name}.json'
            if p.exists():values[name]=read(p)
        p=ROOT/control/'native_gate.json'
        if p.exists():
            g=read(p);values['native_gate']={'passed':g.get('passed'),'complete':g.get('complete'),'cases':len(g.get('rows',[]))}
        if values:result['scai6_c6'][control]=values
    vb=ROOT/'vbench20_20260924'
    if (vb/'plan.json').exists():
        result['vbench20']={'methods':{},'gpu2':{}}
        for method in ('C4','C6','C8','SVG_EAR_p08'):
            p=vb/method/'summary.json'
            result['vbench20']['methods'][method]=read(p) if p.exists() else {'complete':False}
        for name in ('launch','stage','completion','failure'):
            p=vb/'gpu2'/f'{name}.json'
            if p.exists():result['vbench20']['gpu2'][name]=read(p)
    return result
if __name__=='__main__':print(json.dumps(status()))
