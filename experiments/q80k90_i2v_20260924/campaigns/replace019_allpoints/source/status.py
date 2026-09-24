"""Public structured campaign state only."""
import time
from worker import ROOT,read

def status():
    plan=read(ROOT/'experiment_plan.json');quality=read(ROOT/'quality.json') if (ROOT/'quality.json').exists() else {'results':{}}
    methods={}
    for name in plan['methods']:
        methods[name]={'generated':(ROOT/name/'generation_complete.json').exists(),
                       'evaluated':name in quality['results'],'reused':name in plan['reuse']}
    result={'time':time.time(),'methods':methods,'generated':sum(v['generated'] for v in methods.values()),
            'evaluated':sum(v['evaluated'] for v in methods.values()),'expected':18,'new_expected':15,
            'gpus':plan.get('gpus',{'scai6_gpu2':plan['gpu']}),'post_hoc':True,'index':19,'seed':19,
            'restoration_required':plan.get('borrowed_slots',[])}
    for name in ['generation_failure','quality_failure','evaluation_launch','generation_complete','evaluation_complete','evaluation_waiting','rollover_gpu2','rollover_gpu2_failure']:
        path=ROOT/(name+'.json')
        if path.exists():result[name]=read(path)
    result['slots']={}
    controls=list(plan.get('gpus',{}))
    rollover=read(ROOT/'rollover_gpu2.json') if (ROOT/'rollover_gpu2.json').exists() else {}
    if rollover.get('phase')!='complete':controls.append('gpu2')
    for slot in controls:
        c=ROOT/slot;item={}
        for name in ['preflight','launch','stage','completion','generation_failure','generation_complete','borrow_status','restoration','restoration_verified']:
            p=c/(name+'.json')
            if p.exists():item[name]=read(p)
        p=c/'native_gate.json'
        if p.exists():
            g=read(p);item['native_gate']={k:g[k] for k in ['complete','passed','capacities','layers']};item['native_gate']['cases']=len(g['rows'])
        p=c/'monitor.jsonl'
        if p.exists():
            import json
            rows=[]
            for line in p.read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass
            item['monitor']={'checks':len(rows),'foreign_checks':sum(bool(x['foreign_pids']) for x in rows),'latest':rows[-1] if rows else None}
        if item:result['slots'][slot]=item
    return result

if __name__=='__main__':
    import json;print(json.dumps(status()))
