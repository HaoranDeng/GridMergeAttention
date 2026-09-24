"""Small public status; never read private environments or media pixels."""
import os
import time
from worker import ROOT,read

def status():
    plan=read(ROOT/'experiment_plan.json');stage=read(ROOT/'stage.json') if (ROOT/'stage.json').exists() else {}
    methods={m:dict(generated=(ROOT/m/'generation_complete.json').exists()) for m in plan['methods']}
    quality=read(ROOT/'quality.json') if (ROOT/'quality.json').exists() else {}
    for method in methods:methods[method]['evaluated']=method in quality.get('results',{})
    gate=read(ROOT/'native_gate.json') if (ROOT/'native_gate.json').exists() else {}
    failures={name:read(ROOT/name) for name in ['generation_failure.json','quality_failure.json'] if (ROOT/name).exists()}
    complete=read(ROOT/'completion.json') if (ROOT/'completion.json').exists() else {}
    if complete.get('failure'):failures['completion']=complete
    restore=read(ROOT/'restoration_verified.json') if (ROOT/'restoration_verified.json').exists() else {}
    return dict(time=time.time(),stage=stage,methods=methods,generated=sum(x['generated'] for x in methods.values()),
                evaluated=sum(x['evaluated'] for x in methods.values()),gate_cases=len(gate.get('rows',[])),gate_passed=gate.get('passed',False),
                failures=failures,restoration_verified=restore.get('verified',False),index=19,seed=19,
                prompt=plan['replacement']['new']['prompt'],post_hoc_replacement=True)

if __name__=='__main__':
    import json
    print(json.dumps(status(),indent=2))
