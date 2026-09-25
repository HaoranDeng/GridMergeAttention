"""Atomic global claims; one existing evaluator serves every GPU slot."""
import argparse,json,os,subprocess,time,traceback
from worker import ROOT,read,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--slot',required=True);a=p.parse_args();plan=read(ROOT/'experiment_plan.json');control=ROOT/a.slot
    def run(name,args,setting=None):
        env=dict(os.environ)
        if setting:
            if 'top_p' in setting:env['SVG_TOP_P']=str(setting['top_p'])
            if 'sparsity' in setting:env['SVG_SPARSITY']=str(setting['sparsity'])
        with (control/(name+'.log')).open('w') as log:
            child=subprocess.Popen([plan['python'],str(ROOT/'source'/args[0]),*args[1:]],cwd=plan['snapshot'],env=env,stdout=log,stderr=subprocess.STDOUT)
            save(control/'stage.json',dict(stage=name,pid=os.getpid(),worker_pid=child.pid,slot=a.slot,time=time.time()))
            rc=child.wait();assert rc==0,(name,rc)
    try:
        gate=control/'native_gate.json'
        if not gate.exists() or not read(gate).get('passed'):
            run('native_gate',['worker_multi.py','--gate','--capacity','6','--slot',a.slot])
        for method in plan['priorities'][a.slot]:
            if (ROOT/method/'generation_complete.json').exists():continue
            claim=ROOT/'claims'/f'{method}.json';claim.parent.mkdir(exist_ok=True)
            try:
                with claim.open('x') as f:json.dump(dict(pid=os.getpid(),slot=a.slot,host=plan['gpus'][a.slot]['host'],gpu=plan['gpus'][a.slot]['index'],method=method,time=time.time()),f)
            except FileExistsError:continue
            setting=plan['settings'][method]
            args=['worker_multi.py','--capacity',str(setting['capacity']),'--slot',a.slot] if setting['family']=='gca' else ['baseline_worker_multi.py','--method',method,'--slot',a.slot]
            run(method,args,setting)
        save(control/'stage.json',dict(stage='generation_complete',slot=a.slot,time=time.time()))
        save(control/'generation_complete.json',dict(complete=True,slot=a.slot,time=time.time()))
        if all((ROOT/m/'generation_complete.json').exists() for m in plan['methods']):
            save(ROOT/'generation_complete.json',dict(complete=True,methods=plan['methods'],time=time.time()))
    except BaseException:
        save(control/'generation_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise

if __name__=='__main__':main()
