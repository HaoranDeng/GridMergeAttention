"""Verify committed artifacts; the scope update restricts this run to p=.4."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
REMOTE='/local2/denghaoran/GridMergeAttention/experiments/svg_ear_p04_p05_scai6_timing5_20260926'
OPTS=['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','HostName=scai6.cs.ucla.edu']
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
known=read(ROOT/'sync_hashes.json') if (ROOT/'sync_hashes.json').exists() else {}
script='''
import hashlib,io,json,tarfile,time
from pathlib import Path
root=Path(REMOTE);plan=json.loads((root/'experiment_plan.json').read_text())
paths=list(root.glob('*.json'));live=dict(time=time.time(),logs={})
paths += [root/name for name in ['monitor.jsonl','monitor_p04_only.jsonl']]
paths += list((root/'source').glob('*.py')) + list((root/'configs').glob('*.yaml'))
for case in plan['cases']:
    folder=root/case;paths+=list(folder.glob('*.json'))+list((folder/'video5').glob('*.json'))
    rt=folder/'video5/runtime.json';purity=folder/'video5/purity_records.json'
    if rt.exists() and purity.exists():
        proof={r['video']:r for r in json.loads(purity.read_text())}
        for row in json.loads(rt.read_text())['records']:
            p=folder/'video5'/row['output']
            if p.is_file() and p.name in proof:
                assert hashlib.sha256(p.read_bytes()).hexdigest()==proof[p.name]['video_sha256'];paths.append(p)
    log=folder/'worker.log'
    if log.exists():
        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-1000));live['logs'][case]=f.read().decode(errors='replace')
for name in ['supervisor.log','p04_only_supervisor.log']:
    log=root/name
    if log.exists():
        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-1500));live['logs'][name]=f.read().decode(errors='replace')
samples=[]
for name in ['monitor.jsonl','monitor_p04_only.jsonl']:
    monitor=root/name
    if monitor.exists():
        with monitor.open() as f:samples.extend(json.loads(line) for line in f)
live['foreign_events']=[r for r in samples if r['foreign']];live['monitor_samples']=len(samples)
telemetry=[]
for phase in ['A','B','p04_only']:
    for gpu in [2,4]:
        rows=[g for r in samples if r['phase']==phase for g in r['telemetry'] if int(g['index'])==gpu and g['utilization']>=70]
        if rows:
            import statistics
            telemetry.append(dict(phase=phase,gpu=gpu,samples=len(rows),busy_threshold=70,
                **{k:dict(mean=statistics.mean(r[k] for r in rows),min=min(r[k] for r in rows),max=max(r[k] for r in rows))
                   for k in ['sm_clock_mhz','memory_clock_mhz','temperature_c','power_w','power_limit_w']}))
live['telemetry_summary']=telemetry
(root/'live_status.json').write_text(json.dumps(live,indent=2)+'\\n');paths.append(root/'live_status.json')
archive=root/'sync_delta.tar.gz';manifest={}
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(set(paths)):
        if not p.is_file():continue
        name=str(p.relative_to(root));data=p.read_bytes();digest=hashlib.sha256(data).hexdigest();manifest[name]=digest
        if KNOWN.get(name)==digest:continue
        info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
print(json.dumps(dict(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),manifest=manifest)))
'''
p=subprocess.run(['ssh',*OPTS,'scai7','python -'],input='REMOTE='+repr(REMOTE)+'\nKNOWN='+repr(known)+'\n'+script,
    text=True,encoding='utf-8',capture_output=True,timeout=180)
assert p.returncode==0,p.stderr[-2000:]
receipt=json.loads(p.stdout.strip().splitlines()[-1]);archive=ROOT/'sync_delta.tar.gz'
subprocess.run(['scp','-q',*OPTS,'scai7:'+receipt['archive'],str(archive)],check=True,timeout=180)
assert sha(archive)==receipt['sha256'];results=ROOT/'results';results.mkdir(exist_ok=True)
with tarfile.open(archive,'r:gz') as tar:
    for member in tar.getmembers():
        assert member.isfile() and member.name in receipt['manifest']
        dest=(results/member.name).resolve();assert results.resolve() in dest.parents
        data=tar.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==receipt['manifest'][member.name]
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
(ROOT/'sync_hashes.json').write_text(json.dumps(receipt['manifest'],indent=2)+'\n',encoding='utf-8')
plan=read(results/'experiment_plan.json');live=read(results/'live_status.json');cases={}
scope=read(results/'scope_update.json') if (results/'scope_update.json').exists() else {}
active=scope.get('active_cases',list(plan['cases']))
for name,case in plan['cases'].items():
    folder=results/name/'video5';rt=read(folder/'runtime.json') if (folder/'runtime.json').exists() else {}
    records=rt.get('records',[]);proof=read(folder/'purity_records.json') if (folder/'purity_records.json').exists() else []
    for row in records:
        i=row['prompt_index'];item=plan['input_manifest'][i];p,=[v for v in proof if v['index']==i]
        assert row['attention_calls']==row['full_attention_calls']==p['attention_calls']==4000
        assert p['layer_calls']==[100]*40 and p['gpu']['uuid']==case['gpu']['uuid']
        assert p['image_sha256']==item['image_sha256'] and p['prompt_sha256']==item['prompt_sha256']
        assert p['video_sha256']==sha(folder/row['output'])
    cases[name]=dict(generated=len(records),phase=case['phase'],gpu=case['gpu'],top_p=case['top_p'],active=name in active,
        complete=(results/name/'generation_complete.json').exists(),runtime_summary=rt.get('summary'),
        per_video=[{k:r[k] for k in ['prompt_index','mean_attention_ms','mean_full_attention_ms','inference_seconds']} for r in records])
is_p04_only=scope.get('status')=='p04_only'
stagepath=results/('stage_p04_only.json' if is_p04_only else 'stage.json')
completionpath=results/('completion_p04_only.json' if is_p04_only else 'completion.json')
stage=read(stagepath) if stagepath.exists() else {}
completion=read(completionpath) if completionpath.exists() else None
summary=dict(cases=cases,stage=stage,foreign_events=len(live['foreign_events']),monitor_samples=live['monitor_samples'],
    telemetry=live['telemetry_summary'],completion=completion,paired_comparison_ready=False,timing_ready=False,
    scope=scope,active_cases=active,within_gpu={},p04_by_gpu={})
if (results/'failure.json').exists():summary['failure']=read(results/'failure.json')
for name in ['failure_p04_only.json','handoff_failure.json']:
    if (results/name).exists():summary[name]=read(results/name)
for gpu in [2,4]:
    a,b=cases[f'gpu{gpu}_p04'],cases[f'gpu{gpu}_p05']
    if a['active'] and b['active'] and a['generated']==b['generated']==5 and a['complete'] and b['complete']:
        av={r['prompt_index']:r for r in a['per_video']};bv={r['prompt_index']:r for r in b['per_video']}
        pairs=[dict(index=i,p04_ms=av[i]['mean_attention_ms'],p05_ms=bv[i]['mean_attention_ms'],
            p04_minus_p05_ms=av[i]['mean_attention_ms']-bv[i]['mean_attention_ms']) for i in range(1,5)]
        m04=statistics.mean(r['p04_ms'] for r in pairs);m05=statistics.mean(r['p05_ms'] for r in pairs)
        summary['within_gpu'][str(gpu)]=dict(p04_ms=m04,p05_ms=m05,delta_ms=m04-m05,
            p04_relative_to_p05_percent=(m04/m05-1)*100,paired_inputs=pairs,
            note='Four identical steady inputs per case; sample000 separately reported as warmup. GPUs are not pooled.')
if len(summary['within_gpu'])==2 and completion and completion['valid']:
    assert not summary['foreign_events'] and not summary.get('failure');summary['paired_comparison_ready']=True
if is_p04_only:
    for gpu in [2,4]:
        case=cases[f'gpu{gpu}_p04']
        if case['generated']==5 and case['complete']:
            steady=[r for r in case['per_video'] if r['prompt_index'] in [1,2,3,4]]
            assert len(steady)==4
            summary['p04_by_gpu'][str(gpu)]={k:statistics.mean(r[k] for r in steady)
                for k in ['mean_attention_ms','mean_full_attention_ms','inference_seconds']}
    if len(summary['p04_by_gpu'])==2 and completion and completion['valid']:
        assert not summary['foreign_events'] and not any('failure' in k for k in summary)
        summary['timing_ready']=True
(ROOT/'status.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(stage=stage,progress={k:v['generated'] for k,v in cases.items()},
    active_cases=active,foreign_events=summary['foreign_events'],timing_ready=summary['timing_ready'],
    paired_comparison_ready=summary['paired_comparison_ready'],
    failures={k:v for k,v in summary.items() if 'failure' in k})))
