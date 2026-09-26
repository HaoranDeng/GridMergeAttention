"""Independent local audit of the ten completed p=.4 executions; no media decoding."""
import hashlib,json,math,statistics
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
OLD=ROOT.parent/'SVG_EAR_p04_I2V20_20260925/results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-8),(a,b)
def main():
    status=read(ROOT/'status.json');done=read(R/'completion_p04_only.json')
    assert status['timing_ready'] and done['valid'] and done['failure'] is None
    plan=read(R/'experiment_plan.json');oldplan=read(OLD/'experiment_plan.json')
    scope=read(R/'scope_update.json');active=['gpu2_p04','gpu4_p04']
    assert scope['active_cases']==active and scope['p05_worker_exited'] and scope['old_supervisor_exited']
    hashes=read(ROOT/'sync_hashes.json')
    for name,digest in hashes.items():assert sha(R/name)==digest,name
    for name,digest in plan['wrapper_source_sha256'].items():assert sha(R/'source'/name)==digest,name
    for field in ['source_sha256','svg_sha256','config_sha256','input_version']:
        assert plan[field]==oldplan[field],field
    assert plan['input_manifest']==oldplan['input_manifest'][:5]
    monitors=[]
    for name in ['monitor.jsonl','monitor_p04_only.jsonl']:
        monitors.extend(json.loads(x) for x in (R/name).read_text().splitlines())
    assert all(not row['foreign'] for row in monitors)
    oldrt=read(OLD/'svg_ear_p04/video20/runtime.json')['records']
    fields=['mean_attention_ms','mean_full_attention_ms','inference_seconds']
    oldsubset={k:statistics.mean(r[k] for r in oldrt if r['prompt_index'] in (1,2,3,4)) for k in fields}
    evidence=dict(valid=True,checked_at=datetime.now(timezone.utc).isoformat(),videos=10,unique_inputs=5,
        calls=40000,steps_per_video=50,warmup_index=0,steady_indices=[1,2,3,4],
        canceled_p05_excluded=True,source_and_input_match_prior_scai7=True,hash_verified_files=len(hashes),
        foreign_events=0,scai7_same_input_subset=oldsubset,cases={})
    for name in active:
        case=plan['cases'][name];assert case['top_p']==.4
        folder=R/name/'video5';rt=read(folder/'runtime.json')['records'];proof=read(folder/'purity_records.json')
        complete=read(R/name/'generation_complete.json');launch=read(R/name/'launch.json')
        assert complete['complete'] and complete['videos']==5 and launch['top_p']==.4
        assert sorted(r['prompt_index'] for r in rt)==list(range(5))
        assert sorted(r['index'] for r in proof)==list(range(5))
        per=[]
        for row in rt:
            i=row['prompt_index'];p,=[p for p in proof if p['index']==i];item=plan['input_manifest'][i]
            assert row['sample_index']==0 and p['seed']==item['seed']==i
            assert row['attention_calls']==row['full_attention_calls']==p['attention_calls']==4000
            assert p['layer_calls']==[100]*40 and p['dense_self_attention_calls']==0
            assert p['gpu']==case['gpu'] and p['video_sha256']==sha(folder/row['output'])
            for k in ['prompt_sha256','image_sha256']:assert p[k]==item[k]
            close(row['mean_attention_ms'],row['attention_total_ms']/4000)
            close(row['mean_full_attention_ms'],row['full_attention_total_ms']/4000)
            trace=read(folder/f'flops50_{i:03d}.json');rows=trace['rows']
            assert trace['index']==trace['seed']==i and trace['calls']==len(rows)==4000
            assert {(r['step'],r['branch'],r['layer']) for r in rows}=={(s,b,l) for s in range(50) for b in range(2) for l in range(40)}
            assert sum(r['initialization'] for r in rows)==40
            actual=[];upper=[]
            for r in rows:
                limit=50 if r['initialization'] else 2
                assert 0<=r['density']<=1 and 0<r['q_iterations']<=limit and 0<r['k_iterations']<=limit
                a=4*75600**2*40*128*r['density']+4*75600*1000*40*128+2*75600*40*128*(300*r['q_iterations']+1000*r['k_iterations'])
                u=4*75600**2*40*128*r['density']+4*75600*1000*40*128+2*75600*40*128*1300*limit
                close(a,r['major_matmul_flops']);close(u,r['scheduled_upper_flops']);actual.append(a);upper.append(u)
            close(sum(actual),trace['full_50_step_flops']);close(sum(upper),trace['scheduled_upper_full_50_step_flops'])
            close(sum(actual)/1e15,p['full_50_step_pflops'])
            per.append(dict(index=i,**{k:row[k] for k in fields},mean_watts=row['mean_watts'],
                full50_pflops=sum(actual)/1e15,upper_full50_pflops=sum(upper)/1e15,
                density=statistics.mean(r['density'] for r in rows)))
        warmup,=[p for p in proof if p['index']==0];last,=[p for p in proof if p['index']==4]
        samples=[x for x in monitors if launch['time']<=x['time']<=complete['time']]
        assert samples
        busy=[g for x in samples if warmup['saved_at']<x['time']<=last['saved_at'] for g in x['telemetry'] if g['uuid']==case['gpu']['uuid'] and g['utilization']>=70]
        assert busy
        telemetry={k:dict(mean=statistics.mean(x[k] for x in busy),min=min(x[k] for x in busy),max=max(x[k] for x in busy)) for k in ['sm_clock_mhz','memory_clock_mhz','temperature_c','power_w','power_limit_w']}
        steady={k:statistics.mean(x[k] for x in per if x['index']>0) for k in fields+['mean_watts','full50_pflops','upper_full50_pflops','density']}
        for k in fields:close(steady[k],status['p04_by_gpu'][str(case['gpu']['index'])][k])
        evidence['cases'][name]=dict(gpu=case['gpu'],per_video=per,steady=steady,
            scai7_same_input_percent={k:100*(steady[k]/oldsubset[k]-1) for k in fields},
            busy_telemetry=telemetry,busy_samples=len(busy),monitored_samples=len(samples),
            telemetry_scope='After sample000 saved through sample004 saved, utilization>=70%; sampled telemetry, not a thermal/clock causal experiment.')
    for name in ['gpu2_p05','gpu4_p05']:
        assert read(R/name/'canceled.json')['excluded_from_results']
        assert not (R/name/'generation_complete.json').exists()
    (ROOT/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    lines=['# scai6 SVG-EAR p=0.4 timing diagnosis','',
        'Completed ten executions of five fixed sparkler_v1 inputs on two exclusive H200s. Each card used indices000-004 and seed=index. Index000 is warmup; steady values use001-004 separately per card. No p=.5 result is included.','',
        '| Card | Core ms/call | Full attention ms/call | Inference s/video |','|---|---:|---:|---:|']
    for name,c in evidence['cases'].items():
        s=c['steady'];lines.append(f"| scai6 GPU{c['gpu']['index']} | {s['mean_attention_ms']:.6f} | {s['mean_full_attention_ms']:.6f} | {s['inference_seconds']:.6f} |")
    lines+=['','Core and full attention are CUDA-event scopes; inference is the generation pipeline wall time, excluding model load and file export. Four thousand attention calls per video. These are measured timings with the campaign audit/logging enabled.','',
        '## Per video','', '| Card | Index | Role | Core ms | Full ms | Inference s | Full50 PFLOPs |','|---|---:|---|---:|---:|---:|---:|']
    for name,c in evidence['cases'].items():
        for v in c['per_video']:lines.append(f"| GPU{c['gpu']['index']} | {v['index']:03d} | {'warmup' if v['index']==0 else 'steady'} | {v['mean_attention_ms']:.6f} | {v['mean_full_attention_ms']:.6f} | {v['inference_seconds']:.6f} | {v['full50_pflops']:.6f} |")
    lines+=['','## Comparison and limits','',f"Prior scai7 GPU7 p=.4, on the same steady input subset001-004, measured {oldsubset['mean_attention_ms']:.6f} ms core, {oldsubset['mean_full_attention_ms']:.6f} ms full attention and {oldsubset['inference_seconds']:.6f} s inference. Its original19-video steady core mean was106.275035 ms. Compare equal input subsets descriptively; this is not a randomized cross-GPU experiment."]
    for name,c in evidence['cases'].items():
        t=c['busy_telemetry'];delta=c['scai7_same_input_percent']['mean_attention_ms']
        lines+=['',f"GPU{c['gpu']['index']} is {delta:.3f}% slower in core time than scai7's matched four-input subset. Busy telemetry ({c['busy_samples']} samples): SM clock mean {t['sm_clock_mhz']['mean']:.1f} MHz, range {t['sm_clock_mhz']['min']:.0f}-{t['sm_clock_mhz']['max']:.0f}; memory clock {t['memory_clock_mhz']['mean']:.0f} MHz; temperature mean {t['temperature_c']['mean']:.1f} C, max {t['temperature_c']['max']:.0f} C; sampled power mean {t['power_w']['mean']:.1f} W, limit {t['power_limit_w']['mean']:.0f} W."]
    lines+=['','No foreign compute processes were observed during the monitored run intervals. Launch checks found clean cards. The large temperature difference between GPU2 and GPU4 accompanied nearly equal timings; this alone cannot identify the cause. There is no newly measured p=.5 control, so this test does not prove that lowering top_p itself slows the implementation. No same-run Dense reference was requested, so no SDPA speedup is inferred.','',
        '## Verification and provenance','',
        f"Independently checked {len(hashes)} synchronized file hashes, all10 video hashes without decoding, all40,000 trace rows, all50 steps x2 CFG branches x40 layers, seeds, input hashes, source/config identities and full50 FLOPs sums. Source and the first five input identities match the previous scai7 campaign. Actual and scheduled upper FLOPs are stored per video; short diagnostic steps are not extrapolated to full-video FLOPs.",
        '', 'The input set retains its documented post-hoc sparkler019 replacement history; this test only uses000-004. Canceled GPU4 p=.5 partial work and never-started GPU2 p=.5 are excluded. No paper figure, table, equation, text or20-video aggregate was changed. Raw artifacts are in results/, and the independent machine-readable audit is INDEPENDENT_VERIFICATION.json.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(valid=True,calls=evidence['calls'],cases={k:v['steady'] for k,v in evidence['cases'].items()})))
if __name__=='__main__':main()
