"""Combine unchanged original000-018 and measured new019 without mixing times."""
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT.parent/'Q80K90_I2V_C4_C6_C8_C10_20260924'

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    results=ROOT/'results';plan=read(results/'experiment_plan.json');status=read(results/'public_status.json')
    q=read(results/'quality.json')['results'] if (results/'quality.json').exists() else read(results/'reused_quality.json')
    ref=read(ROOT/'baseline_reference.json');refs={r['tag']:r for family in ref['series'].values() for r in family}
    mainplan=read(MAIN/'results/experiment_plan.json')
    for i in range(19):
        for key in ('prompt_sha256','image_sha256'):assert plan['input_manifest'][i][key]==mainplan['input_manifest'][i][key]
    methods={}
    for name in plan['methods']:
        setting=plan['settings'][name];row={'new019_status':status['methods'][name], 'replacement20_ready':False}
        if setting['family']=='gca':
            p=MAIN/f'results/{name}/quality_cpu/quality.json';old=read(p)['videos'] if p.exists() else []
            oldcost=plan['selected_kernels'][name[1:]]['full_50_step_pflops']
            oldcosts=[oldcost*1e15]*20
        else:old=refs[name]['quality'];oldcosts=refs[name]['flops_per_video']
        indexed={int(v['video'][:3]):v for v in old}
        if name in q:
            new=q[name];assert new['index']==new['seed']==19 and new['frames']==81
            assert new['image_sha256']==plan['input_manifest'][19]['image_sha256']
            row['new019']={k:new[k] for k in ['psnr','ssim','lpips']}
            rt=read(results/name/'video20/runtime.json')['records'][0]
            proof=read(results/name/'video20/purity_records.json')[0]
            row['new019_timing']={'gpu':proof['gpu'],'attention_ms_per_call':rt['mean_attention_ms'],
                                 'pipeline_seconds':rt['inference_seconds'],'scope':'Single cold-start video, not pooled into original steady timings.'}
            row['new019_actual_full50_pflops']=proof['full_50_step_pflops']
        if name in q and set(range(20))<=set(indexed):
            row['replacement20_ready']=True
            row['original20']={k:statistics.mean(indexed[i][k] for i in range(20)) for k in ['psnr','ssim','lpips']}
            row['replacement20']={k:statistics.mean([indexed[i][k] for i in range(19)]+[q[name][k]]) for k in ['psnr','ssim','lpips']}
            row['old019']={k:indexed[19][k] for k in ['psnr','ssim','lpips']}
            if setting['family'] in ['ear','svg2']:
                trace=read(results/name/'video20/flops50_019.json');assert len(trace['rows'])==trace['calls']==4000
                if 'scheduled_upper_full_50_step_flops' in trace:newcost=trace['scheduled_upper_full_50_step_flops']
                else:
                    assert setting['family']=='ear'
                    newcost=sum(4*75600**2*40*128*v['density']+2*75600*40*128*1300*(50 if v['initialization'] else 2)+4*75600*1000*40*128 for v in trace['rows'])
                scope='Measured complete50 densities plus the same scheduled clustering upper bound as original20.'
            else:
                newcost=row['new019_actual_full50_pflops']*1e15;scope='Fixed complete50 major-matmul arithmetic.'
            row['replacement20_full50_pflops']=(sum(oldcosts[:19])+newcost)/20/1e15
            row['replacement20_tflops_per_call']=row['replacement20_full50_pflops']/4
            row['original20_full50_pflops']=statistics.mean(oldcosts)/1e15;row['flops_scope']=scope
        methods[name]=row
    summary={'post_hoc':True,'expected_settings':18,'generated':status['generated'],'evaluated':status['evaluated'],
             'ready_settings':sum(v['replacement20_ready'] for v in methods.values()),'methods':methods,
             'original_Q45K80C8':'Pending matched I2V results; not substituted.',
             'plotting_latency_overrides_ms':{'C6':83.6,'C8':104.1,'C10':133.8},
             'scope':'Same replaced019 across all settings, original000-018 retained. Do not claim a prespecified benchmark.'}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    text=['# All-point I2V019 replacement','',f"Generated {status['generated']}/18 settings; CPU quality {status['evaluated']}/18; complete replacement20 aggregates {summary['ready_settings']}/18.",
          '', 'This is a post-hoc replacement of one conditioning image/prompt with fixed seed19; original000-018 remain paired to originalDense and new019 to newDense.',
          '', '| Setting | New019 | Replacement20 PSNR | SSIM | LPIPS | Full50 PFLOPs/video |', '|---|---|---:|---:|---:|---:|']
    for name,row in methods.items():
        vals=row.get('replacement20',{})
        values=[f'{vals[k]:.6f}' if k in vals else 'Pending' for k in ['psnr','ssim','lpips']]
        cost=f"{row['replacement20_full50_pflops']:.6f}" if row['replacement20_ready'] else 'Pending'
        text.append('| '+' | '.join([name,'evaluated' if name in q else 'pending',*values,cost])+' |')
    text+=['', 'SVG-EAR/SVG2 arithmetic uses the same scheduled clustering upper bound for original and replacement samples, with actual full50-step densities. Exact new019 returned-iteration costs remain in summary.json. Timings remain separate by physical GPU and are not pooled with the historical steady means. VBench six-dimension scores for the replacement20 dataset are pending. Original Q45K80C8 matched I2V remains pending.']
    (ROOT/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['generated','evaluated','ready_settings','expected_settings']}))

if __name__=='__main__':main()
