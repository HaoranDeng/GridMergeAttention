"""Local report of actual new results and explicitly separate replacement sensitivity."""
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text())

def main():
    results=ROOT/'results';plan=read(results/'experiment_plan.json');old=read(ROOT/'original_quality_reference.json')
    quality=read(results/'quality.json') if (results/'quality.json').exists() else {'results':{}}
    methods={};metrics=['psnr','ssim','lpips']
    for name in ['Dense','C4','C8','EAR_p08']:
        row={'status':'pending'};rt=results/name/'video20/runtime.json';pur=results/name/'video20/purity_records.json'
        if rt.exists() and pur.exists():
            rec,=read(rt)['records'];proof,=read(pur)
            assert rec['prompt_index']==proof['index']==19 and proof['seed']==19
            assert rec['attention_calls']==rec['full_attention_calls']==4000
            row.update(status='generated',gpu=proof['gpu'],attention_ms_per_call=rec['mean_attention_ms'],
                       full_attention_ms_per_call=rec['mean_full_attention_ms'],pipeline_seconds=rec['inference_seconds'],
                       full_50_step_pflops=proof['full_50_step_pflops'],tflops_per_call=proof['full_50_step_pflops']/4)
        if name in quality['results']:
            new=quality['results'][name];original=old['methods'][name]
            assert new['image_sha256']==plan['replacement']['new']['image_sha256'] and new['seed']==19
            for i in range(19):
                assert old['input_manifest'][i]['prompt_sha256']==plan['input_manifest'][i]['prompt_sha256']
                assert old['input_manifest'][i]['image_sha256']==plan['input_manifest'][i]['image_sha256']
            old_mean={k:statistics.mean(r[k] for r in original) for k in metrics}
            revised={k:statistics.mean([r[k] for r in original[:19]]+[new[k]]) for k in metrics}
            row.update(status='evaluated',new019={k:new[k] for k in metrics},old019={k:original[19][k] for k in metrics},
                       original20=old_mean,replacement20=revised,delta20={k:revised[k]-old_mean[k] for k in metrics})
        methods[name]=row
    restoration=read(results/'restoration_verified.json') if (results/'restoration_verified.json').exists() else {}
    summary=dict(post_hoc=True,original_results_unchanged=True,new_sample=plan['replacement']['new'],methods=methods,
                 restoration_verified=restoration.get('verified',False),original_Q45K80C8='pending matched I2V results',
                 timing_note='Separate single-video actual GPU0 times include cold-start effects; not pooled with old steady means.')
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=['# I2V 019 replacement trial','',
          'Post-hoc sensitivity experiment: only019 changes to “a person holding a sparkler in their hand”, seed19. Original20 and original figures are preserved.','',
          '| Method | New019 PSNR | SSIM | LPIPS | Attention ms/call | Full50 PFLOPs |',
          '|---|---:|---:|---:|---:|---:|']
    f=lambda x:'pending' if x is None else f'{x:.6f}'
    for name,r in methods.items():
        q=r.get('new019',{})
        text.append('| '+' | '.join([name,*[f(q.get(k)) for k in metrics],f(r.get('attention_ms_per_call')),f(r.get('full_50_step_pflops'))])+' |')
    text+=['','| Method | Original20 PSNR / SSIM / LPIPS | Replacement20 PSNR / SSIM / LPIPS |','|---|---|---|']
    for name,r in methods.items():
        if name=='Dense':continue
        text.append('| '+name+' | '+' / '.join(f(r.get('original20',{}).get(k)) for k in metrics)+' | '+' / '.join(f(r.get('replacement20',{}).get(k)) for k in metrics)+' |')
    text+=['','Replacement20 = unchanged000–018 plus new019; each method is paired to its matching Dense. Single-video timing is kept separate from previous steady-video statistics.',
           '',f"Original GPU0 restoration verified: {summary['restoration_verified']}. Original Q45K80C8 matched I2V baseline: pending."]
    (ROOT/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'methods':{k:v['status'] for k,v in methods.items()},'restoration_verified':summary['restoration_verified']}))

if __name__=='__main__':main()
