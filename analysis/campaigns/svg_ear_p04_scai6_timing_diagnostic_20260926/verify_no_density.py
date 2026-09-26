"""Verify the clean native four-step timing prefix independently of GPU execution."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
Q=ROOT/'quick_no_density'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    for name,digest in read(Q/'sync_hashes.json').items():assert hashlib.sha256((Q/name).read_bytes()).hexdigest()==digest,name
    d=read(Q/'diagnostic.json');done=read(Q/'completion.json');plan=read(ROOT/'results/experiment_plan.json')
    assert d['valid'] and done['valid'] and done['returncode']==0
    assert d['top_p']==.4 and d['steps']==4 and d['calls']==len(d['rows'])==320
    assert d['density_logging'] is False and d['extra_iteration_audit'] is False
    assert d['input']==plan['input_manifest'][0]
    for key in ['source_sha256','svg_sha256','config_sha256']:assert d[key]==plan[key]
    assert {(r['step'],r['branch'],r['layer']) for r in d['rows']}=={(s,b,l) for s in range(4) for b in range(2) for l in range(40)}
    assert [r['call'] for r in d['rows']]==list(range(320))
    assert all(r['full_ms']>=r['core_ms']>0 for r in d['rows'])
    monitor=[json.loads(line) for line in (Q/'monitor.jsonl').read_text().splitlines()]
    assert monitor and all(not r['foreign'] for r in monitor)
    per_step={str(s):{k:statistics.mean(r[k] for r in d['rows'] if r['step']==s) for k in ['core_ms','full_ms']} for s in range(4)}
    steady=[r for r in d['rows'] if r['step']>0];assert len(steady)==240
    result=dict(valid=True,top_p=.4,steps_executed=4,warmup_step_zero_based=0,steady_steps_zero_based=[1,2,3],
        steady_calls=240,samples=1,seed=0,gpu=read(Q/'launch.json')['gpu'],density_logging=False,
        extra_iteration_audit=False,per_step=per_step,
        steady_core_ms=statistics.mean(r['core_ms'] for r in steady),
        steady_full_ms=statistics.mean(r['full_ms'] for r in steady),
        foreign_events=0,scope=d['scope'],full50_replacement=False,
        source_sha256=hashlib.sha256((ROOT/'quick_no_density.py').read_bytes()).hexdigest())
    (Q/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# Native p=.4 timing with density logging disabled','',
        f"Steady short-test mean: **{result['steady_core_ms']:.6f} ms/core attention call**, **{result['steady_full_ms']:.6f} ms/full attention call**.",
        '', 'One fixed input000, seed0, on exclusive scai6 H200 GPU2. Execute only the first four steps of the unchanged50-step I2V schedule. Step0 (including k-means initialization) is reported separately; steps1-3 provide240 steady calls across all40 layers and both CFG branches. Density logging and the extra iteration audit are disabled. The original native CUDA-event recorder measures core and full attention, including their original layout-conversion scopes. No stage replay instrumentation is installed.',
        '', '| Step (zero-based) | Calls | Core ms | Full attention ms |','|---|---:|---:|---:|']
    for s,r in per_step.items():lines.append(f"| {s}{' (initialization)' if s=='0' else ''} | 80 | {r['core_ms']:.6f} | {r['full_ms']:.6f} |")
    lines+=['','All320 call identities, source/config hashes and input identity were independently checked. No foreign GPU process was observed. The guard completed successfully and released the GPU. No video was decoded or exported.',
        '', 'This is a short prefix measurement, not a50-step/video or20-video mean. No new PSNR/SSIM/LPIPS or full50 FLOPs are inferred from it. The completed20-video quality and measured50-step arithmetic evidence remain unchanged. If used as the requested plot latency coordinate, its short-test timing scope must be explicit in the source and caption. Do not subtract a constant logging penalty from other points.']
    (Q/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','scope']}))
if __name__=='__main__':main()
