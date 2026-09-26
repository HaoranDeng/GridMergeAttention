"""Recompute matched-QKV logging-ablation evidence without importing GPU code."""
import hashlib,json,statistics
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
Q=ROOT/'quick_diagnostic'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    hashes=read(Q/'sync_hashes.json')
    for name,digest in hashes.items():assert sha(Q/name)==digest,name
    d=read(Q/'diagnostic.json');completion=read(Q/'completion.json')
    assert d['valid'] and completion['valid'] and completion['returncode']==0
    assert d['top_p']==.4 and d['calls']==240 and d['steps']==3
    assert len(d['rows'])==240 and {r['call'] for r in d['rows']}==set(range(240))
    assert {(r['step'],r['branch'],r['layer']) for r in d['rows']}=={(s,b,l) for s in range(3) for b in range(2) for l in range(40)}
    monitors=[json.loads(x) for x in (Q/'monitor.jsonl').read_text().splitlines()]
    assert monitors and all(not x['foreign'] for x in monitors)
    modes=['original_log','log_to_dev_null','no_density_log','no_log_no_audit']
    means={};pairs={};ids={80*s+40*b+l for s in (1,2) for b in (0,1) for l in (0,5,10,15,20,25,30,35)}
    assert len(d['replays'])==128 and all(r['max_abs_difference']==0 for r in d['replays'])
    for mode in modes:
        rows=[r for r in d['replays'] if r['mode']==mode];assert len(rows)==32 and {r['call'] for r in rows}==ids
        means[mode]=dict(core_ms=statistics.mean(r['core_ms'] for r in rows),wall_ms=statistics.mean(r['wall_ms'] for r in rows),
            stages={k:statistics.mean(r['stages'].get(k,0) for r in rows) for k in rows[0]['stages']})
        pairs[mode]={r['call']:r['core_ms'] for r in rows}
    delta=[pairs['original_log'][i]-pairs['log_to_dev_null'][i] for i in sorted(ids)]
    oldscript=ROOT.parent/'SVG_EAR_p05_video20/source/generate_ear.py'
    t2vsource=Path('C:/GMA/GMA-ICLR2027/figure_sources/tradeoff/matched_svg_ear_p05.json')
    provenance=read(t2vsource)['protocol']['source_sha256']
    expected,=[h for p,h in provenance.items() if p.endswith('/generate_ear.py')]
    assert sha(oldscript)==expected
    source=oldscript.read_text();assert "os.environ.pop('SVG_DENSITY_LOG', None)" in source
    assert 'The CUDA end event is already recorded. Density accounting is outside attention timing.' in source
    out=dict(valid=True,checked_at=datetime.now(timezone.utc).isoformat(),top_p=.4,steps=3,trajectory_calls=240,
        paired_qkv_states=32,paired_calls=128,zero_output_difference=True,foreign_events=0,
        timing_scope=d['scope'],modes=means,
        file_io_delta_ms=dict(mean=statistics.mean(delta),median=statistics.median(delta),min=min(delta),max=max(delta),positive_pairs=sum(v>0 for v in delta)),
        t2v_p05_provenance=dict(source=str(oldscript),sha256=expected,accounting='Density calculated after core timing end; per-call file logging disabled.'),
        conclusion='Writing the density file inside the timed region adds measured latency on identical p=.4 QKV/centroids. This establishes timing contamination, not an intrinsic slowdown from reducing top_p. Short sampled core timings are not full50-video replacement timings.')
    (Q/'VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    lines=['# p=0.4 three-step timing diagnosis','',
        'The measured p=.4 attention interval contains synchronous density-file I/O. A same-GPU, same-QKV replay establishes that this adds substantial latency without changing outputs. The previously reported110.50/110.30 ms values are audited end-to-end core intervals including this work; they are not clean operator-only estimates.','',
        '| Replay condition | Mean core ms | Mean wall ms |','|---|---:|---:|']
    for mode in modes:lines.append(f"| {mode} | {means[mode]['core_ms']:.6f} | {means[mode]['wall_ms']:.6f} |")
    lines+=['',f"Original file versus /dev/null difference: mean {statistics.mean(delta):.6f} ms, median {statistics.median(delta):.6f} ms, range {min(delta):.6f}-{max(delta):.6f} ms; all {sum(v>0 for v in delta)}/32 paired differences are positive. /dev/null retains density computation, GPU scalar synchronization and JSON formatting; it removes the original filesystem write destination. Disabling the Python iteration audit does not explain the large gap.",
        '', 'Protocol: one frozen sparkler_v1 input000, seed0, p=.4 only, scai6 GPU2 H200. Execute the first three steps of the original50-step schedule and stop before step4, without decoding or exporting a video. Profile240 real attention calls. The32 ablation states span steps1 and2 (zero-based), both CFG branches and layers0,5,10,15,20,25,30,35. Restore identical pre-call Q/K centroids for every replay; reverse replay order across sampled states. All128 replay outputs match the trajectory output exactly (max absolute difference0). First-step initialization is excluded from the ablation means. No foreign GPU process was observed.',
        '', 'The measured processor core excludes the native wrapper layout conversions. It includes diagnostic event instrumentation. These are short, early-step, sampled-layer diagnostics, not50-step or20-video averages, and no multiplication/extrapolation is used to replace reported FLOPs or paper points.',
        '', 'The T2V p=.5 script underlying the89.48 ms reference explicitly disables SVG_DENSITY_LOG and performs density accounting after the core end event. Its archived source hash was checked against the published provenance. Thus that timing scope differs from this p=.4 campaign. No p=.5 experiment was run during this diagnostic.',
        '', 'Current p=.4 logs reside on /local2, an NFS mount. Historical native I2V runner116 also enabled density logging but used h200_run/density_p50i on a different NFS mount, /xuanwu-tank/east. We did not replay the canceled p=.5 case or prove the exact historical per-write latency, so the full numerical difference to historical I2V p=.5 is not causally apportioned.',
        '', 'The corrective measurement protocol is to capture density/iteration evidence in memory and serialize it after the attention timing end (preferably at the video boundary), while retaining the full trace. That requires separately measured corrected timings; subtracting18.24 ms from all historical points would be unjustified. Frozen raw runs, quality values, paper curves and20-video means are preserved.',
        '', 'Evidence: diagnostic.json, VERIFICATION.json, completion.json, launch.json, preflight.json, monitor.jsonl and sync_hashes.json in this directory. The three-step guard has exited successfully and released its GPU.']
    (Q/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(valid=True,means={m:means[m]['core_ms'] for m in modes},file_io_delta=out['file_io_delta_ms'])))
if __name__=='__main__':main()
