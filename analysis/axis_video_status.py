"""Compact status for active twenty-video runs, including first-five metrics."""
import json
from pathlib import Path
from analysis.quick_axis_c4_qkv import save


def main():
    base=Path('/local2/denghaoran/GridMergeAttention/experiments/axis_video_c4_20260918')
    reference=json.loads((base/'reference_reuse.json').read_text())
    registry=base/'active_video_jobs.json'
    active=json.loads(registry.read_text()) if registry.exists() else {}
    dense=json.loads(Path(reference['sources']['dense']['runtime']).read_text())['videos'][:5]
    dense_ms=sum(v['mean_attention_ms'] for v in dense)/len(dense)
    report={'baseline':reference['c8_first5']['summary'], 'dense_attention_ms_reused':dense_ms,
            'protocol':'Wan2.1 T2V 14B 720p, 81 frames, 50 steps, canonical first20, first5 interim, seed0',
            'methods':{}, 'complete':False}
    for axes in ('tw','wh','th'):
        name=f'q{axes[0]}_k{axes[1]}_c4_current'
        if axes not in active:
            report['methods'][name]={'videos':0,'quality_complete':False,'state':'not_submitted_speed_gate'}
            continue
        folder=Path(active[axes]['output'])
        item={'videos':len(list(folder.glob('*.mp4'))),'quality_complete':False,'recorded_videos':0}
        item['workers']=[{k:w[k] for k in ('host','gpu','indices') if k in w} for w in active[axes].get('workers',[active[axes]])]
        item['speed_exception_authorized']=active[axes].get('speed_exception_authorized',False)
        runtime=folder/'runtime.json'
        if runtime.exists():
            data=json.loads(runtime.read_text()); records=data['videos']
            item.update({k:data.get(k,False) for k in ['generation_complete','quality_complete','first5_quality_complete']})
            item['recorded_videos']=len(records)
            if records:
                item['mean_attention_ms']=sum(v['mean_attention_ms'] for v in records)/len(records)
                item['speedup_vs_sdpa_reused']=dense_ms/item['mean_attention_ms']
                item['mean_inference_seconds']=sum(v['inference_seconds'] for v in records)/len(records)
            for count,filename in [(5,'quality_first5.json'),(20,'quality.json')]:
                if (folder/filename).exists():
                    q=json.loads((folder/filename).read_text())['methods'][name]
                    if len(q['videos'])!=count:continue
                    baseline=reference[f'c8_first{count}']['summary']
                    item[f'quality_first{count}']=q['summary']
                    item[f'delta_first{count}_vs_existing_c8']={k:q['summary'][k]-baseline[k] for k in ['psnr','ssim','lpips']}
        item['guards']=[]
        guard_paths=active[axes].get('guards',[active[axes]['guard']])
        for guard_path in guard_paths:
            guard=Path(guard_path);entry={'path':str(guard),'started':guard.exists()}
            item['guards'].append(entry)
            if not guard.exists():continue
            if (guard/'completion.json').exists():entry['completion']=json.loads((guard/'completion.json').read_text())
            log=guard/'worker.log'
            if log.exists():
                with log.open('rb') as f:
                    f.seek(max(0,log.stat().st_size-2000)); tail=f.read().decode(errors='replace')
                progress=[line.strip() for line in tail.replace('\r','\n').splitlines() if '/50' in line]
                if progress:entry['denoising_progress']=progress[-1]
                if 'Traceback' in tail:entry['error_tail']=tail[-1400:]
        if all('completion' in e for e in item['guards']):
            item['completion']={'valid':all(e['completion']['valid'] for e in item['guards'])}
        report['methods'][name]=item
    report['complete']=all(v['quality_complete'] and v.get('completion',{}).get('valid') for v in report['methods'].values())
    save(base/'status.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
