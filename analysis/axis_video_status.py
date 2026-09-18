"""Compact structured status and matched first-five quality comparison."""
import json
from pathlib import Path
from analysis.quick_axis_c4_qkv import save


def main():
    base=Path('/local2/denghaoran/GridMergeAttention/experiments/axis_video_c4_20260918')
    reference=json.loads((base/'reference_reuse.json').read_text())
    dense=json.loads(Path(reference['sources']['dense']['runtime']).read_text())['videos'][:5]
    dense_ms=sum(v['mean_attention_ms'] for v in dense)/len(dense)
    report={'baseline':reference['c8_first5']['summary'], 'dense_attention_ms_reused':dense_ms,
            'protocol':'Wan2.1 T2V 14B 720p, 81 frames, 50 steps, canonical first5, seed0',
            'methods':{}, 'complete':False}
    for axes in ('tw','wh','th'):
        name=f'q{axes[0]}_k{axes[1]}_c4_current'; folder=base/name
        item={'videos':len(list(folder.glob('*.mp4'))),'quality_complete':False}
        runtime=folder/'runtime.json'
        if runtime.exists():
            data=json.loads(runtime.read_text()); records=data['videos']
            item.update({k:data.get(k,False) for k in ['generation_complete','quality_complete']})
            item['recorded_videos']=len(records)
            if records:
                item['mean_attention_ms']=sum(v['mean_attention_ms'] for v in records)/len(records)
                item['speedup_vs_sdpa_reused']=dense_ms/item['mean_attention_ms']
                item['mean_inference_seconds']=sum(v['inference_seconds'] for v in records)/len(records)
            if (folder/'quality.json').exists():
                q=json.loads((folder/'quality.json').read_text())['methods'][name]
                item['quality']=q['summary']
                item['delta_vs_existing_c8']={k:q['summary'][k]-report['baseline'][k] for k in ['psnr','ssim','lpips']}
        guards=sorted(base.glob(f'run_{axes}_*'))
        if guards:
            guard=guards[-1]; item['guard']=str(guard)
            if (guard/'completion.json').exists(): item['completion']=json.loads((guard/'completion.json').read_text())
            log=guard/'worker.log'
            if log.exists():
                with log.open('rb') as f:
                    f.seek(max(0,log.stat().st_size-2000)); tail=f.read().decode(errors='replace')
                progress=[line.strip() for line in tail.replace('\r','\n').splitlines() if '/50' in line]
                if progress:item['denoising_progress']=progress[-1]
                if 'Traceback' in tail:item['error_tail']=tail[-1400:]
        report['methods'][name]=item
    report['complete']=all(v['quality_complete'] and v.get('completion',{}).get('valid') for v in report['methods'].values())
    save(base/'status.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
