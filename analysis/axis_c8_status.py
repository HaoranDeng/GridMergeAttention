"""Compact read-only progress for the W80/H45/C8 experiment."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path('/local2/denghaoran/GridMergeAttention/experiments/axis_video_c8_20260918'))
    a=p.parse_args();result={}
    for path in sorted(a.base.glob('validation*.json')):
        data=json.loads(path.read_text(encoding='utf-8-sig'))
        result[path.name]={'passed':data.get('passed'),'synthetic_cases':len(data.get('cases',[])),
                           'real_cases':len(data.get('real_qkv',[])),
                           'real_errors':[{'independent':x['independent_reference']['relative_l2'],
                                           'r2_vs_slow':x['r2_vs_slow_reference']['relative_l2']} for x in data.get('real_qkv',[])]}
    for path in sorted(a.base.glob('benchmark*.json')):
        data=json.loads(path.read_text())
        result[path.name]={'passed':data.get('passed'),'complete':data.get('complete'),
                           'samples':[{'c8':x['method_median_ms'],'dense':x['dense_median_ms'],
                                       'speedup':x['speedup_vs_sdpa']} for x in data.get('samples',[])]}
    job=a.base/'active_video_job.json'
    if job.exists():
        data=json.loads(job.read_text());out=Path(data['output']);runtime=out/'runtime.json'
        if runtime.exists():
            r=json.loads(runtime.read_text());videos=r.get('videos',[])
            result['video']={'output':str(out),'videos':len(videos),'generation_complete':r.get('generation_complete'),
                             'first5_quality_complete':r.get('first5_quality_complete'),'quality_complete':r.get('quality_complete'),
                             'mean_attention_ms':sum(v['mean_attention_ms'] for v in videos)/len(videos) if videos else None}
        else:result['video']={'output':str(out),'videos':0,'state':'starting'}
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
