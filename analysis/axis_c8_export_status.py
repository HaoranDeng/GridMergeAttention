"""Read-only compact data for the finite local artifact synchronizer."""
import json
from pathlib import Path


def read(path):
    if not path.exists():return None
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    base=Path('/local2/denghaoran/GridMergeAttention/experiments/axis_video_c8_20260918')
    job=read(base/'active_video_job.json');out=Path(job['output']);guard=Path(job['guard'])
    runtime=read(out/'runtime.json') or {}
    videos=runtime.get('videos',[])
    quality=read(out/'quality.json') or read(out/'quality_first5.json')
    result={'method':'qw_kh_c8_current','host':'scai6','output':str(out),'guard':str(guard),
            'videos':len(videos),'video_names':[v['video'] for v in videos],
            'generation_complete':runtime.get('generation_complete',False),
            'quality_complete':runtime.get('quality_complete',False),
            'guard_completion':read(guard/'completion.json'),
            'mean_attention_ms':sum(v['mean_attention_ms'] for v in videos)/len(videos) if videos else None,
            'mean_inference_seconds':sum(v['inference_seconds'] for v in videos)/len(videos) if videos else None,
            'quality':quality['methods']['qw_kh_c8_current']['summary'] if quality else None}
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
