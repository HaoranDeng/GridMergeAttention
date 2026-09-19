"""Collect two disjoint ten-video shards; no CUDA work or video inspection."""
import argparse
import json
from pathlib import Path
import time

from analysis.benchmark_c8_current import save
from utils.prompt_contract import assert_matching_prompt_manifests


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def collect(root):
    receipt=read(root/'launch.json')
    shards=[Path(p) for p in receipt['shards']]
    out=root/'videos';out.mkdir(exist_ok=True)
    (root/'guard').mkdir(exist_ok=True)
    runtimes=[read(s/'videos/runtime.json') for s in shards]
    completions=[read(s/'guard/completion.json') for s in shards]
    active=[r for r in runtimes if r]
    if not active:return False
    manifest=out/'prompt_manifest.json'
    videos=[];quality_by_name={};progress=[]
    for shard,runtime in zip(shards,runtimes):
        if runtime is None:continue
        source=shard/'videos'
        if not manifest.exists():manifest.symlink_to(source/'prompt_manifest.json')
        assert_matching_prompt_manifests(source,out,prompt_indices=runtime['protocol']['prompt_indices'])
        for record in runtime['videos']:
            destination=out/record['video'];target=source/record['video']
            assert target.is_file()
            if not destination.exists():destination.symlink_to(target)
            assert destination.resolve()==target.resolve()
            videos.append(record)
        p=read(source/'progress.json')
        if p:progress.append({'shard':shard.name,**p})
        for count in (1,5,10):
            quality=read(source/f'quality_{count}.json')
            if quality:
                for row in quality['methods']['q45k80c8_current']['videos']:
                    quality_by_name[row['video']]=row
    videos.sort(key=lambda v:v['index'])
    assert len({v['index'] for v in videos})==len(videos)
    assert all(v['attention_calls']==4000 and v['layer_calls']==[100]*40 for v in videos)
    protocol=dict(active[0]['protocol'],num_videos=20,prompt_indices=list(range(20)),
                  gpus=receipt['gpus'],parallel_shards=[str(s) for s in shards],
                  authorization='User explicitly requested two GPUs for faster delivery')
    for r in active:assert r['protocol']['source_sha256']==protocol['source_sha256']
    generation_complete=len(videos)==20
    if generation_complete:assert [v['index'] for v in videos]==list(range(20))
    rows=sorted(quality_by_name.values(),key=lambda row:row['video'])
    assert all(row['frames']==81 for row in rows)
    if rows:
        summary={key:sum(row[key]*row['frames'] for row in rows)/sum(row['frames'] for row in rows)
                 for key in ('psnr','ssim','lpips')}
        summary.update(videos=len(rows),frames=sum(row['frames'] for row in rows))
        save(out/'quality_20.json',{'reference':protocol['reference'],
            'protocol':{'pairing':'same canonical prompt, seed and frame; union of disjoint verified shards',
                        'expected_videos':20,'expected_frames':81,'complete':len(rows)==20},
            'methods':{'q45k80c8_current':{'summary':summary,'videos':rows}}})
    failed=any(c and not c['valid'] for c in completions)
    all_done=all(c is not None for c in completions)
    quality_complete=len(rows)==20 and all(r and r['quality_complete'] for r in runtimes)
    status='failed' if failed else ('complete' if all_done and generation_complete and quality_complete
                                  else 'evaluating' if generation_complete else 'generating')
    save(out/'runtime.json',{'protocol':protocol,'videos':videos,'generation_complete':generation_complete,
        'quality_complete':quality_complete,'status':status,'shard_completions':completions,
        'quality_failures':[x for r in active for x in r['quality_failures']]})
    if progress:
        current=min(progress,key=lambda p:p['completed_videos'])
        save(out/'progress.json',{**current,'completed_videos':len(videos),'shards':progress})
    if all_done:
        valid=not failed and generation_complete and quality_complete
        save(root/'guard/completion.json',{'valid':valid,'error':None if valid else 'See shard completion and runtime',
                                          'shards':completions,'finished_at':time.time()})
    return all_done


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    while not collect(a.output):time.sleep(10)
