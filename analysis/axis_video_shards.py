"""Publish disjoint video shards with matching prompts and operator sources."""
import fcntl
import json
from pathlib import Path
from analysis.quick_axis_c4_qkv import save
from utils.prompt_contract import ensure_output_prompt_manifest, assert_matching_prompt_manifests


def quality_complete(path,axes,count):
    try:
        data=json.loads(Path(path).read_text())['methods'][f'q{axes[0]}_k{axes[1]}_c4_current']
        return len(data['videos'])==count and all(k in data['summary'] for k in ('psnr','ssim','lpips'))
    except (FileNotFoundError,KeyError,json.JSONDecodeError):return False


def publish(config_path):
    config=json.loads(Path(config_path).read_text())
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    with (output/'publish.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        records={};protocol=None;source_metadata=[]
        for directory in config['sources']:
            source=Path(directory);runtime=source/'runtime.json'
            if not runtime.exists():continue
            data=json.loads(runtime.read_text())
            if protocol is None:protocol=data['protocol']
            assert data['protocol']['source_sha256']==protocol['source_sha256'],'Kernel source mismatch'
            for key in ('steps','seed','axes','capacity','height','width','frames','expected_calls_per_video'):
                assert data['protocol'][key]==protocol[key],key
            manifest=json.loads((source/'prompt_manifest.json').read_text())
            ensure_output_prompt_manifest(output,manifest)
            indices=[v['index'] for v in data['videos']]
            assert_matching_prompt_manifests(source,output,prompt_indices=indices)
            for record in data['videos']:
                i=record['index'];assert i not in records,f'Duplicate video index {i}'
                assert record['attention_calls']==4000 and record['layer_calls']==[100]*40
                assert record['dense_self_attention_calls']==0 and record['seed']==0
                src=source/record['video'];dst=output/record['video']
                assert src.is_file()
                if not dst.exists():dst.symlink_to(src.resolve())
                assert dst.resolve()==src.resolve()
                records[i]={**record,'shard_source':str(source)}
            source_metadata.append({'path':str(source),'recorded_videos':len(indices)})
        if protocol is None:return None
        merged={'protocol':protocol,'videos':[records[i] for i in sorted(records)],
                'generation_complete':set(records)==set(range(20)),
                'first5_quality_complete':quality_complete(output/'quality_first5.json',config['axes'],5),
                'quality_complete':quality_complete(output/'quality.json',config['axes'],20),
                'shards':source_metadata}
        save(output/'runtime.json',merged)
        return merged


def evaluate_ready(config_path,reference,prompt_path,prompts,axes):
    import gc
    import sys
    import torch
    from analysis.evaluate_video_quality import main as evaluate
    from utils.prompt_contract import build_prompt_manifest
    config=json.loads(Path(config_path).read_text());output=Path(config['output'])
    merged=publish(config_path)
    if merged is None:return
    indices={v['index'] for v in merged['videos']}
    with (output/'quality.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        for count,filename in ((5,'quality_first5.json'),(20,'quality.json')):
            destination=output/filename
            if quality_complete(destination,axes,count) or not set(range(count)).issubset(indices):continue
            # A previously finished first-five evaluation can be reused exactly.
            reused=False
            for source in config['sources']:
                src=Path(source)/filename
                if not quality_complete(src,axes,count):continue
                assert_matching_prompt_manifests(source,output,prompt_indices=range(count))
                if all((output/f'{i:03d}-00.mp4').resolve()==(Path(source)/f'{i:03d}-00.mp4').resolve() for i in range(count)):
                    save(destination,json.loads(src.read_text()));reused=True;break
            if reused:continue
            subset=build_prompt_manifest(prompt_path,prompts,selected_count=count,seed=0,num_samples=1,vbench_filenames=False,allow_noncanonical=False)
            views=[]
            for source,name in ((Path(reference),'reference'),(output,'method')):
                target=output/f'evaluation_{count}'/name
                ensure_output_prompt_manifest(target,subset)
                assert_matching_prompt_manifests(source,target,prompt_indices=range(count))
                for i in range(count):
                    src=source/f'{i:03d}-00.mp4';dst=target/src.name
                    if not dst.exists():dst.symlink_to(src.resolve())
                    assert dst.resolve()==src.resolve()
                views.append(target)
            previous=sys.argv
            try:
                sys.argv=['evaluate_video_quality','--reference_dir',str(views[0]),'--method_dir',str(views[1]),
                          '--method_name',f'q{axes[0]}_k{axes[1]}_c4_current','--output',str(destination),
                          '--max_videos',str(count),'--expected_videos',str(count),'--expected_frames','81','--lpips_batch_size','8']
                evaluate()
            finally:
                sys.argv=previous;gc.collect();torch.cuda.empty_cache()
            print(json.dumps({'event':'quality_ready','axes':axes,'videos':count,'path':str(destination)}),flush=True)
        publish(config_path)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);a=p.parse_args()
    result=publish(a.config)
    print(json.dumps({'videos':0 if result is None else len(result['videos'])}))
