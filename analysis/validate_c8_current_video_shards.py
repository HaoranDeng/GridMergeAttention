"""CPU-only checks for disjoint canonical video shards and metric aggregation."""
import json
from pathlib import Path
import tempfile

from analysis.collect_c8_current_video_pair import collect
from analysis.evaluate_video_quality import prepare_comparison
from analysis.benchmark_c8_current import save
from utils.prompt_contract import build_prompt_manifest,ensure_output_prompt_manifest


def main():
    prompt_path=Path(__file__).resolve().parents[1]/'prompts/MovieGenVideoBench_extended.txt'
    prompts=[s.strip() for s in prompt_path.read_text().splitlines() if s.strip()]
    manifest=build_prompt_manifest(prompt_path,prompts,selected_count=20,seed=0,
                                   num_samples=1,vbench_filenames=False,allow_noncanonical=False)
    with tempfile.TemporaryDirectory(prefix='c8_shard_metadata_test_') as temporary:
        root=Path(temporary)
        shards=[root/'first',root/'second']
        save(root/'launch.json',{'gpus':[6,5],'shards':[str(s) for s in shards]})
        for shard,start in zip(shards,(0,10)):
            out=shard/'videos';out.mkdir(parents=True)
            (shard/'guard').mkdir()
            ensure_output_prompt_manifest(out,manifest)
            videos=[];rows=[]
            for i in range(start,start+10):
                name=f'{i:03d}-00.mp4'
                # Metadata-only fixture; never decoded or used as video evidence.
                (out/name).write_bytes(b'metadata-test-placeholder')
                videos.append({'index':i,'video':name,'attention_calls':4000,'layer_calls':[100]*40})
                rows.append({'video':name,'frames':81,'psnr':float(i),'ssim':i/100,'lpips':i/1000})
            save(out/'runtime.json',{'protocol':{'prompt_indices':list(range(start,start+10)),
                'source_sha256':{'test':'same'},'reference':'fixture'},'videos':videos,
                'quality_complete':True,'quality_failures':[]})
            save(out/'quality_10.json',{'methods':{'q45k80c8_current':{'videos':rows}}})
            save(shard/'guard/completion.json',{'valid':True})
        pairs,_=prepare_comparison(shards[1]/'videos',shards[1]/'videos',max_videos=10)
        assert [p.name for p,_ in pairs]==[f'{i:03d}-00.mp4' for i in range(10,20)]
        assert collect(root)
        result=json.loads((root/'videos/runtime.json').read_text())
        assert result['generation_complete'] and result['quality_complete'] and result['status']=='complete'
        assert [v['index'] for v in result['videos']]==list(range(20))
        quality=json.loads((root/'videos/quality_20.json').read_text())['methods']['q45k80c8_current']['summary']
        assert quality['videos']==20 and quality['frames']==1620 and abs(quality['psnr']-9.5)<1e-12
        path=shards[1]/'videos/runtime.json';second=json.loads(path.read_text())
        second['videos'][0]['index']=0;save(path,second)
        try:collect(root)
        except AssertionError:pass
        else:raise AssertionError('Duplicate canonical prompt index was accepted')
    print(json.dumps({'passed':True,'checks':['nonzero-start prompt pairing','20 distinct canonical indices',
        'per-frame weighted metrics','duplicate index rejection']}))


if __name__=='__main__':main()
