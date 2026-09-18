"""Requested interim PSNR/SSIM comparison on completed canonical axis videos."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import statistics

from analysis.evaluate_video_quality import paired_frames
from analysis.quick_axis_c4_qkv import save
from utils.prompt_contract import assert_matching_prompt_manifests
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_pair(index, reference, method):
    before=[digest(reference),digest(method)]
    psnr,ssim=[],[]
    for ref,got in paired_frames(reference,method,expected_frames=81):
        psnr.append(float(peak_signal_noise_ratio(ref,got,data_range=255)))
        ssim.append(float(structural_similarity(ref,got,channel_axis=2,data_range=255)))
    assert before==[digest(reference),digest(method)],'Input changed during evaluation'
    return {'index':index,'video':Path(method).name,'frames':len(psnr),
            'psnr':statistics.fmean(psnr),'ssim':statistics.fmean(ssim),
            'reference_sha256':before[0],'method_sha256':before[1]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count',type=int,required=True)
    p.add_argument('--axes',choices=['wh','tw','th'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert 1<=a.count<=5 and not a.output.exists()
    base=Path('/local2/denghaoran/GridMergeAttention/experiments/axis_video_c4_20260918')
    registry=json.loads((base/'active_video_jobs.json').read_text())
    method=Path(registry[a.axes]['output']);reference=base/'dense_first20'
    runtime=json.loads((method/'runtime.json').read_text())
    assert [v['index'] for v in runtime['videos'][:a.count]]==list(range(a.count))
    contract=assert_matching_prompt_manifests(reference,method,prompt_indices=range(a.count))
    assert_matching_prompt_manifests(reference,base/'c8_first20',prompt_indices=range(a.count))
    reuse=json.loads((base/'reference_reuse.json').read_text())
    baseline=reuse['c8_first5']['videos'][:a.count]
    for index,row in enumerate(baseline):
        assert row['video']==f'{index:03d}.mp4' and row['frames']==81
        receipt=reuse['sources']['c8']['videos'][index]
        assert digest(base/'c8_first20'/f'{index:03d}-00.mp4')==receipt['sha256']
    baseline_mean={m:statistics.fmean(x[m] for x in baseline) for m in ('psnr','ssim')}
    report={'complete':False,'axes':a.axes,'count':a.count,'reference':str(reference),
            'method':str(method),'prompt_contract':contract,'expected_frames':81,
            'psnr_data_range':255,'ssim_data_range':255,'ssim_channel_axis':2,
            'c8_label':reuse['c8_first5']['label'],'c8_source':reuse['c8_first5']['source'],
            'c8_summary':baseline_mean,'videos':[],
            'mean_attention_ms':statistics.fmean(v['mean_attention_ms'] for v in runtime['videos'][:a.count])}
    save(a.output,report)
    with ProcessPoolExecutor(max_workers=min(4,a.count)) as pool:
        futures=[pool.submit(score_pair,i,reference/f'{i:03d}-00.mp4',method/f'{i:03d}-00.mp4') for i in range(a.count)]
        for future in as_completed(futures):
            row=future.result();b=baseline[row['index']]
            row['delta_vs_c8']={m:row[m]-b[m] for m in ('psnr','ssim')}
            row['c8']={m:b[m] for m in ('psnr','ssim')}
            report['videos'].append(row);report['videos'].sort(key=lambda x:x['index'])
            save(a.output,report);print(json.dumps(row),flush=True)
    report['summary']={m:statistics.fmean(x[m] for x in report['videos']) for m in ('psnr','ssim')}
    report['delta_vs_c8']={m:report['summary'][m]-baseline_mean[m] for m in ('psnr','ssim')}
    report['complete']=True;save(a.output,report)
    print(json.dumps({k:report[k] for k in ('summary','c8_summary','delta_vs_c8','mean_attention_ms')}),flush=True)


if __name__=='__main__':main()
