"""Reuse verified historical dense/C8 artifacts without regenerating videos."""
import argparse
import json
from pathlib import Path
from analysis.quick_axis_c4_qkv import save, sha
from utils.prompt_contract import build_prompt_manifest, ensure_output_prompt_manifest, assert_matching_prompt_manifests


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    root = Path('/local2/denghaoran/GridMergeAttention/experiments')
    dense = root/'wan14b_720p_50step_hellinger_p16_c32g45_a0p2_20260829/dense_reference_exact_prompt_20260816'
    c8 = root/'wan14b_720p_50step_gma_c8_new_kernel_20260901/gma_c8_all'
    source_metrics = c8.parent/'results/paired_quality_20videos.json'
    assert_matching_prompt_manifests(dense, c8, prompt_indices=range(5))
    prompt_path = Path(__file__).resolve().parents[1]/'prompts/MovieGenVideoBench_extended.txt'
    prompts = [v.strip() for v in prompt_path.read_text().splitlines() if v.strip()]
    manifest = build_prompt_manifest(prompt_path,prompts,selected_count=5,seed=0,num_samples=1,
                                     vbench_filenames=False,allow_noncanonical=False)
    audit = {'regenerated_videos':0, 'sources':{}}
    for name, source in [('dense',dense),('c8',c8)]:
        target = a.output/(name+'_first5')
        ensure_output_prompt_manifest(target,manifest)
        assert_matching_prompt_manifests(source,target,prompt_indices=range(5))
        links = []
        for i in range(5):
            src,dst = source/f'{i:03d}.mp4',target/f'{i:03d}-00.mp4'
            assert src.is_file()
            if not dst.exists(): dst.symlink_to(src)
            assert dst.resolve() == src.resolve()
            links.append({'video':dst.name,'source':str(src),'sha256':sha(src)})
        runtime = source/('runtime.json' if name=='c8' else 'source_runtime.json')
        audit['sources'][name] = {'directory':str(source),'videos':links,
                                  'runtime':str(runtime),'runtime_sha256':sha(runtime)}
    metrics = json.loads(source_metrics.read_text())['methods']['gma_c8_new_kernel']
    videos = metrics['videos'][:5]
    assert [v['video'] for v in videos] == [f'{i:03d}.mp4' for i in range(5)]
    summary = {metric:sum(v[metric]*v['frames'] for v in videos)/sum(v['frames'] for v in videos)
               for metric in ['psnr','ssim','lpips']}
    audit['c8_first5'] = {'summary':summary,'videos':videos,'source':str(source_metrics),
                         'source_sha256':sha(source_metrics), 'label':'existing C8 standalone c79666e, Hellinger RF0'}
    save(a.output/'reference_reuse.json',audit)
    print(json.dumps(summary))


if __name__ == '__main__': main()
