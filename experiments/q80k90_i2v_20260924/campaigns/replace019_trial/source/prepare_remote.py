"""Prepare a separate one-input replacement; never modify the original benchmark."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.parse
import urllib.request

ROOT = Path('/local2/denghaoran/GridMergeAttention/experiments/i2v_replace019_sparkler_20260924')
ORIGINAL = Path('/local2/denghaoran/GridMergeAttention/experiments/q80k90_i2v_c4_c6_c8_c10_20260924')
BASE = Path('/xuanwu-tank/east/zrl/exp-gridmergeattn')
PROMPT = 'a person holding a sparkler in their hand'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (ROOT/'experiment_plan.json').exists(), 'Already prepared; inspect instead of overwrite'
    (ROOT/'inputs/images').mkdir(parents=True, exist_ok=True)
    (ROOT/'source').mkdir(exist_ok=True)
    old = json.loads((ORIGINAL/'experiment_plan.json').read_text())
    snap = Path(old['snapshot'])
    for rel, h in old['source_sha256'].items():
        assert sha(snap/rel) == h, rel
    metadata = BASE/'third_party/VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json'
    assert sha(metadata) == '8cf4d34dc11779a336c216a1a71d482927cfa24a1bf736ab5c92425609f129f8'
    rows = json.loads(metadata.read_text())
    assert rows[956]['prompt_en'] == PROMPT, rows[956]
    filename = PROMPT+'.jpg'
    req = urllib.request.Request('https://huggingface.co/api/datasets/dc-ai/vbench_i2v', headers={'User-Agent':'I2V-replacement-experiment'})
    with urllib.request.urlopen(req, timeout=60) as response:
        revision = json.load(response)['sha']
    url = 'https://huggingface.co/datasets/dc-ai/vbench_i2v/resolve/'+revision+'/crop/16-9/'+urllib.parse.quote(filename)
    dst = ROOT/'inputs/images'/filename
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    assert len(data) > 1000
    dst.write_bytes(data)
    from PIL import Image
    with Image.open(dst) as im:
        dimensions = list(im.size)
        # Existing inputs also retain native resolution. The unchanged pipeline
        # sends native pixels to CLIP and bicubically resizes only the VAE input.
        assert im.format == 'JPEG' and dimensions[0]*9 == dimensions[1]*16, (im.format,dimensions)
        im.verify()
    prompts = Path(old['prompt_path']).read_text().splitlines()
    names = Path(old['image_list']).read_text().splitlines()
    assert len(prompts) == len(names) == 20
    previous = dict(old['input_manifest'][19])
    prompts[19], names[19] = PROMPT, filename
    manifest = []
    for i, row in enumerate(old['input_manifest']):
        target = ROOT/'inputs/images'/names[i]
        if i != 19:
            assert sha(Path(row['image'])) == row['image_sha256']
            if not target.exists(): target.symlink_to(row['image'])
        manifest.append(dict(index=i, seed=i, prompt=prompts[i], prompt_sha256=hashlib.sha256(prompts[i].encode()).hexdigest(),
                             image=str(target), image_sha256=sha(target)))
    for name, lines in [('prompts.txt',prompts),('images.txt',names)]:
        (ROOT/'inputs'/name).write_text('\n'.join(lines)+'\n')
    # The native generator/kernel snapshot is immutable and reused by hash. Freeze SVG dependencies separately.
    svg_source = BASE/'third_party/Sparse-VideoGen'
    svg_copy = ROOT/'svg_snapshot'
    source_hashes = dict(old['source_sha256'])
    for name in ['configs/base50i_dense_unipc.yaml','configs/base50i_svg_ear_unipc.yaml',
                 'configs/q80k90_i2v_C4.yaml','configs/q80k90_i2v_C8.yaml','attention/svg_family_wan.py']:
        source_hashes[name] = sha(snap/name)
    svg_hashes = {}
    for src in (svg_source/'svg').rglob('*.py'):
        rel = src.relative_to(svg_source); target=svg_copy/rel
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,target)
        svg_hashes[str(rel)] = sha(target)
    plan = dict(old)
    plan.update(root=str(ROOT), status='prepared_not_launched', source_sha256=source_hashes,
                original_experiment=str(ORIGINAL), prompt_path=str(ROOT/'inputs/prompts.txt'),
                image_list=str(ROOT/'inputs/images.txt'), image_dir=str(ROOT/'inputs/images'), input_manifest=manifest,
                source_svg_root=str(svg_copy), svg_sha256=svg_hashes, methods=['Dense','C4','C8','EAR_p08'],
                configs={**old['configs'],'Dense':str(snap/'configs/base50i_dense_unipc.yaml'),
                         'EAR_p08':str(snap/'configs/base50i_svg_ear_unipc.yaml')},
                replacement=dict(index=19,seed=19,old=previous,new=manifest[19],category='other',source_index=956,
                                 selection='Single candidate from previously saved stratified Random(0) proposal; no resampling after quality measurement.',
                                 sampling_seed=0,post_hoc=True,purpose='Separate replacement sensitivity experiment, not an untouched benchmark.',
                                 image_source=url,image_revision=revision,image_size=dimensions,metadata_sha256=sha(metadata)),
                dense_reference=str(ROOT/'Dense/video20'),existing_c8=None,
                timing_note='New single-video runs on scai7 GPU0 only; startup-inclusive timings are not pooled with old steady-video means.',
                ear_parameters=dict(top_p=.8,q_centroids=300,k_centroids=1000,min_kc_ratio_effective=.05,
                                    kmeans_init_max=50,kmeans_step_max=2))
    plan['protocol'] = {**old['protocol'],'inputs':'Original inputs000-018 unchanged; input019 replaced with a same-category VBench sample',
                        'videos_per_capacity':1,'generated_indices':[19],'capacities':[4,8]}
    (ROOT/'experiment_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(prepared=True,root=str(ROOT),prompt=PROMPT,seed=19,image_sha256=sha(dst),image_size=dimensions,
                          svg_python_files=len(svg_hashes),unchanged_original_inputs=19)),flush=True)

if __name__=='__main__': main()
