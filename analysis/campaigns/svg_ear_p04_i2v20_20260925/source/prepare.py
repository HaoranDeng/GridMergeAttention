"""Run once remotely after copying source; no GPU work or launch in this step."""
import copy
import hashlib
from pathlib import Path
import py_compile
import time
from common import ROOT, read, save, sha

assert not (ROOT/'experiment_plan.json').exists(), 'Already prepared; inspect instead of overwriting'
oldroot = ROOT.parent/'i2v_replace019_allpoints_20260924'
old = read(oldroot/'experiment_plan.json')
base = read(ROOT.parent/'q80k90_i2v_c4_c6_c8_c10_20260924/experiment_plan.json')
default = ROOT.parents[1]/'datasets/i2v_default_sparkler_v1'
assert read(default/'verification.json')['images_verified'] == 20
prompts = (default/'inputs/prompts.txt').read_text().splitlines()
names = (default/'inputs/prompts.images.txt').read_text().splitlines()
assert len(prompts) == len(names) == 20
inputs = []
for i, (prompt, name) in enumerate(zip(prompts, names)):
    image = default/'images'/name
    ph = hashlib.sha256(prompt.encode()).hexdigest()
    ih = sha(image)
    assert ph == old['input_manifest'][i]['prompt_sha256']
    assert ih == old['input_manifest'][i]['image_sha256']
    if i < 19:
        dense = Path(base['input_manifest'][i]['dense_video'])
        dh = base['input_manifest'][i]['dense_sha256']
        assert ph == base['input_manifest'][i]['prompt_sha256']
        assert ih == base['input_manifest'][i]['image_sha256']
    else:
        assert prompt == 'a person holding a sparkler in their hand'
        dense = oldroot/'Dense/video20/019-00.mp4'
        proof, = [r for r in read(oldroot/'Dense/video20/purity_records.json') if r['video']=='019-00.mp4']
        dh = proof['video_sha256']
    assert sha(dense) == dh
    inputs.append(dict(index=i, seed=i, prompt=prompt, prompt_sha256=ph, image=str(image),
                       image_sha256=ih, dense_video=str(dense), dense_sha256=dh))
config_source = Path(old['settings']['svg_ear_p05']['config'])
config = ROOT/'configs/base50i_svg_ear_unipc.yaml'
config.parent.mkdir(exist_ok=True)
config.write_bytes(config_source.read_bytes())
protocol = copy.deepcopy(old['protocol'])
for key in ('q', 'k', 'capacities', 'videos_per_capacity', 'pure_all_layers'):
    protocol.pop(key, None)
protocol.update(inputs='vbench_i2v_20_sparkler_v1', videos_per_setting=20,
                generated_indices=list(range(20)), top_p=.4, q_centroids=300, k_centroids=1000,
                min_kc_ratio=.10, kmeans_init=50, kmeans_step=2, attention_backend='ear_all')
plan = dict(created_at=time.time(), root=str(ROOT), source_campaign=str(oldroot),
            snapshot=old['snapshot'], source_svg_root=old['source_svg_root'],
            source_sha256=old['source_sha256'], svg_sha256=old['svg_sha256'],
            python=old['python'], evaluation_python=old['evaluation_python'], config=str(config),
            config_sha256=sha(config), protocol=protocol, input_version='vbench_i2v_20_sparkler_v1',
            input_manifest=inputs, prompt_path=str(default/'inputs/prompts.txt'),
            image_list=str(default/'inputs/prompts.images.txt'), image_dir=str(default/'images'),
            setting='svg_ear_p04', post_hoc_input_selection=True,
            scope='Add only missing I2V SVG-EAR p=.4 point and necessary caption/provenance; preserve existing T2V and I2V points.',
            timing_scope='Single exclusive H200 for all20; exclude video000 from steady timing, preserve all raw timings.',
            source_wrapper_sha256={p.name:sha(p) for p in (ROOT/'source').glob('*.py')})
for root, hashes in [(Path(plan['snapshot']), plan['source_sha256']), (Path(plan['source_svg_root']), plan['svg_sha256'])]:
    for name, digest in hashes.items():
        assert sha(root/name) == digest, name
for file in (ROOT/'source').glob('*.py'):
    py_compile.compile(str(file), doraise=True)
save(ROOT/'experiment_plan.json', plan)
print(__import__('json').dumps(dict(prepared=True, verified_inputs=20, source_files=len(plan['source_sha256']),
                                  svg_files=len(plan['svg_sha256']), config_unchanged=sha(config)==sha(config_source))))
