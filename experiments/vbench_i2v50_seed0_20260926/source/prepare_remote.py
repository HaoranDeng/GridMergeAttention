"""Prepare a new campaign without changing any existing run or input set."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/local2/denghaoran/GridMergeAttention/experiments/q80k90_i2v_c4_c6_c8_c10_20260924')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    from PIL import Image
    import yaml
    assert not (ROOT/'experiment_plan.json').exists(), 'Plan already frozen'
    old = json.loads((OLD/'experiment_plan.json').read_text())
    original = Path(old['snapshot'])
    for rel, h in old['source_sha256'].items():
        assert sha(original/rel) == h, rel
    snap = ROOT/'snapshot'
    if not snap.exists():
        shutil.copytree(original, snap, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    selection = json.loads((ROOT/'inputs/selection.json').read_text())
    assert len(selection['items']) == 50
    assert sha(ROOT/'inputs/prompts.txt') == selection['prompts_sha256']
    assert sha(ROOT/'inputs/images.txt') == selection['image_list_sha256']
    req = urllib.request.Request('https://huggingface.co/api/datasets/dc-ai/vbench_i2v', headers={'User-Agent':'GMA-I2V50'})
    with urllib.request.urlopen(req, timeout=60) as response:
        revision = json.load(response)['sha']
    images = ROOT/'inputs/images'
    images.mkdir(exist_ok=True)
    def fetch(row):
        dst = images/row['image_name']
        url = 'https://huggingface.co/datasets/dc-ai/vbench_i2v/resolve/'+revision+'/crop/16-9/'+urllib.parse.quote(row['image_name'])
        # Always use the same pinned remote revision, regardless of local caches.
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    data = response.read()
                assert len(data) > 1000
                dst.write_bytes(data)
                with Image.open(dst) as im:
                    size = list(im.size)
                    assert im.format == 'JPEG' and size[0]*9 == size[1]*16
                    im.verify()
                return dict(row, prompt=row['prompt_en'], image=str(dst), image_sha256=sha(dst),
                            image_size=size, image_source=url, image_revision=revision)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2*(attempt+1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        inputs = list(pool.map(fetch, selection['items']))
    cfgs = {'Dense':str(snap/'configs/base50i_dense_unipc.yaml'), 'C4':str(snap/'configs/q80k90_i2v_C4.yaml')}
    dense, c4 = [yaml.safe_load(Path(cfgs[k]).read_text()) for k in ('Dense','C4')]
    c4['token_merging_args'] = {'enable':False}
    assert c4 == dense, 'Configs differ outside attention'
    hashes = {str(p.relative_to(snap)):sha(p) for p in snap.rglob('*') if p.is_file() and p.suffix != '.pyc'}
    protocol = dict(old['protocol'], inputs=selection['dataset'], videos_per_method=50, capacities=[4],
                    source_selection_sha256=sha(ROOT/'inputs/selection.json'))
    protocol.pop('videos_per_capacity', None)
    plan = dict(created_at=time.time(), status='prepared', root=str(ROOT), snapshot=str(snap),
                source_snapshot=str(original), source_revision=old['source_revision'], source_sha256=hashes,
                python=old['python'], evaluation_python=old['evaluation_python'], protocol=protocol, configs=cfgs,
                methods=['Dense','C4'], execution='One H200 only; Dense50 then native C4 gate then C4_50',
                gpu=dict(host='scai6', index=2, uuid='GPU-244e87b8-5a10-aadb-858a-f0eea028dbd1',model='NVIDIA H200'),
                prompt_path=str(ROOT/'inputs/prompts.txt'), image_list=str(ROOT/'inputs/images.txt'), image_dir=str(images),
                selected_kernels=old['selected_kernels'], input_manifest=inputs, selection=selection,
                evaluation=dict(device='cpu',frames=81,resize=False,lpips='alex', vbench='pending'))
    (ROOT/'experiment_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (ROOT/'inputs/input_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n')
    print(json.dumps(dict(prepared=str(ROOT), images_verified=len(inputs), revision=revision, source_files=len(hashes))))


if __name__ == '__main__':
    main()
