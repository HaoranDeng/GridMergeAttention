"""Freeze a proportional stratified random sample before generation."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re

CAMERA = re.compile(r",\s*camera\s+(pans?|tilts?|zooms?)\s+\w+$|,\s*camera\s+static$", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    info = json.loads(a.metadata.read_text(encoding='utf-8'))
    unique = {}
    for source_index, row in enumerate(info):
        if not CAMERA.search(row['prompt_en']):
            unique.setdefault(row['image_name'], dict(row, source_index=source_index))
    groups = collections.defaultdict(list)
    for row in unique.values():
        groups[row['image_type']].append(row)
    assert 0 < a.n <= len(unique)
    quota = {k: a.n * len(v) / len(unique) for k, v in groups.items()}
    take = {k: int(v) for k, v in quota.items()}
    order = sorted(groups, key=lambda k: (-(quota[k]-take[k]), -len(groups[k]), k))
    for k in order[:a.n-sum(take.values())]:
        take[k] += 1
    rng = random.Random(a.seed)
    selected = []
    for k in sorted(groups):
        selected.extend(rng.sample(sorted(groups[k], key=lambda r: r['image_name']), take[k]))
    rng.shuffle(selected)
    assert len(selected) == len({r['image_name'] for r in selected}) == a.n
    items = [dict(index=i, seed=i, source_index=r['source_index'], prompt_en=r['prompt_en'],
                  prompt_sha256=hashlib.sha256(r['prompt_en'].encode()).hexdigest(),
                  image_name=r['image_name'], image_type=r['image_type']) for i, r in enumerate(selected)]
    prompts = ''.join(r['prompt_en']+'\n' for r in items).encode('utf-8')
    names = ''.join(r['image_name']+'\n' for r in items).encode('utf-8')
    manifest = dict(dataset='vbench_i2v_50_stratified_seed0_v1',
                    source='VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json',
                    source_sha256=hashlib.sha256(a.metadata.read_bytes()).hexdigest(),
                    source_entries=len(info), population_unique_base_images=len(unique),
                    selection='One base prompt per distinct image; proportional image_type quotas by largest remainder; '
                              'Python Random(seed), sorted categories and image names, random.sample within each category, '
                              'then shuffle the selected order. No outcome-based selection or replacements.',
                    sampling_seed=a.seed, generation_seed_policy='seed = zero-based selected prompt index',
                    prompt_text='raw English prompt_en, no extension',
                    population_counts={k:len(v) for k,v in sorted(groups.items())},
                    selected_counts=dict(sorted(take.items())),
                    prompts_sha256=hashlib.sha256(prompts).hexdigest(),
                    image_list_sha256=hashlib.sha256(names).hexdigest(), items=items)
    outputs = {a.out.with_suffix('.txt'):prompts, a.out.with_suffix('.images.txt'):names,
               a.out.with_suffix('.json'):(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n').encode('utf-8')}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for p, data in outputs.items():
        if p.exists():
            assert p.read_bytes() == data, f'Frozen selection differs: {p}'
        else:
            p.write_bytes(data)
    print(json.dumps({k:manifest[k] for k in ['dataset','population_unique_base_images','selected_counts','prompts_sha256']}))


if __name__ == '__main__':
    main()
