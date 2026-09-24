"""Write the versioned default I2V20 inputs, or explicitly reproduce legacy sampling."""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'prompts/i2v/vbench_i2v_20_sparkler_v1.json'
LEGACY = ROOT / 'prompts/i2v/vbench_i2v_20_legacy.json'
CAMERA = re.compile(r',\s*camera\s+(pans?|tilts?|zooms?)\s+\w+$|,\s*camera\s+static$', re.I)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_entries(info):
    result = {}
    for row in info:
        if not CAMERA.search(row['prompt_en']):
            result.setdefault(row['image_name'], row)
    return result


def stratify(by_image, n):
    groups = collections.defaultdict(list)
    for row in by_image.values():
        groups[row['image_type']].append(row)
    total = sum(map(len, groups.values()))
    if not 0 < n <= total:
        raise ValueError(f'n must be in 1..{total}')
    exact = {name: n * len(rows) / total for name, rows in groups.items()}
    take = {name: int(value) for name, value in exact.items()}
    order = sorted(groups, key=lambda name: (-(exact[name] - take[name]), -len(groups[name]), name))
    for name in order[:n - sum(take.values())]:
        take[name] += 1
    return [row for name in sorted(groups, key=lambda name: (-len(groups[name]), name))
            for row in groups[name][:take[name]]]


def validate(manifest, image_dir=None):
    items = manifest['items']
    if len({row['image_name'] for row in items}) != len(items):
        raise ValueError('Each sample must use a distinct conditioning image')
    for index, row in enumerate(items):
        if row['index'] != index or row.get('seed', index) != index:
            raise ValueError('Index order and seed must remain base0 + prompt_index')
        if Path(row['image_name']).name != row['image_name'] or '\\' in row['image_name']:
            raise ValueError('Image names must be plain filenames')
        prompt = row['prompt_en']
        if '\n' in prompt or '\r' in prompt:
            raise ValueError('Prompts must be single lines')
        if 'prompt_sha256' in row and hashlib.sha256(prompt.encode()).hexdigest() != row['prompt_sha256']:
            raise ValueError(f'Prompt hash mismatch at index {index:03d}')
        if image_dir is not None:
            path = Path(image_dir) / row['image_name']
            if not path.is_file():
                raise ValueError(f'Missing conditioning image: {path}')
            if 'image_sha256' in row and digest(path) != row['image_sha256']:
                raise ValueError(f'Image hash mismatch at index {index:03d}; use the frozen input bytes')


def build(selection='default', n=20, full_info=None):
    if selection == 'default':
        if n != 20:
            raise ValueError('The fixed default has20 samples; larger legacy sampling requires --selection legacy-stratified')
        result = json.loads(DEFAULT.read_text(encoding='utf-8'))
    elif full_info is None and n == 20:
        result = json.loads(LEGACY.read_text(encoding='utf-8'))
        result['name'] = 'vbench_i2v_20_legacy'
        result['default'] = False
    else:
        if full_info is None:
            raise ValueError('--full-info is required to sample a different legacy population')
        path = Path(full_info)
        info = json.loads(path.read_text(encoding='utf-8'))
        chosen = stratify(base_entries(info), n)
        result = dict(name=f'vbench_i2v_legacy_{n}', default=False,
                      selection='Legacy largest-remainder stratification, file order within type',
                      source_sha256=digest(path), items=[dict(index=i, seed=i,
                          **{key: row[key] for key in ('prompt_en', 'image_name', 'image_type')})
                          for i, row in enumerate(chosen)])
    validate(result)
    return copy.deepcopy(result)


def write(manifest, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {'.txt': ''.join(row['prompt_en'] + '\n' for row in manifest['items']),
               '.images.txt': ''.join(row['image_name'] + '\n' for row in manifest['items']),
               '.json': json.dumps(manifest, indent=2, ensure_ascii=False) + '\n'}
    for suffix, content in outputs.items():
        Path(str(stem) + suffix).write_text(content, encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'prompts/vbench_i2v_20')
    parser.add_argument('--selection', choices=['default', 'legacy-stratified'], default='default')
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--full-info', type=Path)
    parser.add_argument('--image-dir', type=Path, help='Verify actual image bytes before writing inputs')
    args = parser.parse_args()
    try:
        manifest = build(args.selection, args.n, args.full_info)
        validate(manifest, args.image_dir)
        write(manifest, args.out)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({'name': manifest.get('name'), 'samples': len(manifest['items']),
                      'output': str(args.out), 'images_verified': args.image_dir is not None}))


if __name__ == '__main__':
    main()
