"""Frozen SVG-EAR backend, p=.4, twenty matching inputs and full-step traces."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from common import ROOT, read, save, sha


def main():
    plan = read(ROOT / 'experiment_plan.json')
    gpu = read(ROOT / 'active_gpu.json')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == gpu['uuid']
    assert float(os.environ['SVG_TOP_P']) == .4
    snap = Path(plan['snapshot'])
    out = ROOT / 'svg_ear_p04/video20'
    for root, hashes in [(snap, plan['source_sha256']), (Path(plan['source_svg_root']), plan['svg_sha256'])]:
        for name, digest in hashes.items():
            assert sha(root / name) == digest, name
    assert sha(plan['config']) == plan['config_sha256']
    for row in plan['input_manifest']:
        assert sha(row['image']) == row['image_sha256']
    out.mkdir(parents=True, exist_ok=True)
    assert not list(out.glob('*.mp4')), 'Existing outputs must be diagnosed, never silently overwritten'
    assert not (out / 'density.jsonl').exists()
    manifest = dict(protocol=plan['protocol'], inputs=plan['input_manifest'], input_version=plan['input_version'])
    save(out / 'input_manifest.json', manifest)
    save(out / 'prompt_manifest.json', manifest)
    os.environ['SVG_DENSITY_LOG'] = str(out / 'density.jsonl')
    os.chdir(snap)
    sys.path.insert(0, str(snap))
    import torch
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    from wan.modules.model import WanSelfAttention
    from attention import svg_family_wan as family
    layers = [0] * 40
    iterations, purity = [], []
    density_offset = 0
    old_forward = WanSelfAttention.forward

    def forward(self, x, grid_size, rope_cache):
        assert tuple(x.shape) == (1, 75600, 5120)
        assert tuple(map(int, grid_size)) == (21, 45, 80)
        assert not self.enable_token_merging and self.attention_backend == 'ear_all'
        layers[self.layer_idx] += 1
        return old_forward(self, x, grid_size, rope_cache)

    WanSelfAttention.forward = forward
    cls = family._load_svg_modules()['ear']
    old_cluster = cls.kmeans_clustering

    def cluster(self, q, k, layer_idx):
        initial = not self.centroids_init
        result = old_cluster(self, q, k, layer_idx)
        iterations.append(dict(layer=layer_idx, step=(layers[layer_idx]-1)//2,
                               branch=(layers[layer_idx]-1)%2, initialization=initial,
                               q_iterations=int(result[3]), k_iterations=int(result[7])))
        return result

    cls.kmeans_clustering = cluster
    spec = importlib.util.spec_from_file_location('frozen_generate', snap / 'generate.py')
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    old_save = gen.save_video_atomic

    def save_video(path, video, **kwargs):
        nonlocal density_offset
        i = len(purity)
        assert Path(path).name == f'{i:03d}-00.mp4'
        assert tuple(video.shape) == (81, 720, 1280, 3) and layers == [100] * 40
        with (out / 'density.jsonl').open() as f:
            f.seek(density_offset)
            densities = [json.loads(line) for line in f]
            density_offset = f.tell()
        assert len(densities) == len(iterations) == 4000
        assert sum(r['initialization'] for r in iterations) == 40
        assert {(r['step'], r['branch'], r['layer']) for r in iterations} == {
            (s, b, l) for s in range(50) for b in range(2) for l in range(40)}
        for row, density in zip(iterations, densities):
            assert row['layer'] == density['layer']
            row['density'] = float(density['avg_density'])
            limit = 50 if row['initialization'] else 2
            assert 0 <= row['density'] <= 1
            assert 0 < row['q_iterations'] <= limit and 0 < row['k_iterations'] <= limit
            sparse = 4 * 75600**2 * 40 * 128 * row['density']
            comp = 4 * 75600 * 1000 * 40 * 128
            row['major_matmul_flops'] = sparse + comp + 2 * 75600 * 40 * 128 * (
                300 * row['q_iterations'] + 1000 * row['k_iterations'])
            row['scheduled_upper_flops'] = sparse + comp + 2 * 75600 * 40 * 128 * 1300 * limit
        actual = sum(r['major_matmul_flops'] for r in iterations)
        upper = sum(r['scheduled_upper_flops'] for r in iterations)
        save(out / f'flops50_{i:03d}.json', dict(index=i, seed=i, calls=4000,
             full_50_step_flops=actual, scheduled_upper_full_50_step_flops=upper,
             rows=iterations, scope='Actual full50 densities and returned iterations; scheduled upper bound retained for existing plot convention.'))
        old_save(path, video, **kwargs)
        item = plan['input_manifest'][i]
        purity.append(dict(video=Path(path).name, index=i, seed=i, attention_calls=4000,
                           layer_calls=list(layers), dense_self_attention_calls=0,
                           video_sha256=sha(path), prompt_sha256=item['prompt_sha256'],
                           image_sha256=item['image_sha256'], full_50_step_pflops=actual/1e15,
                           scheduled_upper_full_50_step_pflops=upper/1e15, gpu=gpu, saved_at=time.time()))
        save(out / 'purity_records.json', purity)
        layers[:] = [0] * 40
        iterations.clear()

    gen.save_video_atomic = save_video
    sys.argv = ['generate.py', '--config', plan['config'], '--checkpoint', 'pretrained',
                '--prompt_path', plan['prompt_path'], '--image_list', plan['image_list'],
                '--image_dir', plan['image_dir'], '--output_dir', str(out),
                '--runtime_json', str(out/'runtime.json'), '--num_videos', '20', '--num_samples', '1', '--seed', '0']
    gen.main()
    rows = read(out / 'runtime.json')['records']
    assert len(rows) == len(purity) == 20
    assert {r['prompt_index'] for r in rows} == set(range(20))
    assert all(r['attention_calls'] == r['full_attention_calls'] == 4000 for r in rows)
    save(ROOT / 'generation_complete.json', dict(complete=True, videos=20, time=time.time()))


if __name__ == '__main__':
    main()
