"""CPU-only paired RGB PSNR for FP8/BF16 videos and their dense reference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.evaluate_video_quality import frames
from utils.prompt_contract import assert_matching_prompt_manifests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'bf16', 'fp8', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--num-videos', type=int, default=15)
    a = p.parse_args()
    indices = range(a.num_videos)
    contracts = [assert_matching_prompt_manifests(a.reference, d, prompt_indices=indices)
                 for d in (a.bf16, a.fp8)]
    expected = [f'{i:03d}-00.mp4' for i in indices]
    assert sorted(x.name for x in a.fp8.glob('*.mp4')) == expected
    result = dict(complete=False, protocol='decoded RGB uint8, data_range=255; mean frame PSNR then mean video PSNR; 81 frames each',
                  reference=str(a.reference), bf16=str(a.bf16), fp8=str(a.fp8), prompt_contracts=contracts,
                  num_videos=a.num_videos, videos=[], files={})
    for name in expected:
        paths = [d/name for d in (a.reference,a.bf16,a.fp8)]
        for path in paths:
            result['files'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        scores = [[],[],[]]
        for ref, bf, fp in zip(*(frames(path) for path in paths), strict=True):
            assert ref.shape == bf.shape == fp.shape == (720,1280,3)
            for values, (x,y) in zip(scores, ((ref,bf),(ref,fp),(bf,fp))):
                values.append(float(peak_signal_noise_ratio(x,y,data_range=255)))
        assert all(len(s) == 81 for s in scores), 'incomplete video'
        keys = ('bf16_vs_dense_psnr','fp8_vs_dense_psnr','fp8_vs_bf16_psnr')
        row = dict(video=name,frames=81,**{k:float(np.mean(s)) for k,s in zip(keys,scores)})
        row['delta_psnr_db'] = row['fp8_vs_dense_psnr']-row['bf16_vs_dense_psnr']
        result['videos'].append(row)
        result['summary'] = {k:float(np.mean([r[k] for r in result['videos']])) for k in (*keys,'delta_psnr_db')}
        result['complete'] = len(result['videos']) == a.num_videos
        a.output.parent.mkdir(parents=True,exist_ok=True)
        tmp = a.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,indent=2),encoding='utf-8')
        tmp.replace(a.output)
        print(json.dumps(row),flush=True)


if __name__ == '__main__':
    main()
