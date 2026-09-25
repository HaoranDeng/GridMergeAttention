"""Stream the unchanged, hash-frozen full-frame CPU quality evaluator."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from generate_family import REPO, REFERENCE, CAPACITIES, read, save, sha

EVALUATOR_SHA = 'ea28e3c629165ef09a123f8145b8701996ea69daed7797bcfc14068910e717bb'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 4), required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1')
    deadline = time.monotonic()+48*3600
    def wait_until(predicate):
        while not predicate():
            assert time.monotonic() < deadline, '48-hour bounded wait expired'
            marker = args.root/'completion.json'
            if marker.exists():
                assert read(marker)['valid'], 'GPU worker failed; quality input incomplete'
            time.sleep(5)
    wait_until(lambda: (args.root/'C4/video20/prompt_manifest.json').exists())
    import fcntl
    lock = (args.root/'quality_stream.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = args.root/'quality_status.json'
    success = False
    try:
        sys.path.insert(0, str(REPO))
        source = REPO/'analysis/evaluate_video_quality.py'
        assert sha(source) == EVALUATOR_SHA
        spec = importlib.util.spec_from_file_location('frozen_quality', source)
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        import torch
        import lpips
        assert not torch.cuda.is_available()
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        assert (Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth').is_file()
        model = lpips.LPIPS(net='alex').to('cpu').eval()
        indices = list(range(0, 10) if args.gpu == 0 else range(10, 20))
        for cap in CAPACITIES:
            method = args.root/f'C{cap}/video20'
            contract = evaluator.assert_matching_prompt_manifests(REFERENCE, method, prompt_indices=range(20))
            manifest = read(method/'prompt_manifest.json')
            assert manifest['seed'] == 0 and manifest['selected_prompt_count'] == 20
            assert manifest['num_samples'] == 1
            assert manifest['prompt_file_sha256'] == '50293c2717c8f2df5773154d4680c74e636ecb08eb4fe13cdce7357d0ab9d121'
            destination = method/'quality_per_video_cpu'
            destination.mkdir(exist_ok=False)
            result = dict(complete=False, device='cpu', evaluator_sha256=EVALUATOR_SHA,
                reference=str(REFERENCE), method=f'Q80K90C{cap}', prompt_contract=contract,
                protocol=dict(expected_videos=10, expected_frames=81, planned_indices=indices,
                    frame_subsampling=False, image_resizing=False, lpips_batch_size=1,
                    psnr_data_range=255, ssim_channel_axis=2, lpips_backbone='alex'), videos=[], summary=None)
            for index in indices:
                name = f'{index:03d}-00.mp4'
                def ready():
                    runtime = method/'runtime.json'
                    return runtime.exists() and any(r['index'] == index for r in read(runtime)['videos'])
                save(output, dict(stage='waiting', capacity=cap, index=index, time=time.time()))
                wait_until(ready)
                record, = [r for r in read(method/'runtime.json')['videos'] if r['index'] == index]
                assert record['attention_calls'] == 4000 and record['layer_calls'] == [100]*40
                assert record['dense_self_attention_calls'] == 0 and record['video'] == name
                paths = [REFERENCE/name, method/name]
                before = [sha(path) for path in paths]
                assert before[1] == record['video_sha256']
                save(output, dict(stage='evaluating', capacity=cap, index=index, time=time.time()))
                started = time.perf_counter()
                row = evaluator.evaluate_pair(*paths, model, 1, torch.device('cpu'), expected_frames=81)
                assert row['frames'] == 81 and all(math.isfinite(row[k]) for k in ('psnr', 'ssim', 'lpips'))
                assert before == [sha(path) for path in paths]
                row.update(index=index, evaluation_seconds=time.perf_counter()-started,
                    input_sha256=dict(reference=before[0], method=before[1]))
                result['videos'].append(row)
                result['summary'] = evaluator.aggregate(result['videos'])
                save(destination/f'{index:03d}.json', row)
                save(destination/'quality.json', result)
                print(json.dumps(dict(capacity=cap, **row)), flush=True)
            result['complete'] = True
            save(destination/'quality.json', result)
        success = True
    except BaseException:
        save(args.root/'quality_failure.json', dict(error=traceback.format_exc(), time=time.time()))
        raise
    finally:
        save(output, dict(stage='complete' if success else 'failed', complete=success, time=time.time()))


if __name__ == '__main__':
    main()
