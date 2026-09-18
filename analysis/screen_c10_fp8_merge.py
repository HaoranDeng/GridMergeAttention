"""Complete H5 storage gate for a new FP8 merge; no attention timing claim.

Compare every compressed K/V byte and bias against the retained FP8 merge
on real layer00. This gate can also be run under Compute Sanitizer before
the separate complete H40 benchmark. It does not queue for a busy GPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--head-column-pack', action='store_true', help='Check and use candidate physical head/column Q packing')
    parser.add_argument('--paired-kv', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    initial = require_clean(args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=initial['gpu']['uuid'], OMP_NUM_THREADS='4',
                      TORCH_EXTENSIONS_DIR=str(args.output.parent / 'torch_extensions'))
    result = {'valid': False, 'scope': 'Complete H5 merge storage, no route or full H40 timing',
              'initial_gpu': initial, 'source': str(args.source.resolve()),
              'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    errors, samples = [], []
    stop = threading.Event()

    def save():
        result.update(monitor_errors=errors, monitor_samples=samples)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(args.output)

    def monitor():
        while not stop.is_set():
            try:
                processes = gpu_processes(initial['gpu']['uuid'])
                samples.append({'utc': now(), 'processes': processes})
                if any(row['pid'] != os.getpid() for row in processes):
                    errors.append('Foreign GPU process')
            except Exception as error:
                errors.append(repr(error))
            stop.wait(3)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        import torch
        from native_fisher.fp8_c10 import FP8C10Attention, retained_control
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        control = retained_control()
        reference = FP8C10Attention(control, 'direct', source=ROOT / 'native_fisher/fisher_c10_fp8_shared_store.cu')
        if args.paired_kv:
            assert not args.head_column_pack
            from native_fisher.fp8_paired_kv import PairedFP8C10Attention
            candidate=PairedFP8C10Attention(control,args.source)
        else:candidate = FP8C10Attention(control, 'direct', source=args.source)
        del control
        result.update(resources=candidate.merge.resources(), reference_resources=reference.merge.resources(),
                      reference_source_sha256=reference.merge.source_sha256)
        assert candidate.merge.source_sha256 == result['source_sha256']
        payload = torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt',
                             map_location='cpu', weights_only=True)
        q, k, v = [payload[name].reshape(1, 21, 45, 80, 40, 128).cuda() for name in ('q', 'k', 'v')]
        del payload
        seeds = reference.select(q, k, 128**-.5)
        reference.merge.pack(q, reference.packed)
        assert reference.packed.data_ptr() == candidate.packed.data_ptr()
        reference.merge_chunk(k, v, seeds, 0, 128**-.5)
        torch.cuda.synchronize()
        result['phase'] = 'candidate_merge'
        save()
        if args.head_column_pack:
            expected=reference.packed.reshape(21,45,80,40,128).permute(3,2,0,1,4).contiguous()
            candidate.merge.pack(q,candidate.packed)
            result['packed_query_bytes_equal']=bool(torch.equal(expected.view(torch.uint8).flatten(),candidate.packed.view(torch.uint8).flatten()))
            result['packed_query_bytes_checked']=expected.numel()*expected.element_size()
            assert result['packed_query_bytes_equal']
            del expected
        candidate.merge_chunk(k, v, seeds, 0, 128**-.5)
        torch.cuda.synchronize()
        if args.paired_kv:
            pair=candidate.pair.view(torch.uint8).reshape(*reference.mk.shape,2)
            result['exact']={'mk':bool(torch.equal(reference.mk.view(torch.uint8),pair[...,0])),
                'mv':bool(torch.equal(reference.mv.view(torch.uint8),pair[...,1])),
                'bias':bool(torch.equal(reference.bias.view(torch.uint8),candidate.bias.view(torch.uint8)))}
            result['paired_byte_storage']=True
        else:
            result['exact'] = {name: bool(torch.equal(getattr(reference, name).view(torch.uint8),
                                                    getattr(candidate, name).view(torch.uint8)))
                               for name in ('mk', 'mv', 'bias')}
        assert all(result['exact'].values()), result['exact']
        assert hashlib.sha256(args.source.read_bytes()).hexdigest() == result['source_sha256']
        assert not errors, errors
        result.update(valid=True, phase='complete', kv_bytes_checked=2*candidate.mk.numel(),
                      bias_elements_checked=candidate.bias.numel())
        print(json.dumps({key: value for key, value in result.items() if not key.startswith('monitor_')}), flush=True)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        stop.set()
        thread.join(5)
        result['valid'] = result['valid'] and not errors
        save()


if __name__ == '__main__':
    main()
