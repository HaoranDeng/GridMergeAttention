"""Compile-only preparation or clean-GPU gate for shared atomic ownership.

This tests arbitrary score bits and padding/ties, not the complete attention
operator or its speed. Full real-QKV validation remains a separate gate.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--gate-source', type=Path, default=ROOT/'native_fisher/owner_key_gate.cu')
    parser.add_argument('--merge-source', type=Path, default=ROOT/'native_fisher/fisher_c10_atomic_owner.cu')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing receipt')
    if args.compile_only and args.gpu is not None:
        parser.error('Compile-only mode does not select a GPU')
    if not args.compile_only and args.gpu is None:
        parser.error('GPU execution requires an explicit clean GPU')
    from analysis.launch_quality_first_videos import require_clean, gpu_processes, now
    result = {'scope': 'C10 shared atomic score/owner gate only',
              'gpu_validated': False, 'performance_measured': False,
              'compile_only': args.compile_only}
    errors, samples = [], []
    stop = threading.Event()
    monitor_thread = None
    if args.compile_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        initial = require_clean(args.gpu)
        result['initial_gpu'] = initial
        os.environ['CUDA_VISIBLE_DEVICES'] = initial['gpu']['uuid']

        def monitor():
            while not stop.is_set():
                try:
                    processes = gpu_processes(initial['gpu']['uuid'])
                    samples.append({'utc': now(), 'processes': processes})
                    if any(row['pid'] != os.getpid() for row in processes):
                        errors.append('foreign GPU process')
                except Exception as exc:
                    errors.append(repr(exc))
                stop.wait(3)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
    os.environ.setdefault('MAX_JOBS', '2')
    os.environ['TORCH_EXTENSIONS_DIR'] = str(args.output.parent/'torch_extensions')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [args.gate_source.resolve(), ROOT/'native_fisher/owner_key.h',
             args.merge_source.resolve()]
    source_bytes = {path.name: path.read_bytes() for path in paths}
    hashes = {name: hashlib.sha256(data).hexdigest()
              for name, data in source_bytes.items()}
    result['source_sha256'] = hashes
    # Include the header in the build identity; edits must not reuse a stale
    # extension when only the score/owner encoding changes.
    build_hash = hashlib.sha256(source_bytes[paths[0].name] +
                                source_bytes['owner_key.h']).hexdigest()
    try:
        import numpy as np
        import torch
        from torch.utils.cpp_extension import load
        extension = load(name='fisher_owner_gate_'+build_hash[:16],
                         sources=[str(paths[0])],
                         extra_cuda_cflags=['-O3', '-lineinfo'], verbose=False)
        assert all(path.read_bytes() == source_bytes[path.name] for path in paths)
        result['compiled'] = True
        result['module_file'] = extension.__file__
        if args.compile_only:
            assert not torch.cuda.is_initialized()
            result['cuda_initialized'] = False
        else:
            assert not errors, errors
            rng = np.random.default_rng(20260909)
            bits = rng.integers(0, 2**32, (512, 64, 80), dtype=np.uint32)
            edges = [0xff800000, 0x7fc00000, 0xffc00000, 0,
                     0x80000000, 0x7f800000, 1, 0x80000001,
                     0x007fffff, 0x807fffff, 0x00800000, 0x80800000]
            for tile, edge in enumerate(edges):
                bits[tile].fill(edge)
            # Repeated scores in every owner position, including both zero
            # signs. Empty winners initialize to -inf / owner0, not a NaN.
            for tile in range(12, 32):
                base = rng.integers(0, 2**32, (6, 80), dtype=np.uint32)
                bits[tile, :60] = np.repeat(base, 10, axis=0)
            for tile in range(32, 52):
                bits[tile].fill(0xffc00000)
                owner = (tile-32)//2
                for col in range(6):
                    bits[tile, col*10+owner] = 0 if tile%2==0 else 0x80000000
                    if owner < 9:
                        bits[tile, col*10+owner+1] = 0x80000000 if tile%2==0 else 0
            bits[:, 60:].fill(0x7f800000)  # Poison all padded rows.
            grouped = bits[:, :60].reshape(512, 6, 10, 80)
            winner = np.full((512, 6, 80), 0xff800000, dtype=np.uint32)
            owner = np.zeros(winner.shape, dtype=np.int32)
            with np.errstate(invalid='ignore'):
                for seed in range(10):
                    score = grouped[:, :, seed]
                    replace = score.view(np.float32) > winner.view(np.float32)
                    winner = np.where(replace, score, winner)
                    owner = np.where(replace, seed, owner)
            # Imports/builds and CPU fixtures can outlast a clean preflight.
            # Abort before allocating GPU input if occupancy changed meanwhile.
            assert not errors, errors
            input_tensor = torch.from_numpy(bits.view(np.int32)).cuda()
            result['rotations'] = []
            for rotation in (0, 7, 19, 39):
                assert not errors, errors
                actual_bits, actual_owner = extension.run(input_tensor, rotation)
                actual_bits = actual_bits.cpu().numpy().view(np.uint32)
                actual_owner = actual_owner.cpu().numpy()
                check = {'rotation': rotation,
                         'score_bit_mismatches': int(np.count_nonzero(actual_bits != winner)),
                         'owner_mismatches': int(np.count_nonzero(actual_owner != owner))}
                result['rotations'].append(check)
                assert check['score_bit_mismatches'] == check['owner_mismatches'] == 0, check
            result['groups_per_rotation'] = int(winner.size)
            result['gpu_validated'] = True
        result['valid'] = not errors
        assert not errors, errors
    except Exception as exc:
        result.update(valid=False, error=repr(exc), gpu_validated=False)
        raise
    finally:
        stop.set()
        if monitor_thread:
            monitor_thread.join(timeout=5)
        if errors:
            result.update(valid=False, gpu_validated=False)
        result.update(monitor_errors=errors, monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'monitor_samples'}))


if __name__ == '__main__':
    main()
