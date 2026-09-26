"""Stop an owned legacy worker after its assigned videos are safely recorded.

The old worker predates explicit index plans. Preserve every saved video and
record the reduced assignment instead of claiming the original run completed.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from run import read, save


def owned(pid, root, mode, shard):
    args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    decoded = [v.decode() for v in args if v]
    assert str(root) in decoded and mode in decoded, decoded
    assert decoded[decoded.index('--shard') + 1] == str(shard), decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--capacity', type=int, required=True)
    parser.add_argument('--count', type=int, required=True)
    args = parser.parse_args()
    folder = args.root / f'worker{args.shard}'
    launch = read(folder / 'launch.json')
    assert launch['host'].split('.')[0] == os.uname().nodename.split('.')[0]
    worker, supervisor = launch['pid'], launch['supervisor_pid']
    owned(worker, args.root, 'worker', args.shard)
    owned(supervisor, args.root, 'supervise', args.shard)
    assert os.getpgid(worker) == worker
    request = dict(pid=os.getpid(), worker_pid=worker, supervisor_pid=supervisor,
                   capacity=args.capacity, indices=list(range(args.count)), time=time.time())
    assert not (folder / 'handoff_request.json').exists()
    save(folder / 'handoff_request.json', request)
    runtime = args.root / f'C{args.capacity}/runtime_worker{args.shard}.json'
    deadline = time.monotonic() + 3600
    while True:
        assert time.monotonic() < deadline, 'Handoff timed out'
        owned(worker, args.root, 'worker', args.shard)
        if runtime.exists():
            records = read(runtime)['records']
            indices = [r['index'] for r in records]
            assert all(i < args.count for i in indices), 'Handoff boundary already crossed'
            if indices == request['indices']:
                break
        time.sleep(0.25)
    # Freeze at the persisted boundary before stopping supervision. Only the
    # current worker process group is signalled; no foreign process is touched.
    owned(supervisor, args.root, 'supervise', args.shard)
    os.killpg(worker, signal.SIGSTOP)
    try:
        assert [r['index'] for r in read(runtime)['records']] == request['indices']
        assert not (folder / 'contention.json').exists()
        os.kill(supervisor, signal.SIGTERM)
        time.sleep(1)
        os.killpg(worker, signal.SIGTERM)
    finally:
        os.killpg(worker, signal.SIGCONT)
    for _ in range(60):
        if not Path(f'/proc/{worker}/cmdline').exists():
            break
        if not Path(f'/proc/{worker}/cmdline').read_bytes():
            break
        time.sleep(1)
    else:
        raise RuntimeError('Worker did not exit after planned termination')
    result = dict(valid=True, scope='assigned subset', campaign_complete=False,
                  reason='planned reassignment of remaining videos to additional idle GPUs',
                  signal='SIGTERM', indices=request['indices'], capacity=args.capacity,
                  previous_assignment_preserved_in_protocol=True, time=time.time())
    save(folder / 'handoff_complete.json', result)
    save(folder / 'completion.json', result)
    save(folder / 'stage.json', dict(stage='handed_off', **result))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
