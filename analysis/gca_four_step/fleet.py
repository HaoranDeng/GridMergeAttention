"""Exclusive GPU guard for a frozen per-GPU campaign plan."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from run import read, save, snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    plan = read(a.plan)
    root = a.plan.parent
    assert os.uname().nodename.split('.')[0] == plan['host']
    lock = (root / 'guard.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / 'launch.json').exists()
    if plan.get('after_completion'):
        # This is a dependency on our already-running job, not a queue for a
        # busy foreign GPU. Clean preflight is still required after completion.
        marker = Path(plan['after_completion'])
        deadline = time.monotonic() + 3600
        while not marker.exists():
            assert time.monotonic() < deadline
            time.sleep(5)
        assert read(marker)['valid']
    gpu_lock = (Path(plan['lock_root']) / f"gca_gpu_{plan['gpu']}.lock").open('w')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for _ in range(3):
        state = snapshot(plan['gpu'])
        assert not state['pids'] and state['memory_mib'] < 100 and state['utilization'] == 0, state
        time.sleep(1)
    save(root / 'preflight.json', state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan['gpu'], PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_JOBS='2',
               TOKENIZERS_PARALLELISM='false', WAN_SDPA_BACKEND='flash',
               TOKEN_MERGING_FORCE_TORCH_RMSNORM='1',
               TORCH_EXTENSIONS_DIR=str(root / 'torch_extensions'),
               FLASHINFER_WORKSPACE_BASE=str(root / 'flashinfer_workspace'),
               TRITON_CACHE_DIR=str(root / 'triton_cache'))
    env['PATH'] = '/usr/local/cuda/bin:' + env.get('PATH', '')
    command = [plan['python'], plan['script'], '--plan', str(a.plan)]
    with (root / 'worker.log').open('w') as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(root / 'launch.json', dict(pid=child.pid, supervisor_pid=os.getpid(),
             command=command, preflight=state, time=time.time()))
        def stop(signum, frame):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, stop)
        valid = True
        while child.poll() is None:
            state = snapshot(plan['gpu'])
            foreign = []
            for pid in state['pids']:
                try:
                    if os.getpgid(pid) != child.pid:
                        foreign.append(pid)
                except ProcessLookupError:
                    pass
            state['foreign_pids'] = foreign
            save(root / 'gpu_latest.json', state)
            if foreign:
                valid = False
                save(root / 'contention.json', state)
                os.killpg(child.pid, signal.SIGTERM)
                break
            time.sleep(3)
        code = child.wait()
    save(root / 'completion.json', dict(valid=valid and code == 0, exit_code=code, time=time.time()))
    return 0 if valid and code == 0 else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        if '--plan' in sys.argv:
            folder = Path(sys.argv[sys.argv.index('--plan') + 1]).parent
            save(folder / 'guard_failure.json', dict(error=traceback.format_exc(), time=time.time()))
        traceback.print_exc()
        raise
