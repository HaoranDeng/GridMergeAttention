"""Exclusive idle-H200 guard for explicitly authorized axis experiments."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from analysis.quick_axis_c4_qkv import gpu_state, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', required=True, type=int)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.command[1:] if a.command[0] == '--' else a.command
    state = gpu_state(a.gpu)
    if 'H200' not in state['name'] or state['pids'] or state['memory_mib'] > 100 or state['utilization']:
        raise RuntimeError(f'GPU not clean; no launch or queue: {state}')
    a.output.mkdir(parents=True, exist_ok=False)
    save(a.output/'preflight.json', state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=state['uuid'], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', TRITON_CACHE_DIR=str(a.output.parent/'triton_cache'))
    child, failure = None, None
    try:
        with (a.output/'worker.log').open('w') as log, (a.output/'gpu_monitor.jsonl').open('w') as monitor:
            state = gpu_state(a.gpu)
            if state['pids'] or state['memory_mib'] > 100 or state['utilization']:
                raise RuntimeError('GPU changed immediately before launch')
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(a.output/'process.json', {'parent_pid':os.getpid(), 'child_pid':child.pid, 'command':cmd})
            while child.poll() is None:
                state = gpu_state(a.gpu)
                # Profilers may launch the CUDA worker as a descendant. Only
                # allow processes in the isolated session we created above.
                owned = {child.pid}
                for pid in state['pids']:
                    try:
                        if os.getsid(pid) == child.pid:
                            owned.add(pid)
                    except ProcessLookupError:
                        pass
                foreign = set(state['pids']) - owned
                monitor.write(json.dumps({'time':time.time(), **state})+'\n'); monitor.flush()
                if foreign:
                    raise RuntimeError(f'Foreign GPU process, timing invalid: {sorted(foreign)}')
                time.sleep(5)
            if child.returncode:
                raise RuntimeError(f'Worker exited {child.returncode}')
    except BaseException as e:
        failure = repr(e)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        raise
    finally:
        save(a.output/'completion.json', {'valid':failure is None, 'error':failure,
              'returncode':None if child is None else child.returncode, 'finished_at':time.time()})


if __name__ == '__main__':
    main()
