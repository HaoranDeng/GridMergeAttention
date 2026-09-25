"""Wait for an exclusively idle H200; one shared lease wins across both hosts."""
import csv
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from common import ROOT, read, save

HOST = socket.gethostname().split('.')[0]


def inventory():
    gpu_rows = csv.reader(subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).splitlines())
    processes = list(csv.reader(subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True).splitlines()))
    return [dict(host=HOST, index=int(r[0]), uuid=r[1].strip(), model=r[2].strip(),
                 memory_mib=int(r[3]), utilization=int(r[4]),
                 pids=[int(p[1]) for p in processes if p and p[0].strip()==r[1].strip()])
            for r in gpu_rows]


def reservations(gpu):
    """Detect this user's CPU-side GPU loaders/guards before nvidia-smi sees them."""
    result = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if str(ROOT) in ' '.join(args) or not args:
                continue
            has_flag = any(args[n] in ('--gpu', '--device') and args[n+1] == str(gpu['index'])
                           for n in range(len(args)-1))
            has_flag |= f'--gpu={gpu["index"]}' in args or f'{HOST}_gpu{gpu["index"]}' in args
            # Only inspect the one routing field; never record or export environments.
            visible = next((x.split(b'=',1)[1].decode() for x in (proc/'environ').read_bytes().split(b'\0')
                            if x.startswith(b'CUDA_VISIBLE_DEVICES=')), '')
            targets = visible.split(',')
            if has_flag or str(gpu['index']) in targets or gpu['uuid'] in targets:
                result.append(int(proc.name))
        except (OSError, ProcessLookupError):
            pass
    return result


def clean(gpu):
    return (gpu['model'] == 'NVIDIA H200' and not gpu['pids'] and
            gpu['memory_mib'] < 100 and gpu['utilization'] == 0 and not reservations(gpu))


def run(gpu):
    plan = read(ROOT/'experiment_plan.json')
    control = ROOT/'guard'
    control.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1', MAX_JOBS='4',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               WAN_MODEL_ROOT=plan['protocol']['model_root'], GRIDMERGE_SDPA_BACKEND='flash_attn2',
               SPARSE_VIDEO_GEN_ROOT=plan['source_svg_root'], SVG_TOP_P='0.4',
               SVG_NUM_Q_CENTROIDS='300', SVG_NUM_K_CENTROIDS='1000', SVG_MIN_KC_RATIO='0.10',
               SVG_KMEANS_INIT='50', SVG_KMEANS_STEP='2',
               FLASHINFER_WORKSPACE_BASE='/xuanwu-tank/east/zrl/exp-gridmergeattn/.cache/environments/monarch-rt/flashinfer_workspace',
               TORCH_EXTENSIONS_DIR=str(control/'torch_extensions'), TRITON_CACHE_DIR=str(control/'triton_cache'),
               PATH='/usr/local/cuda/bin:'+os.environ['PATH'])
    save(ROOT/'active_gpu.json', gpu)
    save(control/'preflight.json', dict(time=time.time(), **gpu))
    child = None
    failure = None
    try:
        with (ROOT/'generation.log').open('x') as log, (control/'monitor.jsonl').open('x') as monitor:
            child = subprocess.Popen([plan['python'], str(ROOT/'source/worker.py')], cwd=ROOT,
                                     env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(ROOT/'launch.json', dict(supervisor=os.getpid(), worker=child.pid, host=HOST,
                                          gpu=gpu, time=time.time()))
            cpu_env = dict(env, CUDA_VISIBLE_DEVICES='')
            with (ROOT/'quality.log').open('x') as qlog:
                evaluator = subprocess.Popen([plan['evaluation_python'], str(ROOT/'source/evaluate.py')],
                    cwd=ROOT, env=cpu_env, stdout=qlog, stderr=subprocess.STDOUT, start_new_session=True)
            save(ROOT/'quality_process.json', dict(pid=evaluator.pid, host=HOST, time=time.time()))
            started = time.monotonic()
            while True:
                current, = [r for r in inventory() if r['uuid']==gpu['uuid']]
                foreign = []
                for pid in current['pids']:
                    try:
                        if os.getsid(pid) != child.pid:
                            foreign.append(pid)
                    except ProcessLookupError:
                        pass
                record = dict(time=time.time(), foreign_pids=foreign, **current)
                monitor.write(__import__('json').dumps(record)+'\n')
                monitor.flush()
                save(control/'latest.json', record)
                if foreign:
                    raise RuntimeError(f'Foreign GPU process detected: {foreign}; this run timing is invalid')
                if child.poll() is not None:
                    assert child.returncode == 0, f'Worker exited {child.returncode}'
                    assert read(ROOT/'generation_complete.json')['videos'] == 20
                    break
                assert time.monotonic()-started < 24*3600, '24-hour generation deadline'
                time.sleep(5)
    except BaseException:
        failure = traceback.format_exc()
        save(ROOT/'failure.json', dict(error=failure, time=time.time(), host=HOST))
    finally:
        if child is not None and child.poll() is None:
            # Only this newly created child session can be stopped here.
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        save(control/'completion.json', dict(valid=failure is None, failure=failure, gpu=gpu,
                                            time=time.time(), restoration_required=False))


def main():
    assert HOST in ('scai6', 'scai7')
    control = ROOT/'watchers'
    control.mkdir(exist_ok=True)
    host_lock = (control/f'{HOST}.lock').open('a')
    fcntl.flock(host_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(control/f'{HOST}.process.json', dict(pid=os.getpid(), host=HOST, started_at=time.time()))
    deadline = time.monotonic()+7*24*3600
    while time.monotonic() < deadline:
        if (ROOT/'launch.json').exists():
            save(control/f'{HOST}.status.json', dict(status='dispatched', launch=read(ROOT/'launch.json'), time=time.time()))
            return
        states = inventory()
        candidates = []
        for gpu in states:
            gpu['reservation_pids'] = reservations(gpu) if not gpu['pids'] else []
            if clean(gpu):
                candidates.append(gpu)
        save(control/f'{HOST}.status.json', dict(status='waiting_for_exclusive_gpu', time=time.time(),
             candidates=[g['index'] for g in candidates], gpus=states))
        if candidates:
            lease = (ROOT/'campaign.lock').open('a')
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                time.sleep(20)
                continue
            try:
                if (ROOT/'launch.json').exists():
                    return
                gpu = candidates[0]
                time.sleep(2)
                second, = [g for g in inventory() if g['uuid']==gpu['uuid']]
                if clean(second):
                    save(control/f'{HOST}.status.json', dict(status='claimed', gpu=second, time=time.time()))
                    run(second)
                    return
            finally:
                lease.close()
        time.sleep(20)
    save(control/f'{HOST}.status.json', dict(status='waiting_deadline_reached', time=time.time()))


if __name__ == '__main__':
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    main()
