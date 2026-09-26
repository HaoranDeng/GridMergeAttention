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
