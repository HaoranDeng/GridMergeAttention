"""Replace this experiment's idle watcher without interrupting a GPU child."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--run-dir',type=Path,required=True)
a=p.parse_args()
root=a.run_dir.resolve()
lock=root/'watcher.lock'
pid=int(lock.read_text())
proc=Path(f'/proc/{pid}')
command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
assert 'analysis/watch_c8_pipeline.py' in command and str(root) in command, command
assert proc.stat().st_uid==os.getuid()
stopped=False
try:
    os.kill(pid,signal.SIGSTOP);stopped=True
    children=(proc/f'task/{pid}/children').read_text().strip()
    if children:
        print(json.dumps({'restarted':False,'reason':'watcher has an active child'}))
        sys.exit(0)
    os.kill(pid,signal.SIGTERM)
    os.kill(pid,signal.SIGCONT);stopped=False
    for _ in range(100):
        if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':
            break
        time.sleep(.05)
    else:
        raise RuntimeError('Old watcher did not exit; refusing duplicate launch')
    if lock.exists():
        assert int(lock.read_text())==pid
        lock.unlink()
finally:
    if stopped:
        os.kill(pid,signal.SIGCONT)

with (root/'watcher.log').open('a') as output:
    child=subprocess.Popen([sys.executable,'-u','analysis/watch_c8_pipeline.py',
                            '--run-dir',str(root)],cwd=root/'source',
                           stdout=output,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({'restarted':True,'old_pid':pid,'new_pid':child.pid}))
