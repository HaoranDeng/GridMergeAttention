"""Finish the in-flight legacy video, then replace worker with disjoint shard."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from analysis.quick_axis_c4_qkv import gpu_state,save

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--legacy-output',type=Path,required=True)
    p.add_argument('--legacy-guard',type=Path,required=True)
    p.add_argument('--stop-after-count',type=int,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();cmd=a.command[1:] if a.command[0]=='--' else a.command
    process=json.loads((a.legacy_guard/'process.json').read_text())
    pid=process['child_pid']
    actual=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert str(a.legacy_output) in actual and 'analysis.run_axis_c4_videos' in actual
    assert os.getsid(pid)==pid
    print(json.dumps({'event':'waiting_for_video_boundary','count':a.stop_after_count,'pid':pid}),flush=True)
    read_failures=0
    while True:
        try:
            records=json.loads((a.legacy_output/'runtime.json').read_text())['videos']
            read_failures=0
        except (OSError,json.JSONDecodeError) as error:
            # Shared storage can briefly reject opens during atomic rename.
            # Retry boundedly without touching the running generation process.
            read_failures+=1
            if read_failures>60:raise
            if read_failures==1:print(json.dumps({'event':'retry_runtime_read','error':repr(error)}),flush=True)
            time.sleep(.5);continue
        if len(records)>=a.stop_after_count:break
        if not Path(f'/proc/{pid}').exists():raise RuntimeError('Legacy worker exited before boundary')
        time.sleep(.25)
    assert len(records)==a.stop_after_count,'Boundary missed; do not risk overlapping prompt indices'
    save(a.receipt,{'reason':'user requested all idle H200s; migrate at completed-video boundary',
                    'legacy_pid':pid,'completed_indices':[v['index'] for v in records],
                    'next_command':cmd,'time':time.time(),'state':'stopping_legacy_after_export'})
    os.killpg(pid,signal.SIGTERM)
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        state=gpu_state(a.gpu)
        if not state['pids'] and state['memory_mib']<=100 and not state['utilization']:break
        time.sleep(1)
    else:raise RuntimeError('GPU did not become exclusively idle; replacement not queued')
    print(json.dumps({'event':'launching_disjoint_replacement','gpu':a.gpu}),flush=True)
    result=subprocess.run(cmd,check=False)
    receipt=json.loads(a.receipt.read_text());receipt.update(state='replacement_finished',returncode=result.returncode)
    save(a.receipt,receipt)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
