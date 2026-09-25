"""Install the frozen experiment and start idempotent host watchers."""
import base64
import json
from pathlib import Path
import subprocess

LOCAL = Path(__file__).resolve().parents[1]
REMOTE = '/local2/denghaoran/GridMergeAttention/experiments/svg_ear_p04_i2v20_20260925'
FILES = ['common.py', 'worker.py', 'evaluate.py', 'watch.py', 'prepare.py']
payload = {name:base64.b64encode((LOCAL/'source'/name).read_bytes()).decode() for name in FILES}
for name in FILES:
    compile((LOCAL/'source'/name).read_text(encoding='utf-8'), name, 'exec')


def remote(script, host='scai7'):
    options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
    if host == 'scai6':
        options += ['-o', 'HostName=scai6.cs.ucla.edu']
    p = subprocess.run(['ssh', *options, 'scai7', 'python -'], input=script,
                       text=True, encoding='utf-8', capture_output=True, timeout=180)
    assert p.returncode == 0, p.stderr[-3500:] + p.stdout[-1000:]
    return p.stdout.strip()


setup = '''
import base64,json,subprocess
from pathlib import Path
root=Path(REMOTE);root.mkdir(parents=True,exist_ok=True)
assert not (root/'experiment_plan.json').exists(), 'Prepared already; do not duplicate'
(root/'source').mkdir(exist_ok=True)
for name,data in PAYLOAD.items():
    (root/'source'/name).write_bytes(base64.b64decode(data))
subprocess.run(['/home/denghaoran/miniconda3/envs/monarch_rt/bin/python',str(root/'source/prepare.py')],check=True)
'''
print(remote('REMOTE='+repr(REMOTE)+'\nPAYLOAD='+repr(payload)+'\n'+setup))
launch = '''
import json,os,socket,subprocess,time
from pathlib import Path
root=Path(REMOTE);host=socket.gethostname().split('.')[0]
control=root/'watchers';control.mkdir(exist_ok=True)
receipt=control/(host+'.launch_request.json')
assert not receipt.exists(), 'Watcher already requested; inspect recorded PID'
with (control/(host+'.log')).open('x') as log:
    p=subprocess.Popen(['/home/denghaoran/miniconda3/envs/monarch_rt/bin/python',str(root/'source/watch.py')],
        cwd=root,env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(host=host,pid=p.pid,time=time.time())
receipt.write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
'''
receipts = [json.loads(remote('REMOTE='+repr(REMOTE)+'\n'+launch, host)) for host in ['scai7','scai6']]
(LOCAL/'launch_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipts))
