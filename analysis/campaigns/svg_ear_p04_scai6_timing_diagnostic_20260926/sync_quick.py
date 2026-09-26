"""Fetch only structured p=.4 diagnostic artifacts with SHA256 verification."""
import base64,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/local2/denghaoran/GridMergeAttention/experiments/svg_ear_p04_p05_scai6_timing5_20260926'
CASE=sys.argv[1] if len(sys.argv)>1 else 'quick_diagnostic'
assert CASE in ['quick_diagnostic','quick_no_density']
script='''
import base64,hashlib,json
from pathlib import Path
r=Path(REMOTE)/CASE;files={}
for p in list(r.glob('*.json'))+list(r.glob('*.jsonl')):
 data=p.read_bytes();files[p.name]=dict(sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode())
print(json.dumps(files))
'''
z=subprocess.run(['ssh','-o','BatchMode=yes','-o','HostName=scai6.cs.ucla.edu','scai7','python -'],input='REMOTE='+repr(REMOTE)+'\nCASE='+repr(CASE)+'\n'+script,text=True,capture_output=True,timeout=90)
assert z.returncode==0,z.stderr[-1200:]
files=json.loads(z.stdout);out=ROOT/CASE;out.mkdir(exist_ok=True)
for name,item in files.items():
 assert Path(name).name==name
 data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256']
 (out/name).write_bytes(data)
(out/'sync_hashes.json').write_text(json.dumps({n:x['sha256'] for n,x in files.items()},indent=2)+'\n')
print(json.dumps(dict(files=len(files),complete='completion.json' in files)))
