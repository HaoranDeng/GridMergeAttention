"""Incremental local synchronization with hashes; never displays videos/frames."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/local2/denghaoran/GridMergeAttention/experiments/q80k90_i2v_c4_c6_c8_c10_20260924'
PYTHON='/home/denghaoran/miniconda3/envs/monarch_rt/bin/python'
OPTS=['-o','ControlPath=none','-o','HostName=scai7.cs.ucla.edu','-o','BatchMode=yes','-o','ConnectTimeout=12']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    result=ROOT/'results';result.mkdir(exist_ok=True)
    known=result/'public_hashes.json';extra=[]
    if known.exists():
        subprocess.run(['scp','-q',*OPTS,str(known),f'scai7:{REMOTE}/local_known_hashes.json'],check=True)
        extra=['--known',REMOTE+'/local_known_hashes.json']
    text=subprocess.check_output(['ssh',*OPTS,'scai7',PYTHON,REMOTE+'/source/export_public.py',*extra],text=True)
    receipt=json.loads(text.strip().splitlines()[-1]);archive=ROOT/'public_delta.tar.gz'
    subprocess.run(['scp','-q',*OPTS,'scai7:'+receipt['archive'],str(archive)],check=True)
    assert sha(archive)==receipt['sha256']
    with tarfile.open(archive,'r:gz') as tar:
        manifest=json.load(tar.extractfile('public_hashes.json'))
        for member in tar.getmembers():
            assert member.isfile() and not member.name.endswith('_environment.json')
            dest=(result/member.name).resolve();assert result.resolve() in dest.parents
            assert member.name=='public_hashes.json' or member.name in manifest
            dest.parent.mkdir(parents=True,exist_ok=True)
            data=tar.extractfile(member).read()
            if member.name!='public_hashes.json':assert hashlib.sha256(data).hexdigest()==manifest[member.name]
            dest.write_bytes(data)
    for name,h in manifest.items():assert sha(result/name)==h,name
    (ROOT/'last_sync.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print((result/'public_status.json').read_text())
if __name__=='__main__':main()
