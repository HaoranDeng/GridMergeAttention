"""Recompile one cached candidate to a separate object and retain diagnostics."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

p=argparse.ArgumentParser()
p.add_argument('compile_json',type=Path)
p.add_argument('source_name')
p.add_argument('output_dir',type=Path)
a=p.parse_args()
report=json.loads(a.compile_json.read_text())
candidate=report['candidates'][a.source_name]
if 'module' in candidate:
    module=Path(candidate['module'])
else:
    match=re.search(r"Error building extension '([^']+)'",candidate['error'])
    if match is None:
        raise ValueError('failed receipt does not identify an extension')
    name=match.group(1)
    module=a.compile_json.resolve().parent/'torch_extensions'/name/(name+'.so')
commands=subprocess.check_output(['ninja','-C',str(module.parent),'-t','commands'],text=True)
command=next(shlex.split(x) for x in commands.splitlines() if '/nvcc ' in x)
a.output_dir.mkdir(parents=True,exist_ok=True)
output=a.output_dir.resolve()
command[command.index('-o')+1]=str(output/'diagnostic.cuda.o')
if '-MF' in command:
    command[command.index('-MF')+1]=str(output/'diagnostic.cuda.o.d')
command+=['--keep','--keep-dir',str(output)]
result=subprocess.run(command,cwd=module.parent,text=True,capture_output=True)
log=result.stdout+result.stderr
(output/'nvcc.log').write_text(log)
(output/'command.json').write_text(json.dumps(command,indent=2))
print(json.dumps({'exit_code':result.returncode,'diagnostics':[
    x for x in log.splitlines() if any(k in x.lower() for k in ('warning','serialized','spill','register','stack'))]}))
raise SystemExit(result.returncode)
