"""CPU-only source/assembly context for H40 local-memory spill sites."""
import argparse
import collections
import json
from pathlib import Path
import re
import subprocess

p=argparse.ArgumentParser()
p.add_argument('compile_json',type=Path)
p.add_argument('source_name')
p.add_argument('output_dir',type=Path)
a=p.parse_args()
report=json.loads(a.compile_json.read_text())
candidate=report['candidates'][a.source_name]
assert candidate['compiled']
a.output_dir.mkdir(parents=True,exist_ok=True)
out=a.output_dir.resolve()
subprocess.run(['/usr/local/cuda/bin/cuobjdump','--extract-elf','all',candidate['module']],
               cwd=out,check=True,capture_output=True,text=True)
cubins=list(out.glob('*.cubin'))
assert len(cubins)==1, cubins
assembly=subprocess.check_output(['/usr/local/cuda/bin/nvdisasm','--print-line-info',
    str(cubins[0])],text=True)
(out/'assembly.txt').write_text(assembly)
active=False
location='unknown'
sites=[]
history=collections.deque(maxlen=4)
for line in assembly.splitlines():
    section=re.search(r'\.section\s+\.text\.([^,\s]+)',line)
    if section:
        active=section.group(1).startswith('_Z12merge_kernelILi21ELi40E')
        history.clear()
    if not active:
        continue
    if '//##' in line:
        location=line.strip()
    if re.search(r'\b(?:LDL|STL)\b',line):
        sites.append(dict(location=location,instruction=line.strip(),before=list(history)))
    if '/*' in line:
        history.append(line.strip())
counts=collections.Counter(x['location'] for x in sites)
result={'source':a.source_name,'spill_sites':sites,'source_counts':dict(counts)}
(out/'spill_sites.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'source':a.source_name,'site_count':len(sites),'source_counts':dict(counts)}))
