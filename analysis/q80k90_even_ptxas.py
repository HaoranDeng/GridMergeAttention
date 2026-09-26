"""Print ptxas resource usage (registers, stack, spills) of one Q80K90 even-capacity kernel build.

python q80k90_even_ptxas.py merge --cap 8 [--flags "-DMERGE_PACKED=1 ..."]
Compiles analysis/q80k90_even_<stem>.cu to a cubin with nvcc (no extension
load, no GPU needed) using the flags native_<stem> would pass.
"""
import argparse,subprocess,sys,tempfile
from pathlib import Path
import torch
from torch.utils.cpp_extension import include_paths

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('stem',choices=['merge','route','select','layout'])
ap.add_argument('--cap',type=int,default=8);ap.add_argument('--hc',type=int,default=8);ap.add_argument('--flags',default='')
a=ap.parse_args()
shape=['-DAXIS_OUTER=21','-DAXIS_HEADS=40',f'-DAXIS_ACTIVE_HEADS={a.hc}',f'-DAXIS_CAP={a.cap}']
extra={'merge':['-DAXIS_FAST_MATH=1'],'route':[],'select':[],'layout':[]}[a.stem]
cutlass=Path(torch.__file__).resolve().parent.parent/'flashinfer/data/cutlass/include'
inc=[f'-I{p}' for p in include_paths(device_type='cuda')]+[f'-I{cutlass}',f'-I{Path(sys.prefix)/"include"}']
import sysconfig;inc.append('-I'+sysconfig.get_paths()['include'])
with tempfile.TemporaryDirectory() as tmp:
    cmd=['nvcc','-cubin','-o',str(Path(tmp)/'k.cubin'),str(HERE/f'q80k90_even_{a.stem}.cu'),'-arch=sm_90a','-O3','--expt-relaxed-constexpr',
         '-std=c++17','-DTORCH_EXTENSION_NAME=probe','--ptxas-options=-v','-lineinfo']+shape+extra+a.flags.split()+inc
    r=subprocess.run(cmd,capture_output=True,text=True)
    for line in (r.stdout+r.stderr).splitlines():
        if any(w in line for w in ('ptxas','spill','stack','rror')):print(line.strip())
    sys.exit(r.returncode)
