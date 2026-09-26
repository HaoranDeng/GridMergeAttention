"""Build Q80K90 even-capacity kernel variants in parallel processes (CPU only).

python q80k90_even_precompile.py --stage merge --cap 8 --variants "" "-DMERGE_SUM_WAYS=2" ...
Each variant is compiled by native_<stage> into TORCH_EXTENSIONS_DIR exactly as
the benchmarks would build it, so a later A/B run only loads the modules.
"""
import argparse,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['merge','route','select'],required=True)
ap.add_argument('--cap',type=int,nargs='+',required=True);ap.add_argument('--hc',type=int,default=8)
ap.add_argument('--variants',nargs='+',default=[''])
a=ap.parse_args()
code=("import sys,ctypes;sys.path.insert(0,{root!r});ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL);"
      "from analysis.q80k90_even_kernel import native_merge,native_route,native_select;")
calls={'merge':'native_merge(21,40,{hc},{cap},1,3,{flags})','route':'native_route(21,40,{hc},{cap},{flags})','select':'native_select(21,40,{cap},{flags})'}
procs=[]
for cap in a.cap:
    for v in a.variants:
        stmt=code.format(root=str(HERE.parent))+calls[a.stage].format(hc=a.hc,cap=cap,flags=repr(tuple(v.split())))+";print('built',{cap},{v!r})".format(cap=cap,v=v)
        procs.append((cap,v,subprocess.Popen([sys.executable,'-c',stmt],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))))
bad=0
for cap,v,p in procs:
    rc=p.wait();bad+=rc!=0
    print(f'[precompile] {a.stage} cap={cap} flags="{v}" exit={rc}',flush=True)
sys.exit(1 if bad else 0)
