"""Run generate.py with the Q90K80 even-capacity native operator as Wan self-attention.

python analysis/q90k80_even_generate.py --gma-cap 8 [--gma-hc 8] [--gma-route-flags "..."] <generate.py args>
--gma-cap 0 runs generate.py unchanged (dense SDPA). Otherwise every
self-attention call of shape [1, 75600, 40, 128] inside the dense config's
"sdpa" branch is replaced by Q90K80Even(cap); cross-attention stays dense.
generate.py's --runtime_json then records the replaced call as the core
attention time and the whole self-attention module as the full time.
"""
import os,sys,ctypes,runpy
from pathlib import Path
REPO=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO))

def pop(name,default,cast=str):
    if name in sys.argv:
        i=sys.argv.index(name);value=cast(sys.argv[i+1]);del sys.argv[i:i+2];return value
    for i,a in enumerate(sys.argv):
        if a.startswith(name+'='):
            del sys.argv[i];return cast(a.split('=',1)[1])
    return default
cap=pop('--gma-cap',8,int);hc=pop('--gma-hc',8,int);rflags=tuple(pop('--gma-route-flags','').split())
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a');os.environ.setdefault('MAX_JOBS','8')
os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(REPO/('.torch_ext_'+os.uname().nodename)))
ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
import torch
import wan.modules.model as wm
if cap:
    from analysis.q90k80_even_kernel import Q90K80Even
    core=Q90K80Even(cap,merge_backend='cuda',head_chunk=hc,route_flags=rflags)
    dense=wm.flash_attention
    calls={'gma':0,'dense':0}
    def attention(q,k,v,*args,**kw):
        if tuple(q.shape)==(1,75600,40,128) and tuple(k.shape)==tuple(q.shape) and not args and set(kw)<={'window_size'} \
           and kw.get('window_size',(-1,-1))==(-1,-1):
            calls['gma']+=1
            return core(q,k,v)
        calls['dense']+=1
        return dense(q,k,v,*args,**kw)
    wm.flash_attention=attention
    import atexit
    atexit.register(lambda:print(f'[q90k80_even_generate] cap={cap} hc={hc} route_flags={list(rflags)} calls={calls}',flush=True))
sys.argv[0]=str(REPO/'generate.py')
runpy.run_path(str(REPO/'generate.py'),run_name='__main__')
