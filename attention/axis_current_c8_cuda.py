"""Q80K45 exact C8 (axes='wh': query on the 80-wide W axis, key/merge candidates
on the 45-wide H axis): current-line selection, r2 packed merge, full
compressed routing.

Selection uses a single reference query-outer frame (qo=0, matching
kernels/best/c8_r2's q[:, :1] probe) instead of recomputing per (qo, ko) --
see axis_current_c8_selector.cu. Merge tile count defaults to 105 (matched
c8_r2's own resweep result over {21,42,105,210}).

Instances share cached scratch on a device and must stay on one CUDA stream.
All Q/K/V permutations, selection, merging and full routing are in __call__.
"""
import ctypes
from functools import lru_cache
import hashlib
import os
import socket
from pathlib import Path
import torch
import triton
from analysis.quick_axis_c4_qkv import candidate_table
from kernels.best.c4.route_reference import _hard_c4_720p_route_kernel

extra_source_files=[]


@lru_cache(maxsize=32)
def _compiled(o,r,l,h,ah,implementation='r2'):
    from torch.utils.cpp_extension import load
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a')
    os.environ.setdefault('MAX_JOBS','1')
    os.environ.setdefault('TORCH_EXTENSIONS_DIR','/local2/denghaoran/GridMergeAttention/axis_cuda_extensions_20260918/'+socket.gethostname())
    here=Path(__file__).resolve().parent
    flags=['-O3','--expt-relaxed-constexpr','-lineinfo',f'-DAXIS_OUTER={o}',f'-DAXIS_ROWS={r}',
           f'-DAXIS_COLS={l}',f'-DAXIS_HEADS={h}',f'-DAXIS_ACTIVE_HEADS={ah}',
           f'-DAXIS_C8_R2={int(implementation == "r2")}',
           '-DAXIS_SELECTOR_WARPS='+os.environ.get('AXIS_C8_SELECTOR_WARPS','4')]
    if os.environ.get('AXIS_C8_FAST_MATH','0')=='1':
        flags+=['--use_fast_math','-DAXIS_FAST_MATH=1']
    cutlass=Path(torch.__file__).resolve().parent.parent/'flashinfer/data/cutlass/include'
    modules=[]
    for name in ('selector','merge'):
        suffix=''
        source=here/f'axis_current_c8_{name}{suffix}.cu'
        digest=hashlib.sha256(source.read_bytes()+' '.join(flags).encode()).hexdigest()[:16]
        module=load(name=f'axis_c8_{name}_{digest}',sources=[str(source)],
                    extra_include_paths=[str(cutlass)],extra_cuda_cflags=flags,
                    extra_ldflags=['-lcuda'],verbose=False)
        modules.append(module)
    return tuple(modules)


@lru_cache(maxsize=32)
def _workspace(o,r,l,h,ah,device):
    table,anchors=candidate_table(r)
    table=[row+[0]*(32-len(row)) for row in table]
    candidates=torch.tensor(table,device=device,dtype=torch.int32)
    anchors=torch.tensor(anchors,device=device,dtype=torch.int32)
    shape=(1,o,l,ah,o,r,8)
    seeds=torch.empty(shape,device=device,dtype=torch.uint8)
    mk=torch.empty((*shape,128),device=device,dtype=torch.bfloat16);mv=torch.empty_like(mk)
    bias=torch.empty(shape,device=device,dtype=torch.float32)
    empty=torch.empty(0,device=device,dtype=torch.float32)
    return candidates,anchors,seeds,mk,mv,bias,empty


def merged_attention(q,k,v,*,head_start=0,head_count=None,audit=False,output=None,packed_q=None,implementation='r2'):
    b,o,r,l,h,d=q.shape;ah=head_count or h
    assert b==1 and d==128 and q.dtype==k.dtype==v.dtype==torch.bfloat16
    assert all(x.is_contiguous() for x in (q,k,v))
    selector,merge=_compiled(o,r,l,h,ah,implementation)
    candidates,anchors,seeds,mk,mv,bias,empty=_workspace(o,r,l,h,ah,str(q.device))
    selector.run(q,k,candidates,anchors,seeds,head_start,128**-.5)
    # Selector only computes the qo=0 reference frame (see axis_current_c8_selector.cu);
    # broadcast it across the full qo extent so merge's unchanged addressing sees a
    # complete seeds tensor. Cheap: seeds is uint8, no D_HEAD factor.
    seeds.copy_(seeds[:,:1].expand_as(seeds))
    if packed_q is None:
        packed_q=torch.empty_like(q);merge.pack(q,packed_q)
    merge.run(packed_q,k,v,seeds,mk,mv,bias,head_start,ah,
              int(os.environ.get('AXIS_C8_CUDA_TILES','105')),128**-.5,empty,0,1)
    out=torch.empty_like(q) if output is None else output
    _hard_c4_720p_route_kernel[(1,b*ah,o*l)](
        q,k,v,mk,mv,bias,bias,bias,bias,bias,out,
        Q_OUTER=o,K_OUTER=o,TOTAL_HEADS=h,ACTIVE_HEADS=ah,HEAD_START=head_start,
        D_HEAD=128,HEAD_MAJOR=False,OUTPUT_HEAD_MAJOR=False,ROWS=r,COLS=l,C=8,
        BLOCK_I=triton.next_power_of_2(r),BLOCK_TOKENS=int(os.environ.get('AXIS_C8_ROUTE_TOKENS','64')),
        LOOP_STAGES=int(os.environ.get('AXIS_C8_ROUTE_STAGES','2')),
        INPUT_PRECISION='ieee',EXACT_INTRA_FRAME=False,EXACT_NEIGHBOR_WINDOW_COLS=0,
        RECORD_DISTORTION=False,QUERY_OUTER_OFFSET=0,SINGLETON_TOP_R=0,
        SINGLETON_QUERY_COLUMN=False,SINGLETON_SECONDARY_FRAME_STRIDE=1,
        SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False,QUERY_COLUMN_MAJOR=False,
        PACKED_SINGLETON_KV=False,SCALED_FP8_KV=False,COMPRESSED_FP8=False,
        SCALE=128**-.5,num_warps=int(os.environ.get('AXIS_C8_ROUTE_WARPS','4')),
        num_stages=int(os.environ.get('AXIS_C8_ROUTE_STAGES','2')))
    if os.environ.get('AXIS_C8_KERNEL_METADATA')=='1':
        print({'selector_resources':selector.resources(),'merge_resources':merge.resources()},flush=True)
    return (out,mk,mv,bias,seeds) if audit else out


class AxisCurrentC8:
    def __init__(self,axes='wh',grid=(21,45,80),implementation='r2'):
        assert axes == 'wh'
        assert implementation in ('r2','slow_reference')
        self.implementation=implementation
        self.axes,self.grid=axes,grid
        self.qa,self.ka=['thw'.index(a) for a in axes]
        self.oa=next(a for a in range(3) if a not in (self.qa,self.ka))
        self.chunk=int(os.environ.get('AXIS_C8_HEAD_CHUNK','4'))
        assert self.chunk>0
        self.calls=0

    def __call__(self,q,k,v):
        b,h,n,d=q.shape
        assert b==1 and n==self.grid[0]*self.grid[1]*self.grid[2] and d==128
        order=(0,2+self.oa,2+self.qa,2+self.ka,1,5)
        relaid=[x.reshape(b,h,*self.grid,d).permute(order).contiguous() for x in (q,k,v)]
        out=torch.empty_like(relaid[0]);packed=torch.empty_like(relaid[0])
        o,r,l=[self.grid[a] for a in (self.oa,self.qa,self.ka)]
        _,merge=_compiled(o,r,l,h,min(h,self.chunk),self.implementation);merge.pack(relaid[0],packed)
        for hs in range(0,h,self.chunk):
            merged_attention(*relaid,head_start=hs,head_count=min(self.chunk,h-hs),output=out,packed_q=packed,implementation=self.implementation)
        self.calls+=1
        inverse=tuple(order.index(i) for i in range(6))
        return out.permute(inverse).reshape(b,h,n,d).contiguous()
