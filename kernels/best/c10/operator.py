"""Hopper C4/C10 merge prototypes with unchanged Fisher selection and SQ1 route.

Experimental only: observed small BF16 output differences and targets unmet.
Each instance owns reusable workspace and must not be shared across streams.
"""
import os
import hashlib
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

def extension(capacity=4,source_path=None,name_suffix=''):
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a')
    os.environ.setdefault('MAX_JOBS','2')
    includes=Path(torch.__file__).resolve().parent.parent/'flashinfer/data/cutlass/include'
    source=Path(source_path) if source_path is not None else Path(__file__).with_name(f'fisher_c{capacity}_merge.cu')
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    # Content-address the extension: interrupted experiments must not make a
    # changed source wait on, or accidentally import, another variant's build.
    module=load(name=f'fisher_c{capacity}_merge_{before[:16]}{name_suffix}',sources=[str(source)],
        extra_include_paths=[str(includes)],extra_ldflags=['-lcuda'],
        extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','-lineinfo','--ptxas-options=-v'],verbose=False)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=before:
        raise RuntimeError('CUDA source changed during extension build; discard this run')
    module.source_sha256=before
    return module

class NativeFisherAttention:
    def __init__(self,device='cuda',capacity=4,tiles=21,route_block=64,route_warps=4,route_stages=3,selector_block=2,selector_backend='probe4',merge_source=None,workspace_columns=1,selector_source=None):
        assert capacity in (4,10)
        if workspace_columns != 1 and not (capacity == 4 and workspace_columns == 16 and merge_source is not None):
            raise ValueError('Experimental workspace requires C4, 16 columns, and explicit matching merge source')
        self.workspace_columns=workspace_columns
        self.capacity=capacity
        self.tiles=tiles
        self.route_block=route_block
        self.route_warps=route_warps
        self.route_stages=route_stages
        self.selector_block=selector_block
        self.selector_backend=selector_backend
        self.native_selector=None
        if selector_backend!='triton':
            from .selector import NativeFisherSelector
            self.native_selector=NativeFisherSelector(capacity,device=device,layout=selector_backend,source_path=selector_source)
        self.merge=extension(capacity,source_path=merge_source)
        self.packed=torch.empty((1,21,45,80,40,128),device=device,dtype=torch.bfloat16)
        self.mk=torch.empty((21*80*5,945*capacity,128),device=device,dtype=torch.bfloat16)
        self.mv=torch.empty_like(self.mk)
        self.bias=torch.empty(self.mk.shape[:-1],device=device,dtype=torch.float32)
        self.unused=torch.empty(1,device=device)
        self.selected=torch.empty(1,device=device,dtype=torch.uint8)
        self.empty=torch.empty(0,device=device)

    def select(self,q,k,scale):
        if self.native_selector is not None:
            return self.native_selector(q[:,:1],k,scale)
        from attention.affine_hellinger_seed_triton import affine_hellinger_seed_rows_triton
        return affine_hellinger_seed_rows_triton(q[:,:1],k,scale=scale,feature_mode='fisher_geometric',num_seeds=self.capacity,block_j=self.selector_block)

    def route(self,q,k,v,out,head,scale):
        from . import route_reference as h
        route=h._hard_c4_720p_route_kernel
        layout_options={}
        if self.workspace_columns != 1:
            from .column_workspace_route import column_workspace_route
            route=column_workspace_route
            layout_options['WORKSPACE_COLUMNS']=self.workspace_columns
        route[(1,5,21*80)](
            q,k,v,self.mk,self.mv,self.unused,self.unused,self.bias,self.selected,self.unused,out,
            21,21,40,5,head,128,False,False,45,80,self.capacity,
            h.HARD_720_ROUTE_BLOCK_I,self.route_block,self.route_stages,h.HARD_720_INPUT_PRECISION,
            False,0,False,0,SINGLETON_TOP_R=1,SINGLETON_QUERY_COLUMN=True,
            SINGLETON_SECONDARY_FRAME_STRIDE=1,SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False,
            QUERY_COLUMN_MAJOR=True,PACKED_SINGLETON_KV=False,SCALED_FP8_KV=False,COMPRESSED_FP8=False,
            SCALE=scale,num_warps=self.route_warps,num_stages=h.HARD_720_ROUTE_NUM_STAGES,**layout_options)

    @torch.no_grad()
    def __call__(self,q,k,v,scale=128**-.5,head_count=40):
        if not all(x.shape==(1,21,45,80,40,128) and x.dtype==torch.bfloat16 and x.is_contiguous() and x.device==self.packed.device for x in (q,k,v)):
            raise ValueError('Require contiguous BF16 native40 Q/K/V on the workspace device')
        if head_count not in (5,40):
            raise ValueError('Only diagnostic H5 or complete H40 calls are supported')
        seeds=self.select(q,k,scale);self.seeds=seeds
        self.merge.pack(q,self.packed)
        out=torch.empty_like(q)
        for head in range(0,head_count,5):
            self.merge.run(self.packed,k,v,seeds,self.mk,self.mv,self.bias,head,5,self.tiles,scale,self.empty,0,1)
            self.route(q,k,v,out,head,scale)
        return out
