"""Hopper exact-capacity C4 production operator.

Experimental only: observed small BF16 output differences and targets unmet.
Each instance owns reusable workspace and must not be shared across streams.

C4 is exactly four merged K/V representatives. It has no SQ1, singleton,
raw-K/V, or residual route.
"""
import os
import hashlib
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from attention.capacity_contract import require_exact_merged_capacity

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
    def __init__(self,device='cuda',capacity=4,tiles=21,route_block=64,route_warps=4,route_stages=3,selector_block=2,selector_backend='probe4',merge_source=None,workspace_columns=1,selector_source=None,route_backend='triton',head_chunk=5):
        if capacity != 4:
            raise ValueError("This package is exact C4 only")
        require_exact_merged_capacity(capacity, 0)
        if head_chunk not in (5,10) or (head_chunk!=5 and (capacity!=4 or workspace_columns!=1 or route_backend!='triton' or merge_source is None)):
            raise ValueError('Experimental H10 chunks require C4, standard workspace, Triton route and explicit matching merge source')
        self.head_chunk=head_chunk
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
        if selector_source is not None and not selector_backend.startswith('fused'):
            raise ValueError('Explicit selector source requires a fused CUDA selector backend')
        self.native_selector=None
        if selector_backend in ('fused','fused_loop','fused_cooperative_loop','fused_cooperative_basis_loop'):
            from .fused_selector import NativeFusedFisherSelector
            self.native_selector=NativeFusedFisherSelector(capacity,device=device,loop_frames=selector_backend.endswith('loop'),cooperative='cooperative' in selector_backend,basis='basis' in selector_backend,source_path=selector_source)
        elif selector_backend!='triton':
            from .selector import NativeFisherSelector
            self.native_selector=NativeFisherSelector(capacity,device=device,layout=selector_backend)
        self.merge=extension(capacity,source_path=merge_source)
        if route_backend not in ('triton','cuda') or (route_backend=='cuda' and (capacity!=4 or workspace_columns!=1)):
            raise ValueError('Native route requires C4 with the standard workspace')
        self.route_backend=route_backend
        self.native_route=extension(4,Path(__file__).with_name('fisher_c4_route.cu'),'_route') if route_backend=='cuda' else None
        self.packed=torch.empty((1,21,45,80,40,128),device=device,dtype=torch.bfloat16)
        self.mk=torch.empty((21*80*head_chunk,945*capacity,128),device=device,dtype=torch.bfloat16)
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

    def route(self,q,k,v,out,head,scale,active_heads=5):
        if self.native_route is not None:
            self.native_route.run(q,k,v,self.mk,self.mv,self.bias,out,head,active_heads,scale)
            return
        from . import route_reference as h
        route=h._hard_c4_720p_route_kernel
        layout_options={}
        if self.workspace_columns != 1:
            from .column_workspace_route import column_workspace_route
            route=column_workspace_route
            layout_options['WORKSPACE_COLUMNS']=self.workspace_columns
        route[(1,active_heads,21*80)](
            q,k,v,self.mk,self.mv,self.unused,self.unused,self.bias,self.selected,self.unused,out,
            21,21,40,active_heads,head,128,False,False,45,80,self.capacity,
            h.HARD_720_ROUTE_BLOCK_I,self.route_block,self.route_stages,h.HARD_720_INPUT_PRECISION,
            False,0,False,0,SINGLETON_TOP_R=0,SINGLETON_QUERY_COLUMN=False,
            SINGLETON_SECONDARY_FRAME_STRIDE=1,SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False,
            QUERY_COLUMN_MAJOR=False,PACKED_SINGLETON_KV=False,SCALED_FP8_KV=False,COMPRESSED_FP8=False,
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
        for head in range(0,head_count,self.head_chunk):
            active_heads=min(self.head_chunk,head_count-head)
            self.merge.run(self.packed,k,v,seeds,self.mk,self.mv,self.bias,head,active_heads,self.tiles,scale,self.empty,0,1)
            self.route(q,k,v,out,head,scale,active_heads=active_heads)
        return out
