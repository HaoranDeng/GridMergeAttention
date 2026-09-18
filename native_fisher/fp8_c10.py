"""Experimental C10 E4M3 intermediate storage. Serial single-stream workspace."""
import hashlib,os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from .operator import NativeFisherAttention,extension

ROOT=Path(__file__).parent

def retained_control(device='cuda'):
    return NativeFisherAttention(device=device,capacity=10,tiles=84,
        merge_source=ROOT/'fisher_c10_long_tiles_tree.cu',selector_source=ROOT/'fisher_gs_register_distribute.cu')

def selector_extension():
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a');os.environ.setdefault('MAX_JOBS','2')
    source=ROOT/'fisher_c10_fused_selector.cu';header=ROOT/'warp_mma.cuh'
    digest=hashlib.sha256(source.read_bytes()+header.read_bytes()).hexdigest()
    module=load(name='fisher_c10_selector_'+digest[:16],sources=[str(source)],
        extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=False)
    assert hashlib.sha256(source.read_bytes()+header.read_bytes()).hexdigest()==digest
    module.source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    return module

class FP8C10Attention(NativeFisherAttention):
    def __init__(self,control,mode='scaled',fused_selector=False,source=None,head_chunk=5):
        assert control.capacity==10 and mode in ('scaled','direct')
        assert head_chunk in (1,5)
        self.__dict__.update(control.__dict__)
        self.mode=mode
        self.head_chunk=head_chunk
        self.merge=extension(10,source or ROOT/f'fisher_c10_fp8_{mode}.cu')
        self.mk=torch.empty((21*80*head_chunk,9450,128),device=control.mk.device,dtype=torch.float8_e4m3fn)
        self.mv=torch.empty_like(self.mk)
        self.bias=torch.empty(self.mk.shape[:2],device=self.mk.device,dtype=torch.float32)
        self.ks=torch.empty_like(self.bias);self.vs=torch.empty_like(self.bias)
        self.fused=selector_extension() if fused_selector else None
        self.seed_buffer=torch.empty_like(control.native_selector.seeds)
        self._route=None
        self.flash_route=None
        self.pipeline_enabled=False
        self.unit_scale_route=False
        self.via_float_route=False
        self.integer_route=False

    def enable_pipeline(self):
        """Two FP8 slots; overlap the next merge with the previous route."""
        assert self.mode=='direct' and self.flash_route is None and self.head_chunk==5
        import copy
        self.pipeline_peer=copy.copy(self)
        self.pipeline_peer.mk=torch.empty_like(self.mk)
        self.pipeline_peer.mv=torch.empty_like(self.mv)
        self.pipeline_peer.bias=torch.empty_like(self.bias)
        self.producer_stream=torch.cuda.Stream(device=self.packed.device)
        self.consumer_stream=torch.cuda.Stream(device=self.packed.device)
        self.ready_events=[torch.cuda.Event() for _ in range(8)]
        self.done_events=[torch.cuda.Event() for _ in range(8)]
        self.pipeline_enabled=True

    def select(self,q,k,scale):
        if self.fused is None:return super().select(q,k,scale)
        self.fused.run(q[:,:1],k,self.native_selector.candidates,self.native_selector.anchors,self.seed_buffer,scale,self.empty)
        return self.seed_buffer

    def merge_chunk(self,k,v,seeds,head,scale):
        self.merge.run(self.packed,k,v,seeds,self.mk,self.mv,self.ks,self.vs,self.bias,
                       head,self.head_chunk,self.tiles,scale,self.empty,0,1)

    def route(self,q,k,v,out,head,scale):
        if self.flash_route is not None:
            self.flash_route(self,q,k,v,out,head,scale)
            return
        from attention import hard_c4_wan21_14b_720p_triton as h
        if self.integer_route:
            from .fp8_integer_route import fp8_integer_route
            kernel=fp8_integer_route
        elif self.via_float_route:
            from .fp8_via_float_route import fp8_via_float_route
            kernel=fp8_via_float_route
        elif self.mode=='direct' and not self.unit_scale_route:
            from .fp8_direct_route import fp8_direct_route
            kernel=fp8_direct_route
        else:kernel=h._hard_c4_720p_route_kernel
        self._route=kernel[(1,self.head_chunk,21*80)](
            q,k,v,self.mk,self.mv,self.ks,self.vs,self.bias,self.selected,self.unused,out,
            21,21,40,self.head_chunk,head,128,False,False,45,80,10,
            h.HARD_720_ROUTE_BLOCK_I,self.route_block,self.route_stages,h.HARD_720_INPUT_PRECISION,
            False,0,False,0,SINGLETON_TOP_R=1,SINGLETON_QUERY_COLUMN=True,
            SINGLETON_SECONDARY_FRAME_STRIDE=1,SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False,
            QUERY_COLUMN_MAJOR=True,PACKED_SINGLETON_KV=False,SCALED_FP8_KV=self.mode=='scaled' or self.unit_scale_route,COMPRESSED_FP8=False,
            SCALE=scale,num_warps=self.route_warps,num_stages=h.HARD_720_ROUTE_NUM_STAGES)

    @torch.no_grad()
    def __call__(self,q,k,v,scale=128**-.5,head_count=40):
        if not all(x.shape==(1,21,45,80,40,128) and x.dtype==torch.bfloat16 and x.is_contiguous() and x.device==self.packed.device for x in (q,k,v)):
            raise ValueError('Require native contiguous BF16 QKV')
        if head_count not in (5,40):raise ValueError('Require H5/H40')
        self.seeds=self.select(q,k,scale);self.merge.pack(q,self.packed)
        out=torch.empty_like(q)
        if self.pipeline_enabled:
            current=torch.cuda.current_stream(q.device)
            self.producer_stream.wait_stream(current)
            self.consumer_stream.wait_stream(current)
            for i,head in enumerate(range(0,head_count,5)):
                # The final H40 chunk occupies self, matching serial storage.
                slot=self if (head_count//5-1-i)%2==0 else self.pipeline_peer
                with torch.cuda.stream(self.producer_stream):
                    if i>=2:self.producer_stream.wait_event(self.done_events[i-2])
                    slot.merge_chunk(k,v,self.seeds,head,scale)
                    self.ready_events[i].record()
                with torch.cuda.stream(self.consumer_stream):
                    self.consumer_stream.wait_event(self.ready_events[i])
                    slot.route(q,k,v,out,head,scale)
                    self.done_events[i].record()
            current.wait_stream(self.consumer_stream)
            out.record_stream(self.consumer_stream)
            return out
        for head in range(0,head_count,self.head_chunk):
            self.merge_chunk(k,v,self.seeds,head,scale);self.route(q,k,v,out,head,scale)
        return out
