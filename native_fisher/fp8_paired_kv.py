"""Experimental paired-byte workspace; original BF16 QKV/weights are intact."""
import torch
from .fp8_c10 import FP8C10Attention

class PairedFP8C10Attention(FP8C10Attention):
    def __init__(self,control,source):
        super().__init__(control,'direct',source=source)
        self.pair=torch.empty(self.mk.shape,device=self.mk.device,dtype=torch.int16)
        self.mk=self.pair;self.mv=self.pair

    def route(self,q,k,v,out,head,scale):
        from attention import hard_c4_wan21_14b_720p_triton as h
        from .fp8_paired_route import fp8_paired_route
        self._route=fp8_paired_route[(1,5,21*80)](
            q,k,v,self.pair,self.pair,self.ks,self.vs,self.bias,self.selected,self.unused,out,
            21,21,40,5,head,128,False,False,45,80,10,
            h.HARD_720_ROUTE_BLOCK_I,self.route_block,self.route_stages,h.HARD_720_INPUT_PRECISION,
            False,0,False,0,SINGLETON_TOP_R=1,SINGLETON_QUERY_COLUMN=True,
            SINGLETON_SECONDARY_FRAME_STRIDE=1,SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False,
            QUERY_COLUMN_MAJOR=True,PACKED_SINGLETON_KV=False,SCALED_FP8_KV=False,COMPRESSED_FP8=False,
            SCALE=scale,num_warps=self.route_warps,num_stages=h.HARD_720_ROUTE_NUM_STAGES)
