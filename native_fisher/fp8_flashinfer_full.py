"""Combine CUDA FP8 compressed attention and unchanged BF16 SQ1 support."""
import torch
import triton
import triton.language as tl
from .fp8_flashinfer_route import CompressedRoute

@triton.jit
def _combine(a,la,b,lb,out,HS:tl.constexpr):
    t=tl.program_id(0);d=tl.arange(0,128)
    x=tl.load(la+t);y=tl.load(lb+t);m=tl.maximum(x,y)
    wa=tl.exp2(x-m);wb=tl.exp2(y-m)
    value=(tl.load(a+t*128+d).to(tl.float32)*wa+tl.load(b+t*128+d)*wb)/(wa+wb)
    row=t%45;g=t//45;head=g%5;col=(g//5)%80;frame=g//400
    dst=(((frame*45+row)*80+col)*40+HS+head)*128+d
    tl.store(out+dst,value)

class FlashinferFullRoute:
    def __init__(self,device='cuda'):
        self.compressed=CompressedRoute(device)
        self.singleton_out=torch.empty((8400*45,1,128),dtype=torch.float32,device=device)
        self.singleton_lse=torch.empty((8400*45,1),dtype=torch.float32,device=device)

    def state(self,op,q,k,v,out,lse,head,scale,compressed=False):
        from attention import hard_c4_wan21_14b_720p_triton as h
        from .fp8_route_states import singleton_state,compressed_state
        kernel=compressed_state if compressed else singleton_state
        return kernel[(1,5,21*80)](
            q,k,v,op.mk,op.mv,op.ks,op.vs,op.bias,op.selected,op.unused,out,lse,
            21,21,40,5,head,128,False,False,45,80,10,
            h.HARD_720_ROUTE_BLOCK_I,64,3,h.HARD_720_INPUT_PRECISION,
            False,0,False,0,SINGLETON_TOP_R=1,SINGLETON_QUERY_COLUMN=True,
            SINGLETON_SECONDARY_FRAME_STRIDE=1,SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False,
            QUERY_COLUMN_MAJOR=True,PACKED_SINGLETON_KV=False,SCALED_FP8_KV=False,COMPRESSED_FP8=False,
            SCALE=scale,num_warps=4,num_stages=h.HARD_720_ROUTE_NUM_STAGES)

    def __call__(self,op,q,k,v,out,head,scale):
        packed_q=q[0,:,:,:,head:head+5,:].permute(0,2,3,1,4).reshape(-1,1,128)
        a,la=self.compressed(packed_q,op.mk,op.mv,op.bias,scale)
        op._route=self.state(op,q,k,v,self.singleton_out,self.singleton_lse,head,scale)
        _combine[(8400*45,)](a,la,self.singleton_out,self.singleton_lse,out,head,num_warps=4)
