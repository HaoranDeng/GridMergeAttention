"""Combine all key partitions; SQ1 is already included in each partial."""
import triton
import triton.language as tl

@triton.jit
def combine_ring_partials(Acc,Max,Den,Out,HEADS:tl.constexpr,HEAD_START:tl.constexpr):
    index=tl.program_id(0);row=index%45;group=index//45
    head=group%HEADS;column=(group//HEADS)%80;frame=group//(HEADS*80)
    part=tl.arange(0,16);dim=tl.arange(0,128)
    base=(group*16+part)*45+row
    maximum=tl.load(Max+base);m=tl.max(maximum,0)
    weight=tl.exp2(maximum-m)
    den=tl.sum(tl.load(Den+base)*weight,0)
    acc=tl.load(Acc+base[:,None]*128+dim[None,:])
    value=tl.sum(acc*weight[:,None],0)/den
    target=(((frame*45+row)*80+column)*40+HEAD_START+head)*128+dim
    tl.store(Out+target,value)

def combine(partial,maximum,denominator,out,heads,head_start=0):
    combine_ring_partials[(21*80*heads*45,)](partial,maximum,denominator,out,heads,head_start,num_warps=4)
