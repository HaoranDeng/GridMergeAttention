"""Experimental merge/route alternation over one head and a column/frame slice."""
from pathlib import Path
import torch
from .operator import NativeFisherAttention


class CacheSlicedFisherAttention(NativeFisherAttention):
    def __init__(self, capacity=4, frames=7, device='cuda'):
        if frames not in (1, 3, 7, 21):
            raise ValueError('Frame slice must divide 21')
        super().__init__(capacity=capacity, device=device,
                         merge_source=Path(__file__).with_name(f'fisher_c{capacity}_cache_slice.cu'))
        self.frames=frames
        self.columns=16 if capacity==4 else 6

    def slice_route(self, q, k, v, out, head_start, head_offset, frame_start, column_start, scale):
        from attention import hard_c4_wan21_14b_720p_triton as h
        from .cache_slice_route import cache_slice_route
        columns=min(self.columns, 80-column_start)
        cache_slice_route[(1,1,columns*self.frames)](
            q,k,v,self.mk,self.mv,self.unused,self.unused,self.bias,self.selected,self.unused,out,
            21,21,40,5,head_start,128,False,False,45,80,self.capacity,
            h.HARD_720_ROUTE_BLOCK_I,self.route_block,self.route_stages,h.HARD_720_INPUT_PRECISION,
            False,0,False,0,SINGLETON_TOP_R=1,SINGLETON_QUERY_COLUMN=True,
            SINGLETON_SECONDARY_FRAME_STRIDE=1,SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False,
            QUERY_COLUMN_MAJOR=True,PACKED_SINGLETON_KV=False,SCALED_FP8_KV=False,COMPRESSED_FP8=False,
            SCALE=scale,HEAD_OFFSET=head_offset,FRAME_START=frame_start,COLUMN_START=column_start,
            FRAME_COUNT=self.frames,num_warps=self.route_warps,num_stages=h.HARD_720_ROUTE_NUM_STAGES)

    @torch.no_grad()
    def __call__(self,q,k,v,scale=128**-.5,head_count=40):
        if not all(x.shape==(1,21,45,80,40,128) and x.dtype==torch.bfloat16
                   and x.is_contiguous() and x.device==self.packed.device for x in (q,k,v)):
            raise ValueError('Require contiguous native40 BF16 QKV on workspace device')
        if head_count not in (5,40):
            raise ValueError('Only H5 diagnostic or complete H40 supported')
        if self.frames not in (1,3,7,21):
            raise ValueError('Frame slice must divide 21')
        seeds=self.select(q,k,scale)
        self.seeds=seeds
        self.merge.pack(q,self.packed)
        out=torch.empty_like(q)
        for head in range(0,head_count,5):
            for head_offset in range(5):
                for column_start in range(0,80,self.columns):
                    tile_column=column_start//self.columns
                    for frame_start in range(0,21,self.frames):
                        self.merge.run_slice(self.packed,k,v,seeds,self.mk,self.mv,self.bias,
                            head,5,self.frames,scale,self.empty,0,1,
                            tile_column*21+frame_start,head_offset,1,self.frames)
                        self.slice_route(q,k,v,out,head,head_offset,frame_start,column_start,scale)
        return out
