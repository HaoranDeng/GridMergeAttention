"""Extract unchanged native WGMMA arithmetic into a five-column shared ring.

Ten merged entries plus the original SQ1 key/value occupy each 16-token ring.
No materialized K/V workspace is written by the device producer function.
This is a component of a pending fusion, not a complete attention kernel.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'native_fisher/fisher_c10_long_tiles_tree.cu').read_text()
prefix=s[:s.index('template<int TILES,int TOTAL_HEADS>')]
prefix=prefix.replace('using bf16 =', 'namespace c10_ring {\nusing bf16 =',1)
body=s[s.index('    int row0=warp*16'):s.index('    #pragma unroll\n    for(int kind=0;kind<2;++kind)')]
start=body.index('    if(debug_scores) {')
end=body.index('    float norm_max',start)
body=body[:start]+body[end:]
body=body.replace('cb*6','column_base')
body=body.replace('    if(tid==0)asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");\n','')
body=body.replace('    if(stop_after==2) continue;\n','')
body=body.replace('int out=((qo*C+col)*AH+head)*(KO*R*10)+kb*10+owner;', 'int out=col_for_row(row)*16+owner;')
body=body.replace('row<60 && col<80','row<50 && col<80')
body=body.replace('__syncthreads();','producer_barrier();')
assert not any(x in body for x in ('debug_scores','stop_after','tma_store','cb*6'))
producer=r'''
constexpr int COLUMNS=5,RING=16;
constexpr int SHARED_BYTES=(2*80*128+2*COLUMNS*RING*128)*2+COLUMNS*RING*4;
__device__ __forceinline__ void producer_barrier(){asm volatile("bar.sync 1, 128;":::"memory");}
__device__ __forceinline__ int ring_offset(int row,int d){return row*128+(d^((row&7)*8));}
__device__ __forceinline__ void load_raw(const bf16* k,const bf16* v,bf16* sk,bf16* sv,int kb,int gh){
  for(int vec=threadIdx.x;vec<80*16;vec+=128){
    int ki=vec/16,d=(vec%16)*8;
    int64_t src=((int64_t(kb)*80+ki)*40+gh)*128+d;
    *reinterpret_cast<uint4*>(sk+boff(ki,d))=*reinterpret_cast<const uint4*>(k+src);
    *reinterpret_cast<uint4*>(sv+boff(ki,d))=*reinterpret_cast<const uint4*>(v+src);
  }
  producer_barrier();
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
}
__device__ __forceinline__ void produce(const bf16* q,const uint8_t* seeds,
    const bf16* sk,const bf16* sv,bf16* mk,bf16* mv,float* bias,
    int qo,int column_base,int kb,int gh,float scale){
  constexpr int KO=21,R=45,C=80,H=40;
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,seed=lane>>2;
  int ko=kb/45,kr=kb%45,rf=0;
  bf16* temp=mk;
  uint64_t cache_last;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(cache_last));
'''+body+r'''
  // Owner scratch aliases MK. Reinitialize only the six non-cluster slots
  // after every producer lane has finished using scratch and packed weights.
  for(int vec=tid;vec<COLUMNS*6*16;vec+=128){
    int col=vec/(6*16),row=10+(vec/16)%6,d=(vec%16)*8;
    uint4 x={0,0,0,0},y=x;
    if(row==10){
      x=*reinterpret_cast<const uint4*>(sk+boff(column_base+col,d));
      y=*reinterpret_cast<const uint4*>(sv+boff(column_base+col,d));
    }
    *reinterpret_cast<uint4*>(mk+ring_offset(col*16+row,d))=x;
    *reinterpret_cast<uint4*>(mv+ring_offset(col*16+row,d))=y;
  }
  if(tid<COLUMNS*6){int col=tid/6,row=10+tid%6;bias[col*16+row]=row==10?0.f:CUDART_INF_F;}
  #pragma unroll
  for(int kind=0;kind<2;++kind){
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    float acc[64];
    using Merge=MMA_64x128x16_F32BF16BF16_RS<Major::K,Major::MN>;
    #pragma unroll
    for(int ki=0;ki<80;ki+=16){
      uint64_t bd=desc((kind==0?sk:sv)+boff(ki,0),10240,1024);
      int wi=(ki/8)*2;
      Merge::fma(packed_weights[wi],packed_weights[wi+1],packed_weights[wi+2],packed_weights[wi+3],bd,E64(acc),ki==0?ScaleOut::Zero:ScaleOut::One);
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
    bf16* dst=kind==0?mk:mv;
    #pragma unroll
    for(int i=0;i<64;i+=2){
      int row=warp*16+seed+8*((i&3)>>1),d=2*(lane&3)+8*(i>>2);
      __nv_bfloat162 pair=__floats2bfloat162_rn(__fmul_rn(acc[i],residual_scales[(i&3)>>1]),__fmul_rn(acc[i+1],residual_scales[(i&3)>>1]));
      if(row<50)*reinterpret_cast<__nv_bfloat162*>(dst+ring_offset(col_for_row(row)*16+seed_for_row(row),d))=pair;
    }
  }
}
} // namespace c10_ring
'''
(root/'native_fisher/c10_wgmma_ring.cuh').write_text('#pragma once\n'+prefix+producer)
print(root/'native_fisher/c10_wgmma_ring.cuh')
