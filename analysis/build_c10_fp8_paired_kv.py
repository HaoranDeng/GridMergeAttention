"""Pair existing E4M3 K/V bytes without combining their matrix accumulators.

One int16 storage word contains K in its low byte and V in its high byte.
Both BF16 MMAs and FP32-to-E4M3 rounding remain unchanged and sequential.
The16KiB shared output fits four CTAs and emits one TMA store per main tile.
The final two columns use vector global stores after both bytes are ready.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fisher_c10_fp8_local_scores.cu').read_text()
a=s.index('      // Quantize only the final compressed storage boundary.')
b=s.index('  if(tma_store==1 && tid==0)',a)
s=s[:a]+r'''
      uint8_t* fp8temp=reinterpret_cast<uint8_t*>(temp);
      #pragma unroll
      for(int i=0;i<64;i+=2) {
        int row=warp*16+seed+8*((i&3)>>1);
        int d=2*(lane&3)+8*(i>>2),offset=row*256+2*d;
        uint16_t bits=__nv_cvt_float2_to_fp8x2(make_float2(acc[i],acc[i+1]),__NV_SATFINITE,__NV_E4M3);
        uint32_t word=uint32_t(bits&255)|(uint32_t(bits>>8)<<16);
        auto* dst=reinterpret_cast<uint32_t*>(fp8temp+(offset^(((offset>>7)&7)<<4)));
        if(kind==0)*dst=word;
        else *dst=*dst|(word<<8);
      }
      // Each thread reads only the words it wrote in the K iteration.
      // No TMA reader uses this tile until both complete byte planes exist.
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncthreads();
    if(cb==13) {
      for(int vec=tid;vec<20*16;vec+=128) {
        int row=vec/16,d=(vec%16)*8,col=cb*6+col_for_row(row);
        int offset=row*256+2*d;
        uint4 word=*reinterpret_cast<const uint4*>(reinterpret_cast<uint8_t*>(temp)+(offset^(((offset>>7)&7)<<4)));
        int64_t oi=(((int64_t(qo)*C+col)*AH+head)*(KO*R*10)+kb*10+seed_for_row(row))*128+d;
        *reinterpret_cast<uint4*>(reinterpret_cast<uint16_t*>(mk)+oi)=word;
      }
      // The next score scratch writes may overlap other threads' tail reads.
      __syncthreads();
    } else if(tid==0) {
      uint32_t p=__cvta_generic_to_shared(temp);
      asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%1,%2,%3,%4,%5}], [%6];"::"l"(&map_k),"r"(0),"r"(kb*20),"r"(head),"r"(cb*6),"r"(qo),"r"(p):"memory");
      asm volatile("cp.async.bulk.commit_group;":::"memory");
    }
  }
''' + s[b:]
s=s.replace('(64*40+6*40)*4','64*128*2')
s=s.replace('x.scalar_type()==at::ScalarType::Float8_e4m3fn','x.scalar_type()==at::kShort')
s=s.replace('"Matching E4M3 workspace required"','"Matching paired E4M3 byte container required"')
s=s.replace(' dim3 grid(ko*r,ah,',' TORCH_CHECK(mk.data_ptr()==mv.data_ptr(),"Paired storage arguments must alias");\n dim3 grid(ko*r,ah,')
s=s.replace('dims[5]={128,uint64_t(ko*r*10)','dims[5]={64,uint64_t(ko*r*10*2)')
s=s.replace('box[5]={128,10,1,6,1}','box[5]={64,20,1,6,1}')
s=s.replace('CU_TENSOR_MAP_DATA_TYPE_UINT8','CU_TENSOR_MAP_DATA_TYPE_UINT16')
s=s.replace('encode(map_k,mk);encode(map_v,mv);','encode(map_k,mk);map_v=map_k;')
(root/'fisher_c10_fp8_paired_kv.cu').write_text('// Paired E4M3 storage; int16 is only a two-byte container.\n'+s)

r=(root/'fp8_via_float_route.py').read_text()
r=r.replace('def fp8_via_float_route(', 'def fp8_paired_route(')
helper='''@triton.jit
def decode_pair(word):
    bits=word.to(tl.uint16,bitcast=True)
    k=(bits & 255).to(tl.uint8).to(tl.float8e4nv,bitcast=True)
    v=(bits >> 8).to(tl.uint8).to(tl.float8e4nv,bitcast=True)
    return _via_float(k),_via_float(v)


'''
r=r.replace('@triton.jit\ndef fp8_paired_route(',helper+'@triton.jit\ndef fp8_paired_route(')
a=r.index('        merged_key = tl.load(\n            merged_k_ptr')
b=r.index('        if SCALED_FP8_KV:',a)
r=r[:a]+'''        pair = tl.load(
            merged_k_ptr + merged_base * D_HEAD + dims[None, :],
            mask=token_mask[:, None],other=0,cache_modifier=".cg",
        )
        merged_key,merged_value=decode_pair(pair)
''' +r[b:]
r=r.replace('    if SINGLETON_TOP_R > 0:\n        tl.static_assert', '    tl.static_assert(not SCALED_FP8_KV and not COMPRESSED_FP8)\n    if SINGLETON_TOP_R > 0:\n        tl.static_assert',1)
(root/'fp8_paired_route.py').write_text(r)
