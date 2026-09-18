"""Avoid a resident16KiB query tile; reuse the existing packed BF16 Q.

Query fragments are reloaded through L1 for each compressed/singleton tile
and become dead after QK. They do not remain live across PV. No extra global
packing or BF16 KV materialization is added to a complete attention call.
"""
from pathlib import Path
root = Path(__file__).resolve().parents[1] / 'native_fisher'
s = (root / 'fisher_c10_fp8_native_route.cu').read_text()
s = s.replace('__global__ __launch_bounds__(128,4) void route_kernel(', r'''
__device__ __forceinline__ uint4 load_packed_query(const bf16* p) {
 uint4 x;
 asm volatile("ld.global.ca.v4.b32 {%0,%1,%2,%3},[%4];"
   :"=r"(x.x),"=r"(x.y),"=r"(x.z),"=r"(x.w):"l"(p):"memory");
 return x;
}
__global__ __launch_bounds__(128,3) void route_kernel(''')
s = s.replace('bf16* route_q=reinterpret_cast<bf16*>(raw);bf16* sk=route_q+64*128;bf16* sv=sk+64*128;',
              'bf16* sk=reinterpret_cast<bf16*>(raw);bf16* sv=sk+64*128;')
start = s.index(' for(int vec=tid;vec<64*16;vec+=128)')
end = s.index(' __syncthreads();', start)
s = s[:start] + s[end:]
fragment = r'''
  uint32_t qreg[32];
  int row0=warp*16+seed,row1=row0+8;
  #pragma unroll
  for(int d=0;d<128;d+=32) {
   uint4 a{},b{};
   if(row0<45) {
    int64_t address=(((int64_t(qo)*45+row0)*80+col)*40+gh)*128+(lane&3)*8+d;
    a=load_packed_query(q+address);
   }
   if(row1<45) {
    int64_t address=(((int64_t(qo)*45+row1)*80+col)*40+gh)*128+(lane&3)*8+d;
    b=load_packed_query(q+address);
   }
   int i=d/4;
   qreg[i]=a.x;qreg[i+1]=b.x;qreg[i+2]=a.y;qreg[i+3]=b.y;
   qreg[i+4]=a.z;qreg[i+5]=b.z;qreg[i+6]=a.w;qreg[i+7]=b.w;
  }
  #pragma unroll
  for(int i=0;i<32;++i)asm volatile("" : "+r"(qreg[i]) :: "memory");
'''
s = s.replace('  asm volatile("wgmma.fence.sync.aligned;":::"memory");\n  float logits[32];',
              fragment + '  asm volatile("wgmma.fence.sync.aligned;":::"memory");\n  float logits[32];', 1)
s = s.replace('MMA_64x64x16_F32BF16BF16_SS', 'MMA_64x64x16_F32BF16BF16_RS')
s = s.replace('QK::fma(desc(route_q+aoff(0,d),16,1024),desc(sk+aoff(0,d),16,1024),',
              'QK::fma(qreg[d/4],qreg[d/4+1],qreg[d/4+2],qreg[d/4+3],desc(sk+aoff(0,d),16,1024),')
s = s.replace('3*64*128*2+8192+64*4+16', '2*64*128*2+8192+64*4+16')
s = s.replace('m.def("run",&run);', 'm.def("run_packed",&run);')
assert 'route_q' not in s
(root / 'fisher_c10_fp8_native_route_rs.cu').write_text(s)
