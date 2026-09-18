"""Generate C10 FP8 storage variants from the retained BF16 native merge.

Both preserve Fisher C10, SQ1, BF16 merge weights and BF16 dot products.
Fixed-scale E4M3 additionally requires representable-range validation.
"""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'native_fisher/fisher_c10_long_tiles_tree.cu'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='22008a819e5a9dac345444a0df2ba0d9d5b7efff8fcd8c8212075f02d4682704'
s=source.read_text().replace('#include <cuda_bf16.h>','#include <cuda_bf16.h>\n#include <cuda_fp8.h>')
s=s.replace('seeds,bf16* __restrict__ mk,bf16* __restrict__ mv,','seeds,uint8_t* __restrict__ mk,uint8_t* __restrict__ mv,\n float* __restrict__ ks,float* __restrict__ vs,')
start=s.index('      if(tma_store==2 || cb==13) {')
end=s.index('\n    }\n  }\n  if(tma_store==1',start)
s=s[:start]+'''
      // Quantize only the final compressed storage boundary. All earlier
      // weights and accumulations retain the BF16 native operator semantics.
      #pragma unroll
      for(int i=0;i<64;++i) acc[i]=__bfloat162float(__float2bfloat16_rn(acc[i]));
      float inv_quant[2]={1.f,1.f};
      if constexpr(SCALED_STORAGE) {
        #pragma unroll
        for(int half=0;half<2;++half) {
          float mx=0.f;
          #pragma unroll
          for(int i=0;i<16;++i)
            mx=fmaxf(mx,fmaxf(fabsf(acc[4*i+2*half]),fabsf(acc[4*i+2*half+1])));
          float quant_scale=fmaxf(max4(mx)*(1.f/448.f),1.e-8f);
          inv_quant[half]=reciprocal(quant_scale);
          int row=warp*16+seed+8*half,col=cb*6+col_for_row(row);
          int64_t oi=((int64_t(qo)*C+col)*AH+head)*(KO*R*10)+kb*10+seed_for_row(row);
          if((lane&3)==0 && row<60 && col<80) (kind==0?ks:vs)[oi]=quant_scale;
        }
      }
      if(cb==13) {
        uint8_t* dst=kind==0?mk:mv;
        #pragma unroll
        for(int i=0;i<64;i+=2) {
          int half=(i&3)>>1,row=warp*16+seed+8*half;
          int d=2*(lane&3)+8*(i>>2),col=cb*6+col_for_row(row);
          int64_t oi=(((int64_t(qo)*C+col)*AH+head)*(KO*R*10)+kb*10+seed_for_row(row))*128+d;
          float2 value=make_float2(acc[i]*inv_quant[half],acc[i+1]*inv_quant[half]);
          uint16_t bits=__nv_cvt_float2_to_fp8x2(value,__NV_SATFINITE,__NV_E4M3);
          if(row<60 && col<80)*reinterpret_cast<uint16_t*>(dst+oi)=bits;
        }
        continue;
      }
      if(tid==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
      __syncthreads();
      uint8_t* fp8temp=reinterpret_cast<uint8_t*>(temp);
      #pragma unroll
      for(int g=0;g<4;++g) {
        uint32_t packed[4];
        #pragma unroll
        for(int j=0;j<4;++j) {
          int half=j&1,ix=g*16+(j>>1)*8+half*2;
          float2 lo=make_float2(acc[ix]*inv_quant[half],acc[ix+1]*inv_quant[half]);
          float2 hi=make_float2(acc[ix+4]*inv_quant[half],acc[ix+5]*inv_quant[half]);
          uint32_t pair0=__nv_cvt_float2_to_fp8x2(lo,__NV_SATFINITE,__NV_E4M3);
          uint32_t pair1=__nv_cvt_float2_to_fp8x2(hi,__NV_SATFINITE,__NV_E4M3);
          uint32_t pair=pair0|(pair1<<16);
          int src=(lane&~3)+2*(lane&1);
          uint32_t a=__shfl_sync(0xffffffff,pair,src),b=__shfl_sync(0xffffffff,pair,src+1);
          int shift=(lane&2)*8;
          packed[j]=((a>>shift)&65535)|((b>>shift)<<16);
        }
        int row=warp*16+(lane&7)+8*((lane>>3)&1);
        int d=g*32+16*(lane>>4),offset=row*128+d;
        uint32_t sp=__cvta_generic_to_shared(fp8temp+(offset^(((offset>>7)&7)<<4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%2,%3,%4};"::"r"(sp),"r"(packed[0]),"r"(packed[1]),"r"(packed[2]),"r"(packed[3]):"memory");
      }
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
      __syncthreads();
      if(tid==0) {
        uint32_t p=__cvta_generic_to_shared(temp);
        const CUtensorMap* map=kind==0?&map_k:&map_v;
        asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%1,%2,%3,%4,%5}], [%6];"::"l"(map),"r"(0),"r"(kb*10),"r"(head),"r"(cb*6),"r"(qo),"r"(p):"memory");
        asm volatile("cp.async.bulk.commit_group;":::"memory");
      }
'''+s[end:]
s=s.replace('torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,int hs','torch::Tensor mk,torch::Tensor mv,torch::Tensor ks,torch::Tensor vs,torch::Tensor bias,int hs')
s=s.replace(' dim3 grid(ko*r,ah,',''' TORCH_CHECK(tma_store==1 && stop_after==0 && debug.numel()==0,"FP8 requires complete TMA merge");
 for(auto x:{k,v})TORCH_CHECK(x.device()==q.device() && x.sizes()==q.sizes() && x.scalar_type()==at::kBFloat16,"Matching BF16 QKV required");
 TORCH_CHECK(seeds.device()==q.device() && seeds.sizes()==at::IntArrayRef({1,1,ko,r,c,h,10}),"Matching C10 seeds required");
 TORCH_CHECK(hs>=0 && hs+ah<=h,"Invalid head range");
 for(auto x:{mk,mv})TORCH_CHECK(x.device()==q.device() && x.is_contiguous() && x.scalar_type()==at::ScalarType::Float8_e4m3fn && x.sizes()==at::IntArrayRef({qo*c*ah,ko*r*10,128}),"Matching E4M3 workspace required");
 for(auto x:{ks,vs,bias})TORCH_CHECK(x.device()==q.device() && x.is_contiguous() && x.scalar_type()==at::kFloat && x.sizes()==mk.sizes().slice(0,2),"Matching FP32 metadata required");
 dim3 grid(ko*r,ah,''')
s=s.replace('(80*128*2)*2+64*128*2','(80*128*2)*2+(64*40+2*6*40)*4')
s=s.replace('dims[5]={64,uint64_t(ko*r*20)','dims[5]={128,uint64_t(ko*r*10)')
s=s.replace('box[5]={64,20,1,6,1}','box[5]={128,10,1,6,1}')
s=s.replace('CU_TENSOR_MAP_DATA_TYPE_BFLOAT16','CU_TENSOR_MAP_DATA_TYPE_UINT8')
s=s.replace('(bf16*)mk.data_ptr(),(bf16*)mv.data_ptr(),bias.data_ptr<float>()','(uint8_t*)mk.data_ptr(),(uint8_t*)mv.data_ptr(),ks.data_ptr<float>(),vs.data_ptr<float>(),bias.data_ptr<float>()')
for scaled,name in [(True,'scaled'),(False,'direct')]:
    path=ROOT/f'native_fisher/fisher_c10_fp8_{name}.cu'
    path.write_text('// Experimental FP8 intermediate KV, no production promotion.\nconstexpr bool SCALED_STORAGE='+str(scaled).lower()+';\n'+s)
    print(path.name,hashlib.sha256(path.read_bytes()).hexdigest())

selector=ROOT/'native_fisher/fisher_c10_fused_selector_base.cu'
z=selector.read_text().replace('fisher_from_registers<4,4>','fisher_from_registers<10,4>').replace('seeds.numel()==21*45*80*40*4','seeds.numel()==21*45*80*40*10').replace('Native C4 seeds required','Native C10 seeds required')
# Nine pivots share one loop body, avoiding excessive unrolled code size.
z=z.replace('  #pragma unroll\n  for(int step=1;step<SEEDS;++step)', '  #pragma unroll 1\n  for(int step=1;step<SEEDS;++step)')
(ROOT/'native_fisher/fisher_c10_fused_selector.cu').write_text(z)
print('fisher_c10_fused_selector.cu')
