"""Overlap score/owner preparation with merged-KV MMA and FP8 output.

Two warp groups have distinct roles and exchange BF16 weights through two
shared slots. Unlike the old multiwarpgroup experiment, they do not just
compute more independent columns in lockstep. No attention route is fused.
All original normalization, ownership, BF16 rounding and FP8 conversion stay.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'native_fisher'
s = (root / 'fisher_c10_fp8_shared_store.cu').read_text()
kernel_start = s.index('template<int TILES,int TOTAL_HEADS>')
loop_start = s.index('  int total=Q*((C+5)/6),start=blockIdx.z*tile_count;')
weight_start = s.index('    uint32_t packed_weights[20];')
merge_start = s.index('    #pragma unroll\n    for(int kind=0;kind<2;++kind)')
kernel_end = s.index('// Pack each 32-BF16 dimension slice')

helpers = r'''
// The descriptor leaves swizzle base_offset at zero. Align the raw arena
// and every swizzled tile to a complete 1024-byte swizzle period, including
// the FP8 TMA source; 128-byte alignment alone is insufficient here.
constexpr int ROLE_WEIGHT_OFFSET=53248;
constexpr int ROLE_RESIDUAL_OFFSET=86016;
constexpr int ROLE_OUTPUT_OFFSET=87040;
constexpr int ROLE_BARRIER_OFFSET=95232;
constexpr int ROLE_SHARED_BYTES=95360;
static_assert(ROLE_WEIGHT_OFFSET%1024==0 && ROLE_OUTPUT_OFFSET%1024==0);
__device__ __forceinline__ void role_barrier_producer() {
  asm volatile("bar.sync 1,128;":::"memory");
}
__device__ __forceinline__ void role_barrier_consumer() {
  asm volatile("bar.sync 2,128;":::"memory");
}
__device__ __forceinline__ void role_arrive(uint64_t* barrier) {
  uint32_t address=__cvta_generic_to_shared(barrier);
  asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(address):"memory");
}
__device__ __forceinline__ void role_wait(uint64_t* barrier,int phase) {
  uint32_t address=__cvta_generic_to_shared(barrier);
  // Diagnostic fail-fast bound prevents a broken experimental handoff from
  // hanging indefinitely. Normal handoffs finish in far fewer iterations.
  for(int attempt=0;attempt<10000000;++attempt) {
    int ready;
    asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2; selp.b32 %0,1,0,p;}"
                 :"=r"(ready):"r"(address),"r"(phase):"memory");
    if(ready)return;
  }
  asm volatile("trap;");
}
'''

prefix = s[kernel_start:loop_start]
prefix = prefix.replace('__launch_bounds__(128,4)', '__launch_bounds__(256,2)')
prefix = prefix.replace('__shared__ __align__(128)', '__shared__ __align__(1024)')
prefix = prefix.replace('  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,seed=lane>>2;',
'''  int global_tid=threadIdx.x,tid=global_tid&127;
  int lane=tid&31,warp=tid>>5,seed=lane>>2;
  uint64_t* barriers=reinterpret_cast<uint64_t*>(raw+ROLE_BARRIER_OFFSET);
  if(global_tid==0) {
    #pragma unroll
    for(int i=0;i<4;++i) {
      uint32_t address=__cvta_generic_to_shared(barriers+i);
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(address):"memory");
    }
  }''')
prefix = prefix.replace('for(int vec=tid;vec<80*16;vec+=128)', 'for(int vec=global_tid;vec<80*16;vec+=256)')
loop = '''  int total=Q*((C+5)/6),start=blockIdx.z*tile_count;
  for(int tile=start;tile<min(start+tile_count,total);++tile) {
    int ordinal=tile-start,slot=ordinal&1;
    int qo=tile%Q,cb=tile/Q,rf=0;
'''
body_start = s.index('    int row0=warp*16+seed,col0=cb*6+col_for_row(row0);')
producer = s[body_start:weight_start]
producer = producer.replace('    if(tid==0)asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");\n', '')
producer = producer.replace('__syncthreads();', 'role_barrier_producer();')
weights = s[weight_start:merge_start]
weights = weights.replace('    uint32_t packed_weights[20];', '''    if(tid==0 && ordinal>=2)role_wait(barriers+2+slot,((ordinal/2)-1)&1);
    role_barrier_producer();
    bf16* weight_slot=reinterpret_cast<bf16*>(raw+ROLE_WEIGHT_OFFSET)+slot*8192;
    float* residual_slot=reinterpret_cast<float*>(raw+ROLE_RESIDUAL_OFFSET)+slot*64;''')
assert weights.count('        packed_weights[2*i+half]=pair.bits;') == 1
weights = weights.replace('        packed_weights[2*i+half]=pair.bits;',
                          '        *reinterpret_cast<uint32_t*>(weight_slot+aoff(row,ki))=pair.bits;')
weights = weights.replace('    __syncthreads(); // All weights now reside in registers; shared scores may be reused.\n', '')
weights = weights.replace('    if(stop_after==2) continue;\n', '')
weights += '''    #pragma unroll
    for(int half=0;half<2;++half)
      if((lane&3)==0)residual_slot[warp*16+seed+8*half]=residual_scales[half];
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    role_barrier_producer();
    if(tid==0)role_arrive(barriers+slot);
  }
'''
consumer = s[merge_start:kernel_end]
assert consumer.endswith('}\n\n')
# Retain the two original kind epilogues, including tail handling and the
# final TMA wait. Supply weights and residual scales from the selected slot.
consumer = consumer.replace('      using Merge=MMA_64x128x16_F32BF16BF16_RS<Major::K,Major::MN>;',
                            '      using Merge=MMA_64x128x16_F32BF16BF16_SS<Major::K,Major::MN>;')
consumer = consumer.replace('        int wi=(ki/8)*2;\n', '')
old_call = 'Merge::fma(packed_weights[wi],packed_weights[wi+1],packed_weights[wi+2],packed_weights[wi+3],bd,E64(acc),ki==0?ScaleOut::Zero:ScaleOut::One);'
assert consumer.count(old_call) == 1
consumer = consumer.replace(old_call, 'Merge::fma(desc(weight_slot+aoff(0,ki),16,1024),bd,E64(acc),ki==0?ScaleOut::Zero:ScaleOut::One);')
consumer = consumer.replace('__syncthreads();', 'role_barrier_consumer();')
old_tail = '''    }
  }
  if(tma_store==1 && tid==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
}

'''
assert consumer.endswith(old_tail)
consumer = consumer[:-len(old_tail)] + '''    }
    // Both WGMMA groups have completed every read of the weight slot. The
    // output TMA reads a disjoint buffer; its own epilogue wait stays intact.
    role_barrier_consumer();
    if(tid==0)role_arrive(barriers+2+slot);
  }
  if(tid==0)asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
'''
consumer_prefix = '''    if(tid==0)role_wait(barriers+slot,(ordinal/2)&1);
    role_barrier_consumer();
    bf16* weight_slot=reinterpret_cast<bf16*>(raw+ROLE_WEIGHT_OFFSET)+slot*8192;
    float* residual_slot=reinterpret_cast<float*>(raw+ROLE_RESIDUAL_OFFSET)+slot*64;
    float residual_scales[2]={residual_slot[warp*16+seed],residual_slot[warp*16+seed+8]};
'''
kernel = prefix + '  if(global_tid<128) {\n' + loop + producer + weights
kernel += '  } else {\n    temp=reinterpret_cast<bf16*>(raw+ROLE_OUTPUT_OFFSET);\n' + loop + consumer_prefix + consumer + '  }\n}\n\n'
suffix = s[kernel_end:]
assert suffix.count('(80*128*2)*2+(64*40+2*6*40)*4') == 2
suffix = suffix.replace('(80*128*2)*2+(64*40+2*6*40)*4', 'ROLE_SHARED_BYTES')
suffix = suffix.replace('<<<grid,128,bytes,stream>>>', '<<<grid,256,bytes,stream>>>')
suffix = suffix.replace('merge_kernel<21,40>,128,bytes', 'merge_kernel<21,40>,256,bytes')
output = s[:kernel_start] + helpers + kernel + suffix
assert 'packed_weights' not in output
assert output.count('__syncthreads();') == 1
assert output.count('role_arrive(barriers+') == 2
(root / 'fisher_c10_fp8_role_pipeline.cu').write_text(output)
