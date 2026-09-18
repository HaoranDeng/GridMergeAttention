"""C10 warp-local ownership with 16 MMA rows but only ten stored tokens.

The six unused owners have masked scores and zero weights. They never add
route tokens. Extra MMA work trades against eliminating cross-warp shared
ownership exchanges. FP8 output is compacted in shared memory before TMA.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'native_fisher/fisher_c10_fp8_pad16_base.cu').read_text()
control=(root/'native_fisher/fisher_c10_long_tiles_tree.cu').read_text()
s=s.replace('return (row>>3)*2+(row&1);','return row>>4;').replace('return (row&7)>>1;','return row&15;')
anchor='__device__ __forceinline__ int storeoff'
s=s.replace(anchor,'''__device__ __forceinline__ int compact_for_row(int row) {
 int owner=seed_for_row(row),col=col_for_row(row);
 return owner<10?col*10+owner:40+col*6+owner-10;
}
'''+anchor)
s=s.replace('cb*16','cb*4').replace('(C/16)','(C/4)')
s=s.replace('*H+gh)*4+seed_for_row(row0);','*H+gh)*10+min(seed_for_row(row0),9);')
s=s.replace('*H+gh)*4+seed_for_row(row0+8);','*H+gh)*10+min(seed_for_row(row0+8),9);')
s=s.replace('(KO*R*4)','(KO*R*10)').replace('+kb*4+','+kb*10+')
s=s.replace('col_for_row(row)*4+seed_for_row(row)','compact_for_row(row)')
s=s.replace('if((lane&3)==0)bias[out]', 'if((lane&3)==0 && owner<10)bias[out]')
s=s.replace('if((lane&3)==0) (kind==0?ks:vs)[oi]', 'if((lane&3)==0 && seed_for_row(row)<10) (kind==0?ks:vs)[oi]')
# Use the retained C10 normalization and log instruction to isolate layout
# changes from FP32 arithmetic changes in this first candidate.
a=s.index('    float norm_max[2],norm_sum[2];')
b=s.index('    // Each four-lane row group',a)
ca=control.index('    float norm_max[2],norm_sum[2];')
cb=control.index('    // Each four-lane row group',ca)
s=s[:a]+control[ca:cb]+s[b:]
s=s.replace('  return log2f(x);', '  float y; asm("lg2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;')
a=s.index('        float best=s[ix];')
b=s.index('        winners[i]=best;',a)
s=s[:a]+'''        int ix0=(i/2)*4+(i&1);
        float low=seed<10?s[ix0]:-CUDART_INF_F;
        float high=seed+8<10?s[ix0+2]:-CUDART_INF_F;
        float best=fmaxf(low,high);
        best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,4));
        best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,8));
        best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,16));
        uint32_t mask=0x11111111u<<(lane&3);
        uint32_t low_matches=__ballot_sync(0xffffffff,seed<10 && low==best)&mask;
        uint32_t high_matches=__ballot_sync(0xffffffff,seed+8<10 && high==best)&mask;
        int winner=low_matches?(__ffs(low_matches)-1)/4:8+(__ffs(high_matches)-1)/4;
        bool wins=owner<10 && owner==winner;
'''+s[b:]
# Explicitly zero unused MMA rows, including the rare all-minus-inf case.
s=s.replace('pair.values[0]=__float2bfloat16_rn(ki==col?0.f:', 'pair.values[0]=__float2bfloat16_rn(owner>=10 || ki==col?0.f:')
s=s.replace('pair.values[1]=__float2bfloat16_rn(ki+1==col?0.f:', 'pair.values[1]=__float2bfloat16_rn(owner>=10 || ki+1==col?0.f:')
a=s.index('      float inv_quant[2];')
b=s.index('      if(tid==0) asm volatile("cp.async.bulk.wait_group.read 0;',a)
quant=s[a:b]
quant=quant.replace('      float inv_quant[2];','      float inv_quant[2]={1.f,1.f};\n      if constexpr(SCALED_STORAGE) {')+'      }\n'
s=s[:a]+quant+s[b:]
s=s.replace('"r"(kb*4)','"r"(kb*10)')
s=s.replace('dim3 grid(ko*r,ah,(qo*5+tiles-1)/tiles)','dim3 grid(ko*r,ah,(qo*20+tiles-1)/tiles)')
s=s.replace('seeds.sizes()==at::IntArrayRef({1,1,ko,r,c,h,4})','seeds.sizes()==at::IntArrayRef({1,1,ko,r,c,h,10})')
s=s.replace('x.sizes()==at::IntArrayRef({qo*c*ah,ko*r*4,128})','x.sizes()==at::IntArrayRef({qo*c*ah,ko*r*10,128})')
s=s.replace('dims[5]={128,uint64_t(ko*r*4)','dims[5]={128,uint64_t(ko*r*10)')
s=s.replace('box[5]={128,4,1,16,1}','box[5]={128,10,1,4,1}')
s=s.replace('tiles==1 || tiles==3 || tiles==7 || tiles==21','tiles==1 || tiles==3 || tiles==7 || tiles==21 || tiles==84 || tiles==420')
for scaled in (False,True):
    name='scaled' if scaled else 'direct'
    path=root/f'native_fisher/fisher_c10_fp8_pad16_{name}.cu'
    path.write_text('// Experimental warp-local C10 ownership with compact FP8 storage.\nconstexpr bool SCALED_STORAGE='+str(scaled).lower()+';\n'+s)
    print(path.name)
