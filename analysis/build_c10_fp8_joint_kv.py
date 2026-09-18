"""Issue one N256 WGMMA for the adjacent K and V shared-memory panels."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fisher_c10_fp8_shared_store.cu').read_text()
s=s.replace('__launch_bounds__(128,4)','__launch_bounds__(128,2)')
s=s.replace('#define E64(a) E32(a),E8(a,32),E8(a,40),E8(a,48),E8(a,56)',
'''#define E64(a) E32(a),E8(a,32),E8(a,40),E8(a,48),E8(a,56)
#define E128(a) E64(a),E8(a,64),E8(a,72),E8(a,80),E8(a,88),E8(a,96),E8(a,104),E8(a,112),E8(a,120)''')
a=s.index('    #pragma unroll\n    for(int kind=0;kind<2;++kind) {')
b=s.index('      if(stop_after==3) continue;',a)
s=s[:a]+'''    // boff's column panels are adjacent: V starts at column 128.
    // Preserve K=80 accumulation order, but issue both outputs together.
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    float all_acc[128];
    using MergeJoint=MMA_64x256x16_F32BF16BF16_RS<Major::K,Major::MN>;
    #pragma unroll
    for(int ki=0;ki<80;ki+=16) {
      auto bd=desc(sk+boff(ki,0),10240,1024);
      int wi=(ki/8)*2;
      MergeJoint::fma(packed_weights[wi],packed_weights[wi+1],packed_weights[wi+2],packed_weights[wi+3],bd,E128(all_acc),ki==0?ScaleOut::Zero:ScaleOut::One);
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
    #pragma unroll
    for(int kind=0;kind<2;++kind) {
      float* acc=all_acc+kind*64;
'''+s[b:]
(root/'fisher_c10_fp8_joint_kv.cu').write_text(s)
(root/'fisher_c10_fp8_joint_kv_regs.cu').write_text(s.replace('__launch_bounds__(128,2)','__launch_bounds__(128,3)'))
