"""Generate paired-query-tile scheduling controls from the retained exact C8.

Only scheduling changes: original score shape, arithmetic, reduction order,
owner keys, BF16 conversions, K/V reuse, TMA stores and route remain fixed.
The odd 21st query frame is an explicit one-tile tail.
"""

import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/best/c8_r2/gma_c8_d10a_fused_argmax_v0.cu"


def make_source(base: str, overlap: bool, residency: int) -> str:
    begin = base.index("    uint32_t qreg[32];")
    end = base.index("  if(tma_store==1 && lane==0)", begin)
    math_begin = base.index("    float norm_max[2],norm_sum[2];", begin)
    math_end = base.index("    if(stop_after==2) continue;", math_begin)
    arithmetic = base[math_begin:math_end]
    merge_begin = base.index("    #pragma unroll\n    for(int kind=0;kind<2;++kind)", math_end)
    merge_tail = base[merge_begin:end]
    assert merge_tail.endswith("  }\n  }\n")
    merge = merge_tail[:-len("  }\n  }\n")]
    prefetch_start = merge.index("        {\n          int pf=")
    prefetch_end = merge.index("        continue;", prefetch_start)
    # Pair prologues load Q explicitly; the original epilogue prefetch would
    # overwrite the in-flight B WGMMA register operands.
    merge = merge[:prefetch_start] + merge[prefetch_end:]

    def score(target, frame):
        return f'''      #pragma unroll
      for(int d=0;d<128;d+=32) {{
        uint4 a=query_vector(q+qi0+({frame})*R*C*H*128+d,cache_last,lane);
        uint4 b=query_vector(q+qi1+({frame})*R*C*H*128+d,cache_last,lane);
        int i=d/4;
        qreg[i]=a.x;qreg[i+1]=b.x;qreg[i+2]=a.y;qreg[i+3]=b.y;
        qreg[i+4]=a.z;qreg[i+5]=b.z;qreg[i+6]=a.w;qreg[i+7]=b.w;
      }}
      #pragma unroll
      for(int i=0;i<32;++i) asm volatile("" : "+r"(qreg[i]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;":::"memory");
      using Score=MMA_64x80x16_F32BF16BF16_RS<Major::K,Major::K>;
      #pragma unroll
      for(int d=0;d<128;d+=16) {{
        int i=d/4;
        Score::fma(qreg[i],qreg[i+1],qreg[i+2],qreg[i+3],
          desc(sk+boff(0,d),16,1024),E40({target}),d==0?ScaleOut::Zero:ScaleOut::One);
      }}
      asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
'''

    wait = '''      asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
      // Pin register A operands until the asynchronous reader has completed.
      #pragma unroll
      for(int i=0;i<32;++i) asm volatile("" : "+r"(qreg[i]) :: "memory");
'''
    loop = '''    for(int pair_qo=0;pair_qo<Q;pair_qo+=2) {
      uint32_t qreg[32];
      float s[40],s_next[40];
      {
'''+score("s", "pair_qo")+wait+'''      }
      if(pair_qo+1<Q) {
'''+score("s_next", "pair_qo+1")+("" if overlap else wait)+'''      }
      #pragma unroll
      for(int member=0;member<2;++member) {
        if(pair_qo+member>=Q) break;
        int qo=pair_qo+member;
        if(member==1) {
          #pragma unroll
          for(int i=0;i<40;++i) s[i]=s_next[i];
        }
'''+arithmetic
    if overlap:
        loop += '''        if(member==0 && pair_qo+1<Q) {
'''+wait+'''        }
'''
    loop += merge+'''      }
    }
  }
'''
    result = base[:begin] + loop + base[end:]
    result = result.replace("__launch_bounds__(128,4)", f"__launch_bounds__(128,{residency})")
    return (f"// Generated paired-tile {'overlap' if overlap else 'serial'} control.\n"
            f"// Retained-source SHA256: {hashlib.sha256(base.encode()).hexdigest()}\n"
            "// No arithmetic changes; exactly C8; no singleton or residual route.\n" + result)


def make_fragment_pipeline(base: str, overlap: bool, width=64, min_ctas=4, compact=False, late_query=False, query_prefetch=True) -> str:
    """Split N128 into N64 products; two 32-float banks replace one 64 bank.

    A completed fragment is staged while the next independent fragment runs.
    K0,K1,V0,V1 keep the same K reduction order and BF16 conversion boundary.
    Each per-warp TMA box is issued only after both halves have been staged.
    """
    begin=base.index('    #pragma unroll\n    for(int kind=0;kind<2;++kind)')
    end=base.index('  if(tma_store==1 && lane==0)',begin)
    epilogue=base[begin:end]
    prefetch=epilogue[epilogue.index('        {\n          int pf='):epilogue.index('        continue;',epilogue.index('        {\n          int pf='))]
    issue_tma=epilogue[epilogue.index('        asm volatile("fence.proxy.async.shared::cta;'):epilogue.index('        {\n          int pf=')]

    fragments=128//width
    def issue(stage):
        kind=stage//fragments; n=width*(stage%fragments); bank=stage%2
        return f'''    {{
      asm volatile("wgmma.fence.sync.aligned;":::"memory");
      using Merge=MMA_64x{width}x16_F32BF16BF16_RS<Major::K,Major::MN>;
      #pragma unroll
      for(int ki=0;ki<80;ki+=16) {{
        auto bd=desc({'sk' if kind==0 else 'sv'}+boff(ki,{n}),10240,1024);
        int wi=(ki/8)*2;
        Merge::fma(packed_weights[wi],packed_weights[wi+1],packed_weights[wi+2],packed_weights[wi+3],bd,E{width//2}(acc{bank}),ki==0?ScaleOut::Zero:ScaleOut::One);
      }}
      asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    }}
'''
    def stage_fragment(stage):
        kind=stage//fragments; n=width*(stage%fragments); bank=stage%2
        s=f'''    {{
      constexpr int kind={kind};
      bf16* buf=temp+warp*{16*width if compact else 2048};
'''
        if n==0 or compact:
            s+='''      if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
      __syncwarp();
'''
        s+=f'''      #pragma unroll
      for(int g=0;g<{width//16};++g) {{
        uint32_t packed[4];
        #pragma unroll
        for(int j=0;j<4;++j) {{
          __nv_bfloat162 value=__floats2bfloat162_rn(acc{bank}[g*8+j*2],acc{bank}[g*8+j*2+1]);
          packed[j]=*reinterpret_cast<uint32_t*>(&value);
        }}
        int row=warp*16+(lane&7)+8*((lane>>3)&1);
        int canonical=col_for_row(row)*8+seed_for_row(row)-warp*16;
        int d={0 if compact else n}+g*16+8*(lane>>4),offset=canonical*{width if compact else 128}+d;
        uint32_t sp=__cvta_generic_to_shared(buf+(offset^(((offset>>{5 if compact and width==32 else 6})&{3 if compact and width==32 else 7})<<3)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{{%1,%2,%3,%4}};"::"r"(sp),"r"(packed[0]),"r"(packed[1]),"r"(packed[2]),"r"(packed[3]):"memory");
      }}
'''
        if compact:
            s+=f'''      asm volatile("fence.proxy.async.shared::cta;":::"memory");
      __syncwarp();
      if(lane==0) {{
        const CUtensorMap* map=kind==0?&map_k:&map_v;
        #pragma unroll
        for(int w=0;w<4;++w) if(warp==w) {{
          uint32_t p=__cvta_generic_to_shared(temp+w*{16*width});
          asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {{%1,%2,%3,%4,%5}}], [%6];"::"l"(map),"r"({n}),"r"(0),"r"(kb),"r"(head),"r"(qo*C+cb*8+2*w),"r"(p):"memory");
          asm volatile("cp.async.bulk.commit_group;":::"memory");
        }}
      }}
'''
        elif n==128-width:
            s+=issue_tma
        if late_query and stage==2*fragments-1:
            s+='''      int pf=(qo<Q-1)?R*C*H*128:0;
      #pragma unroll
      for(int d=0;d<128;d+=32) {
        uint4 a=query_vector(q+qi0+pf+d,cache_last,lane);
        uint4 b=query_vector(q+qi1+pf+d,cache_last,lane);
        int i=d/4;
        qreg[i]=a.x;qreg[i+1]=b.x;qreg[i+2]=a.y;qreg[i+3]=b.y;
        qreg[i+4]=a.z;qreg[i+5]=b.z;qreg[i+6]=a.w;qreg[i+7]=b.w;
      }
'''
        elif n==128-width and not late_query and query_prefetch:
            s+=prefetch
        return s+'    }\n'
    wait='    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n'
    body=f'    float acc0[{width//2}],acc1[{width//2}];\n'+issue(0)+wait
    for stage in range(1,2*fragments):
        if overlap:
            body+=issue(stage)+stage_fragment(stage-1)+wait
        else:
            body+=stage_fragment(stage-1)+issue(stage)+wait
    body+=stage_fragment(2*fragments-1)+'  }\n  }\n'
    source=base[:begin]+body+base[end:]
    if compact:
        assert width in (32,64)
        source=source.replace('#define E32(a)', '#define E16(a) E8(a,0),E8(a,8)\n#define E32(a)',1)
        source=source.replace('__launch_bounds__(128,4)',f'__launch_bounds__(128,{min_ctas})')
        source=source.replace('(80*128*2+64*128)*2',f'(80*128*2+64*{width})*2')
        source=source.replace('uint64_t dims[5]={64,uint64_t(ko*r*16),uint64_t(ah),80,uint64_t(qo)};',
                              'uint64_t dims[5]={128,8,uint64_t(ko*r),uint64_t(ah),uint64_t(qo*80)};')
        source=source.replace('uint64_t stride=128;','uint64_t stride=256;')
        source=source.replace('uint32_t box[5]={64,16,1,2,1}',f'uint32_t box[5]={{{width},8,1,1,2}}')
        if width==32:
            source=source.replace('CU_TENSOR_MAP_SWIZZLE_128B','CU_TENSOR_MAP_SWIZZLE_64B')
    return f'// N{width} fragment epilogue pipeline, overlap={overlap}; exact C8.\n'+source


def main():
    base = BASE.read_text(encoding="utf-8")
    manifest = {"baseline": str(BASE.relative_to(ROOT)),
                "baseline_sha256": hashlib.sha256(base.encode()).hexdigest(), "sources": {}}
    for overlap, residency in ((False, 3), (True, 3), (True, 4)):
        name = f"fisher_c8_pair_{'overlap' if overlap else 'serial'}_lb{residency}_v0.cu"
        source = make_source(base, overlap, residency)
        out = ROOT / "native_fisher" / name
        out.write_text(source, encoding="utf-8", newline="\n")
        manifest["sources"][name] = {"sha256": hashlib.sha256(source.encode()).hexdigest(),
                                     "overlap": overlap, "min_ctas": residency}
    return manifest


def pack_owner_labels(source: str) -> str:
    """Store twenty exact three-bit owners in two words, without FP changes."""
    helper='''__device__ __forceinline__ uint32_t unpack_owner(uint32_t word,int shift) {
  uint32_t value;
  asm volatile("bfe.u32 %0,%1,%2,3;":"=r"(value):"r"(word),"r"(shift));
  return value;
}
'''
    source=source.replace('__device__ __forceinline__ float lg2(float x)',helper+'__device__ __forceinline__ float lg2(float x)',1)
    source=source.replace('int labels[20];','uint32_t labels_packed[2]={0,0};',1)
    source=source.replace('labels[i]=(int)(key&7u);',
                          'asm volatile("bfi.b32 %0,%1,%0,%2,3;":"+r"(labels_packed[i/10]):"r"(key),"r"(3*(i%10)));',1)
    source=re.sub(r'labels\[([^\]]+)\]',lambda m:f'unpack_owner(labels_packed[({m[1]})/10],3*(({m[1]})%10))',source)
    return '// Exact three-bit owner packing reduces live integer registers.\n'+source


def split_score_query_registers(source: str) -> str:
    """Keep 16 Q words live by finishing each ordered K64 score half."""
    source=source.replace('uint32_t qreg[32];','uint32_t qreg[16];')
    source=source.replace('for(int i=0;i<32;++i) asm volatile','for(int i=0;i<16;++i) asm volatile')
    source=source.replace('for(int d=0;d<128;d+=32)','for(int d=0;d<64;d+=32)')
    begin=source.index('    float s[40];')
    end=source.index('    if(debug_scores)',begin)
    source=source[:begin]+'''    float s[40];
    using Score=MMA_64x80x16_F32BF16BF16_RS<Major::K,Major::K>;
    #pragma unroll
    for(int part=0;part<2;++part) {
      if(part==1) {
        #pragma unroll
        for(int d=0;d<64;d+=32) {
          uint4 a=query_vector(q+qi0+64+d,cache_last,lane);
          uint4 b=query_vector(q+qi1+64+d,cache_last,lane);
          int i=d/4;
          qreg[i]=a.x;qreg[i+1]=b.x;qreg[i+2]=a.y;qreg[i+3]=b.y;
          qreg[i+4]=a.z;qreg[i+5]=b.z;qreg[i+6]=a.w;qreg[i+7]=b.w;
        }
      }
      #pragma unroll
      for(int i=0;i<16;++i) asm volatile("" : "+r"(qreg[i]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;":::"memory");
      #pragma unroll
      for(int d=0;d<64;d+=16) {
        int i=d/4;
        Score::fma(qreg[i],qreg[i+1],qreg[i+2],qreg[i+3],desc(sk+boff(0,part*64+d),16,1024),E40(s),part==0 && d==0?ScaleOut::Zero:ScaleOut::One);
      }
      asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
    }
'''+source[end:]
    return '// Ordered K64 score halves halve live query operands.\n'+source


def make_two_warpgroup(base: str, pingpong: bool, all_stages: bool = False, tiles=None) -> str:
    """Two independent WG register files share one immutable K/V tile.

    WG0 handles frames 0,2,...,20; WG1 handles 1,3,...,19. In pingpong mode,
    named barriers 1 and 2 alternate QK issue, without sharing score registers.
    Every barrier cycle receives 128 arrivals from each WG. The final even
    tail consumes WG1's last barrier-2 arrival and does not signal barrier 1.
    """
    s=base.replace('__launch_bounds__(128,4)','__launch_bounds__(256,2)')
    s=s.replace('bf16* temp=sv+80*128;','bf16* temp=sv+80*128+(threadIdx.x/128)*64*128;')
    s=s.replace('int tid=threadIdx.x,lane=tid&31,warp=tid>>5,seed=lane>>2;',
                'int tid=threadIdx.x,wg=tid/128,lane=tid&31,warp=(tid%128)>>5,seed=lane>>2;')
    s=s.replace('for(int vec=tid;vec<80*16;vec+=128)',
                'for(int vec=tid;vec<80*16;vec+=256)')
    s=s.replace('uint32_t qreg[32];','qi0+=wg*R*C*H*128;qi1+=wg*R*C*H*128;\n    uint32_t qreg[32];',1)
    s=s.replace('for(int qo=0;qo<Q;++qo,qi0+=R*C*H*128,qi1+=R*C*H*128)',
                'for(int qo=wg;qo<Q;qo+=2,qi0+=2*R*C*H*128,qi1+=2*R*C*H*128)')
    s=s.replace('int pf=(qo<Q-1)?R*C*H*128:0;',
                'int pf=(qo<Q-2)?2*R*C*H*128:0;')
    s=s.replace('(80*128*2+64*128)*2','(80*128*2+2*64*128)*2')
    s=s.replace('<<<grid,128,bytes,stream>>>','<<<grid,256,bytes,stream>>>')
    s=s.replace('merge_kernel<21,40>,128,bytes','merge_kernel<21,40>,256,bytes')
    if pingpong:
        s=s.replace('    float s[40];\n', '''    // CTA barrier 0 belongs to initial K/V loading. Barriers 1/2 have
    // explicit 256-thread counts and coordinate two complete warpgroups.
    if(wg==1) asm volatile("bar.sync 1,256;":::"memory");
    else if(qo>0) asm volatile("bar.sync 2,256;":::"memory");
    float s[40];
''',1)
        marker='    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n'
        s=s.replace(marker,marker+'''    if(wg==0 && qo+1<Q) asm volatile("bar.arrive 1,256;":::"memory");
    else if(wg==1) asm volatile("bar.arrive 2,256;":::"memory");
''',1)
        if all_stages:
            s=s.replace('    for(int kind=0;kind<2;++kind) {', '''    for(int kind=0;kind<2;++kind) {
      // The unpaired even tail performs its two merges without a peer.
      if(wg==1) asm volatile("bar.sync 1,256;":::"memory");
      else if(qo<Q-1) asm volatile("bar.sync 2,256;":::"memory");
''',1)
            marker='      asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n'
            s=s.replace(marker,marker+'''      if(wg==0 && qo<Q-1) asm volatile("bar.arrive 1,256;":::"memory");
      else if(wg==1) asm volatile("bar.arrive 2,256;":::"memory");
''',1)
    if tiles is not None:
        s=s.replace(' int qo=q.size(1)',f' tiles={tiles}; // Candidate-specific scheduling span, no arithmetic change.\n int qo=q.size(1)',1)
    return (f'// Two independent warpgroups, shared K/V, pingpong={pingpong}.\n'
            '// Exact C8; unchanged arithmetic and TMA buffer-reuse waits.\n'+s)


def make_norm_handoff(base: str, early: bool) -> str:
    """Start the peer QK only after scalar work, staggering SFU and TC phases."""
    source=make_two_warpgroup(base,True)
    signal='''    if(wg==0 && qo+1<Q) asm volatile("bar.arrive 1,256;":::"memory");
    else if(wg==1) asm volatile("bar.arrive 2,256;":::"memory");
'''
    assert source.count(signal)==1
    source=source.replace(signal,'',1)
    marker='    float winners[20];' if early else '    if(stop_after==2) continue;'
    source=source.replace(marker,signal+marker,1)
    return f'// Peer QK handoff after {"normalization" if early else "complete weights"}.\n'+source


def specialize_two_wg(source: str) -> str:
    """Keep shared TMA source offsets compile-time constant within each WG."""
    begin=source.index('  int total=Q*(C/8),start=blockIdx.z*tile_count;')
    end=source.index('\n}\n\n// Pack each',begin)
    body=source[begin:end]
    paths=[]
    for group in (0,1):
        path=re.sub(r'\bwg\b',str(group),body)
        paths.append(f'    bf16* temp=sv+80*128+{group}*64*128;\n'+path)
    return ('// Specialize each warpgroup path to eliminate dynamic TMA uniformization.\n'+source[:begin]+
            '  if(tid<128) {\n'+paths[0]+'\n  } else {\n'+paths[1]+'\n  }'+source[end:])


def make_column_pipeline(base: str, handoff: bool) -> str:
    """Split ten column groups 5/5, keeping all 21 frames in each WG."""
    source=make_two_warpgroup(base,False,tiles=210)
    source=source.replace('threadIdx.x/128','int(__all_sync(0xffffffff,threadIdx.x>=128))')
    source=source.replace('wg=tid/128','wg=int(__all_sync(0xffffffff,tid>=128))')
    source=source.replace('for(int span=start;span<min(start+tile_count,total);span+=Q)',
                          'for(int span=start+wg*Q;span<min(start+tile_count,total);span+=2*Q)')
    source=source.replace('qi0+=wg*R*C*H*128;qi1+=wg*R*C*H*128;','')
    source=source.replace('for(int qo=wg;qo<Q;qo+=2,qi0+=2*R*C*H*128,qi1+=2*R*C*H*128)',
                          'for(int qo=0;qo<Q;++qo,qi0+=R*C*H*128,qi1+=R*C*H*128)')
    source=source.replace('int pf=(qo<Q-2)?2*R*C*H*128:0;','int pf=(qo<Q-1)?R*C*H*128:0;')
    if handoff:
        source=source.replace('    float s[40];','''    if(wg==1) asm volatile("bar.sync 1,256;":::"memory");
    else if(span!=start || qo>0) asm volatile("bar.sync 2,256;":::"memory");
    float s[40];''',1)
        source=source.replace('    if(stop_after==2) continue;','''    if(wg==0) asm volatile("bar.arrive 1,256;":::"memory");
    else if(qo+1<Q || span+Q<min(start+tile_count,total))
      asm volatile("bar.arrive 2,256;":::"memory");
    if(stop_after==2) continue;''',1)
    return '// Column-group distribution removes the 11/10 query-frame imbalance.\n'+source


def advance_query_prefetch(source: str, stage: str) -> str:
    """Place next-tile register loads inside an in-flight merge product."""
    begin=source.index('        {\n          int pf=')
    end=source.index('        continue;',begin)
    prefetch=source[begin:end]
    assert prefetch.count('query_vector(')==2
    source=source[:begin]+source[end:]
    marker='      asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n'
    assert source.count(marker)==1
    condition={'k':'kind==0','v':'kind==1','both':'true'}[stage]
    source=source.replace(marker,marker+f'      if({condition}) '+prefetch.lstrip(),1)
    late=f'        if(!({condition})) '+prefetch.lstrip()
    tail='''        continue;
      }
      #pragma unroll
      for(int i=0;i<64;i+=2)'''
    assert source.count(tail)==1
    source=source.replace(tail,late+tail,1)
    # New versions remove whitespace left by the archived frame-offset edit.
    source='\n'.join(line.rstrip() for line in source.splitlines())+'\n'
    return f'// Next-Q prefetch while {stage} merge WGMMA is in flight.\n'+source


def make_cross_query_pipeline(base: str) -> str:
    """Issue next QK while staging the last completed N64 V fragment."""
    source=make_fragment_pipeline(base,True)
    loop='    for(int qo=0;qo<Q;++qo,qi0+=R*C*H*128,qi1+=R*C*H*128) {\n'
    start=source.index(loop)+len(loop)
    wait='    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n'
    end=source.index(wait,start)
    score=source[start:end]
    assert score.count('float s[40];')==1
    score=score.replace('    float s[40];\n','')
    # One score accumulator per query. After normalization consumes it, the
    # next query may write it while the completed V fragment is staged.
    source=source[:start]+source[end:]
    source=source.replace(loop,'    float s[40];\n    {\n'+score+'    }\n'+loop,1)
    prefetch_start=source.rindex('        {\n          int pf=')
    prefetch_end=source.index('    }\n  }\n  }\n',prefetch_start)
    prefetch=source[prefetch_start:prefetch_end]
    assert prefetch.count('query_vector(')==2
    source=source[:prefetch_start]+source[prefetch_end:]
    final_wait=source.rindex(wait)+len(wait)
    issue='    {\n      constexpr int kind=1;\n'+prefetch+'    }\n'
    issue+='    if(qo+1<Q) {\n'+score+'    }\n'
    source=source[:final_wait]+issue+source[final_wait:]
    return '// Next-query QK overlaps the final V-fragment store epilogue.\n'+source


def make_role_pipeline(base: str, stages: int, split_merge=False) -> str:
    """Separate score/weight production from K/V products using exact RS words.

    Barriers 1..stages publish filled slots; stages+1..2*stages release
    copied slots. Each phase has 128 producer and 128 consumer arrivals.
    The consumer returns a slot only after every RS word is in registers.
    """
    assert stages in (1,2)
    begin=base.index('  int total=Q*(C/8),start=blockIdx.z*tile_count;')
    merge_begin=base.index('    #pragma unroll\n    for(int kind=0;kind<2;++kind)',begin)
    drain=base.index('  if(tma_store==1 && lane==0)',merge_begin)
    producer=base[begin:merge_begin]
    producer=producer.replace('++qo,qi0+=','++qo,++iteration,qi0+=')
    assert producer.count('++iteration')==1
    producer+=f'''    int slot=iteration%{stages};
    int filled=1+slot,freed=1+{stages}+slot;
    if(iteration>={stages}) asm volatile("bar.sync %0,256;"::"r"(freed):"memory");
    #pragma unroll
    for(int i=0;i<20;++i) transfer[slot*128*20+i*128+(tid&127)]=packed_weights[i];
    asm volatile("bar.arrive %0,256;"::"r"(filled):"memory");
    // The next query belongs only to the producer. Loading it can overlap
    // the consumer's two tensor products and TMA output handling.
    int pf=(qo<Q-1)?R*C*H*128:0;
    #pragma unroll
    for(int d=0;d<128;d+=32) {{
      uint4 a=query_vector(q+qi0+pf+d,cache_last,lane);
      uint4 b=query_vector(q+qi1+pf+d,cache_last,lane);
      int i=d/4;
      qreg[i]=a.x;qreg[i+1]=b.x;qreg[i+2]=a.y;qreg[i+3]=b.y;
      qreg[i+4]=a.z;qreg[i+5]=b.z;qreg[i+6]=a.w;qreg[i+7]=b.w;
    }}
  }}
  }}
'''
    if split_merge:
        fragment=make_fragment_pipeline(base,False,query_prefetch=False)
        fragment_begin=fragment.index('    float acc0[32],acc1[32];')
        fragment_end=fragment.index('  if(tma_store==1 && lane==0)',fragment_begin)
        merge=fragment[fragment_begin:fragment_end]
    else:
        merge=base[merge_begin:drain]
        pf_begin=merge.index('        {\n          int pf=')
        pf_end=merge.index('        continue;',pf_begin)
        merge=merge[:pf_begin]+merge[pf_end:]
    assert merge.endswith('  }\n  }\n')
    merge=merge[:-len('  }\n  }\n')]
    consumer=f'''  int total=Q*(C/8),start=blockIdx.z*tile_count;
  for(int span=start;span<min(start+tile_count,total);span+=Q) {{
    int cb=span/Q;
    for(int qo=0;qo<Q;++qo,++iteration) {{
    int slot=iteration%{stages};
    int filled=1+slot,freed=1+{stages}+slot;
    asm volatile("bar.sync %0,256;"::"r"(filled):"memory");
    uint32_t packed_weights[20];
    #pragma unroll
    for(int i=0;i<20;++i) packed_weights[i]=transfer[slot*128*20+i*128+(tid&127)];
    // Force all shared reads to resolve before releasing the slot. WGMMA
    // consumes these independent register copies, never the shared slot.
    #pragma unroll
    for(int i=0;i<20;++i) asm volatile("" : "+r"(packed_weights[i]) :: "memory");
    asm volatile("bar.arrive %0,256;"::"r"(freed):"memory");
'''+merge+'  }\n  }\n'
    roles=('  uint32_t* transfer=reinterpret_cast<uint32_t*>(temp+64*128);\n'
           '  int iteration=0;\n'
           '  if(__all_sync(0xffffffff,tid<128)) {\n'+producer+
           '  } else {\n'+consumer+'  }\n')
    source=base[:begin]+roles+base[drain:]
    source=source.replace('__launch_bounds__(128,4)','__launch_bounds__(256,2)')
    source=source.replace('int tid=threadIdx.x,lane=tid&31,warp=tid>>5,seed=lane>>2;',
                          'int tid=threadIdx.x,lane=tid&31,warp=(tid&127)>>5,seed=lane>>2;')
    source=source.replace('for(int vec=tid;vec<80*16;vec+=128)',
                          'for(int vec=tid;vec<80*16;vec+=256)')
    source=source.replace('(80*128*2+64*128)*2',f'((80*128*2+64*128)*2+{stages}*128*20*4)')
    source=source.replace('<<<grid,128,bytes,stream>>>','<<<grid,256,bytes,stream>>>')
    source=source.replace('merge_kernel<21,40>,128,bytes','merge_kernel<21,40>,256,bytes')
    return f'// Exact RS-weight producer/consumer pipeline, shared slots={stages}.\n'+source


def role_shared_query(source: str, residency: int) -> str:
    """Reuse a 16KiB slot for producer SS queries and then exact RS weights."""
    assert source.count('slot=iteration%1')==2
    source=source.replace('((80*128*2+64*128)*2+1*128*20*4)',
                          '((80*128*2+64*128)*2+64*128*2)')
    source=source.replace('uint32_t* transfer=reinterpret_cast<uint32_t*>(temp+64*128);',
                          'uint32_t* transfer=reinterpret_cast<uint32_t*>(temp+64*128);\n'
                          '  bf16* shared_query=reinterpret_cast<bf16*>(transfer);',1)
    begin=source.index('    uint32_t qreg[32];')
    loop=source.index('    for(int qo=0;',begin)
    source=source[:begin]+source[loop:]
    begin=source.index('    #pragma unroll\n    for(int i=0;i<32;++i)')
    end=source.index('    float s[40];',begin)
    source=source[:begin]+'''    int slot=0,filled=1,freed=2;
    if(iteration>=1) asm volatile("bar.sync 2,256;":::"memory");
    // The inverse register-fragment transpose restores original Q words.
    // Each four-lane group writes the same two rows as the RS score operand.
    #pragma unroll
    for(int d=0;d<128;d+=32) {
      uint4 a=fragment_transpose(query_vector(q+qi0+d,cache_last,lane),lane);
      uint4 b=fragment_transpose(query_vector(q+qi1+d,cache_last,lane),lane);
      int column=d+(lane&3)*8;
      *reinterpret_cast<uint4*>(shared_query+aoff(row0,column))=a;
      *reinterpret_cast<uint4*>(shared_query+aoff(row0+8,column))=b;
    }
    asm volatile("bar.sync 5,128;":::"memory");
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
'''+source[end:]
    source=source.replace('using Score=MMA_64x80x16_F32BF16BF16_RS<Major::K,Major::K>;',
                          'using Score=MMA_64x80x16_F32BF16BF16_SS<Major::K,Major::K>;',1)
    source=source.replace('Score::fma(qreg[i],qreg[i+1],qreg[i+2],qreg[i+3],desc(sk+boff(0,d),16,1024),',
                          'Score::fma(desc(shared_query+aoff(0,d),16,1024),desc(sk+boff(0,d),16,1024),',1)
    old='''    int slot=iteration%1;
    int filled=1+slot,freed=1+1+slot;
    if(iteration>=1) asm volatile("bar.sync %0,256;"::"r"(freed):"memory");
'''
    assert source.count(old)==1
    source=source.replace(old,'',1)
    begin=source.index('    // The next query belongs only to the producer.')
    end=source.index('  }\n  }\n  } else {',begin)
    source=source[:begin]+source[end:]
    assert 'qreg' not in source
    # Retain mul.rn arithmetic but rematerialize the inexpensive multiply at
    # owner normalization, rather than keeping its earlier CSE value live.
    helper='''__device__ __forceinline__ float owner_rescale(float x,float scale) {
  float result;
  asm volatile("mul.rn.f32 %0,%1,%2;":"=f"(result):"f"(x),"f"(scale));
  return result;
}
'''
    source=source.replace('__device__ __forceinline__ float ex2(float x)',helper+
                          '__device__ __forceinline__ float ex2(float x)',1)
    source=source.replace('__fmul_rn(s[ix],scale)','owner_rescale(s[ix],scale)',1)
    source=source.replace('__launch_bounds__(256,2)',f'__launch_bounds__(256,{residency})')
    return '// Shared Q/weight slot and rematerialized owner scale.\n'+source


def make_packed_epilogue_pipeline(base: str, overlap: bool, cross_query=False, late_groups=4, early_query=True) -> str:
    """Overlap a finished BF16 half-output with a full-N128 matrix product."""
    begin=base.index('    #pragma unroll\n    for(int kind=0;kind<2;++kind)')
    end=base.index('  if(tma_store==1 && lane==0)',begin)
    original=base[begin:end]
    pf_start=original.index('        {\n          int pf=')
    prefetch=original[pf_start:original.index('        continue;',pf_start)]
    tma=original[original.index('        asm volatile("fence.proxy.async.shared::cta;'):pf_start]
    wait='    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n'

    def product(kind):
        return f'''    float acc{kind}[64];
    {{
      asm volatile("wgmma.fence.sync.aligned;":::"memory");
      using Merge=MMA_64x128x16_F32BF16BF16_RS<Major::K,Major::MN>;
      #pragma unroll
      for(int ki=0;ki<80;ki+=16) {{
        int wi=(ki/8)*2;
        Merge::fma(packed_weights[wi],packed_weights[wi+1],packed_weights[wi+2],packed_weights[wi+3],
          desc({'sk' if kind==0 else 'sv'}+boff(ki,0),10240,1024),E64(acc{kind}),ki==0?ScaleOut::Zero:ScaleOut::One);
      }}
      asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    }}
'''

    def pack(kind):
        return f'''    uint32_t output{kind}[32];
    #pragma unroll
    for(int i=0;i<32;++i) {{
      __nv_bfloat162 pair=__floats2bfloat162_rn(acc{kind}[2*i],acc{kind}[2*i+1]);
      output{kind}[i]=*reinterpret_cast<uint32_t*>(&pair);
    }}
'''

    def stage(kind, first, last):
        result=''
        if first==0:
            result+='''    if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
    __syncwarp();
'''
        result+=f'''    {{
      bf16* buf=temp+warp*2048;
      #pragma unroll
      for(int g={first};g<{last};++g) {{
        int row=warp*16+(lane&7)+8*((lane>>3)&1);
        int canonical=col_for_row(row)*8+seed_for_row(row)-warp*16;
        int d=g*16+8*(lane>>4),offset=canonical*128+d;
        uint32_t sp=__cvta_generic_to_shared(buf+(offset^(((offset>>6)&7)<<3)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{{%1,%2,%3,%4}};"::
          "r"(sp),"r"(output{kind}[4*g]),"r"(output{kind}[4*g+1]),
          "r"(output{kind}[4*g+2]),"r"(output{kind}[4*g+3]):"memory");
      }}
    }}
'''
        return result

    def enqueue(kind):
        return f'    {{ constexpr int kind={kind};\n'+tma+'    }\n'

    def query_prefetch(kind):
        return f'    {{ constexpr int kind={kind};\n'+prefetch+'    }\n'

    body=product(0)+wait+pack(0)
    if overlap:
        body+=stage(0,0,8-late_groups)+product(1)+stage(0,8-late_groups,8)+enqueue(0)
        body+=(query_prefetch(0)+wait) if early_query else (wait+query_prefetch(0))
    else:
        body+=stage(0,0,8)+enqueue(0)+query_prefetch(0)+product(1)+wait
    body+=pack(1)
    source=base
    if cross_query:
        loop='    for(int qo=0;qo<Q;++qo,qi0+=R*C*H*128,qi1+=R*C*H*128) {\n'
        score_begin=source.index(loop)+len(loop)
        score_end=source.index(wait,score_begin)
        score=source[score_begin:score_end].replace('    float s[40];\n','')
        source=source[:score_begin]+source[score_end:]
        source=source.replace(loop,'    float s[40];\n    {\n'+score+'    }\n'+loop,1)
        body+=stage(1,0,8-late_groups)+query_prefetch(1)
        body+='    if(qo+1<Q) {\n'+score+'    }\n'+stage(1,8-late_groups,8)+enqueue(1)
    else:
        body+=stage(1,0,8)+enqueue(1)+query_prefetch(1)
    begin=source.index('    #pragma unroll\n    for(int kind=0;kind<2;++kind)')
    end=source.index('  if(tma_store==1 && lane==0)',begin)
    return (f'// Packed BF16 epilogue overlap={overlap}, next-query overlap={cross_query}.\n'+
            source[:begin]+body+'  }\n  }\n'+source[end:])


def reset_carried_scores(source: str) -> str:
    """End old score liveness before a ScaleOut::Zero read/write asm operand."""
    marker='    if(qo+1<Q) {\n'
    assert source.count(marker)==1
    return source.replace(marker,marker+'''    #pragma unroll
    for(int reset=0;reset<40;++reset) s[reset]=0.f;
''',1)


def make_pair_packed_pipeline(base: str, overlap_kv: bool) -> str:
    """Overlap next QK within a two-query body, with no loop-carried score."""
    source=make_packed_epilogue_pipeline(base,overlap_kv)
    loop='    for(int qo=0;qo<Q;++qo,qi0+=R*C*H*128,qi1+=R*C*H*128) {\n'
    begin=source.index(loop)+len(loop)
    wait='    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n'
    score_end=source.index(wait,begin)
    score=source[begin:score_end]
    kv_begin=source.index('    float acc0[64];',score_end)
    math=source[score_end+len(wait):kv_begin]
    store_begin=source.index('    if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;',
                             source.index('    uint32_t output1[32];',kv_begin))
    tma_begin=source.index('    { constexpr int kind=1;\n',store_begin)
    prefetch_begin=source.rindex('    { constexpr int kind=1;\n')
    end=source.index('  }\n  }\n  if(tma_store==1 && lane==0)',prefetch_begin)
    stage=source[store_begin:tma_begin]
    enqueue=source[tma_begin:prefetch_begin]
    prefetch=source[prefetch_begin:end]
    assert stage.count('for(int g=0;g<8;++g)')==1
    first=stage.replace('g<8','g<4').replace('output1','carried')
    late=stage.replace('''    if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
    __syncwarp();
''','',1).replace('g=0;g<8','g=4;g<8').replace('output1','carried')
    tile0=source[begin:store_begin].replace('    uint32_t output1[32];\n','').replace('output1','carried')
    kv=source[kv_begin:store_begin]
    body='    uint32_t carried[32];\n    {\n'+tile0+'    }\n'+first+prefetch
    body+='    if(qo+1<Q) {\n'+score+late+enqueue+wait
    body+='    ++qo;qi0+=R*C*H*128;qi1+=R*C*H*128;\n'
    body+=math+kv+stage+enqueue+prefetch
    body+='    } else {\n'+late+enqueue+'    }\n'
    return '// Two-query body scopes score accumulators locally; odd tail retained.\n'+source[:begin]+body+source[end:]


def uniform_pair_index(source: str) -> str:
    """Keep the outer trip count independent of the optional second query."""
    loop='    for(int qo=0;qo<Q;++qo,qi0+=R*C*H*128,qi1+=R*C*H*128) {\n'
    pos=source.index(loop)
    prefix=re.sub(r'\bqi0\b','qi_base0',source[:pos])
    prefix=re.sub(r'\bqi1\b','qi_base1',prefix)
    tail=source[pos+len(loop):]
    tail=tail.replace('if(qo+1<Q)','if(__all_sync(0xffffffff,pair+1<Q))',1)
    return prefix+'''    for(int pair=0;pair<Q;pair+=2) {
    int qo=pair;
    int qi0=qi_base0+pair*R*C*H*128,qi1=qi_base1+pair*R*C*H*128;
'''+tail


def ordered_query_prefetch(source: str) -> str:
    """Keep Q0 loads after V completes using a value-dependent zero offset."""
    marker='    { constexpr int kind=0;\n        {\n          int pf='
    begin=source.index(marker)
    end=source.index('    uint32_t output1[32];',begin)
    prefetch=source[begin:end]
    assert prefetch.count('q+qi0+')==1 and prefetch.count('q+qi1+')==1
    prefetch=prefetch.replace('q+qi0+','q+qi0+query_dependency+')
    prefetch=prefetch.replace('q+qi1+','q+qi1+query_dependency+')
    dependency='''    uint32_t query_dependency;
    // The zero offset is exact. Its input makes the load depend on the
    // completed V accumulator, preventing ptxas from hoisting it into V.
    asm volatile("and.b32 %0,%1,0;":"=r"(query_dependency):"r"(__float_as_uint(acc1[0])));
'''
    return source[:begin]+dependency+prefetch+source[end:]


def rotate_packed_epilogue(base: str, kv_overlap=False, ordered_prefetch=False) -> str:
    """Carry only completed BF16 V words across the loop's back edge."""
    source=make_packed_epilogue_pipeline(base,kv_overlap,early_query=not ordered_prefetch)
    if ordered_prefetch:
        source=ordered_query_prefetch(source)
    loop='    for(int qo=0;qo<Q;++qo,qi0+=R*C*H*128,qi1+=R*C*H*128) {\n'
    begin=source.index(loop)+len(loop)
    wait='    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n'
    score_end=source.index(wait,begin)
    pack_begin=source.index('    uint32_t output1[32];',score_end)
    store_begin=source.index('    if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;',pack_begin)
    tma_begin=source.index('    { constexpr int kind=1;\n',store_begin)
    prefetch_begin=source.rindex('    { constexpr int kind=1;\n')
    end=source.index('  }\n  }\n  if(tma_store==1 && lane==0)',prefetch_begin)
    stage=source[store_begin:tma_begin]
    enqueue=source[tma_begin:prefetch_begin]
    prefetch=source[prefetch_begin:end]
    low=stage.replace('g<8','g<4')
    high=stage.replace('''    if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
    __syncwarp();
''','',1).replace('g=0;g<8','g=4;g<8')
    high=high.replace('output1[4*g','carried[4*g-16')
    body=source[begin:score_end]
    body+='    if(__all_sync(0xffffffff,qo>0)) {\n    int previous_qo=qo-1;\n'
    body+=high+re.sub(r'\bqo\b','previous_qo',enqueue)+'    }\n'
    body+=source[score_end:pack_begin]
    body+='''    uint32_t output1[16];
    #pragma unroll
    for(int i=0;i<16;++i) {
      __nv_bfloat162 lo=__floats2bfloat162_rn(acc1[2*i],acc1[2*i+1]);
      __nv_bfloat162 hi=__floats2bfloat162_rn(acc1[32+2*i],acc1[33+2*i]);
      output1[i]=*reinterpret_cast<uint32_t*>(&lo);
      carried[i]=*reinterpret_cast<uint32_t*>(&hi);
    }
'''+low+prefetch+'  }\n'
    body+='    { constexpr int qo=Q-1;\n'+high+enqueue+'    }\n'
    # The footer is inside the column span. The original final TMA drain
    # remains after all spans; each next K store also waits before reuse.
    return ('// Rotate the V epilogue under the next local QK product.\n'+
            source[:begin-len(loop)]+'    uint32_t carried[16];\n'+loop+body+source[end+len('  }\n'):])


def rotate_full_packed_epilogue(base: str, kv_overlap=False, ordered_prefetch=False) -> str:
    """Carry all final V words so the next QK overlaps its full staging."""
    source=rotate_packed_epilogue(base,kv_overlap,ordered_prefetch)
    # Restrict the transform to V staging: an overlapped K epilogue also
    # contains a g=4..7 loop but still reads output0, not carried words.
    starts=[source.rfind('for(int g=4;g<8;++g)',0,m.start())
            for m in re.finditer(re.escape('"r"(carried[4*g-16])'),source)]
    assert len(starts)==2 and all(x>=0 for x in starts)
    for start in reversed(starts):
        source=source[:start]+source[start:].replace('for(int g=4;g<8;++g)',
                                                    'for(int g=0;g<8;++g)',1)
    source=source.replace('uint32_t carried[16];','uint32_t carried[32];',1)
    source=source.replace('carried[4*g-16','carried[4*g')
    begin=source.index('    uint32_t output1[16];')
    end=source.index('    { constexpr int kind=1;\n        {',begin)
    source=source[:begin]+'''    #pragma unroll
    for(int i=0;i<32;++i) {
      __nv_bfloat162 pair=__floats2bfloat162_rn(acc1[2*i],acc1[2*i+1]);
      carried[i]=*reinterpret_cast<uint32_t*>(&pair);
    }
    // Reserve the same shared buffer before the next iteration stages V.
    if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
    __syncwarp();
'''+source[end:]
    return '// Full V staging is rotated under the next QK.\n'+source


def wait_before_score_backedge(source: str) -> str:
    """Carry completed scores; finish the next QK after the V epilogue."""
    loop='    for(int qo=0;qo<Q;++qo,qi0+=R*C*H*128,qi1+=R*C*H*128) {\n'
    wait='    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n'
    assert source.count(loop)==1 and loop+wait in source
    # The prologue and each loop tail now finish their QK before s[] crosses
    # a back edge. All arithmetic still consumes the same completed scores.
    source=source.replace(loop+wait,wait+loop,1)
    end=source.index('  }\n  }\n  if(tma_store==1 && lane==0)')
    return ('// Wait for next QK after V staging, before the loop back edge.\n'+
            source[:end]+wait+source[end:])


def preserve_query_dependency(source: str) -> str:
    """Two lane permutations retain an exact zero that ptxas cannot fold."""
    old='    asm volatile("and.b32 %0,%1,0;":"=r"(query_dependency):"r"(__float_as_uint(acc1[0])));'
    assert source.count(old)==1
    new='''    // XOR-1 twice restores every lane's original bits, including NaNs.
    // Every lane participates; the final XOR is zero without FP arithmetic.
    asm volatile("{.reg .b32 a,b;"
                 "shfl.sync.bfly.b32 a,%1,1,31,0xffffffff;"
                 "shfl.sync.bfly.b32 b,a,1,31,0xffffffff;"
                 "xor.b32 %0,%1,b;}"
                 :"=r"(query_dependency):"r"(__float_as_uint(acc1[0])));'''
    return '// Value dependency survives constant-zero folding.\n'+source.replace(old,new,1)


def rematerialize_owner_scale(source: str) -> str:
    """Retain mul.rn but avoid carrying the early scaled-score CSE values."""
    helper='''__device__ __forceinline__ float owner_rescale(float x,float scale) {
  float result;
  asm volatile("mul.rn.f32 %0,%1,%2;":"=f"(result):"f"(x),"f"(scale));
  return result;
}
'''
    owner='s[ix]=__fadd_rn(offset,-__fmul_rn(s[ix],scale));'
    assert source.count(owner)==1
    source=source.replace('__device__ __forceinline__ float ex2(float x)',
                          helper+'__device__ __forceinline__ float ex2(float x)',1)
    return source.replace(owner,owner.replace('__fmul_rn(s[ix],scale)','owner_rescale(s[ix],scale)'),1)


def fixed_launch_span(source: str, tiles: int) -> str:
    """Change only the launch span passed to the same generated kernel."""
    assert tiles in (21,42,105,210)
    marker='int hs,int ah,int tiles,double scale,torch::Tensor debug,int stop_after,int tma_store) {'
    assert source.count(marker)==1
    return f'// Experimental fixed CTA span: {tiles} query tiles.\n'+source.replace(
        marker,marker+f'\n tiles={tiles};',1)


def reuse_normalization_exponents(source: str) -> str:
    """Keep routing arithmetic; replace the second exp with cached exp / sum.

    This preserves the C8 formula but changes FP32 rounding. It must use a
    separately declared numerical-error gate, never the bitwise contract.
    """
    source=source.replace('    float norm_max[2],norm_sum[2];',
                          '    float cached_exp[40];\n    float norm_max[2],norm_sum[2];',1)
    for name,expression in (
        ('direct','ex2(__fmul_rn(s[4*(i/2)+2*half+(i&1)],scale))'),
        ('bs','ex2(__fmaf_rn(s[4*(i/2)+2*half+(i&1)],scale,-mx))')):
        old=f'      for(int i=0;i<20;++i)\n        {name}[i]={expression};'
        assert source.count(old)==1
        new=f'''      for(int i=0;i<20;++i) {{
        {name}[i]={expression};
        cached_exp[4*(i/2)+2*half+(i&1)]={name}[i];
      }}'''
        source=source.replace(old,new,1)
    marker='    float norm_log=lg2((lane&1)?norm_sum[1]:norm_sum[0]);\n'
    assert source.count(marker)==1
    source=source.replace(marker,marker+'''    float reciprocal_norm=reciprocal((lane&1)?norm_sum[1]:norm_sum[0]);
    float inverse_norm0=__shfl_sync(0xffffffff,reciprocal_norm,lane&~1);
    float inverse_norm1=__shfl_sync(0xffffffff,reciprocal_norm,(lane&~1)+1);
''',1)
    old='      winners[i]=best;winner_probabilities[i]=ex2(best);'
    assert source.count(old)==1
    source=source.replace(old,'''      winners[i]=best;
      winner_probabilities[i]=u0<=u1 ? __fmul_rn(cached_exp[ix],inverse_norm0)
                                    : __fmul_rn(cached_exp[ix+2],inverse_norm1);''',1)
    return '// Numerical-roundoff experiment: reuse normalized exponentials.\n'+source


def polynomial_winner_exponential(source: str, degree: int) -> str:
    """Move only the winner exponential to a small FP32 FMA polynomial."""
    assert degree in (5,6)
    coefficients=[math.log(2)**i/math.factorial(i) for i in range(degree+1)]
    helper='''__device__ __forceinline__ float winner_exp(float x) {
  // Match the retained exp2 FTZ domain and retain its exceptional fallback.
  if(x < -126.f) return 0.f;
  if(!(x <= 126.f)) return ex2(x);
  int n=__float2int_rn(x);
  float r=__fadd_rn(x,-float(n));
'''+f'  float p={coefficients[-1]:.17e}f;\n'
    for c in reversed(coefficients[:-1]):
        helper+=f'  p=__fmaf_rn(p,r,{c:.17e}f);\n'
    helper+='''  return __fmul_rn(p,__int_as_float((n+127)<<23));
}
'''
    source=source.replace('__device__ __forceinline__ float lg2(float x)',
                          helper+'__device__ __forceinline__ float lg2(float x)',1)
    old='winners[i]=best;winner_probabilities[i]=ex2(best);'
    assert source.count(old)==1
    return (f'// Numerical-roundoff experiment: degree-{degree} winner exp2.\n'+
            source.replace(old,'winners[i]=best;winner_probabilities[i]=winner_exp(best);',1))


def fuse_kv_product(source: str, residency: int) -> str:
    """One N256 product reads the adjacent K and V shared-memory tiles."""
    assert residency in (2,3)
    a=source.index('    float acc0[64];')
    b=source.index('    uint32_t output0[32];',a)
    c=source.index('    float acc1[64];',b)
    d=source.index('    uint32_t output1[',c)
    product='''    float acc_kv[128];
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    using MergeKV=MMA_64x256x16_F32BF16BF16_RS<Major::K,Major::MN>;
    #pragma unroll
    for(int ki=0;ki<80;ki+=16) {
      int wi=(ki/8)*2;
      // N64 panels are contiguous: K low/high, then V low/high. The
      // panel stride remains 80*64 BF16 elements, as in either N128 arm.
      MergeKV::fma(packed_weights[wi],packed_weights[wi+1],packed_weights[wi+2],packed_weights[wi+3],
          desc(sk+boff(ki,0),10240,1024),E128(acc_kv),ki==0?ScaleOut::Zero:ScaleOut::One);
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
'''
    source=source[:a]+product+source[b:c]+source[d:]
    source=re.sub(r'acc0\[([^\]]+)\]',r'acc_kv[\1]',source)
    source=re.sub(r'acc1\[([^\]]+)\]',r'acc_kv[64+\1]',source)
    macro='#define E128(a) '+','.join(f'E8(a,{i})' for i in range(0,128,8))+'\n'
    source=source.replace('__device__ __forceinline__ uint64_t desc(',
                          macro+'\n__device__ __forceinline__ uint64_t desc(',1)
    source=source.replace('__launch_bounds__(128,4)',f'__launch_bounds__(128,{residency})')
    return '// Fused K/V product; exact C8 capacity and per-output K16 accumulation order.\n'+source


def dynamic_register_kv(base: str, normal: int, handoff=False) -> str:
    """Two independent WGs borrow a CTA's spare registers for fused SS KV."""
    assert normal in (96,104)
    expanded=256-normal
    source=fuse_kv_product(make_packed_epilogue_pipeline(make_column_pipeline(base,False),False),2)
    assert '__launch_bounds__(256,2)' in source
    sync='    asm volatile("bar.sync %0,128;"::"r"(3+wg):"memory");\n'
    decrease=f'    asm volatile("setmaxnreg.dec.sync.aligned.u32 {normal};":::"memory");\n'+sync
    increase=f'    asm volatile("setmaxnreg.inc.sync.aligned.u32 {expanded};":::"memory");\n'+sync
    marker='  int total=Q*(C/8),start=blockIdx.z*tile_count;'
    assert source.count(marker)==1
    source=source.replace(marker,decrease+('    int iteration=0;\n' if handoff else '')+marker,1)
    stage='''    // Wait for each warp's previous TMA read before aliasing its scratch.
    if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
    __syncwarp();
    #pragma unroll
    for(int d=0;d<96;d+=32) {
      int i=d/4;
      uint4 a={packed_weights[i],packed_weights[i+2],
               i+4<20?packed_weights[i+4]:0u,i+6<20?packed_weights[i+6]:0u};
      uint4 b={packed_weights[i+1],packed_weights[i+3],
               i+5<20?packed_weights[i+5]:0u,i+7<20?packed_weights[i+7]:0u};
      int row=warp*16+seed,column=d+(lane&3)*8;
      *reinterpret_cast<uint4*>(temp+aoff(row,column))=fragment_transpose(a,lane);
      *reinterpret_cast<uint4*>(temp+aoff(row+8,column))=fragment_transpose(b,lane);
    }
'''+sync+'''    asm volatile("fence.proxy.async.shared::cta;":::"memory");
'''+increase
    source=source.replace('    float acc_kv[128];',stage+'    float acc_kv[128];',1)
    if handoff:
        # Do not use a scarce register pool as a WG mutex: independent warps
        # could otherwise acquire partial allocations and block each other.
        enter='''    if(wg==1) asm volatile("bar.sync 1,256;":::"memory");
    else if(iteration>0) asm volatile("bar.sync 2,256;":::"memory");
'''
        assert source.count(increase)==1
        source=source.replace(increase,enter+increase,1)
    source=source.replace('using MergeKV=MMA_64x256x16_F32BF16BF16_RS<Major::K,Major::MN>;',
                          'using MergeKV=MMA_64x256x16_F32BF16BF16_SS<Major::K,Major::MN>;',1)
    source=source.replace('MergeKV::fma(packed_weights[wi],packed_weights[wi+1],packed_weights[wi+2],packed_weights[wi+3],',
                          'MergeKV::fma(desc(temp+aoff(0,ki),16,1024),',1)
    end=source.index('  }\n  }\n  if(tma_store==1 && lane==0)')
    release='''    if(wg==0) asm volatile("bar.arrive 1,256;":::"memory");
    else if(qo+1<Q || span+2*Q<min(start+tile_count,total))
      asm volatile("bar.arrive 2,256;":::"memory");
    ++iteration;
''' if handoff else ''
    source=source[:end]+decrease+release+source[end:]
    prefix=f'// Dynamic register pool: {normal} for QK/scalars, {expanded} for SS KV.\n'
    if handoff:
        prefix='// Explicit release-to-acquire WG handoff precedes register expansion.\n'+prefix
    return prefix+source


def specialize_dynamic_groups(source: str) -> str:
    """Give the two groups distinct control-flow regions for register allocation."""
    begin=source.index('    asm volatile("setmaxnreg.dec.sync.aligned.u32')
    end=source.index('\n}\n',begin)
    body=source[begin:end]
    specialized=[]
    for group in (0,1):
        specialized.append(f'    constexpr int wg={group};\n'+body)
    source=source[:begin]+'  if(threadIdx.x<128) {\n'+specialized[0]+'\n  } else {\n'+specialized[1]+'\n  }'+source[end:]
    return '// Separate compile-time WG paths; original v34 handoff preserved.\n'+source


def interleave_normalization(source: str) -> str:
    """Stream the two independent left-fold sums without reassociation."""
    old='''    float norm_max[2],norm_sum[2];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float direct[20];
      #pragma unroll
      for(int i=0;i<20;++i)
        direct[i]=ex2(__fmul_rn(s[4*(i/2)+2*half+(i&1)],scale));
      float sum=reduce80(direct);
'''
    new='''    float norm_max[2],norm_sum[2];
    float direct_sum0=ex2(__fmul_rn(s[0],scale));
    float direct_sum1=ex2(__fmul_rn(s[2],scale));
    #pragma unroll
    for(int i=1;i<20;++i) {
      int ix=4*(i/2)+(i&1);
      direct_sum0=__fadd_rn(direct_sum0,ex2(__fmul_rn(s[ix],scale)));
      direct_sum1=__fadd_rn(direct_sum1,ex2(__fmul_rn(s[ix+2],scale)));
    }
    direct_sum0=sum4(direct_sum0);
    direct_sum1=sum4(direct_sum1);
    #pragma unroll
    for(int half=0;half<2;++half) {
      float sum=half==0?direct_sum0:direct_sum1;
'''
    assert source.count(old)==1
    return '// Interleaved normalization; exact original per-row reduction order.\n'+source.replace(old,new,1)


def interleave_winner_moments(source: str) -> str:
    """Accumulate both rows' sums/dots as each winner becomes available."""
    decl='    uint32_t oc0=(uint32_t)seed_for_row(warp*16+seed),oc1=oc0+4u;'
    assert source.count(decl)==1
    source=source.replace(decl,'''    float sums[2],dots[2],first_probability[2];
'''+decl,1)
    marker='      winners[i]=best;winner_probabilities[i]=ex2(best);'
    update='''
      #pragma unroll
      for(int half=0;half<2;++half) {
        float probability=labels[i]==int(oc0+4u*half)?winner_probabilities[i]:0.f;
        if(i==0) {
          sums[half]=probability;
          first_probability[half]=probability;
        } else {
          sums[half]=__fadd_rn(sums[half],probability);
          if(i==1) dots[half]=__fmaf_rn(first_probability[half],winners[0],__fmul_rn(probability,best));
          else dots[half]=__fmaf_rn(probability,best,dots[half]);
        }
      }'''
    assert source.count(marker)==1
    source=source.replace(marker,marker+update,1)
    a=source.index('    float sums[2],dots[2];')
    b=source.index('    float lane_sum=',a)
    source=source[:a]+'''    #pragma unroll
    for(int half=0;half<2;++half) {
      sums[half]=sum4(sums[half]);
      dots[half]=sum4(dots[half]);
    }
'''+source[b:]
    return '// Interleaved winner moments; original left-fold and FMA order.\n'+source


def trace_merge_phases(source: str, predicated=False) -> str:
    """Sample one steady-state iteration per column tile; diagnostic only."""
    helper='''__device__ __forceinline__ void trace_phase(float* output,int cb,int phase) {
  uint64_t timestamp;
  asm volatile("mov.u64 %0, %%clock64;":"=l"(timestamp)::"memory");
  int slot=(int(blockIdx.y)*945+int(blockIdx.x))*10+cb;
  reinterpret_cast<volatile uint64_t*>(output)[slot*7+phase]=timestamp;
}

'''
    if predicated:
        helper='''__device__ __forceinline__ void trace_phase(float* output,int cb,int phase,int sample) {
  int slot=(int(blockIdx.y)*945+int(blockIdx.x))*10+cb;
  uint64_t address=reinterpret_cast<uint64_t>(output)+uint64_t(slot*7+phase)*8;
  asm volatile("{ .reg .pred p; .reg .u64 timestamp;\\n"
               "setp.ne.u32 p,%0,0;\\n"
               "@p mov.u64 timestamp,%%clock64;\\n"
               "@p st.global.u64 [%1],timestamp; }"::"r"(sample),"l"(address):"memory");
}

'''
    source=source.replace('template<int TILES,int TOTAL_HEADS>',helper+'template<int TILES,int TOTAL_HEADS>',1)
    def mark(phase):
        if predicated:
            return f'    trace_phase(debug_scores_arg,cb,{phase},int(debug_scores_arg!=nullptr)&int(tid==0)&int(qo==10));\n'
        return f'    if(debug_scores_arg && tid==0 && qo==10) trace_phase(debug_scores_arg,cb,{phase});\n'
    source=source.replace('    float s[40];',mark(0)+'    float s[40];',1)
    wait='    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n'
    assert source.count(wait)==3
    parts=source.split(wait)
    source=parts[0]+wait+mark(1)+parts[1]+mark(2)+wait+mark(3)+parts[2]+mark(4)+wait+mark(5)+parts[3]
    tail='  }\n    { constexpr int qo=Q-1;'
    assert source.count(tail)==1
    source=source.replace(tail,mark(6)+tail,1)
    old=' if(debug.numel()) grid=dim3(1,1,1);'
    new=''' if(debug.numel()) TORCH_CHECK(debug.scalar_type()==at::kFloat &&
     debug.numel()==int64_t(ah)*ko*r*10*7*2,"Require complete clock trace storage");'''
    assert source.count(old)==1
    source=source.replace(old,new,1)
    return '// Instrumented thread-0 clock samples; timings are diagnostic only.\n'+source


def tma_store_cache_hint(source: str, fraction: float) -> str:
    """Prefer evicting streamed merged outputs over repeatedly reused queries."""
    assert fraction in (0.5,1.0)
    marker='  uint64_t cache_last,cache_first=0;'
    assert source.count(marker)==1
    source=source.replace(marker,'  uint64_t cache_last,cache_first;\n'+
        f'  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,{fraction};":"=l"(cache_first));',1)
    instruction='cp.async.bulk.tensor.5d.global.shared::cta.bulk_group '
    assert source.count(instruction)==3
    source=source.replace(instruction,instruction.rstrip()+'.L2::cache_hint ')
    assert source.count('[%6];"::')==3
    source=source.replace('[%6];"::','[%6],%7;"::')
    assert source.count('"r"(p):"memory");')==3
    source=source.replace('"r"(p):"memory");','"r"(p),"l"(cache_first):"memory");')
    return f'// Streaming TMA output L2 eviction hint, fraction={fraction}; unchanged synchronization.\n'+source


def separate_store_buffers(source: str) -> str:
    """Two alternating TMA groups protect separate K and V shared buffers."""
    pointer='  bf16* temp=sv+80*128;'
    assert source.count(pointer)==1
    source=source.replace(pointer,pointer+'\n  bf16* temp_v=temp+64*128;',1)
    ranges=[]
    a=source.index('    if(__all_sync(0xffffffff,qo>0)) {')
    b=source.index('    asm volatile("wgmma.wait_group.sync.aligned 0;',a)
    ranges.append((a,b))
    a=source.index('    uint32_t output1[16];')
    b=source.index('  if(tma_store==1 && lane==0)',a)
    ranges.append((a,b))
    for a,b in reversed(ranges):
        source=source[:a]+source[a:b].replace('temp+','temp_v+')+source[b:]
    wait='    if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");'
    assert source.count(wait)==2
    source=source.replace(wait,wait.replace('read 0','read 1'))
    old='(80*128*2+64*128)*2'
    assert source.count(old)==2
    source=source.replace(old,'(80*128*2+2*64*128)*2')
    return '''// Separate K/V store buffers, three resident CTAs by shared-memory capacity.
// Issued groups alternate K,V. Before reusing either buffer, wait for all
// groups except the newest (opposite buffer). The kernel exit drains all.
'''+source


def just_in_time_queries(source: str) -> str:
    """Limit Q fragment liveness to the query that immediately consumes it."""
    marker='        {\n          int pf='
    while marker in source:
        begin=source.index(marker)
        first=source.index('{',begin)
        depth=1
        end=first+1
        while depth:
            depth+=(source[end]=='{')-(source[end]=='}')
            end+=1
        source=source[:begin]+source[end:]
    begin=source.index('    uint32_t qreg[32];')
    loop=source.index('    for(int qo=',begin)
    loading=source[begin:loop]
    a=loading.index('    // Span prologue:')
    b=loading.index('    #pragma unroll',a)
    loading=loading[:a]+'    // Load only the query consumed by the following score product.\n'+loading[b:]
    source=source[:begin]+source[loop:]
    loop=source.index('    for(int qo=',begin)
    end=source.index('\n',loop)+1
    source=source[:end]+loading+source[end:]
    assert 'int pf=' not in source
    return source


def five_warpgroup_columns(source: str) -> str:
    """Five independent column groups share raw K/V, with one CTA per SM."""
    vote='+'.join(f'int(__all_sync(0xffffffff,threadIdx.x>={128*i}))' for i in range(1,5))
    source=source.replace('__launch_bounds__(128,4)','__launch_bounds__(640,1)',1)
    source=source.replace('bf16* temp=sv+80*128;',f'bf16* temp=sv+80*128+({vote})*64*128;',1)
    source=source.replace('int tid=threadIdx.x,lane=tid&31,warp=tid>>5,seed=lane>>2;',
                          f'int tid=threadIdx.x,wg={vote},lane=tid&31,warp=(tid%128)>>5,seed=lane>>2;',1)
    source=source.replace('for(int vec=tid;vec<80*16;vec+=128)',
                          'for(int vec=tid;vec<80*16;vec+=640)',1)
    source=source.replace('for(int span=start;span<min(start+tile_count,total);span+=Q)',
                          'for(int span=start+wg*Q;span<min(start+tile_count,total);span+=5*Q)',1)
    assert source.count('(80*128*2+64*128)*2')==2
    source=source.replace('(80*128*2+64*128)*2','(80*128*2+5*64*128)*2')
    source=source.replace('<<<grid,128,bytes,stream>>>','<<<grid,640,bytes,stream>>>',1)
    source=source.replace('merge_kernel<21,40>,128,bytes','merge_kernel<21,40>,640,bytes',1)
    return '// Five independent WGs, exact C8, balanced columns and just-in-time Q registers.\n'+source


def shared_query_per_group(source: str) -> str:
    """Reuse the output tile for SS QK after every warp's TMA read finishes."""
    begin=source.index('    uint32_t qreg[32];')
    end=source.index('    float s[40];',begin)
    stage='''    if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
    // Query swizzling crosses the old output-warp slices: synchronize the
    // whole WG after all four issuing lanes finish their previous TMA reads.
    asm volatile("bar.sync %0,128;"::"r"(1+wg):"memory");
    #pragma unroll
    for(int d=0;d<128;d+=32) {
      uint4 a=fragment_transpose(query_vector(q+qi0+d,cache_last,lane),lane);
      uint4 b=fragment_transpose(query_vector(q+qi1+d,cache_last,lane),lane);
      int column=d+(lane&3)*8;
      *reinterpret_cast<uint4*>(temp+aoff(row0,column))=a;
      *reinterpret_cast<uint4*>(temp+aoff(row0+8,column))=b;
    }
    asm volatile("bar.sync %0,128;"::"r"(1+wg):"memory");
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
'''
    source=source[:begin]+stage+source[end:]
    source=source.replace('using Score=MMA_64x80x16_F32BF16BF16_RS<Major::K,Major::K>;',
                          'using Score=MMA_64x80x16_F32BF16BF16_SS<Major::K,Major::K>;',1)
    source=source.replace('Score::fma(qreg[i],qreg[i+1],qreg[i+2],qreg[i+3],desc(sk+boff(0,d),16,1024),',
                          'Score::fma(desc(temp+aoff(0,d),16,1024),desc(sk+boff(0,d),16,1024),',1)
    assert 'qreg' not in source
    return '// Shared Q staging aliases the output buffer only after a full-WG read-completion barrier.\n'+source


def six_warpgroup_frame_columns(source: str) -> str:
    """Two column lanes times three frame lanes give 35 queries per WG."""
    old='+'.join(f'int(__all_sync(0xffffffff,threadIdx.x>={128*i}))' for i in range(1,5))
    new=old+'+int(__all_sync(0xffffffff,threadIdx.x>=640))'
    assert source.count(old)==2
    source=source.replace(old,new)
    source=source.replace('__launch_bounds__(640,1)','__launch_bounds__(768,1)',1)
    source=source.replace('vec+=640','vec+=768',1)
    source=source.replace('start+wg*Q;span<min(start+tile_count,total);span+=5*Q',
                          'start+(wg/3)*Q;span<min(start+tile_count,total);span+=2*Q',1)
    loop='    for(int qo=0;qo<Q;++qo,qi0+=R*C*H*128,qi1+=R*C*H*128) {'
    assert source.count(loop)==1
    source=source.replace(loop,'''    qi0+=(wg%3)*R*C*H*128;qi1+=(wg%3)*R*C*H*128;
    for(int qo=wg%3;qo<Q;qo+=3,qi0+=3*R*C*H*128,qi1+=3*R*C*H*128) {''',1)
    source=source.replace('+5*64*128','+6*64*128')
    source=source.replace('<<<grid,640,bytes,stream>>>','<<<grid,768,bytes,stream>>>',1)
    source=source.replace('merge_kernel<21,40>,640,bytes','merge_kernel<21,40>,768,bytes',1)
    source=source.replace(' int qo=q.size(1)',' tiles=210; // Ten columns by 21 frames divide evenly over six WGs.\n int qo=q.size(1)',1)
    return '// Six independent WGs; two column lanes by three frame lanes.\n'+source


def four_warpgroup_two_ctas(source: str) -> str:
    """Two column lanes times two frame lanes; target eight resident WGs."""
    old='+'.join(f'int(__all_sync(0xffffffff,threadIdx.x>={128*i}))' for i in range(1,5))
    new='+'.join(f'int(__all_sync(0xffffffff,threadIdx.x>={128*i}))' for i in range(1,4))
    assert source.count(old)==2
    source=source.replace(old,new)
    source=source.replace('__launch_bounds__(640,1)','__launch_bounds__(512,2)',1)
    source=source.replace('vec+=640','vec+=512',1)
    source=source.replace('start+wg*Q;span<min(start+tile_count,total);span+=5*Q',
                          'start+(wg/2)*Q;span<min(start+tile_count,total);span+=2*Q',1)
    loop='    for(int qo=0;qo<Q;++qo,qi0+=R*C*H*128,qi1+=R*C*H*128) {'
    assert source.count(loop)==1
    source=source.replace(loop,'''    qi0+=(wg%2)*R*C*H*128;qi1+=(wg%2)*R*C*H*128;
    for(int qo=wg%2;qo<Q;qo+=2,qi0+=2*R*C*H*128,qi1+=2*R*C*H*128) {''',1)
    source=source.replace('+5*64*128','+4*64*128')
    source=source.replace('<<<grid,640,bytes,stream>>>','<<<grid,512,bytes,stream>>>',1)
    source=source.replace('merge_kernel<21,40>,640,bytes','merge_kernel<21,40>,512,bytes',1)
    source=source.replace(' int qo=q.size(1)',' tiles=210; // Two column lanes and two frame lanes cover all queries.\n int qo=q.size(1)',1)
    return '// Four independent WGs; target two CTAs per SM with a 64-register budget.\n'+source


def rematerialize_query_addresses(source: str) -> str:
    """Avoid keeping eight per-thread shared Q addresses across the main loop."""
    old='''    for(int d=0;d<128;d+=32) {
      uint4 a=fragment_transpose(query_vector(q+qi0+d,cache_last,lane),lane);
      uint4 b=fragment_transpose(query_vector(q+qi1+d,cache_last,lane),lane);
      int column=d+(lane&3)*8;
      *reinterpret_cast<uint4*>(temp+aoff(row0,column))=a;
      *reinterpret_cast<uint4*>(temp+aoff(row0+8,column))=b;
    }'''
    new='''    for(int d=0;d<128;d+=32) {
      unsigned stage_tid;
      // Volatile S2R prevents loop-invariant address hoisting and spilling.
      asm volatile("mov.u32 %0, %%tid.x;":"=r"(stage_tid));
      int stage_lane=stage_tid&31,stage_warp=(stage_tid&127)>>5;
      int stage_row=stage_warp*16+(stage_lane>>2),column=d+(stage_lane&3)*8;
      bf16* stage_temp=reinterpret_cast<bf16*>(raw)+2*80*128+(stage_tid>>7)*64*128;
      uint4 a=fragment_transpose(query_vector(q+qi0+d,cache_last,stage_lane),stage_lane);
      uint4 b=fragment_transpose(query_vector(q+qi1+d,cache_last,stage_lane),stage_lane);
      *reinterpret_cast<uint4*>(stage_temp+aoff(stage_row,column))=a;
      *reinterpret_cast<uint4*>(stage_temp+aoff(stage_row+8,column))=b;
    }'''
    assert source.count(old)==1
    return '// Recompute shared Q addresses at use instead of retaining them across iterations.\n'+source.replace(old,new,1)


def stage_half_scores(source: str) -> str:
    """Use N40 score banks and stage the first half in private warp scratch."""
    source=source.replace('#define E32(a)',
        '#define E20(a) E8(a,0),E8(a,8),a[16],a[17],a[18],a[19]\n#define E32(a)',1)
    begin=source.index('    float s[40];')
    end=source.index('    if(debug_scores)',begin)
    score='''    float s[40];
    using Score=MMA_64x40x16_F32BF16BF16_RS<Major::K,Major::K>;
    #pragma unroll
    for(int half=0;half<2;++half) {
      float scores[20];
      asm volatile("wgmma.fence.sync.aligned;":::"memory");
      #pragma unroll
      for(int d=0;d<128;d+=16) {
        int i=d/4;
        Score::fma(qreg[i],qreg[i+1],qreg[i+2],qreg[i+3],
          desc(sk+boff(half*40,d),16,1024),E20(scores),d==0?ScaleOut::Zero:ScaleOut::One);
      }
      asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
      // Each warp uses only its own previous 4-KiB output slice. The
      // transposed 20x32 FP32 scratch occupies 2560 bytes without bank conflicts.
      if(half==0) {
        if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
        __syncwarp();
        #pragma unroll
        for(int i=0;i<20;++i) {
          uint32_t address=__cvta_generic_to_shared(temp+warp*2048)+i*128+lane*4;
          asm volatile("st.shared.f32 [%0],%1;"::"r"(address),"f"(scores[i]):"memory");
        }
      } else {
        #pragma unroll
        for(int i=0;i<20;++i) {
          uint32_t address=__cvta_generic_to_shared(temp+warp*2048)+i*128+lane*4;
          float first;
          asm volatile("ld.shared.f32 %0,[%1];":"=f"(first):"r"(address):"memory");
          s[i]=first;s[i+20]=scores[i];
        }
      }
    }
'''
    return '// N40 score halves retain the original K16 accumulation order and exact FP32 bits.\n'+source[:begin]+score+source[end:]


def quarter_score_registers(source: str) -> str:
    """Combine N40 score staging with ordered K64 steps and sixteen Q words."""
    begin=source.index('    uint32_t qreg[32];')
    end=source.index('    float s[40];',begin)
    source=source[:begin]+source[end:]
    begin=source.index('      asm volatile("wgmma.fence.sync.aligned;',begin)
    end=source.index('      // Each warp uses only its own previous',begin)
    score='''      #pragma unroll
      for(int part=0;part<2;++part) {
        uint32_t qreg[16];
        #pragma unroll
        for(int d=0;d<64;d+=32) {
          uint4 a=query_vector(q+qi0+part*64+d,cache_last,lane);
          uint4 b=query_vector(q+qi1+part*64+d,cache_last,lane);
          int i=d/4;
          qreg[i]=a.x;qreg[i+1]=b.x;qreg[i+2]=a.y;qreg[i+3]=b.y;
          qreg[i+4]=a.z;qreg[i+5]=b.z;qreg[i+6]=a.w;qreg[i+7]=b.w;
        }
        #pragma unroll
        for(int i=0;i<16;++i) asm volatile("" : "+r"(qreg[i]) :: "memory");
        asm volatile("wgmma.fence.sync.aligned;":::"memory");
        #pragma unroll
        for(int d=0;d<64;d+=16) {
          int i=d/4;
          Score::fma(qreg[i],qreg[i+1],qreg[i+2],qreg[i+3],
            desc(sk+boff(half*40,part*64+d),16,1024),E20(scores),part==0 && d==0?ScaleOut::Zero:ScaleOut::One);
        }
        asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
      }
'''
    return '// Ordered K64-by-N40 score quarters; sixteen live Q words.\n'+source[:begin]+score+source[end:]


def compact_shared_weights(base: str, groups: int, normalization: bool, shared_query=False) -> str:
    """Keep exact BF16 weights in 10 KiB and N64 TMA output in another 8 KiB."""
    assert groups in (3,4,6)
    source=just_in_time_queries(make_fragment_pipeline(base,False,64,compact=True,query_prefetch=False))
    if normalization:
        source=interleave_normalization(source)
    source=pack_owner_labels(rematerialize_owner_scale(interleave_winner_moments(source)))
    if not shared_query:
        source=quarter_score_registers(stage_half_scores(source))
    helpers='''// Dense K-major interleaved 8x8 atoms: 64 rows by 80 BF16 weights.
__device__ __forceinline__ int woff(int row,int k) {
  return (row&7)*8+(k&7)+(row>>3)*640+(k>>3)*64;
}
__device__ __forceinline__ uint64_t weight_desc(const bf16* p) {
  uint64_t s=__cvta_generic_to_shared(p);
  return ((s&0x3ffff)>>4) | (uint64_t(128>>4)<<16) | (uint64_t(1280>>4)<<32);
}
'''
    source=source.replace('__device__ __forceinline__ int col_for_row',helpers+'__device__ __forceinline__ int col_for_row',1)
    vote='+'.join(f'int(__all_sync(0xffffffff,threadIdx.x>={128*i}))' for i in range(1,groups))
    threads=128*groups
    frames=3 if groups==3 else groups//2
    source=source.replace('__launch_bounds__(128,4)',f'__launch_bounds__({threads},{2 if groups in (3,4) else 1})',1)
    source=source.replace('bf16* temp=sv+80*128;',
        f'bf16* weight=sv+80*128+({vote})*(64*80+64*64);\n  bf16* temp=weight+64*80;',1)
    source=source.replace('int tid=threadIdx.x,lane=tid&31,warp=tid>>5,seed=lane>>2;',
        f'int tid=threadIdx.x,wg={vote},lane=tid&31,warp=(tid%128)>>5,seed=lane>>2;',1)
    source=source.replace('vec+=128)',f'vec+={threads})',1)
    if groups!=3:
        source=source.replace('for(int span=start;span<min(start+tile_count,total);span+=Q)',
            f'for(int span=start+(wg/{frames})*Q;span<min(start+tile_count,total);span+=2*Q)',1)
    loop='    for(int qo=0;qo<Q;++qo,qi0+=R*C*H*128,qi1+=R*C*H*128) {'
    assert source.count(loop)==1
    source=source.replace(loop,f'''    qi0+=(wg%{frames})*R*C*H*128;qi1+=(wg%{frames})*R*C*H*128;
    for(int qo=wg%{frames};qo<Q;qo+={frames},qi0+={frames}*R*C*H*128,qi1+={frames}*R*C*H*128) {{''',1)
    if groups==3:
        source=source.replace('wg%3','wg')
    source=source.replace('temp+warp*2048','weight+warp*1280')
    source=source.replace('    uint32_t packed_weights[20];\n','',1)
    source=source.replace('        packed_weights[2*i+half]=pair.bits;',
        '        *reinterpret_cast<uint32_t*>(weight+woff(row,8*i+2*(lane&3)))=pair.bits;',1)
    source=source.replace('    if(stop_after==2) continue;', '''    if(stop_after==2) continue;
    // The weight layout crosses warp boundaries; publish the whole WG.
    asm volatile("bar.sync %0,128;"::"r"(1+wg):"memory");
    asm volatile("fence.proxy.async.shared::cta;":::"memory");''',1)
    source=source.replace('MMA_64x64x16_F32BF16BF16_RS<Major::K,Major::MN>',
                          'MMA_64x64x16_F32BF16BF16_SS<Major::K,Major::MN>')
    old='Merge::fma(packed_weights[wi],packed_weights[wi+1],packed_weights[wi+2],packed_weights[wi+3],bd,'
    assert source.count(old)==4
    source=source.replace(old,'Merge::fma(weight_desc(weight+woff(0,ki)),bd,')
    assert 'packed_weights[' not in source
    source=source.replace('(80*128*2+64*64)*2',f'(80*128*2+{groups}*(64*80+64*64))*2')
    source=source.replace('<<<grid,128,bytes,stream>>>',f'<<<grid,{threads},bytes,stream>>>',1)
    source=source.replace('merge_kernel<21,40>,128,bytes',f'merge_kernel<21,40>,{threads},bytes',1)
    source=source.replace(' int qo=q.size(1)',' tiles=210; // Two column lanes and independent frame lanes.\n int qo=q.size(1)',1)
    if shared_query:
        begin=source.index('    uint32_t qreg[32];')
        end=source.index('    float s[40];',begin)
        loading='''    // The temporary Q matrix spans weights and part of the output buffer.
    // Retire each warp's prior TMA reads before any warp writes this Q tile.
    if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
    asm volatile("bar.sync %0,128;"::"r"(1+wg):"memory");
    #pragma unroll
    for(int d=0;d<128;d+=32) {
      unsigned stage_tid;
      asm volatile("mov.u32 %0, %%tid.x;":"=r"(stage_tid));
      int stage_lane=stage_tid&31,stage_warp=(stage_tid&127)>>5;
      int stage_row=stage_warp*16+(stage_lane>>2),column=d+(stage_lane&3)*8;
      bf16* stage_temp=reinterpret_cast<bf16*>(raw)+2*80*128+(stage_tid>>7)*(64*80+64*64);
      uint4 a=fragment_transpose(query_vector(q+qi0+d,cache_last,stage_lane),stage_lane);
      uint4 b=fragment_transpose(query_vector(q+qi1+d,cache_last,stage_lane),stage_lane);
      *reinterpret_cast<uint4*>(stage_temp+aoff(stage_row,column))=a;
      *reinterpret_cast<uint4*>(stage_temp+aoff(stage_row+8,column))=b;
    }
    asm volatile("bar.sync %0,128;"::"r"(1+wg):"memory");
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
'''
        source=source[:begin]+loading+source[end:]
        source=source.replace('MMA_64x80x16_F32BF16BF16_RS<Major::K,Major::K>',
                              'MMA_64x80x16_F32BF16BF16_SS<Major::K,Major::K>',1)
        old='Score::fma(qreg[i],qreg[i+1],qreg[i+2],qreg[i+3],desc(sk+boff(0,d),16,1024),'
        assert source.count(old)==1
        source=source.replace(old,'Score::fma(desc(weight+aoff(0,d),16,1024),desc(sk+boff(0,d),16,1024),',1)
        assert 'qreg[' not in source
    return f'// {groups} WGs with compact shared BF16 weights and independent N64 output buffers.\n'+source


def rotate_half_tma_buffers(base: str, pending: int) -> str:
    """Two 8-KiB TMA sources share v23's existing 16-KiB output storage."""
    assert pending in (0,1)
    source=rotate_packed_epilogue(base)

    def stage(kind,part,values,offset,query):
        return f'''    if(lane==0) asm volatile("cp.async.bulk.wait_group.read {pending};":::"memory");
    __syncwarp();
    {{
      bf16* buf=temp+warp*2048+{part*1024};
      #pragma unroll
      for(int g=0;g<4;++g) {{
        int row=warp*16+(lane&7)+8*((lane>>3)&1);
        int canonical=col_for_row(row)*8+seed_for_row(row)-warp*16;
        int d=g*16+8*(lane>>4),offset=canonical*64+d;
        uint32_t sp=__cvta_generic_to_shared(buf+(offset^(((offset>>6)&7)<<3)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{{%1,%2,%3,%4}};"::
          "r"(sp),"r"({values}[{offset}+4*g]),"r"({values}[{offset}+4*g+1]),
          "r"({values}[{offset}+4*g+2]),"r"({values}[{offset}+4*g+3]):"memory");
      }}
    }}
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncwarp();
    if(lane==0) {{
      #pragma unroll
      for(int w=0;w<4;++w) if(warp==w) {{
        uint32_t p=__cvta_generic_to_shared(temp+w*2048+{part*1024});
        asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {{%1,%2,%3,%4,%5}}], [%6];"::
          "l"(&{'map_k' if kind==0 else 'map_v'}),"r"({part*64}),"r"(0),"r"(kb),"r"(head),
          "r"(({query})*C+cb*8+2*w),"r"(p):"memory");
        asm volatile("cp.async.bulk.commit_group;":::"memory");
      }}
    }}
'''
    begin=source.index('    if(__all_sync(0xffffffff,qo>0)) {')
    end=source.index('    asm volatile("wgmma.wait_group.sync.aligned 0;',begin)
    source=source[:begin]+'''    if(__all_sync(0xffffffff,qo>0)) {
    int previous_qo=qo-1;
'''+stage(1,1,'carried',0,'previous_qo')+'    }\n'+source[end:]
    begin=source.index('    if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;',source.index('    uint32_t output0[32];'))
    end=source.index('    { constexpr int kind=0;\n        {\n          int pf=',begin)
    source=source[:begin]+stage(0,0,'output0',0,'qo')+stage(0,1,'output0',16,'qo')+source[end:]
    begin=source.index('    if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;',source.index('    uint32_t output1[16];'))
    end=source.index('    { constexpr int kind=1;\n        {\n          int pf=',begin)
    source=source[:begin]+stage(1,0,'output1',0,'qo')+source[end:]
    begin=source.index('    { constexpr int qo=Q-1;')
    end=source.index('  if(tma_store==1 && lane==0)',begin)
    source=source[:begin]+'    { constexpr int qo=Q-1;\n'+stage(1,1,'carried',0,'qo')+'    }\n  }\n'+source[end:]
    old='uint64_t dims[5]={64,uint64_t(ko*r*16),uint64_t(ah),80,uint64_t(qo)};'
    assert source.count(old)==1
    source=source.replace(old,'uint64_t dims[5]={128,8,uint64_t(ko*r),uint64_t(ah),uint64_t(qo*80)};',1)
    source=source.replace('uint64_t stride=128;','uint64_t stride=256;',1)
    source=source.replace('uint32_t box[5]={64,16,1,2,1}','uint32_t box[5]={64,8,1,1,2}',1)
    return f'// Alternating half-output TMA buffers; allow {pending} more recent read group.\n'+source


def overlap_bias_epilogue(base: str, mode: str) -> str:
    """Overlap completed bias stores or exact per-row correction with K MMA."""
    assert mode in ('stores', 'lane_early', 'lane_late')
    source=rotate_packed_epilogue(base)
    begin=source.index('      float log_safe=__shfl_sync(0xffffffff,lane_log,')
    end=source.index('      #pragma unroll\n      for(int i=0;i<10;++i)',begin)
    old=source[begin:end]
    row='      int row=warp*16+seed+8*half;\n'
    if mode=='stores':
        replacement=old[:old.index('      int row=')]
        replacement+='      bias_pending[half]=sums[half]>0?correction:CUDART_INF_F;\n'+row
        source=source[:begin]+replacement+source[end:]
        source=source.replace('    float lane_sum=', '    float bias_pending[2];\n    float lane_sum=',1)
        deferred='''    #pragma unroll
    for(int half=0;half<2;++half) {
      int row=warp*16+seed+8*half,col=cb*8+col_for_row(row);
      int out=((qo*C+col)*AH+head)*(KO*R*8)+kb*8+seed_for_row(row);
      if((lane&3)==0) bias[out]=bias_pending[half];
    }
'''
    else:
        source=source[:begin]+row+source[end:]
        marker='    float lane_inv=reciprocal(lane_safe),lane_log=lg2(lane_safe);\n'
        assert source.count(marker)==1
        source=source.replace(marker,marker+'''    // Retain the original storing lane's high-half dot, even for edge
    // cases whose reduction bit pattern could depend on lane order.
    float high_dot=__shfl_sync(0xffffffff,dots[1],lane&~3);
    float lane_dot=(lane&1)?high_dot:dots[0];
    bool lane_nonempty=lane_sum>0;
''',1)
        deferred='''    {
      float correction=__fmul_rn(__fadd_rn(__fmul_rn(lane_dot,lane_inv),-lane_log),0.6931471805599453f);
      int half=lane&1,row=warp*16+seed+8*half,col=cb*8+col_for_row(row);
      int out=((qo*C+col)*AH+head)*(KO*R*8)+kb*8+seed_for_row(row);
      if((lane&3)<2) bias[out]=lane_nonempty?correction:CUDART_INF_F;
    }
'''
    if mode=='lane_early':
        marker='    if(stop_after==2) continue;\n'
        assert source.count(marker)==1
        source=source.replace(marker,deferred+marker,1)
    else:
        begin=source.index('    float acc0[64];')
        wait=source.index('    asm volatile("wgmma.wait_group.sync.aligned 0;',begin)
        source=source[:wait]+deferred+source[wait:]
    return f'// Exact bias epilogue scheduling: {mode}.\n'+source


def group_tma_issuers(base: str, warps: int, halves: bool) -> str:
    """Combine adjacent warps' output boxes behind a group-local barrier."""
    assert warps in (2, 4)
    source=rotate_half_tma_buffers(base,1) if halves else rotate_packed_epilogue(base)
    begin=source.index('__global__ __launch_bounds__(128,4) void merge_kernel(')
    end=source.index('// Pack each 32-BF16',begin)
    body=source[begin:end]
    assert body.count('for(int w=0;w<4;++w)')==(5 if halves else 3)
    body=body.replace('for(int w=0;w<4;++w)',f'for(int w=0;w<4;w+={warps})')
    barrier=f'asm volatile("bar.sync %0,{warps*32};"::"r"(1+warp/{warps}):"memory");'
    body=body.replace('__syncwarp();',barrier)
    if halves:
        for part in (0,1):
            body=body.replace(f'bf16* buf=temp+warp*2048+{part*1024};',
                f'bf16* buf=temp+(warp/{warps})*{warps*2048}+{part*warps*1024}+(warp%{warps})*1024;')
            body=body.replace(f'__cvta_generic_to_shared(temp+w*2048+{part*1024})',
                f'__cvta_generic_to_shared(temp+w*2048+{part*warps*1024})')
    else:
        body=body.replace('// Four issue sites with COMPILE-TIME-CONSTANT operands. Written as',
                          f'// {4//warps} issue sites with COMPILE-TIME-CONSTANT operands. Written as')
        body=body.replace('// Exactly one predicate is true per warp and selects the same\n'
                          '          // (address, column) pair the warp-indexed form computed.',
                          '// Only the first warp in each group issues its combined box.')
    source=source[:begin]+body+source[end:]
    old='uint32_t box[5]={64,8,1,1,2}' if halves else 'uint32_t box[5]={64,16,1,2,1}'
    new=f'uint32_t box[5]={{64,8,1,1,{2*warps}}}' if halves else f'uint32_t box[5]={{64,16,1,{2*warps},1}}'
    assert source.count(old)==1
    source=source.replace(old,new,1)
    return f'// Combine TMA sources for {warps} adjacent warps; half buffers={halves}.\n'+source


def split_shared_score_64_16(source: str) -> str:
    """Use separate N64/N16 accumulators instead of one N80 register bank."""
    begin=source.index('    float s[40];')
    end=source.index('    if(debug_scores)',begin)
    score='    float s[40];\n'
    for width,offset in ((64,0),(16,64)):
        operands='E32(s)' if width==64 else 'E8(s,32)'
        score+=f'''    {{
      asm volatile("wgmma.fence.sync.aligned;":::"memory");
      using Score=MMA_64x{width}x16_F32BF16BF16_SS<Major::K,Major::K>;
      #pragma unroll
      for(int d=0;d<128;d+=16) {{
        Score::fma(desc(weight+aoff(0,d),16,1024),desc(sk+boff({offset},d),16,1024),
          {operands},d==0?ScaleOut::Zero:ScaleOut::One);
      }}
      asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
    }}
'''
    return '// N64 + N16 score banks retain the original per-element K16 accumulation order.\n'+source[:begin]+score+source[end:]


def specialize_native_h40(source: str) -> str:
    """Keep the requested H40 path while excluding the optional H1 diagnostic."""
    old='TORCH_CHECK((h==1 && ah==1) || (h==40 && ah==5),"Require native H40/AH5 or diagnostic H1/AH1");'
    assert source.count(old)==1
    source=source.replace(old,'TORCH_CHECK(h==40 && ah==5,"Require native H40/AH5");',1)
    old='''   if(h==40) launch_heads(std::integral_constant<int,40>{});
   else launch_heads(std::integral_constant<int,1>{});'''
    assert source.count(old)==1
    return '// Native H40 only; the optional H1 diagnostic is not instantiated.\n'+source.replace(
        old,'   launch_heads(std::integral_constant<int,40>{});',1)


def outline_stable_normalizer(source: str) -> str:
    """Move the rare, unchanged stable-exp fallback outside the hot loop."""
    begin=source.index('      float mx=-CUDART_INF_F;')
    end=source.index('      norm_max[half]=mx;norm_sum[half]=reduce80(bs);',begin)
    end+=len('      norm_max[half]=mx;norm_sum[half]=reduce80(bs);')
    params=','.join(f'float a{i}' for i in range(20))
    values=','.join(f'a{i}' for i in range(20))
    helper=f'''__device__ __noinline__ float2 stable_normalizer({params},float scale) {{
  float a[20]={{{values}}};
  float mx=-CUDART_INF_F;
  #pragma unroll
  for(int i=0;i<10;++i) mx=fmaxf(mx,fmaxf(__fmul_rn(a[2*i],scale),__fmul_rn(a[2*i+1],scale)));
  mx=max4(mx);
  float bs[20];
  #pragma unroll
  for(int i=0;i<20;++i) bs[i]=ex2(__fmaf_rn(a[i],scale,-mx));
  return make_float2(mx,reduce80(bs));
}}

'''
    args=','.join(f's[{(i//2)*4}+2*half+{i%2}]' for i in range(20))
    call=f'      float2 stable=stable_normalizer({args},scale);\n      norm_max[half]=stable.x;norm_sum[half]=stable.y;'
    source=source[:begin]+call+source[end:]
    marker='// K/V stay resident across query tiles.'
    assert source.count(marker)==1
    return '// Unchanged stable-normalization fallback outlined to reduce hot code footprint.\n'+source.replace(marker,helper+marker,1)


def reorder_pipeline_blocks(source: str, order: str) -> str:
    """Change only CTA traversal, retaining the exact per-CTA pipeline."""
    assert order in ('head','span','head_span')
    old='  int block=blockIdx.x,head=blockIdx.y,gh=HS+head;'
    assert source.count(old)==1
    if order=='head':
        index='int head=linear%AH,block=(linear/AH)%(KO*R),slice=linear/(AH*KO*R);'
    elif order=='span':
        index='int slice=linear%slices,block=(linear/slices)%(KO*R),head=linear/(slices*KO*R);'
    else:
        index='int slice=linear%slices,head=(linear/slices)%AH,block=linear/(slices*AH);'
    source=source.replace(old,'  int linear=blockIdx.x,slices=(Q*(C/8)+tile_count-1)/tile_count;\n  '+index+'\n  int gh=HS+head;',1)
    source=source.replace('start=blockIdx.z*tile_count','start=slice*tile_count',1)
    source=source.replace('dim3 grid(ko*r,ah,(qo*10+tiles-1)/tiles);',
                          'dim3 grid(ko*r*ah*((qo*10+tiles-1)/tiles));',1)
    return f'// CTA traversal order: {order}; per-CTA arithmetic and overlap unchanged.\n'+source


def issue_value_before_key_store(source: str, prefetch: bool) -> str:
    """Launch V after K shared staging, overlap only TMA issue and optional Q loads."""
    begin=source.index('    float acc1[64];')
    end=source.index('    asm volatile("wgmma.wait_group.sync.aligned 0;',begin)
    issue=source[begin:end]
    source=source[:begin]+source[end:]
    marker='    { constexpr int kind=0;\n        asm volatile("fence.proxy.async.shared::cta;'
    assert source.count(marker)==1
    source=source.replace(marker,issue+marker,1)
    if not prefetch:
        begin=source.index('    { constexpr int kind=0;\n        {\n          int pf=')
        end=source.index('    asm volatile("wgmma.wait_group.sync.aligned 0;',begin)
        query=source[begin:end]
        source=source[:begin]+source[end:]
        marker='    uint32_t output1[16];'
        assert source.count(marker)==1
        source=source.replace(marker,query+marker,1)
    return f'// Issue V after K shared staging; overlap TMA and Q prefetch={prefetch}.\n'+source


def stagger_cta_start(source: str, gap: int) -> str:
    """Test initial phase spacing without adding a steady-state handshake."""
    assert gap in (1024,2048)
    marker='  int total=Q*(C/8),start=blockIdx.z*tile_count;'
    assert source.count(marker)==1
    delay=f'''  // ID-based startup spacing is a heuristic; no block-to-SM mapping is assumed.
  if(tid==0) {{
    uint64_t beginning,current;
    uint64_t delay=uint64_t(blockIdx.x%5)*{gap};
    asm volatile("mov.u64 %0, %%clock64;":"=l"(beginning)::"memory");
    do {{
      asm volatile("mov.u64 %0, %%clock64;":"=l"(current)::"memory");
      if(current-beginning>=delay) break;
      asm volatile("nanosleep.u32 64;");
    }} while(true);
  }}
  __syncthreads();
'''
    return f'// One-time CTA phase-spacing experiment, {gap} SM clocks per class.\n'+source.replace(marker,delay+marker,1)


if __name__ == "__main__":
    manifest=main()
    base=BASE.read_text(encoding='utf-8')
    additional=[]
    for pingpong in (False,True):
        name=f"fisher_c8_two_wg_{'pingpong' if pingpong else 'independent'}_v1.cu"
        (ROOT/'native_fisher'/name).write_text(make_two_warpgroup(base,pingpong),encoding='utf-8',newline='\n')
        additional.append(name)
    for name,pingpong,all_stages,tiles in (
        ('fisher_c8_two_wg_allstages_v2.cu',True,True,None),
        ('fisher_c8_two_wg_independent_t210_v2.cu',False,False,210),
        ('fisher_c8_two_wg_allstages_t210_v2.cu',True,True,210)):
        (ROOT/'native_fisher'/name).write_text(make_two_warpgroup(base,pingpong,all_stages,tiles),encoding='utf-8',newline='\n')
        additional.append(name)
    for overlap in (False,True):
        name=f"fisher_c8_fragment64_{'overlap' if overlap else 'serial'}_v3.cu"
        (ROOT/'native_fisher'/name).write_text(make_fragment_pipeline(base,overlap),encoding='utf-8',newline='\n')
        additional.append(name)
    name='fisher_c8_pair_packed_uniform_v22.cu'
    source=uniform_pair_index(make_pair_packed_pipeline(base,False))
    (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
    additional.append(name)
    for label,candidate in (('serial',base),('rotated',rotate_packed_epilogue(base))):
        name=f'fisher_c8_reuse_exponential_{label}_v30.cu'
        (ROOT/'native_fisher'/name).write_text(reuse_normalization_exponents(candidate),encoding='utf-8',newline='\n')
        additional.append(name)
    for degree in (5,6):
        name=f'fisher_c8_polynomial_exp{degree}_rotated_v31.cu'
        (ROOT/'native_fisher'/name).write_text(polynomial_winner_exponential(rotate_packed_epilogue(base),degree),encoding='utf-8',newline='\n')
        additional.append(name)
    for residency in (2,3):
        name=f'fisher_c8_fused_kv_rotated_lb{residency}_v32.cu'
        (ROOT/'native_fisher'/name).write_text(fuse_kv_product(rotate_packed_epilogue(base),residency),encoding='utf-8',newline='\n')
        additional.append(name)
    for normal in (96,104):
        name=f'fisher_c8_dynamic_registers_{normal}_v33.cu'
        (ROOT/'native_fisher'/name).write_text(dynamic_register_kv(base,normal),encoding='utf-8',newline='\n')
        additional.append(name)
        name=f'fisher_c8_dynamic_handoff_{normal}_v34.cu'
        (ROOT/'native_fisher'/name).write_text(dynamic_register_kv(base,normal,True),encoding='utf-8',newline='\n')
        additional.append(name)
    name='fisher_c8_dynamic_specialized_96_v35.cu'
    (ROOT/'native_fisher'/name).write_text(specialize_dynamic_groups(dynamic_register_kv(base,96,True)),encoding='utf-8',newline='\n')
    additional.append(name)
    for pending in (0,1):
        name=f'fisher_c8_half_tma_wait{pending}_v58.cu'
        (ROOT/'native_fisher'/name).write_text(rotate_half_tma_buffers(base,pending),encoding='utf-8',newline='\n')
        additional.append(name)
    for warps in (2,4):
        for halves in (False,True):
            name=f"fisher_c8_grouped_tma_{warps}w_{'half' if halves else 'full'}_v59.cu"
            (ROOT/'native_fisher'/name).write_text(group_tma_issuers(base,warps,halves),encoding='utf-8',newline='\n')
            additional.append(name)
    for mode in ('stores','lane_early','lane_late'):
        name=f'fisher_c8_bias_{mode}_v60.cu'
        (ROOT/'native_fisher'/name).write_text(overlap_bias_epilogue(base,mode),encoding='utf-8',newline='\n')
        additional.append(name)
    name='fisher_c8_shared_operands_4wg_h40_v56.cu'
    (ROOT/'native_fisher'/name).write_text(specialize_native_h40(compact_shared_weights(base,4,False,True)),encoding='utf-8',newline='\n')
    additional.append(name)
    for normalization in (False,True):
        name=f"fisher_c8_split_shared_score_{'both' if normalization else 'moments'}_v57.cu"
        source=specialize_native_h40(split_shared_score_64_16(compact_shared_weights(base,4,normalization,True)))
        (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
        additional.append(name)
    for normalization in (False,True):
        name=f"fisher_c8_shared_operands_3wg_{'both' if normalization else 'moments'}_v55.cu"
        (ROOT/'native_fisher'/name).write_text(compact_shared_weights(base,3,normalization,True),encoding='utf-8',newline='\n')
        additional.append(name)
    for groups,normalization in ((4,False),(4,True),(6,False)):
        name=f"fisher_c8_shared_weights_{groups}wg_{'both' if normalization else 'moments'}_v53.cu"
        (ROOT/'native_fisher'/name).write_text(compact_shared_weights(base,groups,normalization),encoding='utf-8',newline='\n')
        additional.append(name)
        name=f"fisher_c8_shared_operands_{groups}wg_{'both' if normalization else 'moments'}_v54.cu"
        (ROOT/'native_fisher'/name).write_text(compact_shared_weights(base,groups,normalization,True),encoding='utf-8',newline='\n')
        additional.append(name)
    rotated=rotate_packed_epilogue(base)
    name='fisher_c8_outlined_stable_v51.cu'
    (ROOT/'native_fisher'/name).write_text(outline_stable_normalizer(rotated),encoding='utf-8',newline='\n')
    additional.append(name)
    for order in ('head','span','head_span'):
        name=f'fisher_c8_block_order_{order}_v50.cu'
        (ROOT/'native_fisher'/name).write_text(reorder_pipeline_blocks(rotated,order),encoding='utf-8',newline='\n')
        additional.append(name)
    for prefetch in (False,True):
        name=f"fisher_c8_value_early_{'prefetch' if prefetch else 'store'}_v48.cu"
        (ROOT/'native_fisher'/name).write_text(issue_value_before_key_store(rotated,prefetch),encoding='utf-8',newline='\n')
        additional.append(name)
    for gap in (1024,2048):
        name=f'fisher_c8_staggered_{gap}_v47.cu'
        (ROOT/'native_fisher'/name).write_text(stagger_cta_start(rotated,gap),encoding='utf-8',newline='\n')
        additional.append(name)
    narrow=just_in_time_queries(make_fragment_pipeline(base,False,query_prefetch=False))
    wide=just_in_time_queries(make_packed_epilogue_pipeline(base,False))
    for label,source in (('n64',narrow),('n64_packed',pack_owner_labels(narrow)),
                         ('n128_packed',pack_owner_labels(wide))):
        name=f'fisher_c8_five_wg_{label}_v42.cu'
        (ROOT/'native_fisher'/name).write_text(five_warpgroup_columns(source),encoding='utf-8',newline='\n')
        additional.append(name)
    shared_narrow=shared_query_per_group(five_warpgroup_columns(pack_owner_labels(narrow)))
    shared_wide=shared_query_per_group(five_warpgroup_columns(pack_owner_labels(wide)))
    for label,source in (('five_n64',shared_narrow),('five_n128',shared_wide),
                        ('six_n64',six_warpgroup_frame_columns(rematerialize_owner_scale(shared_narrow)))):
        name=f'fisher_c8_sharedq_{label}_v43.cu'
        (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
        additional.append(name)
    for label,source in (('moments',interleave_winner_moments(narrow)),
                        ('both',interleave_winner_moments(interleave_normalization(narrow)))):
        source=shared_query_per_group(five_warpgroup_columns(pack_owner_labels(source)))
        source=six_warpgroup_frame_columns(rematerialize_owner_scale(source))
        name=f'fisher_c8_six_wg_{label}_v44.cu'
        (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
        additional.append(name)
        if label=='moments':
            six_moments=source
    for label,source in (('five_n64',shared_narrow),('six_moments',six_moments)):
        name=f'fisher_c8_remat_addresses_{label}_v45.cu'
        (ROOT/'native_fisher'/name).write_text(rematerialize_query_addresses(source),encoding='utf-8',newline='\n')
        additional.append(name)
    for label,source in (('plain',narrow),('moments',interleave_winner_moments(narrow)),
                         ('both',interleave_winner_moments(interleave_normalization(narrow)))):
        source=pack_owner_labels(rematerialize_owner_scale(source))
        source=stage_half_scores(four_warpgroup_two_ctas(five_warpgroup_columns(source)))
        name=f'fisher_c8_eight_resident_{label}_v49.cu'
        (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
        additional.append(name)
        if label in ('moments','both'):
            name=f'fisher_c8_score_quarters_eight_{label}_v52.cu'
            (ROOT/'native_fisher'/name).write_text(quarter_score_registers(source),encoding='utf-8',newline='\n')
            additional.append(name)
    six_registers=six_warpgroup_frame_columns(five_warpgroup_columns(
        pack_owner_labels(rematerialize_owner_scale(interleave_winner_moments(narrow)))))
    name='fisher_c8_score_quarters_six_moments_v52.cu'
    (ROOT/'native_fisher'/name).write_text(quarter_score_registers(stage_half_scores(six_registers)),encoding='utf-8',newline='\n')
    additional.append(name)
    for label,source in (('five_n64',five_warpgroup_columns(pack_owner_labels(narrow))),
                        ('five_n128',five_warpgroup_columns(pack_owner_labels(wide))),
                        ('six_n64',six_registers)):
        name=f'fisher_c8_staged_scores_{label}_v46.cu'
        (ROOT/'native_fisher'/name).write_text(stage_half_scores(source),encoding='utf-8',newline='\n')
        additional.append(name)
    name='fisher_c8_separate_store_buffers_v41.cu'
    (ROOT/'native_fisher'/name).write_text(separate_store_buffers(rotated),encoding='utf-8',newline='\n')
    additional.append(name)
    for label,fraction in (('half',0.5),('all',1.0)):
        name=f'fisher_c8_tma_stream_{label}_v40.cu'
        (ROOT/'native_fisher'/name).write_text(tma_store_cache_hint(rotated,fraction),encoding='utf-8',newline='\n')
        additional.append(name)
    name='fisher_c8_clock_phases_v37.cu'
    (ROOT/'native_fisher'/name).write_text(trace_merge_phases(rotated),encoding='utf-8',newline='\n')
    additional.append(name)
    name='fisher_c8_clock_predicated_v38.cu'
    (ROOT/'native_fisher'/name).write_text(trace_merge_phases(rotated,True),encoding='utf-8',newline='\n')
    additional.append(name)
    name='fisher_c8_clock_lb3_v39.cu'
    source=trace_merge_phases(rotated,True).replace('__launch_bounds__(128,4)','__launch_bounds__(128,3)')
    (ROOT/'native_fisher'/name).write_text('// Diagnostic occupancy changes from four to at most three CTAs.\n'+source,encoding='utf-8',newline='\n')
    additional.append(name)
    for label,source in (
        ('normalization',interleave_normalization(rotated)),
        ('moments',interleave_winner_moments(rotated)),
        ('both',interleave_winner_moments(interleave_normalization(rotated)))):
        name=f'fisher_c8_interleaved_{label}_v36.cu'
        (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
        additional.append(name)
    name='fisher_c8_rotated_packed_v23.cu'
    (ROOT/'native_fisher'/name).write_text(rotate_packed_epilogue(base),encoding='utf-8',newline='\n')
    additional.append(name)
    for tiles in (21,42,210):
        name=f'fisher_c8_rotated_span{tiles}_v29.cu'
        (ROOT/'native_fisher'/name).write_text(fixed_launch_span(rotate_packed_epilogue(base),tiles),encoding='utf-8',newline='\n')
        additional.append(name)
    for label,overlap in (('serial',False),('kv',True)):
        name=f'fisher_c8_completed_scores_{label}_v25.cu'
        source=wait_before_score_backedge(make_packed_epilogue_pipeline(base,overlap,True))
        (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
        additional.append(name)
    for label,candidate in (
        ('kv',ordered_query_prefetch(make_packed_epilogue_pipeline(base,True,early_query=False))),
        ('rotated',rotate_packed_epilogue(base,True,True)),
        ('rotated_full',rotate_full_packed_epilogue(base,True,True))):
        name=f'fisher_c8_ordered_prefetch_{label}_v26.cu'
        (ROOT/'native_fisher'/name).write_text(candidate,encoding='utf-8',newline='\n')
        additional.append(name)
        name=f'fisher_c8_preserved_prefetch_{label}_v27.cu'
        (ROOT/'native_fisher'/name).write_text(preserve_query_dependency(candidate),encoding='utf-8',newline='\n')
        additional.append(name)
    combined=preserve_query_dependency(rotate_packed_epilogue(base,True,True))
    for label,candidate in (
        ('owners',pack_owner_labels(combined)),
        ('rescale',rematerialize_owner_scale(combined)),
        ('both',pack_owner_labels(rematerialize_owner_scale(combined)))):
        name=f'fisher_c8_compact_rotated_{label}_v28.cu'
        (ROOT/'native_fisher'/name).write_text(candidate,encoding='utf-8',newline='\n')
        additional.append(name)
    for label,candidate in (
        ('full',rotate_full_packed_epilogue(base)),
        ('columns',rotate_packed_epilogue(make_column_pipeline(base,False))),
        ('full_columns',rotate_full_packed_epilogue(make_column_pipeline(base,False)))):
        name=f'fisher_c8_rotated_{label}_v24.cu'
        candidate='\n'.join(line.rstrip() for line in candidate.splitlines())+'\n'
        (ROOT/'native_fisher'/name).write_text(candidate,encoding='utf-8',newline='\n')
        additional.append(name)
    for overlap,min_ctas in ((False,5),(True,4),(True,5)):
        name=f"fisher_c8_fragment32_{'overlap' if overlap else 'serial'}_lb{min_ctas}_v4.cu"
        (ROOT/'native_fisher'/name).write_text(make_fragment_pipeline(base,overlap,32,min_ctas,True),encoding='utf-8',newline='\n')
        additional.append(name)
    name='fisher_c8_fragment32_lateq_lb5_v4.cu'
    (ROOT/'native_fisher'/name).write_text(make_fragment_pipeline(base,True,32,5,True,True),encoding='utf-8',newline='\n')
    additional.append(name)
    for late in (False,True):
        name=f"fisher_c8_fragment32_packedowners_{'lateq' if late else 'prefetch'}_lb5_v5.cu"
        source=pack_owner_labels(make_fragment_pipeline(base,True,32,5,True,late))
        (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
        additional.append(name)
    name='fisher_c8_fragment32_scorehalves_lb5_v6.cu'
    score_half_source=split_score_query_registers(make_fragment_pipeline(base,True,32,5,True,True))
    (ROOT/'native_fisher'/name).write_text(score_half_source,encoding='utf-8',newline='\n')
    additional.append(name)
    for handoff in (False,True):
        name=f"fisher_c8_two_wg_uniform_{'weightshandoff' if handoff else 'independent'}_v11.cu"
        source=make_norm_handoff(base,False) if handoff else make_two_warpgroup(base,False)
        source=source.replace('threadIdx.x/128','int(__all_sync(0xffffffff,threadIdx.x>=128))')
        source=source.replace('wg=tid/128','wg=int(__all_sync(0xffffffff,tid>=128))')
        (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
        additional.append(name)
    name='fisher_c8_fragment32_scorehalves_pinned_lb5_v6.cu'
    cut=score_half_source.index('    uint32_t packed_weights[20];')
    wait='asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");'
    front=score_half_source[:cut].replace(wait,wait+'\n      #pragma unroll\n      for(int pin=0;pin<16;++pin) asm volatile("" : "+r"(qreg[pin]) :: "memory");')
    tail=score_half_source[cut:].replace(wait,wait+'\n    #pragma unroll\n    for(int pin=0;pin<20;++pin) asm volatile("" : "+r"(packed_weights[pin]) :: "memory");')
    (ROOT/'native_fisher'/name).write_text(front+tail,encoding='utf-8',newline='\n')
    additional.append(name)
    # Preserve v4-v6 receipts with their rejected SW64 index. CuTe's
    # Swizzle<2,3,3> for BF16 XORs bits 3:4 with 6:7, not 5:6.
    for name,candidate in (
        ('fisher_c8_fragment32_sw64_lb4_v7.cu',make_fragment_pipeline(base,True,32,4,True)),
        ('fisher_c8_fragment32_sw64_scorehalves_pinned_lb5_v7.cu',front+tail)):
        fixed=candidate.replace('(offset>>5)&3','(offset>>6)&3')
        assert fixed!=candidate
        (ROOT/'native_fisher'/name).write_text(fixed,encoding='utf-8',newline='\n')
        additional.append(name)
    for early in (False,True):
        name=f"fisher_c8_two_wg_{'norm' if early else 'weights'}handoff_v8.cu"
        (ROOT/'native_fisher'/name).write_text(make_norm_handoff(base,early),encoding='utf-8',newline='\n')
        additional.append(name)
    for handoff in (False,True):
        name=f"fisher_c8_two_wg_static_{'weightshandoff' if handoff else 'independent'}_v9.cu"
        source=make_norm_handoff(base,False) if handoff else make_two_warpgroup(base,False)
        (ROOT/'native_fisher'/name).write_text(specialize_two_wg(source),encoding='utf-8',newline='\n')
        additional.append(name)
    name='fisher_c8_two_wg_vote_weightshandoff_v10.cu'
    source=specialize_two_wg(make_norm_handoff(base,False))
    assert source.count('if(tid<128)')==1
    source=source.replace('if(tid<128)','if(__all_sync(0xffffffff,tid<128))',1)
    (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
    additional.append(name)
    for handoff in (False,True):
        name=f"fisher_c8_two_wg_columns_{'weightshandoff' if handoff else 'independent'}_v12.cu"
        (ROOT/'native_fisher'/name).write_text(make_column_pipeline(base,handoff),encoding='utf-8',newline='\n')
        additional.append(name)
    for stage in ('k','v','both'):
        name=f'fisher_c8_two_wg_columns_earlyq_{stage}_v13.cu'
        source=advance_query_prefetch(make_column_pipeline(base,False),stage)
        (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
        additional.append(name)
    name='fisher_c8_fragment64_crossquery_v14.cu'
    (ROOT/'native_fisher'/name).write_text(make_cross_query_pipeline(base),encoding='utf-8',newline='\n')
    additional.append(name)
    for stages in (1,2):
        name=f'fisher_c8_role_ring{stages}_v15.cu'
        (ROOT/'native_fisher'/name).write_text(make_role_pipeline(base,stages),encoding='utf-8',newline='\n')
        additional.append(name)
    for residency in (2,3):
        name=f'fisher_c8_role_n64_packed_lb{residency}_v16.cu'
        source=pack_owner_labels(make_role_pipeline(base,1,split_merge=True))
        source=source.replace('__launch_bounds__(256,2)',f'__launch_bounds__(256,{residency})')
        (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
        additional.append(name)
    for residency in (2,3):
        name=f'fisher_c8_role_sharedq_lb{residency}_v17.cu'
        source=pack_owner_labels(make_role_pipeline(base,1,split_merge=True))
        source=role_shared_query(source,residency)
        (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
        additional.append(name)
    for label,overlap,cross_query in (('serial',False,False),('kv',True,False),('crossquery',True,True)):
        name=f'fisher_c8_packed_epilogue_{label}_v18.cu'
        source=make_packed_epilogue_pipeline(base,overlap,cross_query)
        (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
        additional.append(name)
    for label,candidate in (('fragment',make_cross_query_pipeline(base)),
                            ('packed',make_packed_epilogue_pipeline(base,True,True))):
        name=f'fisher_c8_crossquery_reset_{label}_v19.cu'
        (ROOT/'native_fisher'/name).write_text(reset_carried_scores(candidate),encoding='utf-8',newline='\n')
        additional.append(name)
    for label,late_groups,early_query in (('quarter_earlyq',2,True),('half_lateq',4,False)):
        name=f'fisher_c8_packed_epilogue_{label}_v20.cu'
        source=make_packed_epilogue_pipeline(base,True,late_groups=late_groups,early_query=early_query)
        (ROOT/'native_fisher'/name).write_text(source,encoding='utf-8',newline='\n')
        additional.append(name)
    for overlap in (False,True):
        name=f'fisher_c8_pair_packed_{"kv" if overlap else "serialkv"}_v21.cu'
        (ROOT/'native_fisher'/name).write_text(make_pair_packed_pipeline(base,overlap),encoding='utf-8',newline='\n')
        additional.append(name)
    for name in additional:
        manifest['sources'][name]={'sha256':hashlib.sha256((ROOT/'native_fisher'/name).read_bytes()).hexdigest()}
    print(json.dumps(manifest,indent=2))
