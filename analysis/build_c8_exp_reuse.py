"""Build exp-reuse candidates from the retained r2 source without editing it."""
import argparse
import hashlib
import json
from pathlib import Path


def replace_once(source, old, new):
    assert source.count(old) == 1, old[:100]
    return source.replace(old, new, 1)


def build(source, signed_stream):
    source = replace_once(source, '    float norm_max[2],norm_sum[2];',
                          '    float cached_exp[40];\n    float norm_max[2],norm_sum[2];')
    for name, expr in (
        ('direct', 'ex2(__fmul_rn(s[4*(i/2)+2*half+(i&1)],scale))'),
        ('bs', 'ex2(__fmaf_rn(s[4*(i/2)+2*half+(i&1)],scale,-mx))'),
    ):
        old = f'      for(int i=0;i<20;++i)\n        {name}[i]={expr};'
        new = (f'      for(int i=0;i<20;++i) {{\n        {name}[i]={expr};\n'
               f'        cached_exp[4*(i/2)+2*half+(i&1)]={name}[i];\n      }}')
        source = replace_once(source, old, new)
    start = source.index('    float winners[20];')
    end = source.index('    if(stop_after==2) continue;', start)
    if signed_stream:
        body = r'''    // Preserve the first exponentials, then retire each score immediately.
    // A signed probability encodes which of this thread's two seed rows owns
    // the key; zero denotes another owner. No per-key label/log-score arrays.
    float signed_probabilities[20];
    float sum0=0.f,sum1=0.f,dot0=0.f,dot1=0.f;
    uint32_t oc0=(uint32_t)seed_for_row(warp*16+seed),oc1=oc0+4u;
    #pragma unroll
    for(int i=0;i<20;++i) {
      int ix=(i/2)*4+(i&1);
      float u0=s[ix],u1=s[ix+2];
      uint32_t key=min(owner_key(u0,oc0),owner_key(u1,oc1));
      key=min(key,__shfl_xor_sync(0xffffffff,key,8));
      key=min(key,__shfl_xor_sync(0xffffffff,key,16));
      uint32_t owner=key&7u;
      float p0=owner==oc0?cached_exp[ix]:0.f;
      float p1=owner==oc1?cached_exp[ix+2]:0.f;
      signed_probabilities[i]=owner==oc1?-p1:p0;
      sum0=__fadd_rn(sum0,p0);sum1=__fadd_rn(sum1,p1);
      dot0=__fmaf_rn(p0,__fadd_rn(1.f,-u0),dot0);
      dot1=__fmaf_rn(p1,__fadd_rn(1.f,-u1),dot1);
    }
    float sums[2]={sum4(sum0),sum4(sum1)};
    float dots[2]={sum4(dot0),sum4(dot1)};
'''
    else:
        body = r'''    // Reuse unnormalized exponentials: the row denominator cancels in W.
    // Keep the current r2 packed assignment, selecting its actual owner row.
    float winners[20],winner_probabilities[20];
    int labels[20];
    uint32_t oc0=(uint32_t)seed_for_row(warp*16+seed),oc1=oc0+4u;
    #pragma unroll
    for(int i=0;i<20;++i) {
      int ix=(i/2)*4+(i&1);
      float u0=s[ix],u1=s[ix+2];
      uint32_t key=min(owner_key(u0,oc0),owner_key(u1,oc1));
      key=min(key,__shfl_xor_sync(0xffffffff,key,8));
      key=min(key,__shfl_xor_sync(0xffffffff,key,16));
      labels[i]=(int)(key&7u);
      bool upper=labels[i]==oc1;
      winners[i]=__fadd_rn(1.f,-(upper?u1:u0));
      winner_probabilities[i]=upper?cached_exp[ix+2]:cached_exp[ix];
    }
    float sums[2],dots[2];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float probabilities[20];
      #pragma unroll
      for(int i=0;i<20;++i)
        probabilities[i]=labels[i]==seed_for_row(warp*16+seed+half*8)?winner_probabilities[i]:0.f;
      sums[half]=reduce80(probabilities);
      dots[half]=dot80(probabilities,winners);
    }
'''
    body += r'''    uint32_t packed_weights[20];
    float lane_sum=(lane&1)?sums[1]:sums[0];
    float lane_safe=lane_sum>0?lane_sum:1.f;
    float lane_inv=reciprocal(lane_safe),lane_log=lg2(lane_safe);
    #pragma unroll
    for(int half=0;half<2;++half) {
      float inv=__shfl_sync(0xffffffff,lane_inv,(lane&~1)+half);
      float log_safe=__shfl_sync(0xffffffff,lane_log,(lane&~1)+half);
      // log P = log E - log Z. For raw-E weights, restore +log Z so that
      // bias = sum_j W_j log P_j - log(sum_j M_j E_j) + log Z = sum W log W.
      float log_z=__shfl_sync(0xffffffff,norm_log,(lane&~1)+half);
      float correction=__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(dots[half],inv),-log_safe),log_z),0.6931471805599453f);
      int row=warp*16+seed+8*half,col=cb*8+col_for_row(row);
      int out=((qo*C+col)*AH+head)*(KO*R*8)+kb*8+seed_for_row(row);
      if((lane&3)==0) bias[out]=sums[half]>0?correction:CUDART_INF_F;
      #pragma unroll
      for(int i=0;i<10;++i) {
        union { bf16 values[2]; uint32_t bits; } pair;
'''
    if signed_stream:
        body += r'''        float sign=half==0?1.f:-1.f;
        float p0=fmaxf(__fmul_rn(signed_probabilities[2*i],sign),0.f);
        float p1=fmaxf(__fmul_rn(signed_probabilities[2*i+1],sign),0.f);
'''
    else:
        body += r'''        float p0=labels[2*i]==seed_for_row(row)?winner_probabilities[2*i]:0.f;
        float p1=labels[2*i+1]==seed_for_row(row)?winner_probabilities[2*i+1]:0.f;
'''
    body += r'''        pair.values[0]=__float2bfloat16_rn(__fmul_rn(p0,inv));
        pair.values[1]=__float2bfloat16_rn(__fmul_rn(p1,inv));
        packed_weights[2*i+half]=pair.bits;
      }
    }
'''
    return source[:start] + body + source[end:]


def build_probability(source):
    source = build(source, True)
    start = source.index('    float cached_exp[40];')
    end = source.index('    float signed_probabilities[20];', start)
    normalization = r'''    // Convert one seed row at a time to FP32 probabilities in place.
    // Original scores are needed only until this row's range check completes.
    #pragma unroll
    for(int half=0;half<2;++half) {
      float probabilities[20];
      #pragma unroll
      for(int i=0;i<20;++i)
        probabilities[i]=ex2(__fmul_rn(s[4*(i/2)+2*half+(i&1)],scale));
      float sum=reduce80(probabilities);
      bool direct_ok=__all_sync(0xffffffff,sum>0x1p-64f && sum<0x1p64f);
      if(!direct_ok) {
        float mx=-CUDART_INF_F;
        #pragma unroll
        for(int i=0;i<20;++i)
          mx=fmaxf(mx,__fmul_rn(s[4*(i/2)+2*half+(i&1)],scale));
        mx=max4(mx);
        #pragma unroll
        for(int i=0;i<20;++i)
          probabilities[i]=ex2(__fmaf_rn(s[4*(i/2)+2*half+(i&1)],scale,-mx));
        sum=reduce80(probabilities);
      }
      float inv=reciprocal(sum);
      #pragma unroll
      for(int i=0;i<20;++i)
        s[4*(i/2)+2*half+(i&1)]=__fmul_rn(probabilities[i],inv);
    }
    // Compare normalized probabilities directly; retain only signed winners.
'''
    source = source[:start] + normalization + source[end:]
    source = replace_once(source,
        '''      float u0=s[ix],u1=s[ix+2];
      uint32_t key=min(owner_key(u0,oc0),owner_key(u1,oc1));
      key=min(key,__shfl_xor_sync(0xffffffff,key,8));
      key=min(key,__shfl_xor_sync(0xffffffff,key,16));
      uint32_t owner=key&7u;
      float p0=owner==oc0?cached_exp[ix]:0.f;
      float p1=owner==oc1?cached_exp[ix+2]:0.f;''',
        '''      float probability0=s[ix],probability1=s[ix+2];
      // Positive FP32 probability order equals unsigned bit order. Reverse
      // the owner code so MAX breaks represented ties toward the lower seed.
      uint32_t key=max(owner_key(probability0,7u-oc0),owner_key(probability1,7u-oc1));
      key=max(key,__shfl_xor_sync(0xffffffff,key,8));
      key=max(key,__shfl_xor_sync(0xffffffff,key,16));
      uint32_t owner=7u-(key&7u);
      float p0=owner==oc0?probability0:0.f;
      float p1=owner==oc1?probability1:0.f;
      float probability=owner==oc1?p1:p0;
      float logp=lg2(probability>0.f?probability:1.f);''')
    source = replace_once(source,
        '''      dot0=__fmaf_rn(p0,__fadd_rn(1.f,-u0),dot0);
      dot1=__fmaf_rn(p1,__fadd_rn(1.f,-u1),dot1);''',
        '''      dot0=__fmaf_rn(p0,logp,dot0);
      dot1=__fmaf_rn(p1,logp,dot1);''')
    source = replace_once(source,
        '''      // log P = log E - log Z. For raw-E weights, restore +log Z so that
      // bias = sum_j W_j log P_j - log(sum_j M_j E_j) + log Z = sum W log W.
      float log_z=__shfl_sync(0xffffffff,norm_log,(lane&~1)+half);
      float correction=__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(dots[half],inv),-log_safe),log_z),0.6931471805599453f);''',
        '''      // Probabilities already include the row normalizer.
      float correction=__fmul_rn(__fadd_rn(__fmul_rn(dots[half],inv),-log_safe),0.6931471805599453f);''')
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    source = args.baseline.read_text(encoding='utf-8')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'baseline': str(args.baseline.resolve()),
                'baseline_sha256': hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                'files': {}}
    for name, signed in (('c8_exp_reuse_raw_v1.cu', False),
                         ('c8_exp_reuse_signed_v2.cu', True),
                         ('c8_exp_reuse_signed_lb3_v3.cu', True),
                         ('c8_probability_inplace_v4.cu', True)):
        path = args.output_dir / name
        content = build_probability(source) if '_inplace_' in name else build(source, signed)
        if '_lb3_' in name:
            content = replace_once(content, '__launch_bounds__(128,4) void merge_kernel(',
                                   '__launch_bounds__(128,3) void merge_kernel(')
        assert not path.exists() or path.read_text() == content
        path.write_text(content, encoding='utf-8', newline='\n')
        manifest['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
