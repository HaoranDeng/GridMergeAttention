"""Exact ten-way ownership with two/four lanes per key; no padded MMA work."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'native_fisher/fisher_c10_half_owner.cu').read_text()
start = source.index('      for(int index=tid;index<6*40;index+=128) {')
end = source.index('      __syncthreads();', start)
for width in (2, 4):
    replacement = f'''      for(int index=tid/{width};index<6*40;index+=128/{width}) {{
        int column=index/40,ki=index%40,sub=lane%{width};
        float best=-CUDART_INF_F;int owner=0;
        #pragma unroll
        for(int c=sub;c<10;c+={width}) {{
          float score=normalized[(column*10+c)*40+ki];
          if(score>best){{best=score;owner=c;}}
        }}
        #pragma unroll
        for(int delta=1;delta<{width};delta*=2) {{
          float other=__shfl_xor_sync(0xffffffff,best,delta,{width});
          int other_owner=__shfl_xor_sync(0xffffffff,owner,delta,{width});
          if(other>best || (other==best && other_owner<owner)){{best=other;owner=other_owner;}}
        }}
        if(sub==0){{selected_winners[index]=best;selected_owners[index]=owner;}}
      }}
'''
    path = ROOT / f'native_fisher/fisher_c10_owner_lanes{width}.cu'
    path.write_text(source[:start] + replacement + source[end:])
    print(path)
