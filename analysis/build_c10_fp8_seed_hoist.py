"""Hoist frame-invariant seed indices into an explicit outer column loop.

Every old tile still runs once, in the same order, including partial first/
last columns for diagnostic tile counts. Both original KV tiles remain
resident; seed selection, packed Q layout, arithmetic and output are intact.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fisher_c10_fp8_local_scores.cu').read_text()
old='''  int total=Q*((C+5)/6),start=blockIdx.z*tile_count;
  for(int tile=start;tile<min(start+tile_count,total);++tile) {
    int qo=tile%Q,cb=tile/Q,rf=0;'''
new='''  int total=Q*((C+5)/6),start=blockIdx.z*tile_count,end=min(start+tile_count,total);
  for(int cb=start/Q;cb<(end+Q-1)/Q;++cb) {
    int rf=0;'''
assert s.count(old)==1;s=s.replace(old,new)
old='    int sr0=seeds[si0],sr1=seeds[si1];'
assert s.count(old)==1
s=s.replace(old,old+'''
    // These two row indices are invariant across all query frames here.
    for(int qo=max(0,start-cb*Q);qo<min(Q,end-cb*Q);++qo) {''')
old='  if(tma_store==1 && tid==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");'
assert s.count(old)==1;s=s.replace(old,'  } // Outer query-column loop.\n'+old)
(root/'fisher_c10_fp8_seed_hoist.cu').write_text(s)
