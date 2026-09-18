"""Aggregate complete matched H40 receipts; preserve each raw measurement."""
import argparse, copy, json, statistics
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('receipt',type=Path)
p.add_argument('--assessment',required=True)
args=p.parse_args()
x=json.loads(args.receipt.read_text())
assert x['valid'] and not x['monitor_errors']
rows=x['layers']
assert [r['layer'] for r in rows]==['00','19','39']
assert all(r['valid'] and r['seed_mismatch_count']==0 and r['bias_bitwise_equal'] for r in rows)
out=copy.deepcopy(x)
out.pop('layers')
out.pop('pending_correctness',None)
out['layer']='00/19/39'
out['raw_receipt']=args.receipt.name
for method in ('candidate','native_control','sdpa'):
    samples=[v for row in rows for v in row[method]['samples_ms']]
    out[method]={'mean_ms':statistics.fmean(samples),'samples_ms':samples}
out['speedup_vs_sdpa']=out['sdpa']['mean_ms']/out['candidate']['mean_ms']
out['seed_mismatch_count']=0
out['bias_bitwise_equal']=True
out['native_control_correctness']={
    'bitwise_equal':all(r['native_control_correctness']['bitwise_equal'] for r in rows),
    'finite':all(r['native_control_correctness']['finite'] for r in rows),
    **{key:max(r['native_control_correctness'][key] for r in rows)
       for key in ('relative_l2','max_abs','mean_abs')}}
out['correctness_aggregation']='Conservative maximum of the three per-layer errors, not a pooled L2'
dest=args.receipt.with_name(args.receipt.stem+'_aggregate.json')
dest.write_text(json.dumps(out,indent=2))
dest.with_suffix('.assessment.json').write_text(json.dumps({'validation':args.assessment},indent=2))
print(json.dumps({key:out[key] for key in ('layer','native_control_correctness','resources','route_resources','speedup_vs_sdpa') if key in out}))
print('candidate_ms',out['candidate']['mean_ms'],'control_ms',out['native_control']['mean_ms'])
