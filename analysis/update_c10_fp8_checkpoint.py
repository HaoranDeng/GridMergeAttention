"""Record current measured evidence without changing goal completion state."""
import argparse,hashlib,json,statistics,shutil
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('artifacts',type=Path);p.add_argument('--used-percent',type=float,required=True)
p.add_argument('--best-result',default='shared_store_via_float_sweep.json')
p.add_argument('--worker-state',type=Path,required=True,help='Fresh process observation with observed_utc and processes fields')
p.add_argument('--previous-turn-classification',choices=('progress','verified_wait','no_progress'),default='progress')
args=p.parse_args();root=Path(__file__).resolve().parents[1]
workers=json.loads(args.worker_state.read_text())
assert workers.get('observed_utc') and isinstance(workers.get('processes'),list)
best=args.artifacts/'scai7'/args.best_result
x=json.loads(best.read_text());assert x['valid'] and not x['monitor_errors']
for name,digest in x['source_files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        prior_snapshot=args.artifacts/'scai7/best_source_snapshot'/name
        assert prior_snapshot.exists() and hashlib.sha256(prior_snapshot.read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((root/'native_fisher/fisher_c10_fp8_shared_store.cu').read_bytes()).hexdigest()==x['source_sha256']
for name in list(x['source_files'])+['native_fisher/fisher_c10_fp8_shared_store.cu']:
    snapshot=args.artifacts/'scai7/best_source_snapshot'/name
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if name not in x['source_files'] or hashlib.sha256((root/name).read_bytes()).hexdigest()==x['source_files'][name]:
        shutil.copyfile(root/name,snapshot)
results=[]
for f in sorted((args.artifacts/'scai7').glob('*.json')):
    if '_row' in f.stem:continue
    r=json.loads(f.read_text())
    if not isinstance(r,dict) or not r.get('valid') or r.get('monitor_errors') or not r.get('layers'):continue
    rows=r['layers']
    if not all('candidate' in row for row in rows):continue
    results.append({'file':f.name,'layer_count':len(rows),'mean_ms':statistics.fmean(row['candidate']['mean_ms'] for row in rows),
                    'control_ms':statistics.fmean(row['native_control']['mean_ms'] for row in rows),
                    'relative_l2':[row['native_control_correctness']['relative_l2'] for row in rows]})
rows=x['layers'];ms=statistics.fmean(row['candidate']['mean_ms'] for row in rows)
sdpa=statistics.fmean(row['sdpa']['mean_ms'] for row in rows)
checkpoint={'objective':'Optimize GMA-C10 FP8 intermediate KV to 150 ms and leave at least 20 percent token allowance',
    'status':'active','target_achieved':False,'utc':datetime.now(timezone.utc).isoformat(),
    'last_usage_used_percent':args.used_percent,'stop_new_search_by_used_percent':79,
    'no_own_workers_active':not workers['processes'],'worker_observation':workers,
    'previous_goal_turn_classification':args.previous_turn_classification,
    'current_goal_turn_classification':'progress','blocked_audit_count':0,
    'best':{'file':str(best),'mean_attention_ms':ms,'speedup_vs_sdpa':sdpa/ms,
            'same_run_sdpa_ms':sdpa,'layers':[row['layer'] for row in rows],'samples_per_layer':16,
            'source_sha256':x['source_sha256'],'route':x['chosen_route'],
            'source_files':x['source_files'],'generation_validation':workers.get('generation_validation','not verified; required before promotion')},
    'completed_measurements':results,
    'next_work':['Improve C10 merge scheduling/normalization beyond the rejected N256 joint-KV family.',
                 'Preserve exact Fisher seeds, BF16 weight rounding, SQ1 and full compressed attention.',
                 'Do not repeat FlashInfer fa2, padded16, vector global stores, ordinary unchanged route sweeps or N256 joint-KV trials.',
                 'Complete matched generation/error validation before declaring the 150 ms goal achieved.']}
(args.artifacts/'goal_checkpoint.json').write_text(json.dumps(checkpoint,indent=2))
print(json.dumps(checkpoint['best'],indent=2))
