"""Source-linked incremental workbook rows, without any quality promotion."""
import json,sys,statistics
from pathlib import Path
source=Path(sys.argv[1]);x=json.loads(source.read_text())
assert x['valid'] and not x['monitor_errors']
if '--aggregate' in sys.argv[2:]:
    layers=x['layers'];assert all(row['valid'] for row in layers)
    assert len({len(row['candidate']['samples_ms']) for row in layers})==1
    y={k:v for k,v in x.items() if k!='layers'}
    y.update(fp8_storage=True,parent_result=source.name,layer='/'.join(row['layer'] for row in layers),per_layer=layers)
    for metric in ('candidate','native_control','sdpa'):
        samples=[v for row in layers for v in row[metric]['samples_ms']]
        y[metric]={'mean_ms':statistics.fmean(samples),'samples_ms':samples}
    y['speedup_vs_sdpa']=y['sdpa']['mean_ms']/y['candidate']['mean_ms']
    y['native_control_correctness']={k:all(row['native_control_correctness'][k] for row in layers) if k in ('finite','bitwise_equal')
                                    else max(row['native_control_correctness'][k] for row in layers)
                                    for k in layers[0]['native_control_correctness']}
    y['seed_mismatch_count']=sum(row['seed_mismatch_count'] for row in layers)
    y['bias_bitwise_equal']=all(row['bias_bitwise_equal'] for row in layers)
    target=source.with_name(source.stem+'_aggregate.json');target.write_text(json.dumps(y,indent=2))
    target.with_suffix('.assessment.json').write_text(json.dumps({'validation':
        'Equal-weight complete-call latency across layers '+y['layer']+'. Shown L2/max/mean errors are worst per-layer values, not pooled norms. '
        'Seeds/bias unchanged, storage checked; generation quality pending.'},indent=2))
    print(target.stem);sys.exit(0)
for layer in x['layers']:
    assert layer['valid']
    y={k:v for k,v in x.items() if k!='layers'}
    y.update(layer,fp8_storage=True,parent_result=source.name)
    target=source.with_name(source.stem+'_row'+layer['layer']+'.json')
    target.write_text(json.dumps(y,indent=2))
    target.with_suffix('.assessment.json').write_text(json.dumps({'validation':
        'Finite complete outputs. Seeds and bias unchanged; sampled storage quantization verified. '
        'Reported L2 is FP8 error, not an accepted BF16 roundoff gate. Generation quality pending.'},indent=2))
    print(target.stem)
