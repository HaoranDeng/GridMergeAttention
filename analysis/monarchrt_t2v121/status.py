"""Compact metadata only; no generated video/frame inspection."""
import json
import time
from common import ROOT,read,save

rt=ROOT/'video20/runtime.json';runtime=read(rt) if rt.exists() else {}
quality=ROOT/'quality_cpu/quality.json';q=read(quality) if quality.exists() else {}
slots={}
for folder in ROOT.glob('scai*_gpu*'):
    row={f:read(folder/f'{f}.json') for f in ('active','completion','preflight') if (folder/f'{f}.json').exists()}
    if (folder/'gate.json').exists():
        gate=read(folder/'gate.json');row['gate']={k:gate.get(k) for k in ('complete','passed','mean_method_ms','mean_sdpa_ms')}
        row['gate']['cases']=len(gate['rows']);row['gate']['max_relative_l2']=max((r['relative_l2'] for r in gate['rows']),default=None)
    slots[folder.name]=row
failure=read(ROOT/'quality_failure.json') if (ROOT/'quality_failure.json').exists() else {}
launch=read(ROOT/'quality_launch.json') if (ROOT/'quality_launch.json').exists() else {}
status=dict(time=time.time(),method='T2V121',target=20,videos=len(runtime.get('videos',[])),runtime=runtime.get('summary'),quality=q.get('summary'),
    vbench=(ROOT/'vbench/summary.json').exists(),slots=slots,quality_failed=failure.get('time',0)>launch.get('time',0))
save(ROOT/'latest_status.json',status);print(json.dumps(status,ensure_ascii=False))
