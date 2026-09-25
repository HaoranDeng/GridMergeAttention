"""Read only the current state and aggregate complete-call results."""
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
state=json.loads((root/'watcher_state.json').read_text())
completed=state.get('completed_jobs',{})
last=next(reversed(completed),None)
print(json.dumps({**{k:state.get(k) for k in ('pid','status','updated_utc')},
                  'completed_count':len(completed),'last_completed':last,
                  'last_result':completed[last]['result'] if last else None}))
active=state.get('active_job')
if active:
    print(json.dumps({k:active.get(k) for k in ('job','pid','result','exit_code','invalid_reason')}))
    path=Path(active['result'])
    if path.exists():
        result=json.loads(path.read_text())
        print(json.dumps({k:result.get(k) for k in ('status','valid_timing','error')}))
        for name,c in result.get('candidates',{}).items():
            print(json.dumps({'name':name,'resources':c.get('resources'),'layers':{
                k:{'gate':v.get('passed'),'ms':v.get('complete_call',{}).get('candidate_mean_ms'),
                   'control':v.get('complete_call',{}).get('control_mean_ms')}
                for k,v in c.get('layers',{}).items()}}))
