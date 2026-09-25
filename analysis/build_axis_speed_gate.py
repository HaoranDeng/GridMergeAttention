"""Build a source-bound launch receipt from independent correctness/timing runs."""
import argparse
import json
from pathlib import Path
from analysis.quick_axis_c4_qkv import save
from analysis.axis_c4_speed_gate import verify_gate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--axes',required=True,choices=['tw','wh','th'])
    p.add_argument('--validation',required=True,type=Path)
    p.add_argument('--validation-guard',required=True,type=Path)
    p.add_argument('--benchmark',required=True,type=Path,nargs='+')
    p.add_argument('--benchmark-guard',required=True,type=Path,nargs='+')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--user-speed-exception',help='Explicit user authorization for slower T/H experiment only')
    a=p.parse_args();v=json.loads(a.validation.read_text())
    assert v['passed'] is True and v['axes']==a.axes
    assert json.loads((a.validation_guard/'completion.json').read_text())['valid'] is True
    assert len(a.benchmark)==len(a.benchmark_guard)
    samples=[];environment=None
    for path,guard in zip(a.benchmark,a.benchmark_guard):
        b=json.loads(path.read_text())
        assert b['complete'] and b['axes']==a.axes and b['input_kind']=='captured_real_qkv'
        assert json.loads((guard/'completion.json').read_text())['valid'] is True
        assert b['source_sha256']==v['source_sha256'] and b['module']==v['module']
        assert len(b['trials'])==1
        trial=b['trials'][0]
        assert trial['median_ms']<b['sdpa_ms'] or (a.axes=='th' and a.user_speed_exception)
        current=dict(trial.get('environment',b['environment']),AXIS_C4_CUDA_TILES=trial['config'].split(':')[2])
        defaults={'AXIS_C4_CHOLESKY':'0','AXIS_C4_FAST_MATH':'0','AXIS_C4_SELECTOR_WARPS':'4','AXIS_C4_SELECTOR_BACKEND':'mma',
                  'AXIS_C4_ROUTE_TOKENS':'64','AXIS_C4_ROUTE_WARPS':'4','AXIS_C4_ROUTE_STAGES':'2',
                  'AXIS_C4_CUDA_TILES':'21','AXIS_C4_HEAD_CHUNK':str({'tw':4,'wh':8,'th':2}[a.axes])}
        for key,default in defaults.items():
            assert current.get(key,default)==v['environment'].get(key,default),key
        assert environment is None or environment==current
        environment=current
        samples.append({'input_kind':b['input_kind'],'capture':b['capture'],
                        'method_median_ms':trial['median_ms'],'dense_median_ms':b['sdpa_ms'],
                        'benchmark':str(path),'guard':str(guard),'samples_ms':trial['samples_ms']})
    gate={'passed':True,'shape':[1,40,75600,128],'dtype':'bfloat16',
          'scope':'full attention including layouts and all40 heads','source_sha256':v['source_sha256'],
          'methods':{a.axes:{'validated':True,'exclusive_gpu':True,'module':v['module'],
                              'environment':environment,'samples':samples,'validation':str(a.validation)}}}
    gate['passed']=all(s['method_median_ms']<s['dense_median_ms'] for s in samples)
    if a.user_speed_exception:
        assert a.axes=='th'
        gate['user_speed_exception']={'axes':'th','instruction':a.user_speed_exception,
                                      'scope':'20 videos, 50 steps, first-five immediate quality evaluation'}
    save(a.output,gate)
    verify_gate(a.output,Path(__file__).resolve().parents[1],a.axes,bool(a.user_speed_exception))
    print(json.dumps({'passed':gate['passed'],'authorized_speed_exception':bool(a.user_speed_exception),'axes':a.axes,'samples':samples}),flush=True)

if __name__=='__main__':main()
