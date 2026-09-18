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
        for key in ('AXIS_C4_CHOLESKY','AXIS_C4_FAST_MATH','AXIS_C4_SELECTOR_WARPS'):
            assert b['environment'].get(key)==v['environment'].get(key),key
        assert len(b['trials'])==1
        trial=b['trials'][0];assert trial['median_ms']<b['sdpa_ms']
        current=dict(b['environment'],AXIS_C4_CUDA_TILES=trial['config'].split(':')[2])
        assert environment is None or environment==current
        environment=current
        samples.append({'input_kind':b['input_kind'],'capture':b['capture'],
                        'method_median_ms':trial['median_ms'],'dense_median_ms':b['sdpa_ms'],
                        'benchmark':str(path),'guard':str(guard),'samples_ms':trial['samples_ms']})
    gate={'passed':True,'shape':[1,40,75600,128],'dtype':'bfloat16',
          'scope':'full attention including layouts and all40 heads','source_sha256':v['source_sha256'],
          'methods':{a.axes:{'validated':True,'exclusive_gpu':True,'module':v['module'],
                              'environment':environment,'samples':samples,'validation':str(a.validation)}}}
    save(a.output,gate)
    verify_gate(a.output,Path(__file__).resolve().parents[1],a.axes)
    print(json.dumps({'passed':True,'axes':a.axes,'samples':samples}),flush=True)

if __name__=='__main__':main()
