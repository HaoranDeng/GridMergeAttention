"""Source/config-bound C8 receipts. Video inference requires a faster full call."""
import json
import os
from pathlib import Path
from analysis.quick_axis_c4_qkv import sha,save

DEFAULTS={'AXIS_C8_HEAD_CHUNK':'4','AXIS_C8_CUDA_TILES':'21',
          'AXIS_C8_SELECTOR_WARPS':'4','AXIS_C8_FAST_MATH':'0',
          'AXIS_C8_ROUTE_TOKENS':'64','AXIS_C8_ROUTE_WARPS':'4','AXIS_C8_ROUTE_STAGES':'2'}
FILES=['attention/axis_current_c8_cuda.py','attention/axis_current_c8_selector.cu',
       'attention/axis_current_c8_merge.cu','attention/axis_current_c8_reference.py',
       'kernels/best/c4/route_reference.py','kernels/best/c4/warp_mma.cuh',
       'analysis/quick_axis_c4_qkv.py','analysis/validate_axis_current_c8.py',
       'analysis/benchmark_axis_c8.py','analysis/axis_c8_speed_gate.py',
       'analysis/run_axis_c8_videos.py','analysis/benchmark_wan14b_720p_methods.py']


def environment():
    return {key:os.environ.get(key,value) for key,value in DEFAULTS.items()}


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    return {file:sha(root/file) for file in FILES}


def verify_gate(path,root=None,axes='wh'):
    g=json.loads(Path(path).read_text())
    assert axes=='wh' and g['axes']=='wh' and g['capacity']==8 and g['passed']
    assert g['source_sha256']==source_hashes(),'Stale C8 source receipt'
    assert g['scope']=='full attention including layouts and all40 heads'
    assert g['shape']==[1,40,75600,128] and g['dtype']=='bfloat16'
    assert g['exclusive_gpu'] and g['validated'] and len(g['samples'])>=4
    assert all(s['method_median_ms']<s['dense_median_ms'] and s['input_kind']=='captured_real_qkv' for s in g['samples'])
    assert set(g['environment'])==set(DEFAULTS)
    return g


def main():
    import argparse
    p=argparse.ArgumentParser()
    for name in ('validation','benchmark','guard','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();v=json.loads(a.validation.read_text());b=json.loads(a.benchmark.read_text())
    guard=json.loads((a.guard/'completion.json').read_text())
    assert guard['valid'] and guard['returncode']==0
    assert v['passed'] and b['complete'] and b['passed']
    assert v['source_sha256']==b['source_sha256']==source_hashes()
    assert v['environment']==b['environment']
    g={'passed':True,'axes':'wh','capacity':8,'implementation':'r2','validated':True,
       'exclusive_gpu':True,'scope':b['scope'],'shape':b['shape'],'dtype':'bfloat16',
       'source_sha256':b['source_sha256'],'environment':b['environment'],'samples':b['samples'],
       'validation':str(a.validation),'benchmark':str(a.benchmark),'guard':str(a.guard),
       'module':'attention.axis_current_c8_cuda','dense_mean_attention_ms':b['dense_mean_attention_ms'],
       'mean_attention_ms':b['mean_attention_ms'],'speedup_vs_sdpa':b['speedup_vs_sdpa']}
    save(a.output,g);verify_gate(a.output)
    print(json.dumps({key:g[key] for key in ('passed','mean_attention_ms','speedup_vs_sdpa')}))


if __name__=='__main__':main()
