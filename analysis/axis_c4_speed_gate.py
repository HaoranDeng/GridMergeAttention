"""Fail closed per method; a fast method need not wait for other candidates."""
import json
from pathlib import Path
from analysis.quick_axis_c4_qkv import sha


def verify_gate(path,root,axes,allow_user_speed_exception=False):
    gate=json.loads(Path(path).read_text())
    exception=gate.get('user_speed_exception')
    authorized=(allow_user_speed_exception and axes=='th' and exception
                and exception.get('axes')=='th' and bool(exception.get('instruction')))
    assert gate['passed'] is True or authorized
    assert axes in gate['methods']
    assert set(gate['methods'])<={'tw','wh','th'}
    assert gate['scope']=='full attention including layouts and all40 heads'
    assert gate['dtype']=='bfloat16' and gate['shape']==[1,40,75600,128]
    assert gate['source_sha256']
    for file,digest in gate['source_sha256'].items():
        assert sha(root/file)==digest, f'Speed receipt is stale: {file}'
    for axes,method in gate['methods'].items():
        assert method['validated'] is True and method['exclusive_gpu'] is True
        assert method['samples'] and all(s['input_kind']=='captured_real_qkv' for s in method['samples'])
        assert all(s['method_median_ms']<s['dense_median_ms'] for s in method['samples']) or (authorized and axes=='th'), axes
        assert method['module'].startswith('attention.axis_current_c4_')
    return gate
