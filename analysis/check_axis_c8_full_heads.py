"""Check the complete H40/chunked call against separately validated single heads."""
import argparse
import gc
import json
from pathlib import Path
import torch
from attention.axis_current_c8_cuda import AxisCurrentC8,_workspace
from analysis.axis_c8_speed_gate import source_hashes,environment
from analysis.quick_axis_c4_qkv import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    capture=Path('/local2/denghaoran/GridMergeAttention/experiments/full40_bf16_dense_capture_20260905bb/captures/prompt000_call033_layer33.pt')
    payload=torch.load(capture,map_location='cpu',mmap=True,weights_only=True)
    tensors=[payload[n].permute(0,2,1,3).to('cuda').contiguous() for n in ('q','k','v')]
    del payload
    actual=AxisCurrentC8()(*tensors)
    assert torch.isfinite(actual).all()
    _workspace.cache_clear();gc.collect();torch.cuda.empty_cache()
    report={'passed':False,'source_sha256':source_hashes(),'environment':environment(),
            'capture':str(capture),'heads':[],'diagnostic_relative_l2_limit':0.003}
    for head in (0,10,20,30,39):
        expected=AxisCurrentC8()(*[x[:,head:head+1].contiguous() for x in tensors])
        got=actual[:,head:head+1].float();ref=expected.float()
        relative=float((got-ref).norm()/ref.norm())
        report['heads'].append({'head':head,'relative_l2':relative,'max_absolute':float((got-ref).abs().max())})
        assert relative<0.003,report['heads'][-1]
    report['passed']=True;save(a.output,report)
    print(json.dumps(report['heads']),flush=True)


if __name__=='__main__':main()
