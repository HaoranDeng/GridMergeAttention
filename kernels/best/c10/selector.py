"""Experimental FP64 Fisher GS on unchanged BF16/FP32 probe logits."""
import os
import hashlib
from pathlib import Path
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load

@triton.jit
def _logits(q,k,candidates,out):
    pid=tl.program_id(0);head=tl.program_id(1)
    row=pid//20;column_group=pid%20
    flat=tl.arange(0,128);candidate=flat%32;column=column_group*4+flat//32
    rows=tl.load(candidates+row*32+candidate)
    d=tl.arange(0,128)
    queries=tl.load(q+((rows[:,None]*80+column[:,None])*40+head)*128+d[None,:])
    probes=tl.arange(0,16)*5
    for frame in tl.range(0,21):
        keys=tl.load(k+((((frame*45+row)*80+probes[:,None])*40+head)*128+d[None,:]))
        value=tl.dot(queries,tl.trans(keys),input_precision='ieee')
        offsets=(((((frame*45+row)*80+column[:,None])*40+head)*32+candidate[:,None])*16+tl.arange(0,16)[None,:]).to(tl.int64)
        tl.store(out+offsets,value)

class NativeFisherSelector:
    def __init__(self,capacity,device='cuda',layout='candidate',source_path=None):
        assert capacity in (4,10)
        from attention.adaptive_seed_pytorch import multiscale_candidate_table
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a');os.environ.setdefault('MAX_JOBS','2')
        if layout not in ('candidate','probe4','probe8','probe16'):raise ValueError('Unsupported GS layout')
        self.probe_lanes=0 if layout=='candidate' else int(layout[5:])
        source='fisher_gs.cu' if layout=='candidate' else 'fisher_gs_distributed.cu'
        self.source_path=Path(source_path) if source_path is not None else Path(__file__).with_name(source)
        digest=hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        self.extension=load(name=f'fisher_gs_{digest[:16]}',sources=[str(self.source_path)],
            extra_cuda_cflags=['-O3','--ptxas-options=-v'],verbose=False)
        if hashlib.sha256(self.source_path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Fisher GS source changed during compilation')
        self.candidates,self.anchors=multiscale_candidate_table(45,32,torch.device(device),geometry_count=45)
        self.candidates=self.candidates.int().contiguous();self.anchors=self.anchors.int().contiguous()
        self.logits=torch.empty((21,45,80,40,32,16),device=device,dtype=torch.float32)
        self.seeds=torch.empty((1,1,21,45,80,40,capacity),device=device,dtype=torch.uint8)

    def __call__(self,q,k,scale=128**-.5):
        _logits[(45*20,40)](q,k,self.candidates,self.logits,num_warps=4,num_stages=1)
        self.gs(scale)
        return self.seeds

    def gs(self,scale=128**-.5):
        args=(self.logits,self.candidates,self.anchors,self.seeds,scale)
        self.extension.run(*args,self.probe_lanes) if self.probe_lanes else self.extension.run(*args)
        return self.seeds
