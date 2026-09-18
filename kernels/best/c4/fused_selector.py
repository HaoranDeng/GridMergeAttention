"""Experimental CUDA Uniform16 logits/FP64 Fisher C4 fusion."""
import os,hashlib
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

def extension(loop_frames=False,cooperative=False,basis=False,source_path=None):
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a')
    os.environ.setdefault('MAX_JOBS','2')
    source=Path(__file__).with_name('fisher_c4_fused_selector'+('_cooperative' if cooperative else '')+('_basis' if basis else '')+('_loop' if loop_frames else '')+'.cu')
    if source_path is not None:source=Path(source_path).resolve()
    header=source.with_name('warp_mma.cuh')
    digest=hashlib.sha256(source.read_bytes()+header.read_bytes()).hexdigest()
    module=load(name='fisher_c4_fused_selector_'+digest[:16],sources=[str(source)],
                extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=False)
    if hashlib.sha256(source.read_bytes()+header.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Fused-selector sources changed during build')
    module.source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    module.source_path=str(source)
    return module

class NativeFusedFisherSelector:
    def __init__(self,capacity=4,device='cuda',loop_frames=False,cooperative=False,basis=False,source_path=None):
        if capacity!=4:raise ValueError('Experimental fusion supports C4 only')
        from attention.adaptive_seed_pytorch import multiscale_candidate_table
        self.extension=extension(loop_frames,cooperative,basis,source_path)
        self.source_path=Path(self.extension.source_path)
        candidates,anchors=multiscale_candidate_table(45,32,torch.device(device),geometry_count=45)
        self.candidates=candidates.int().contiguous();self.anchors=anchors.int().contiguous()
        self.seeds=torch.empty((1,1,21,45,80,40,4),device=device,dtype=torch.uint8)
        self.empty=torch.empty(0,device=device,dtype=torch.float32)

    def __call__(self,q,k,scale=128**-.5):
        self.extension.run(q,k,self.candidates,self.anchors,self.seeds,scale,self.empty)
        return self.seeds
