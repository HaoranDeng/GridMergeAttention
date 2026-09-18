"""MI300 BF16 probe logits followed by a native HIP FP64 Fisher GS kernel."""
import ctypes
import hashlib
import os
import math
from pathlib import Path
import subprocess

import torch
from native_fisher.selector import _logits
from attention.adaptive_seed_pytorch import multiscale_candidate_table


def build():
    source = Path(__file__).with_name('fisher_gs.hip')
    compiler = Path(os.environ.get('ROCM_PATH', '/opt/rocm-7.2.0')) / 'bin/hipcc'
    flags = ['-O3', '-std=c++17', '-shared', '-fPIC', '--offload-arch=gfx942', '-ffp-contract=off']
    digest = hashlib.sha256(source.read_bytes() + str(compiler).encode() + repr(flags).encode()).hexdigest()
    cache = Path(os.environ.get('GMA_HIP_CACHE', '/tmp/gma_hip_cache')) / digest
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / 'fisher_gs.so'
    if not library.exists():
        temporary = cache / f'fisher_gs.{os.getpid()}.so'
        subprocess.run([str(compiler), *flags, str(source), '-o', str(temporary)], check=True)
        temporary.replace(library)
    lib = ctypes.CDLL(str(library))
    lib.run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 + [ctypes.c_double, ctypes.c_void_p]
    lib.run.restype = ctypes.c_int
    return lib, digest


class HipFisherSelector:
    """Serial, single-stream reusable selector for full Wan14B H40 tensors."""
    def __init__(self, capacity=4, device='cuda:0', wave=32, probes=4, threads=256):
        if torch.version.hip is None:
            raise RuntimeError('ROCm PyTorch is required')
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if 'gfx942' not in torch.cuda.get_device_properties(self.device).gcnArchName:
            raise ValueError('This experiment is compiled for MI300 gfx942')
        if capacity not in (4, 8, 10) or (wave, probes) not in [(32,4),(32,8),(32,16),(64,2),(64,4),(64,8),(64,16)] or threads not in (128,256):
            raise ValueError('Unsupported HIP launch configuration')
        self.capacity, self.wave, self.probes, self.threads = capacity, wave, probes, threads
        self.lib, self.source_sha256 = build()
        candidates, anchors = multiscale_candidate_table(45,32,self.device,geometry_count=45)
        self.candidates, self.anchors = candidates.int().contiguous(), anchors.int().contiguous()
        self.logits = torch.empty((21,45,80,40,32,16), device=self.device, dtype=torch.float32)
        self.seeds = torch.empty((1,1,21,45,80,40,capacity), device=self.device, dtype=torch.uint8)

    def gs(self, scale=128**-.5):
        if not math.isfinite(scale):
            raise ValueError('Scale must be finite')
        with torch.cuda.device(self.device):
            code = self.lib.run(self.logits.data_ptr(),self.candidates.data_ptr(),self.anchors.data_ptr(),self.seeds.data_ptr(),
                21*45*80*40,40,self.capacity,self.wave,self.probes,self.threads,float(scale),torch.cuda.current_stream().cuda_stream)
        if code:
            raise RuntimeError(f'HIP launch failed: {code}')
        return self.seeds

    @torch.no_grad()
    def __call__(self,q,k,scale=128**-.5):
        if tuple(q.shape) not in [(1,1,45,80,40,128),(1,21,45,80,40,128)] or tuple(k.shape)!=(1,21,45,80,40,128):
            raise ValueError('Require full H40 Wan14B shapes, Q may be RF0')
        for x in (q,k):
            if x.dtype != torch.bfloat16 or x.device != self.device or not x.is_contiguous() or x.requires_grad:
                raise ValueError('Require contiguous inference BF16 inputs on the instance device')
        with torch.cuda.device(self.device):
            _logits[(45*20,40)](q,k,self.candidates,self.logits,num_warps=4,num_stages=1)
            self.gs(scale)
        return self.seeds
