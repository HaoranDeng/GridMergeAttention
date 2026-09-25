"""Current-frame Hellinger C8; retains r2's C32/g45, A8U8 and alpha=0.2.

An instance owns scratch and supports serial inference on one CUDA stream.
RF0 control and current-frame paths use identical arithmetic and eight slots.
The only algorithmic change is selecting independently for every query frame.
"""
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


@lru_cache(None)
def extension(name, heads):
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0a')
    os.environ.setdefault('MAX_JOBS', '2')
    source = HERE / (name + '.cu')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    includes = Path(torch.__file__).resolve().parent.parent / 'flashinfer/data/cutlass/include'
    return load(name=f'c8_current_{name}_{heads}_{digest}', sources=[str(source)],
                extra_include_paths=[str(includes)], extra_ldflags=['-lcuda'],
                extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-lineinfo',
                                  f'-DC8_HEADS={heads}'], verbose=False)


class CurrentFrameC8:
    def __init__(self, device='cuda', heads=40):
        if heads not in (1, 40):
            raise ValueError('Production H40 or diagnostic H1 required')
        self.device = torch.empty(0, device=device).device
        if torch.cuda.get_device_capability(self.device) != (9, 0):
            raise ValueError('Hopper SM90a required')
        self.heads, self.chunk = heads, 5 if heads == 40 else 1
        self.shape = (1, 21, 45, 80, heads, 128)
        with torch.cuda.device(self.device):
            self.probe, self.selector, self.merge, self.route = [extension(n, heads) for n in
                                                               ('probe', 'selector', 'merge', 'route')]
            table = json.loads((ROOT / 'kernels/best/c8_r2/gma_c8_candidate_table.json').read_text())
            self.candidates = torch.tensor(table['candidates'], device=self.device, dtype=torch.int32)
            self.anchors = torch.tensor(table['anchors'], device=self.device, dtype=torch.int32)
            self.probes = torch.empty((1, 1, 21, 45, 20, heads, 16), device=self.device, dtype=torch.int32)
            self.seeds = torch.empty((1, 21, 21, 45, 80, heads, 8), device=self.device, dtype=torch.uint8)
            self.mk = torch.empty((21*80*self.chunk, 21*45*8, 128), device=self.device, dtype=torch.bfloat16)
            self.mv = torch.empty_like(self.mk)
            self.bias = torch.empty(self.mk.shape[:-1], device=self.device, dtype=torch.float32)
            self.packed_q = torch.empty(self.shape, device=self.device, dtype=torch.bfloat16)
            self.empty = torch.empty(0, device=self.device)

    def _check(self, x):
        if (tuple(x.shape) not in (self.shape, (1, 75600, self.heads, 128)) or
                x.dtype != torch.bfloat16 or x.device != self.device or
                not x.is_contiguous() or x.requires_grad):
            raise ValueError('Contiguous inference BF16 token-major QKV on instance device required')
        return x.view(self.shape)

    @torch.no_grad()
    def select(self, q, k, scale=128**-.5, *, policy='current'):
        if policy not in ('current', 'rf0'):
            raise ValueError('policy must be current or rf0')
        q, k = self._check(q), self._check(k)
        frames = 21 if policy == 'current' else 1
        for frame in range(frames):
            self.probe.run(q[:, frame:frame+1], k, self.probes, scale)
            self.selector.run(q[:, frame:frame+1], k, self.probes, self.candidates,
                              self.anchors, self.seeds[:, frame:frame+1], .2, scale)
        return self.seeds[:, :frames]

    @torch.no_grad()
    def run_with_seeds(self, q, k, v, seeds, scale=128**-.5, output=None):
        original = q.shape
        q, k, v = [self._check(x) for x in (q, k, v)]
        if (seeds.shape not in ((1, 1, 21, 45, 80, self.heads, 8), self.seeds.shape)
                or seeds.dtype != torch.uint8 or seeds.device != self.device or not seeds.is_contiguous()):
            raise ValueError('Exact-eight current or RF0 seed tensor required')
        output = torch.empty_like(q) if output is None else self._check(output)
        if any(output.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for x in (q,k,v)):
            raise ValueError('Output must not alias inputs')
        self.merge.pack(q, self.packed_q)
        for head in range(0, self.heads, self.chunk):
            self.merge.run(self.packed_q, k, v, seeds, self.mk, self.mv, self.bias,
                           head, self.chunk, 105, scale, self.empty, 0, 1)
            self.route.run(q, self.mk, self.mv, self.bias, output, head, self.chunk, scale)
        return output.view(original)

    @torch.no_grad()
    def __call__(self, q, k, v, scale=128**-.5, output=None, *, policy='current'):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Positive finite attention scale required')
        seeds = self.select(q, k, scale, policy=policy)
        return self.run_with_seeds(q, k, v, seeds, scale, output)
