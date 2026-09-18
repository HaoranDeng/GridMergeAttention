"""GMA-C8 r2 exact-capacity complete H40 C8 attention call.

This retained implementation fixes the measured merge tile span at 105 and
uses the screened D13 fused-packed-argmax merge source.  The selector, probe,
route, candidate table and buffer layout remain fixed.  Historical campaign
arms are recorded in evidence.json but are not runtime choices.

Each instance owns about 33 GiB of reusable GPU buffers and must be called
serially on one CUDA stream.

C8 is exactly eight merged K/V representatives per key row. No SQ1,
singleton, raw-K/V, residual, or ninth route token is permitted.
"""
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load
from attention.capacity_contract import require_exact_merged_capacity

_extensions = {}
_HERE = Path(__file__).resolve().parent
_SHAPE = (1, 21, 45, 80, 40, 128)
_FLAT = (1, 75600, 40, 128)
C8_REPRESENTATIVES = require_exact_merged_capacity(8, 0)

# Distinct from the packaged c8 snapshot's "_20260908" suffix so that importing
# both packages in one process cannot alias two different sources onto one
# cached extension name.
_SUFFIX = '_c8r2_20260912'

#: Tile span forwarded to merge.run.  See the module docstring: this is a
#: measured tuning constant, not an arbitrary default.  Re-swept in round D10.
MERGE_TILES = 105
MERGE_SOURCE = 'gma_c8_d10a_fused_argmax_v0'


def _extension(stem, driver=False):
    if stem not in _extensions:
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0a')
        os.environ.setdefault('MAX_JOBS', '2')
        includes = Path(torch.__file__).resolve().parent.parent / 'flashinfer/data/cutlass/include'
        _extensions[stem] = load(
            name=stem + _SUFFIX, sources=[str(_HERE / (stem + '.cu'))],
            extra_include_paths=[str(includes)],
            extra_ldflags=['-lcuda'] if driver else None,
            extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-lineinfo',
                               '--ptxas-options=-v'], verbose=False)
    return _extensions[stem]


class NativeC8R2:
    """Complete C8 attention: probe -> greedy select -> pack -> merge -> route."""

    def __init__(self, device='cuda'):
        self.device = torch.empty(0, device=device).device
        with torch.cuda.device(self.device):
            if torch.cuda.get_device_capability(self.device) != (9, 0):
                raise ValueError('This prototype requires Hopper SM90a')
            self.probe = _extension('gma_c8_cuda_probe')
            self.gs = _extension('b1_gs_regpipe_v0')
            self.merge = _extension(MERGE_SOURCE, True)
            self.route = _extension('gma_c8_route_pipeline', True)
            table = json.loads((_HERE / 'gma_c8_candidate_table.json').read_text())
            if (table['rows'], table['candidate_count'], table['geometry_count']) != (45, 32, 45):
                raise ValueError('Unexpected C32/CG45 candidate table')
            self.candidates = torch.tensor(table['candidates'], device=self.device, dtype=torch.int32)
            self.anchors = torch.tensor(table['anchors'], device=self.device, dtype=torch.int32)
            self.probes = torch.empty((1, 1, 21, 45, 20, 40, 16), device=self.device, dtype=torch.int32)
            self.seeds = torch.empty((1, 1, 21, 45, 80, 40, C8_REPRESENTATIVES), device=self.device, dtype=torch.uint8)
            self.mk = torch.empty((21 * 80 * 5, 21 * 45 * C8_REPRESENTATIVES, 128), device=self.device, dtype=torch.bfloat16)
            self.mv = torch.empty_like(self.mk)
            self.bias = torch.empty((21 * 80 * 5, 21 * 45 * C8_REPRESENTATIVES), device=self.device, dtype=torch.float32)
            self.packed_q = torch.empty(_SHAPE, device=self.device, dtype=torch.bfloat16)
            self.empty = torch.empty(0, device=self.device)

    def _check(self, x):
        if (tuple(x.shape) not in (_SHAPE, _FLAT) or x.dtype != torch.bfloat16
                or x.device != self.device or not x.is_contiguous()):
            raise ValueError('Require contiguous BF16 [1,75600,40,128] or '
                             '[1,21,45,80,40,128] on the instance device')
        if x.requires_grad:
            raise ValueError('This prototype supports inference only')
        return x.view(_SHAPE)

    @torch.no_grad()
    def __call__(self, q, k, v, scale=128 ** -0.5, output=None):
        shape = q.shape
        q, k, v = [self._check(x) for x in (q, k, v)]
        if output is None:
            output = torch.empty_like(q)
        else:
            output = self._check(output)
            if output.data_ptr() in (q.data_ptr(), k.data_ptr(), v.data_ptr()):
                raise ValueError('Output must not alias an input')
        self.probe.run(q[:, :1], k, self.probes, scale)
        self.gs.run(q[:, :1], k, self.probes, self.candidates, self.anchors, self.seeds, 0.2, scale)
        # Lossless register-fragment layout; original Q is retained for routing.
        self.merge.pack(q, self.packed_q)
        for head in range(0, 40, 5):
            # 105 is the measured tuning constant; 21 is correct but slower.
            self.merge.run(self.packed_q, k, v, self.seeds, self.mk, self.mv, self.bias,
                           head, 5, MERGE_TILES, scale, self.empty, 0, 1)
            self.route.run(q, self.mk, self.mv, self.bias, output, head, 5, scale)
        return output.view(shape)

    def resources(self):
        """ptxas/occupancy facts for the active merge kernel."""
        registers, local_bytes, smem_bytes, ctas_per_sm = self.merge.resources()
        return {'tiles': MERGE_TILES, 'registers': registers,
                'local_bytes': local_bytes, 'smem_bytes': smem_bytes,
                'ctas_per_sm': ctas_per_sm}
