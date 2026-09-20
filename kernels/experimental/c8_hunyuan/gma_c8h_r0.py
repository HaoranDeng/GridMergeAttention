"""GMA-C8 "Hunyuan" port: round 0, get-it-running.

Ports kernels/best/c8_r2 (probe -> greedy select -> pack -> merge -> route)
from the Wan shape [1,21,45,80,40,128] to the Hunyuan Video 13B shape
[1,33,30,45,24,128]. See the four .cu files in this directory for the
constant-by-constant derivation; this wrapper just wires them together and
owns the intermediate buffers, mirroring kernels/best/c8_r2/gma_c8_r2.py.

STATUS: round-0 correctness port. Not measured for speed, not screened for
numerical fidelity against any Hunyuan reference (there isn't one yet in
this repo). See the docstrings below and the port's final report for the
specific open questions (uniform-probe-column placement approximation, the
C_OUT=48 padding scheme, the geometry_count=20 candidate table choice).

Shape notes baked into every kernel here:
  temporal (KO for K/V, Q for Q) 21 -> 33
  rows / H                       45 -> 30
  real cols / W                  80 -> 45   (NOT a multiple of 4 or 8: see
                                             the .cu file headers for how
                                             each kernel pads/masks this)
  heads                          40 -> 24
  head-chunk width (AH)           5 -> 8    (24/8 = 3 launches per call,
                                             vs the Wan wrapper's 40/5 = 8)
  candidate_count                32 -> 16
  geometry_count                 45 -> 20   (geometry_count=rows=30 is
                                             infeasible for candidate_count
                                             16 -- multiscale_candidate_table
                                             raises "more mandatory rows than
                                             target pool"; 20 is the largest
                                             feasible value, see
                                             generate_candidate_table.py)
  MERGE_TILES / LEGAL_TILES   {21,42,105,210} -> {33,66,99,198}, default 198
                                             (single grid.z; this round did
                                             not retune the tile span)
"""
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_extensions = {}
_HERE = Path(__file__).resolve().parent
_SHAPE = (1, 33, 30, 45, 24, 128)
_FLAT = (1, 33 * 30 * 45, 24, 128)

_SUFFIX = '_c8h_20260919'

#: Tile span forwarded to merge.run. 198 = 33*6 is the whole span in one
#: grid.z step (CG8 = ceil(45/8) = 6 column-groups per temporal frame); this
#: round did not sweep smaller legal spans for speed.
MERGE_TILES = 198
LEGAL_TILES = (33, 66, 99, 198)

#: Head-chunk width per merge/route launch. 24 heads / 8 per chunk = 3
#: launches (the Wan shape did 40/5 = 8 launches).
AH = 8
HEADS = 24


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


class NativeC8Hunyuan:
    """Complete C8-Hunyuan attention: probe -> greedy select -> pack -> merge -> route.

    Only the "numerics" tier (fused packed argmax) is ported -- the Wan
    package's bitwise tier and overlap wrapper are not part of this round.
    """

    def __init__(self, device='cuda', tiles=MERGE_TILES):
        if tiles not in LEGAL_TILES:
            raise ValueError(f'tiles must be one of {LEGAL_TILES}; got {tiles!r}')
        self.device = torch.empty(0, device=device).device
        self.tiles = int(tiles)
        with torch.cuda.device(self.device):
            if torch.cuda.get_device_capability(self.device) != (9, 0):
                raise ValueError('This prototype requires Hopper SM90a')
            self.probe = _extension('gma_c8h_cuda_probe')
            self.gs = _extension('b1h_gs_regpipe_v0')
            self.merge = _extension('gma_c8h_merge_v0', True)
            self.route = _extension('gma_c8h_route_pipeline', True)
            table = json.loads((_HERE / 'gma_c8h_candidate_table.json').read_text())
            if (table['rows'], table['candidate_count'], table['geometry_count']) != (30, 16, 20):
                raise ValueError('Unexpected C16/CG20/rows30 candidate table')
            self.candidates = torch.tensor(table['candidates'], device=self.device, dtype=torch.int32)
            self.anchors = torch.tensor(table['anchors'], device=self.device, dtype=torch.int32)
            # probes: [B,RF,KO,R,CG,H,16], CG=ceil(45/4)=12
            self.probes = torch.empty((1, 1, 33, 30, 12, 24, 16), device=self.device, dtype=torch.int32)
            # seeds: [B,RF,KO,R,W,H,8], real W=45 (no output padding here --
            # the selector kernel skips stores for the padding sub-columns)
            self.seeds = torch.empty((1, 1, 33, 30, 45, 24, 8), device=self.device, dtype=torch.uint8)
            # mk/mv/bias use the PADDED column stride C_OUT=48 (see
            # gma_c8h_merge_v0.cu's header for why 45 can't be used directly).
            KO, C_OUT, KEYPOOL = 33, 48, 7920
            self.mk = torch.empty((KO * C_OUT * AH, KEYPOOL, 128), device=self.device, dtype=torch.bfloat16)
            self.mv = torch.empty_like(self.mk)
            self.bias = torch.empty((KO * C_OUT * AH, KEYPOOL), device=self.device, dtype=torch.float32)
            self.packed_q = torch.empty(_SHAPE, device=self.device, dtype=torch.bfloat16)
            self.empty = torch.empty(0, device=self.device)

    def _check(self, x):
        if (tuple(x.shape) not in (_SHAPE, _FLAT) or x.dtype != torch.bfloat16
                or x.device != self.device or not x.is_contiguous()):
            raise ValueError('Require contiguous BF16 [1,44550,24,128] or '
                             '[1,33,30,45,24,128] on the instance device')
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
        self.merge.pack(q, self.packed_q)
        for head in range(0, HEADS, AH):
            self.merge.run(self.packed_q, k, v, self.seeds, self.mk, self.mv, self.bias,
                           head, AH, self.tiles, scale, self.empty, 0, 1)
            self.route.run(q, self.mk, self.mv, self.bias, output, head, AH, scale)
        return output.view(shape)

    def resources(self):
        registers, local_bytes, smem_bytes, ctas_per_sm = self.merge.resources()
        return {'tiles': self.tiles, 'registers': registers,
                'local_bytes': local_bytes, 'smem_bytes': smem_bytes,
                'ctas_per_sm': ctas_per_sm}


def create(device='cuda'):
    return NativeC8Hunyuan(device=device)
