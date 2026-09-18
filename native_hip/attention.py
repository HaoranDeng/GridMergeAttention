"""Opt-in MI300 C4/SQ1 attention, retaining full compressed attention.

BF16 rounding differs slightly from the stock launch. Instances own selector
scratch and are intended for sequential calls on one device/stream.
"""
from types import FunctionType, SimpleNamespace

import torch

from attention import hard_c4_wan21_14b_720p_triton as hard
from attention.singleton_c4_wan21_14b_720p_triton import (
    _hard_c4_720p_temporal_shared_global_mass_attention_triton_blocks as _stock,
)
from native_hip.merge_tiled import merge_kernel
from native_hip.selector import HipFisherSelector


class _Launch:
    def __init__(self, kernel, **options):
        self.kernel, self.options = kernel, options

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.update(self.options)
            return self.kernel[grid](*args, **kwargs)
        return launch


class Mi300C4Attention:
    """Fixed Wan 720p BF16 [1,21,45,80,40,128] attention entry point."""

    def __init__(self):
        self.selector = HipFisherSelector()
        # Private launch namespace: importing/using this operator never mutates
        # the stock module's kernel objects or launch constants.
        private = SimpleNamespace(**vars(hard))
        private.HARD_720_MERGE_NUM_WARPS = 2
        private.HARD_720_ROUTE_NUM_WARPS = 4
        private.HARD_720_ROUTE_BLOCK_TOKENS = 64
        private._hard_c4_720p_temporal_shared_merge_kernel = _Launch(
            merge_kernel, matrix_instr_nonkdim=16, RESIDUAL_DIV_MODE=2)
        private._hard_c4_720p_route_kernel = _Launch(
            hard._hard_c4_720p_route_kernel,
            matrix_instr_nonkdim=16, waves_per_eu=2)
        namespace = dict(_stock.__globals__, hard_c4=private)
        self._run = FunctionType(_stock.__code__, namespace, _stock.__name__,
                                 _stock.__defaults__, _stock.__closure__)
        self._run.__kwdefaults__ = _stock.__kwdefaults__

    def __call__(self, q, k, v, *, output=None):
        for x in (q, k, v):
            if (tuple(x.shape) != (1, 21, 45, 80, 40, 128)
                    or x.dtype != torch.bfloat16 or not x.is_cuda
                    or not x.is_contiguous() or x.device != q.device):
                raise ValueError('Expected contiguous same-device BF16 [1,21,45,80,40,128] QKV')
        if output is None:
            output = torch.empty_like(q)
        if (output.shape != q.shape or output.dtype != q.dtype
                or output.device != q.device or not output.is_contiguous()):
            raise ValueError('Output must match Q shape, dtype, device and contiguous layout')
        if any(output.data_ptr() == x.data_ptr() for x in (q, k, v)):
            raise ValueError('Output must not alias QKV')
        seeds = self.selector(q, k)
        for head in (0, 20):
            self._run(q, k, v, seeds, top_r=1, head_start=head, head_count=20,
                      query_column_selector=True, seed_rows_native_layout=True,
                      query_column_major=True, split_kv_accum=True, block_j=16,
                      merge_loop_stages=1, route_loop_stages=1,
                      merge_disable_licm=False, output=output)
        return output
