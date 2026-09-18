"""Generate matched Wan2.1-14B 720p videos for supported attention methods.

The script uses the official Sparse-VideoGen Diffusers pipeline for every
method so all quality comparisons share one checkpoint, scheduler, prompt,
seed, and outer attention implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import sys
import time
import types
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F


GCA_STABLE_BACKEND = "gca_fisher_anchor_stable_rf0_all"
GCA_HW_SWAP_BACKEND = "gca_fisher_anchor_stable_rf0_hw_swap_all"
GCA_STABLE_BACKENDS = ()
GCA_STABLE_METHOD = "fisher_anchor_stable_sq1_rf0"
GCA_STABLE_KERNEL_SHA256 = "fa2a729139cdbfee1abba9265eedf463c7f68b0f7de1f4402d6d48c2fe3a8e68"
GCA_MERGE_SYMBOL = "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks"


class DiffusersDispatchAudit:
    """Per-video branch counts; no tensor reads or additional synchronization."""

    def __init__(self, backend, num_steps, num_layers, first_steps, first_layers,
                 guidance_scale=5.0):
        self.backend, self.num_steps, self.num_layers = backend, num_steps, num_layers
        self.cfg_passes = 2 if guidance_scale > 1.0 else 1
        self.first_steps, self.first_layers = first_steps, first_layers
        self.expected_calls = num_steps * num_layers * self.cfg_passes
        self.gca_core, self.active_index, self.videos = None, None, []
        self.enabled = True

    def begin_video(self, index):
        if self.active_index is not None:
            raise RuntimeError("previous dispatch audit video did not finish")
        self.active_index = index
        self.layers = [dict(dense=0, sparse=0, gca=0) for _ in range(self.num_layers)]
        self.gca_before = self.gca_core.counts() if self.gca_core is not None else None

    def record(self, layer, branch):
        if not self.enabled:
            return
        if self.active_index is None or not 0 <= layer < self.num_layers:
            raise RuntimeError("attention dispatch outside an active video/layer")
        if branch not in {"dense", "sparse", "gca"}:
            raise ValueError("unknown attention branch")
        self.layers[layer][branch] += 1

    def finish_video(self, core_calls, full_calls):
        totals = {branch: sum(row[branch] for row in self.layers)
                  for branch in ("dense", "sparse", "gca")}
        expected_dense = self.cfg_passes * (
            self.first_steps * self.num_layers
            + (self.num_steps - self.first_steps) * self.first_layers)
        expected = (dict(dense=expected_dense, sparse=0, gca=self.expected_calls - expected_dense)
                    if self.gca_core is not None or self.backend == "gma_c4_mix" else
                    dict(dense=expected_dense, sparse=self.expected_calls - expected_dense, gca=0))
        if (self.active_index is None or totals != expected
                or core_calls != self.expected_calls or full_calls != self.expected_calls
                or any(sum(row.values()) != self.num_steps * self.cfg_passes for row in self.layers)):
            raise RuntimeError("incomplete/mixed Diffusers attention dispatch")
        row = {"index": self.active_index, "total_calls": core_calls, **totals,
               "per_layer": deepcopy(self.layers)}
        if self.gca_core is not None:
            counts = self.gca_core.counts()
            delta = {key: counts[key] - self.gca_before[key] for key in counts}
            expected_gca = {"core_calls": core_calls, "selector_calls": core_calls,
                            "merge_wrapper_calls": 5 * core_calls, "stable_kernel_calls": 5 * core_calls}
            if delta != expected_gca:
                raise RuntimeError("GCA selector/stable merge dispatch did not cover every attention")
            self.gca_core.verify_kernel()
            row["gca_dispatch"] = delta
        self.videos.append(row)
        self.active_index = None
        return row

    def summary(self):
        result = {"backend": self.backend, "cfg_passes_per_step": self.cfg_passes,
                  "expected_calls_per_video": self.expected_calls,
                  "videos": deepcopy(self.videos),
                  "instrumentation": "host dispatch counters; no additional tensor reads or CUDA synchronization"}
        if self.gca_core is not None:
            result["gca"] = self.gca_core.metadata()
        return result


class GCAStableRF0Core:
    """Use the reviewed selector and stable merge inside the original full route."""

    def __init__(self, token, selector, adapter, receipt):
        from analysis.quality_first_mode_contract import VIDEO_MODES, method_contract
        self.token, self.selector, self.adapter, self.receipt = token, selector, adapter, receipt
        self.seed_mode = VIDEO_MODES[GCA_STABLE_METHOD]
        self.declaration = method_contract(GCA_STABLE_METHOD)
        self.original_selector = getattr(token, self.declaration["selector_symbol"])
        self.original_merge = getattr(token, GCA_MERGE_SYMBOL)
        self.original_core = token._hard_c4_attention
        self.core_calls = self.selector_calls = 0
        self.active = False

    def verify_kernel(self):
        if (self.receipt["sha256"] != GCA_STABLE_KERNEL_SHA256
                or hashlib.sha256(Path(self.receipt["path"]).read_bytes()).hexdigest() != GCA_STABLE_KERNEL_SHA256):
            raise RuntimeError("generated stable SQ1 kernel differs from the validated kernel")

    def counts(self):
        return {"core_calls": self.core_calls, "selector_calls": self.selector_calls,
                "merge_wrapper_calls": self.adapter.calls, "stable_kernel_calls": self.adapter.kernel_calls}

    def metadata(self):
        return {"declaration": self.declaration, "seed_mode": self.seed_mode,
                "head_chunk": 8, "merge_block_j": 16, "generated_kernel": self.receipt,
                "dispatch_counts": self.counts(), "dense_fallback": False, "fp8": False}

    def select(self, query, key, scale, **kwargs):
        expected = {"candidate_count": 32, "candidate_geometry_count": 45, "block_j": 4,
                    "probe_count": 16, "global_probe_count": 256, "gs_num_warps": 4,
                    "output_dtype": torch.uint8}
        if (not self.active or self.seeds is not None or kwargs != expected or scale != 128 ** -.5
                or tuple(query.shape) != (1, 1, 45, 80, 40, 128)
                or tuple(key.shape) != (1, 21, 45, 80, 40, 128)
                or query.data_ptr() != self.qkv[0].data_ptr() or key.data_ptr() != self.qkv[1].data_ptr()):
            raise RuntimeError("GCA requires the unchanged RF0-only real-row Anchor-Fisher selector")
        self.seeds = self.selector(query, key, scale, **kwargs)
        self.selector_calls += 1
        return self.seeds

    def merge(self, query, key, value, seeds, scale, **kwargs):
        if (not self.active or seeds is not self.seeds or seeds is None
                or any(tensor.data_ptr() != original.data_ptr()
                       for tensor, original in zip((query, key, value), self.qkv))):
            raise RuntimeError("GCA merge did not consume current QKV and the observed seed table")
        return self.adapter(query, key, value, seeds, scale, **kwargs)

    def __call__(self, query, key, value):
        if (self.active or getattr(self.token, self.declaration["selector_symbol"]) is not self.original_selector
                or getattr(self.token, GCA_MERGE_SYMBOL) is not self.original_merge):
            raise RuntimeError("nested or externally changed GCA bindings")
        for tensor in (query, key, value):
            if (tuple(tensor.shape) != (1, 40, 75600, 128) or tensor.dtype != torch.bfloat16
                    or not tensor.is_cuda or tensor.device != query.device):
                raise ValueError("GCA stable RF0 requires original 720p/81-frame CUDA BF16 QKV")
        self.qkv = tuple(tensor.transpose(1, 2).contiguous() for tensor in (query, key, value))
        before = self.counts()
        self.seeds, self.active = None, True
        try:
            with patch.object(self.token, self.declaration["selector_symbol"], self.select), patch.object(
                    self.token, GCA_MERGE_SYMBOL, self.merge):
                output = self.original_core(
                    *self.qkv, h=45, w=80, h_reduce=1, w_reduce=1, sm_scale=128 ** -.5,
                    query_outer_chunk=21, head_chunk=8, hard_route_backend="hard_c4",
                    exact_intra_frame_attention=False, exact_neighbor_window_columns=0,
                    exact_query_rows=0, first_frame_full_attention=False,
                    full_attention_anchor_time_count=0, full_attention_anchor_row_count=0,
                    full_attention_anchor_scale=1.0, seed_mode=self.seed_mode)
            after = self.counts()
            if (after["selector_calls"] - before["selector_calls"] != 1
                    or after["merge_wrapper_calls"] - before["merge_wrapper_calls"] != 5
                    or after["stable_kernel_calls"] - before["stable_kernel_calls"] != 5):
                raise RuntimeError("GCA must execute one selector and five stable merge kernels per core")
            self.core_calls += 1
            return output.transpose(1, 2).contiguous()
        finally:
            self.active = False
            self.qkv = self.seeds = None


def create_gca_stable_rf0_core(output_dir):
    from analysis.stable_sq1_video_adapter import StableSQ1MergeAdapter, load_generated_kernel
    from attention.affine_hellinger_seed_triton import fisher_anchor_seed_rows_triton
    token = importlib.import_module("attention.token_merging")
    hard = importlib.import_module("attention.hard_c4_wan21_14b_720p_triton")
    kernel, receipt = load_generated_kernel(hard._hard_c4_720p_temporal_shared_merge_kernel,
                                          Path(output_dir) / "stable_kernel.py")
    adapter = StableSQ1MergeAdapter(getattr(token, GCA_MERGE_SYMBOL), hard, kernel)
    core = GCAStableRF0Core(token, fisher_anchor_seed_rows_triton, adapter, receipt)
    core.verify_kernel()
    return core


def create_gca_hw_swap_core(output_dir):
    from attention.spatial_transpose_gca_triton import create_spatial_transpose_core
    core = create_spatial_transpose_core(output_dir)
    core.verify_kernel()
    return core


NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen "
    "limbs, fused fingers, still picture, messy background, three legs, many "
    "people in the background, walking backwards"
)


def install_rocm_import_stubs():
    """Stub optional Sparse-VideoGen imports when an environment omits them."""
    if torch.version.hip is not None and "flashinfer" not in sys.modules:
        sys.modules["flashinfer"] = types.ModuleType("flashinfer")

    try:
        from cuvs.cluster import kmeans as _cuvs_kmeans  # noqa: F401
    except ModuleNotFoundError:
        cuvs = types.ModuleType("cuvs")
        cluster = types.ModuleType("cuvs.cluster")
        kmeans = types.ModuleType("cuvs.cluster.kmeans")
        kmeans.KMeansParams = object
        kmeans.fit = None
        cuvs.cluster = cluster
        cluster.kmeans = kmeans
        sys.modules.update(
            {
                "cuvs": cuvs,
                "cuvs.cluster": cluster,
                "cuvs.cluster.kmeans": kmeans,
            }
        )


def install_flashinfer_workspace_compat():
    """Accept the legacy SVG workspace arguments on newer FlashInfer."""
    import flashinfer

    wrapper_type = flashinfer.sparse.VariableBlockSparseAttentionWrapper
    reset = wrapper_type.reset_workspace_buffer
    if getattr(reset, "__svg_workspace_compat__", False):
        return
    if "vector_sparse_indices_buffer" in inspect.signature(reset).parameters:
        return

    original_init = wrapper_type.__init__

    def compatible_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._vector_sparse_indices_buffer = torch.empty(
            0, dtype=torch.uint8, device=self.device
        )
        self._vector_sparse_indptr_buffer = torch.empty(
            0, dtype=torch.int32, device=self.device
        )

    def compatible_reset(
        self,
        float_workspace_buffer,
        int_workspace_buffer,
        vector_sparse_indices_buffer=None,
        vector_sparse_indptr_buffer=None,
    ):
        del vector_sparse_indices_buffer, vector_sparse_indptr_buffer
        return reset(self, float_workspace_buffer, int_workspace_buffer)

    compatible_reset.__svg_workspace_compat__ = True
    wrapper_type.__init__ = compatible_init
    wrapper_type.reset_workspace_buffer = compatible_reset


class AttentionTimer:
    """Collect asynchronous CUDA-event timings without synchronizing each layer."""

    def __init__(self):
        self.events = []
        self.enabled = True

    def reset(self):
        self.events.clear()

    def measure(self, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function(*args, **kwargs)
        end.record()
        self.events.append((start, end))
        return output

    def finish(self):
        torch.cuda.synchronize()
        total_ms = sum(start.elapsed_time(end) for start, end in self.events)
        return total_ms, len(self.events)


class AttentionErrorStats:
    """Aggregate local attention-output error against dense SDPA on GPU."""

    def __init__(self, max_calls):
        self.max_calls = max_calls
        self.enabled = False
        self.calls = 0
        self.numel = 0
        self.sqerr = None
        self.ref_sq = None
        self.max_abs = None

    def should_record(self):
        return self.enabled and (self.max_calls < 0 or self.calls < self.max_calls)

    def add(self, output, reference):
        if not self.should_record():
            return
        output_f = output.float()
        reference_f = reference.float()
        difference = output_f - reference_f
        sqerr = difference.square().sum()
        ref_sq = reference_f.square().sum()
        max_abs = difference.abs().max()
        self.sqerr = sqerr if self.sqerr is None else self.sqerr + sqerr
        self.ref_sq = ref_sq if self.ref_sq is None else self.ref_sq + ref_sq
        self.max_abs = max_abs if self.max_abs is None else torch.maximum(self.max_abs, max_abs)
        self.numel += difference.numel()
        self.calls += 1

    def summary(self):
        if not self.calls:
            return {
                "calls": 0,
                "numel": 0,
                "squared_error_sum": 0.0,
                "reference_squared_sum": 0.0,
                "mse_vs_dense": 0.0,
                "rel_l2_vs_dense": 0.0,
                "max_abs_vs_dense": 0.0,
            }
        sqerr = float(self.sqerr.item())
        ref_sq = float(self.ref_sq.item())
        return {
            "calls": self.calls,
            "numel": self.numel,
            "squared_error_sum": sqerr,
            "reference_squared_sum": ref_sq,
            "mse_vs_dense": sqerr / self.numel,
            "rel_l2_vs_dense": math.sqrt(sqerr / max(ref_sq, 1e-20)),
            "max_abs_vs_dense": float(self.max_abs.item()),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=(
            "dense",
            "svg",
            "svg_all",
            "svg2",
            "svg2_all",
            "ear",
            "ear_all",
            "gma",
            "gma_c4_mix",
            "gma_c4_mix_no_entropy_bias",
            "gma_c4_all_no_entropy_bias",
            "gma_c5_mix",
            "gma_c6_mix",
            "gma_c6_all",
            "gma_hellinger_p16_all",
            "gma_first_frame_global_hellinger_all",
            "gma_c4_mix_transpose",
            "gma_grid24x80",
            "gma_grid48x40",
            "vsa_training_free_all",
        ),
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--sparse_video_gen_root", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--canonical_names", action="store_true",
                        help="Use canonical zero-based NNN-00.mp4 filenames; default retains NNN.mp4.")
    parser.add_argument("--original_script_seeding", action="store_true",
                        help="Reset Python/NumPy/CPU/CUDA RNGs per video like independent upstream prompt runs.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument(
        "--svg_pattern_density",
        type=float,
        default=0.30,
        help=(
            "SVG upstream pattern-area parameter. The realized mask density also "
            "includes the first-frame sink and 128-token width rounding."
        ),
    )
    parser.add_argument(
        "--svg_top_p",
        type=float,
        default=0.9,
        help="SVG-2/EAR cumulative attention-mass threshold; not realized density.",
    )
    parser.add_argument(
        "--svg_min_kc_ratio",
        type=float,
        default=0.10,
        help="Minimum retained key-cluster ratio for SVG-2/EAR.",
    )
    parser.add_argument(
        "--attention_error_calls",
        type=int,
        default=0,
        help="Separate untimed pass: 0 disables, -1 records every self-attention call.",
    )
    parser.add_argument(
        "--attention_error_inline",
        action="store_true",
        help="Measure error during video generation instead of a separate pass.",
    )
    parser.add_argument(
        "--attention_error_sparse_region_only",
        action="store_true",
        help=(
            "For GMA, accumulate error only where SVG-2/EAR use sparse attention "
            "so method comparisons cover the same layers and denoise steps."
        ),
    )
    parser.add_argument(
        "--wan_forward_backend",
        choices=("diffusers", "svg"),
        default="svg",
        help="Transformer-block implementation used by dense and GMA backends.",
    )
    parser.add_argument(
        "--qk_norm_backend",
        choices=("pytorch", "triton"),
        default="triton",
    )
    parser.add_argument(
        "--rope_backend",
        choices=("legacy", "svg"),
        default="svg",
    )
    parser.add_argument(
        "--gma_preprocess_backend",
        choices=("svg", "fused"),
        default="fused",
        help="GMA Q/K norm, QKV layout, and RoPE implementation.",
    )
    parser.add_argument(
        "--gma_output_layout",
        choices=("head_major", "token"),
        default="token",
        help="Store GMA output head-major or directly in projection-ready token layout.",
    )
    parser.add_argument("--gma_head_chunk", type=int, default=10)
    parser.add_argument("--vsa_sparsity", type=float, default=0.9)
    return parser.parse_args()


def configure_processors(
    pipe,
    backend,
    timer,
    full_attention_timer,
    error_stats,
    sparse_video_gen_root,
    num_steps,
    height,
    width,
    num_frames,
    attention_error_sparse_region_only=False,
    wan_forward_backend="svg",
    qk_norm_backend="triton",
    rope_backend="svg",
    gma_preprocess_backend="fused",
    gma_output_layout="token",
    gma_head_chunk=10,
    vsa_sparsity=0.9,
    svg_pattern_density=0.30,
    svg_top_p=0.9,
    svg_min_kc_ratio=0.10,
    output_dir=None,
):
    if backend in GCA_STABLE_BACKENDS and (
            (height, width, num_frames, gma_head_chunk, len(pipe.transformer.blocks)) != (720, 1280, 81, 8, 40)
            or output_dir is None):
        raise ValueError("stable GCA requires 720p/81 frames, all 40 layers, head_chunk=8 and output_dir")
    scheduler = deepcopy(pipe.scheduler)
    scheduler.set_timesteps(num_steps)
    all_layer_gma_backends = {
        "gma_c4_all_no_entropy_bias",
        "gma_c6_all",
        "gma_hellinger_p16_all",
        *GCA_STABLE_BACKENDS,
        "gma_first_frame_global_hellinger_all",
        "gma_grid24x80",
    }
    all_layer_backends = {
        "svg_all",
        "svg2_all",
        "ear_all",
        "vsa_training_free_all",
        *all_layer_gma_backends,
    }
    use_dense_fallback = backend not in all_layer_backends
    first_times_count = math.floor(0.2 * num_steps) if use_dense_fallback else 0
    first_times_fp = (
        int(scheduler.timesteps[first_times_count - 1].item()) - 1
        if first_times_count
        else 1001
    )
    first_layers_fp = (
        math.floor(0.03 * len(pipe.transformer.blocks))
        if use_dense_fallback
        else 0
    )
    dispatch_audit = (DiffusersDispatchAudit(backend, num_steps, len(pipe.transformer.blocks),
                                           first_times_count, first_layers_fp)
                      if backend in {"svg", "svg2", "ear", "gma_c4_mix", *GCA_STABLE_BACKENDS} else None)

    sparse_root = Path(sparse_video_gen_root).resolve()
    sys.path.insert(0, str(sparse_root))
    sys.path.insert(0, str(sparse_root / "svg" / "kernels" / "build"))
    install_rocm_import_stubs()

    mixed_gma_backends = {
        "gma_c4_mix",
        "gma_c4_mix_no_entropy_bias",
        "gma_c5_mix",
        "gma_c6_mix",
        "gma_c4_mix_transpose",
    }
    grid_gma_backends = {"gma_grid24x80", "gma_grid48x40"}
    gma_backends = {
        "gma",
        *mixed_gma_backends,
        *all_layer_gma_backends,
        *grid_gma_backends,
    }
    direct_token_output = (
        backend in {"gma", "gma_c4_mix"} and gma_output_layout == "token"
    )
    optimized_backends = {"dense", "vsa_training_free_all", *gma_backends}

    triton_rmsnorm_forward = None
    svg_apply_rotary_emb = None
    if backend in optimized_backends and (
        wan_forward_backend == "svg"
        or qk_norm_backend == "triton"
        or rope_backend == "svg"
    ):
        # SVG compiles flex_attention at import time, although Dense/GMA never
        # call it. Skipping that unused compile also lets Torch 2.6 use a newer
        # standalone Triton for the custom kernels below.
        torch_compile = torch.compile
        torch.compile = lambda function, *args, **kwargs: function
        try:
            from svg.kernels.triton.rmsnorm import triton_rmsnorm_forward
            from svg.models.wan.attention import (
                apply_rotary_emb as svg_apply_rotary_emb,
            )
            from svg.models.wan.custom_models import replace_sparse_forward
        finally:
            torch.compile = torch_compile

        if wan_forward_backend == "svg":
            replace_sparse_forward()

    gma_attention_fn = None
    token_gma_attention_fn = None
    gca_core = None
    vsa_attention_fn = None
    fused_preprocess_qkv_rope = None
    if backend in gma_backends and gma_preprocess_backend == "fused":
        from attention.wan_qkv_preprocess_triton import (
            preprocess_qkv_head_major_rope as fused_preprocess_qkv_rope,
        )

    if backend in GCA_STABLE_BACKENDS:
        factory = create_gca_stable_rf0_core if backend == GCA_STABLE_BACKEND else create_gca_hw_swap_core
        gca_core = factory(output_dir)
        token_gma_attention_fn = gca_core
        dispatch_audit.gca_core = gca_core

    if backend in {
        "gma_hellinger_p16_all",
        "gma_first_frame_global_hellinger_all",
    }:
        from attention.token_merging import _hard_c4_attention

        seed_mode = {
            "gma_hellinger_p16_all": (
                "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_cg45_"
                "sq1_qm_fp8score_skv_nl_ml2"
            ),
            "gma_first_frame_global_hellinger_all": (
                "hellinger_span_alpha0p2_c32_p80_a2_j4_t1_rf0_ffg_cg45_"
                "sq1_qm_fp8score_skv_nl_ml2"
            ),
        }[backend]

        def token_gma_attention_fn(query, key, value):
            query_token = query.transpose(1, 2).contiguous()
            key_token = key.transpose(1, 2).contiguous()
            value_token = value.transpose(1, 2).contiguous()
            output_token = _hard_c4_attention(
                query_token,
                key_token,
                value_token,
                h=height // 16,
                w=width // 16,
                h_reduce=1,
                w_reduce=1,
                sm_scale=query.size(-1) ** -0.5,
                query_outer_chunk=(num_frames - 1) // 4 + 1,
                head_chunk=gma_head_chunk,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=seed_mode,
            )
            return output_token.transpose(1, 2).contiguous()

    if backend in gma_backends and token_gma_attention_fn is None:
        if backend in grid_gma_backends:
            from attention.hard_c4_grid_wan21_14b_720p_triton import (
                hard_c4_grid24x80_720p_attention_triton_head_major,
                hard_c4_grid48x40_720p_attention_triton_head_major,
            )

            gma_attention_fn = (
                hard_c4_grid24x80_720p_attention_triton_head_major
                if backend == "gma_grid24x80"
                else hard_c4_grid48x40_720p_attention_triton_head_major
            )
        elif backend == "gma_c4_mix_transpose":
            from attention.hard_c4_wan21_14b_720p_transpose_triton import (
                hard_c4_720p_transpose_attention_triton_head_major,
            )

            gma_attention_fn = hard_c4_720p_transpose_attention_triton_head_major
        elif backend in {
            "gma_c4_mix_no_entropy_bias",
            "gma_c4_all_no_entropy_bias",
        }:
            from attention.hard_c4_wan21_14b_720p_triton import (
                hard_c4_720p_attention_triton_head_major_no_entropy_bias,
            )

            gma_attention_fn = (
                hard_c4_720p_attention_triton_head_major_no_entropy_bias
            )
        elif backend in {"gma_c5_mix", "gma_c6_mix", "gma_c6_all"}:
            from attention.hard_c5_c6_wan21_14b_720p_triton import (
                hard_c5_720p_attention_triton_head_major,
                hard_c6_720p_attention_triton_head_major,
            )

            gma_attention_fn = (
                hard_c5_720p_attention_triton_head_major
                if backend == "gma_c5_mix"
                else hard_c6_720p_attention_triton_head_major
            )
        else:
            if direct_token_output:
                from attention.hard_c4_wan21_14b_720p_triton import (
                    hard_c4_720p_attention_triton_head_major_to_token,
                )

                gma_attention_fn = (
                    hard_c4_720p_attention_triton_head_major_to_token
                )
            else:
                from attention.hard_c4_wan21_14b_720p_triton import (
                    hard_c4_720p_attention_triton_head_major,
                )

                gma_attention_fn = hard_c4_720p_attention_triton_head_major

    if backend == "vsa_training_free_all":
        from attention.vsa_training_free import vsa_training_free_attention

        vsa_attention_fn = vsa_training_free_attention

    class TimedWanProcessor:
        """Diffusers Wan processor with a timed dense or custom attention core."""

        def __init__(self, use_gma, layer_idx):
            self.use_gma = use_gma
            self.layer_idx = layer_idx

        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            rotary_emb=None,
            timestep=None,
        ):
            return full_attention_timer.measure(
                self.forward,
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                rotary_emb,
                timestep,
            )

        def forward(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            rotary_emb=None,
            timestep=None,
        ):
            if gca_core is not None and (encoder_hidden_states is not None
                    or attention_mask is not None or attn.add_k_proj is not None):
                raise ValueError("pure GCA supports only unmasked T2V self-attention, without added dense paths")
            encoder_hidden_states_img = None
            if attn.add_k_proj is not None:
                image_context_length = encoder_hidden_states.shape[1] - 512
                encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
                encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states

            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            if (
                fused_preprocess_qkv_rope is not None
                and qk_norm_backend == "triton"
                and rope_backend == "svg"
                and attn.norm_q is not None
                and attn.norm_k is not None
                and rotary_emb is not None
            ):
                if not isinstance(rotary_emb, tuple):
                    rotary_emb = (
                        rotary_emb.real.squeeze(0).squeeze(0).contiguous().float(),
                        rotary_emb.imag.squeeze(0).squeeze(0).contiguous().float(),
                    )
                query, key, value = fused_preprocess_qkv_rope(
                    query,
                    key,
                    value,
                    attn.norm_q.weight,
                    attn.norm_k.weight,
                    attn.norm_q.eps,
                    attn.norm_k.eps,
                    attn.heads,
                    rotary_emb[0],
                    rotary_emb[1],
                )
            else:
                if attn.norm_q is not None:
                    query = (
                        triton_rmsnorm_forward(
                            query, attn.norm_q.weight, attn.norm_q.eps
                        )
                        if qk_norm_backend == "triton"
                        else attn.norm_q(query)
                    )
                if attn.norm_k is not None:
                    key = (
                        triton_rmsnorm_forward(
                            key, attn.norm_k.weight, attn.norm_k.eps
                        )
                        if qk_norm_backend == "triton"
                        else attn.norm_k(key)
                    )

                query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
                key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
                value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
                if rotary_emb is not None:
                    if rope_backend == "svg":
                        if not isinstance(rotary_emb, tuple):
                            rotary_emb = (
                                rotary_emb.real.squeeze(0).squeeze(0).contiguous().float(),
                                rotary_emb.imag.squeeze(0).squeeze(0).contiguous().float(),
                            )
                        query, key = svg_apply_rotary_emb(query, key, rotary_emb)
                    else:
                        if isinstance(rotary_emb, tuple):
                            rotary_emb = torch.complex(*rotary_emb)[None, None]
                        query = self.apply_legacy_rotary(query, rotary_emb)
                        key = self.apply_legacy_rotary(key, rotary_emb)

            hidden_states_img = None
            if encoder_hidden_states_img is not None:
                key_img = attn.norm_added_k(attn.add_k_proj(encoder_hidden_states_img))
                value_img = attn.add_v_proj(encoder_hidden_states_img)
                key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
                value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
                hidden_states_img = F.scaled_dot_product_attention(
                    query,
                    key_img,
                    value_img,
                    dropout_p=0.0,
                    is_causal=False,
                )
                hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)

            use_gma = self.use_gma and not (
                backend in mixed_gma_backends
                and (
                    self.layer_idx < first_layers_fp
                    or pipe._current_timestep > first_times_fp
                )
            )
            if gca_core is not None and not use_gma:
                raise RuntimeError("GCA unexpectedly selected dense attention")
            hidden_states = timer.measure(
                self.gma_attention if use_gma else F.scaled_dot_product_attention,
                query,
                key,
                value,
                **(
                    {}
                    if use_gma
                    else {
                        "attn_mask": attention_mask,
                        "dropout_p": 0.0,
                        "is_causal": False,
                    }
                ),
            )
            if dispatch_audit is not None:
                dispatch_audit.record(self.layer_idx, "gca" if use_gma else "dense")
            token_output = use_gma and direct_token_output
            hidden_states_flat = (
                hidden_states
                if token_output
                else hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)
            )
            record_error = use_gma and error_stats.should_record()
            if attention_error_sparse_region_only:
                record_error = record_error and not (
                    self.layer_idx < first_layers_fp
                    or pipe._current_timestep > first_times_fp
                )
            if record_error:
                dense = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
                dense = dense.transpose(1, 2).flatten(2, 3).type_as(query)
                error_stats.add(hidden_states_flat, dense)
            hidden_states = hidden_states_flat
            if hidden_states_img is not None:
                hidden_states = hidden_states + hidden_states_img.type_as(query)
            return attn.to_out[1](attn.to_out[0](hidden_states))

        @staticmethod
        def apply_legacy_rotary(hidden_states, frequencies):
            dtype = (
                torch.float32
                if hidden_states.device.type == "mps"
                else torch.float64
            )
            rotated = torch.view_as_complex(
                hidden_states.to(dtype).unflatten(3, (-1, 2))
            )
            return torch.view_as_real(rotated * frequencies).flatten(3, 4).type_as(
                hidden_states
            )

        @staticmethod
        def gma_attention(query, key, value):
            if token_gma_attention_fn is not None:
                return token_gma_attention_fn(query, key, value)
            if vsa_attention_fn is not None:
                return vsa_attention_fn(
                    query,
                    key,
                    value,
                    video_shape=(
                        (num_frames - 1) // 4 + 1,
                        height // 16,
                        width // 16,
                    ),
                    sparsity=vsa_sparsity,
                )
            output = (
                torch.empty(
                    (query.size(0), query.size(2), query.size(1) * query.size(3)),
                    dtype=query.dtype,
                    device=query.device,
                )
                if direct_token_output
                else torch.empty_like(query)
            )
            head_chunk = gma_head_chunk
            for head_start in range(0, query.size(1), head_chunk):
                gma_attention_fn(
                    query,
                    key,
                    value,
                    output=output,
                    head_start=head_start,
                    head_count=min(head_chunk, query.size(1) - head_start),
                )
            return output

    if backend in optimized_backends:
        for layer_idx, block in enumerate(pipe.transformer.blocks):
            processor = TimedWanProcessor(
                use_gma=backend != "dense", layer_idx=layer_idx
            )
            block.attn1.set_processor(processor)
        return dispatch_audit

    if backend in {"svg2", "svg2_all", "ear", "ear_all"} and torch.version.hip is None:
        install_flashinfer_workspace_compat()
        # Source checkouts may omit the package data directory. Keep a wheel's
        # version-matched JIT templates when they are already available.
        from flashinfer.jit import env as flashinfer_env

        required_templates = (
            "batch_prefill_customize_config.jinja",
            "batch_prefill_sm90_customize_config.jinja",
        )
        if not all(
            (Path(flashinfer_env.FLASHINFER_CSRC_DIR) / name).is_file()
            for name in required_templates
        ):
            flashinfer_root = (
                sparse_root / "svg" / "kernels" / "3rdparty" / "flashinfer"
            )
            flashinfer_env.FLASHINFER_CSRC_DIR = flashinfer_root / "csrc"
            flashinfer_env.FLASHINFER_INCLUDE_DIR = flashinfer_root / "include"
            flashinfer_env.CUTLASS_INCLUDE_DIRS = [
                flashinfer_root / "3rdparty" / "cutlass" / "include",
                flashinfer_root
                / "3rdparty"
                / "cutlass"
                / "tools"
                / "util"
                / "include",
            ]
            flashinfer_env.SPDLOG_INCLUDE_DIR = (
                flashinfer_root / "3rdparty" / "spdlog" / "include"
            )
    from svg.models.wan import attention as svg_attention
    from svg.models.wan.attention import (
        WanAttn_EARAttn_Processor,
        WanAttn_SAPAttn_Processor,
        WanAttn_SVGAttn_Processor2_0,
    )
    from svg.models.wan.inference import replace_wan_attention

    if torch.version.hip is not None:
        from svg.kmeans_utils import dynamic_block_sparse_fwd_triton
        from attention.svg_ear_rocm import dynamic_block_sparse_prune_fwd_triton_rocm

        def dynamic_block_sparse_fwd_rocm(
            query,
            key,
            value,
            dynamic_map,
            query_cluster_sizes,
            key_cluster_sizes,
            **_kwargs,
        ):
            """Exact ROCm fallback for SVG-2's variable block-sparse attention."""
            return dynamic_block_sparse_fwd_triton(
                query,
                key,
                value,
                dynamic_map,
                query_cluster_sizes,
                key_cluster_sizes,
            )

        svg_attention.dynamic_block_sparse_fwd_flashinfer = dynamic_block_sparse_fwd_rocm
        svg_attention.dynamic_block_sparse_prune_fwd_flashinfer = (
            dynamic_block_sparse_prune_fwd_triton_rocm
        )

    class TimedSparseProcessor:
        """Preserve Diffusers' processor signature while timing sparse attention."""

        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            rotary_emb=None,
            timestep=None,
        ):
            return full_attention_timer.measure(
                super().__call__,
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                rotary_emb,
                timestep,
            )

        def attention_core_logic(self, query, key, value, timestep):
            sparse_call = not (
                self.layer_idx < self.first_layers_fp
                or timestep[0] > self.first_times_fp
            )
            parent = super().attention_core_logic
            output = timer.measure(parent, query, key, value, timestep)
            if dispatch_audit is not None:
                dispatch_audit.record(self.layer_idx, "sparse" if sparse_call else "dense")
            if sparse_call and error_stats.should_record():
                dense = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
                error_stats.add(output, dense)
            return output

    class TimedSVG(TimedSparseProcessor, WanAttn_SVGAttn_Processor2_0):
        pass

    class TimedSVG2(TimedSparseProcessor, WanAttn_SAPAttn_Processor):
        pass

    class TimedEAR(TimedSparseProcessor, WanAttn_EARAttn_Processor):
        pass

    if backend in {"svg", "svg_all"}:
        replace_wan_attention(
            pipe,
            height,
            width,
            num_frames,
            first_layers_fp=first_layers_fp,
            first_times_fp=first_times_fp,
            pattern="SVG",
            num_sampled_rows=64,
            sparsity=svg_pattern_density,
        )
        processor_type = TimedSVG
    elif backend in {"svg2", "svg2_all"}:
        replace_wan_attention(
            pipe,
            height,
            width,
            num_frames,
            first_layers_fp=first_layers_fp,
            first_times_fp=first_times_fp,
            pattern="SAP",
            num_q_centroids=300,
            num_k_centroids=1000,
            top_p_kmeans=svg_top_p,
            min_kc_ratio=svg_min_kc_ratio,
            kmeans_iter_init=50,
            kmeans_iter_step=2,
            logging_file=os.environ.get("SVG_DENSITY_LOG"),
        )
        processor_type = TimedSVG2
    elif backend in {"ear", "ear_all"}:
        replace_wan_attention(
            pipe,
            height,
            width,
            num_frames,
            first_layers_fp=first_layers_fp,
            first_times_fp=first_times_fp,
            pattern="EAR",
            num_q_centroids=300,
            num_k_centroids=1000,
            top_p_kmeans=svg_top_p,
            min_kc_ratio=svg_min_kc_ratio,
            kmeans_iter_init=50,
            kmeans_iter_step=2,
            logging_file=os.environ.get("SVG_DENSITY_LOG"),
        )
        processor_type = TimedEAR
    for layer_idx, block in enumerate(pipe.transformer.blocks):
        block.attn1.set_processor(processor_type(layer_idx=layer_idx))
    return dispatch_audit


def reset_video_state(pipe, backend):
    if backend not in {"svg2", "svg2_all", "ear", "ear_all"}:
        return
    for block in pipe.transformer.blocks:
        processor = block.attn1.processor
        processor.centroids_init = False
        processor.q_centroids = None
        processor.k_centroids = None


def summarize(records):
    ordered = [records[index] for index in sorted(records)]
    steady = ordered[1:] if len(ordered) > 1 else ordered
    total_attention_ms = sum(item["attention_total_ms"] for item in ordered)
    total_calls = sum(item["attention_calls"] for item in ordered)
    steady_attention_ms = sum(item["attention_total_ms"] for item in steady)
    steady_calls = sum(item["attention_calls"] for item in steady)
    summary = {
        "mp4_count": len(ordered),
        "attention_calls": total_calls,
        "mean_attention_ms": total_attention_ms / total_calls,
        "mean_inference_seconds": float(np.mean([item["inference_seconds"] for item in ordered])),
        "steady_state_videos": len(steady),
        "steady_state_mean_attention_ms": steady_attention_ms / steady_calls,
        "steady_state_mean_inference_seconds": float(
            np.mean([item["inference_seconds"] for item in steady])
        ),
        "mean_export_seconds": float(np.mean([item["export_seconds"] for item in ordered])),
        "max_memory_gib": max(item["max_memory_gib"] for item in ordered),
    }
    if all("full_attention_total_ms" in item for item in ordered):
        summary.update(
            {
                "mean_full_attention_ms": sum(
                    item["full_attention_total_ms"] for item in ordered
                )
                / sum(item["full_attention_calls"] for item in ordered),
                "steady_state_mean_full_attention_ms": sum(
                    item["full_attention_total_ms"] for item in steady
                )
                / sum(item["full_attention_calls"] for item in steady),
            }
        )
    return summary


def main():
    args = parse_args()
    if args.backend in GCA_STABLE_BACKENDS and (args.gma_head_chunk != 8 or args.attention_error_calls):
        raise ValueError("stable GCA requires --gma_head_chunk 8 and no dense attention-error pass")
    sparse_root = Path(args.sparse_video_gen_root).resolve()
    sys.path.insert(0, str(sparse_root))
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import (
        UniPCMultistepScheduler,
    )
    from diffusers.utils import export_to_video

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.preferred_linalg_library(backend="magma")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = output_dir / "runtime.json"
    records = {}
    if runtime_path.exists():
        records = {
            item["index"]: item
            for item in json.loads(runtime_path.read_text())["videos"]
        }

    prompts = [
        line.strip()
        for line in Path(args.prompt_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = range(args.start, min(args.start + args.count, len(prompts)))

    if args.canonical_names:
        from utils.prompt_contract import (
            build_prompt_manifest, ensure_output_prompt_manifest,
            validate_canonical_prompt_contract,
        )
        validate_canonical_prompt_contract(args.prompt_path, prompts)
        if args.start != 0 or len(indices) != args.count:
            raise ValueError("canonical comparison requires start=0 and complete prompt coverage")
        ensure_output_prompt_manifest(output_dir, build_prompt_manifest(
            args.prompt_path, prompts, selected_count=args.count, seed=args.seed,
            num_samples=1, vbench_filenames=False, allow_noncanonical=False))

    if os.environ.get("GMA_MI300_HYBRID_PREFLIGHT") == "1":
        from analysis.preflight_mi300_hybrid import run as preflight
        preflight(sys.modules[__name__], args.backend, sparse_root, output_dir.parent)

    vae = AutoencoderKLWan.from_pretrained(
        args.model,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=5.0,
    )
    pipe = WanPipeline.from_pretrained(
        args.model,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.scheduler = scheduler
    pipe.to("cuda")

    timer = AttentionTimer()
    full_attention_timer = AttentionTimer()
    error_stats = AttentionErrorStats(args.attention_error_calls)
    dispatch_audit = configure_processors(
        pipe,
        args.backend,
        timer,
        full_attention_timer,
        error_stats,
        sparse_root,
        args.num_steps,
        args.height,
        args.width,
        args.num_frames,
        args.attention_error_sparse_region_only,
        args.wan_forward_backend,
        args.qk_norm_backend,
        args.rope_backend,
        args.gma_preprocess_backend,
        args.gma_output_layout,
        args.gma_head_chunk,
        args.vsa_sparsity,
        args.svg_pattern_density,
        args.svg_top_p,
        args.svg_min_kc_ratio,
        output_dir=output_dir,
    )
    if args.attention_error_inline and args.backend != "dense":
        error_stats.enabled = True

    slice_started = time.monotonic()
    slice_videos = 0
    for index in indices:
        video_path = output_dir / (f"{index:03d}-00.mp4" if args.canonical_names else f"{index:03d}.mp4")
        if video_path.exists() and index in records:
            continue
        if (slice_videos and os.environ.get("GMA_MI300_SLICE_SECONDS")
                and time.monotonic() - slice_started > float(os.environ["GMA_MI300_SLICE_SECONDS"])):
            print("Saved complete videos; ending allocation slice", flush=True)
            break

        if args.original_script_seeding:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        reset_video_state(pipe, args.backend)
        if dispatch_audit is not None:
            dispatch_audit.begin_video(index)
        timer.reset()
        full_attention_timer.reset()
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        frames = pipe(
            prompt=prompts[index],
            negative_prompt=NEGATIVE_PROMPT,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=5.0,
            num_inference_steps=args.num_steps,
            generator=generator,
        ).frames[0]
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        attention_total_ms, attention_calls = timer.finish()
        full_attention_total_ms, full_attention_calls = full_attention_timer.finish()
        dispatch_record = (dispatch_audit.finish_video(attention_calls, full_attention_calls)
                           if dispatch_audit is not None else None)

        export_started = time.perf_counter()
        export_to_video(frames, str(video_path), fps=16)
        export_seconds = time.perf_counter() - export_started
        records[index] = {
            "index": index,
            "prompt": prompts[index],
            "seed": args.seed,
            "video": video_path.name,
            "attention_calls": attention_calls,
            "attention_total_ms": attention_total_ms,
            "mean_attention_ms": attention_total_ms / attention_calls,
            "full_attention_calls": full_attention_calls,
            "full_attention_total_ms": full_attention_total_ms,
            "mean_full_attention_ms": full_attention_total_ms
            / full_attention_calls,
            "inference_seconds": inference_seconds,
            "export_seconds": export_seconds,
            "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
            "first_video_in_allocation": slice_videos == 0,
        }
        if dispatch_record is not None:
            records[index]["attention_dispatch"] = dispatch_record
        payload = {
            "protocol": {
                "model": str(Path(args.model).resolve()),
                "resolution": f"{args.width}x{args.height}",
                "num_frames": args.num_frames,
                "num_steps": args.num_steps,
                "guidance_scale": 5.0,
                "prompt_path": str(Path(args.prompt_path).resolve()),
                "seed": args.seed,
                "canonical_names": args.canonical_names,
                "original_script_seeding": args.original_script_seeding,
                "wan_forward_backend": args.wan_forward_backend,
                "qk_norm_backend": args.qk_norm_backend,
                "rope_backend": args.rope_backend,
                "gma_preprocess_backend": args.gma_preprocess_backend,
                "gma_output_layout": args.gma_output_layout,
                "gma_head_chunk": args.gma_head_chunk,
                "vsa_sparsity": args.vsa_sparsity,
                "svg_pattern_density": args.svg_pattern_density,
                "svg_top_p": args.svg_top_p,
                "svg_min_kc_ratio": args.svg_min_kc_ratio,
                "svg_repository": "https://github.com/svg-project/Sparse-VideoGen",
                "timing": (
                    "CUDA events for core and full self-attention; wall time for "
                    "pipeline; model loading excluded"
                ),
            },
            "backend": args.backend,
            "videos": [records[i] for i in sorted(records)],
            "summary": summarize(records),
        }
        if dispatch_audit is not None:
            payload["dispatch_audit"] = dispatch_audit.summary()
            (output_dir / "dispatch_audit.json").write_text(
                json.dumps(payload["dispatch_audit"], indent=2), encoding="utf-8")
        if args.backend in {"svg", "svg2", "ear"}:
            payload["protocol"]["attention_schedule"] = {
                "dense_first_steps": math.floor(0.2 * args.num_steps),
                "dense_first_layers": math.floor(0.03 * len(pipe.transformer.blocks)),
                "transformer_layers": len(pipe.transformer.blocks), "cfg_passes_per_step": 2,
                "remaining_attention": {"svg": "SVG", "svg2": "SVG-2 (SAP)", "ear": "SVG-EAR"}[args.backend],
                "method_classification": "original upstream hybrid external baseline"}
        if args.backend in GCA_STABLE_BACKENDS:
            payload["protocol"].update(
                method_classification="training-free pure all-layer GCA", pure_all_layer_gma=True,
                entropy_routing_bias=True, gma_seed_mode=dispatch_audit.gca_core.seed_mode,
                gca_stable=dispatch_audit.gca_core.metadata())
        if args.backend in {
            "gma_c4_mix",
            "gma_c4_mix_no_entropy_bias",
            "gma_c5_mix",
            "gma_c6_mix",
            "gma_c4_mix_transpose",
        }:
            merge_tokens = {
                "gma_c4_mix": 4,
                "gma_c4_mix_no_entropy_bias": 4,
                "gma_c5_mix": 5,
                "gma_c6_mix": 6,
                "gma_c4_mix_transpose": 4,
            }[args.backend]
            payload["protocol"]["attention_schedule"] = {
                "dense_first_steps": math.floor(0.2 * args.num_steps),
                "dense_first_layers": math.floor(
                    0.03 * len(pipe.transformer.blocks)
                ),
                "transformer_layers": len(pipe.transformer.blocks),
                "remaining_attention": f"GMA C{merge_tokens}",
            }
            payload["protocol"]["entropy_routing_bias"] = (
                args.backend != "gma_c4_mix_no_entropy_bias"
            )
        all_layer_labels = {
            "svg_all": "SVG",
            "svg2_all": "SVG-2 (SAP)",
            "ear_all": "SVG-EAR",
            "gma_c4_all_no_entropy_bias": "GMA C4",
            "gma_c6_all": "GMA C6",
            "gma_hellinger_p16_all": (
                "Hellinger-Span C4 C32-G45 alpha=0.2 P16 J4 T1"
            ),
            GCA_STABLE_BACKEND: "Grid Compressed Attention: Anchor-Fisher RF0, stable C4+SQ1",
            GCA_HW_SWAP_BACKEND: "Grid Compressed Attention: Anchor-Fisher RF0, stable C4+SQ1, key columns / query rows",
            "gma_first_frame_global_hellinger_all": (
                "First-frame global Hellinger C4 (dense router)"
            ),
            "gma_grid24x80": "GMA C4 (h_reduce=2, w_reduce=1)",
            "vsa_training_free_all": "VSA training-free",
        }
        if args.backend in all_layer_labels:
            payload["protocol"]["attention_schedule"] = {
                "dense_first_steps": 0,
                "dense_first_layers": 0,
                "transformer_layers": len(pipe.transformer.blocks),
                "remaining_attention": all_layer_labels[args.backend],
            }
        if args.backend == "gma_c4_all_no_entropy_bias":
            payload["protocol"]["entropy_routing_bias"] = False
        if args.backend == "gma_c6_all":
            payload["protocol"]["entropy_routing_bias"] = True
        if args.backend == "gma_hellinger_p16_all":
            payload["protocol"]["method_classification"] = (
                "training-free pure all-layer GMA"
            )
            payload["protocol"]["pure_all_layer_gma"] = True
            payload["protocol"]["entropy_routing_bias"] = True
            payload["protocol"]["gma_seed_mode"] = (
                "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_cg45_"
                "sq1_qm_fp8score_skv_nl_ml2"
            )
        if args.backend == "gma_first_frame_global_hellinger_all":
            payload["protocol"]["method_classification"] = (
                "dense-router diagnostic; not pure GMA"
            )
            payload["protocol"]["pure_all_layer_gma"] = False
            payload["protocol"]["entropy_routing_bias"] = True
            payload["protocol"]["gma_seed_mode"] = (
                "hellinger_span_alpha0p2_c32_p80_a2_j4_t1_rf0_ffg_cg45_"
                "sq1_qm_fp8score_skv_nl_ml2"
            )
            payload["protocol"]["dense_router"] = {
                "query_frames": 1,
                "query_tokens": (args.height // 16) * (args.width // 16),
                "key_tokens": (
                    ((args.num_frames - 1) // 4 + 1)
                    * (args.height // 16)
                    * (args.width // 16)
                ),
                "recomputed_per_attention_call": True,
                "dense_output_rescue": False,
            }
        if args.backend == "vsa_training_free_all":
            payload["protocol"]["vsa"] = {
                "implementation": "FastVideo official Triton fallback",
                "tile_size": [4, 4, 4],
                "block_tokens": 64,
                "sparsity": args.vsa_sparsity,
                "trained_compression_gate": False,
            }
        if args.backend == "gma_c4_mix_transpose":
            payload["protocol"]["attention_layout"] = {
                "queries_per_grid": 80,
                "valid_kv_per_grid": 45,
                "padded_kv_per_grid": 48,
                "compressed_kv_per_grid": 4,
            }
        if args.backend in {"gma_grid24x80", "gma_grid48x40"}:
            query_rows, key_cols = {
                "gma_grid24x80": (24, 80),
                "gma_grid48x40": (48, 40),
            }[args.backend]
            payload["protocol"]["attention_layout"] = {
                "physical_query_rows": 45,
                "physical_key_cols": 80,
                "padded_query_rows": 48,
                "query_rows_per_grid": query_rows,
                "key_cols_per_grid": key_cols,
                "compressed_kv_per_grid": 4,
                "full_second_stage_attention": True,
            }
        if args.backend == "gma_grid24x80":
            payload["protocol"]["method_classification"] = (
                "training-free pure all-layer GMA"
            )
            payload["protocol"]["pure_all_layer_gma"] = True
            payload["protocol"]["num_merge_tokens"] = 4
            payload["protocol"]["h_reduce"] = 2
            payload["protocol"]["w_reduce"] = 1
        runtime_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(records[index]), flush=True)
        slice_videos += 1

    if args.attention_error_calls:
        if dispatch_audit is not None:
            dispatch_audit.enabled = False
        if args.backend == "dense":
            attention_error = error_stats.summary()
            attention_error["baseline"] = "exact dense SDPA"
        elif args.attention_error_inline:
            error_stats.enabled = False
            attention_error = error_stats.summary()
        else:
            reset_video_state(pipe, args.backend)
            timer.enabled = False
            error_stats.enabled = True
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            pipe(
                prompt=prompts[args.start],
                negative_prompt=NEGATIVE_PROMPT,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                guidance_scale=5.0,
                num_inference_steps=args.num_steps,
                generator=generator,
                output_type="latent",
            )
            torch.cuda.synchronize()
            error_stats.enabled = False
            attention_error = error_stats.summary()

        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        payload["protocol"]["attention_error"] = (
            "Inline during generation"
            if args.attention_error_inline
            else "Separate untimed inference"
        ) + (
            "; sparse method output versus dense SDPA on the same call-local QKV; "
            "SVG/SVG-2/EAR dense fallback calls excluded."
        )
        payload["protocol"]["attention_error_sparse_region_only"] = (
            args.attention_error_sparse_region_only
        )
        payload["attention_error_vs_dense"] = attention_error
        runtime_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"attention_error_vs_dense": attention_error}), flush=True)


if __name__ == "__main__":
    main()
