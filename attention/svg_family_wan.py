"""Pure all-layer SVG-family attention for the native Wan model."""

from __future__ import annotations

import inspect
import os
import sys
import types

import torch


_PROCESSORS = {}
_CONFIGURED = set()
_MODULES = None


def _install_flashinfer_workspace_compat():
    """Accept the legacy SVG workspace arguments on newer FlashInfer."""
    if torch.version.hip is not None:
        return

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


def _install_rocm_import_stubs():
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


def _load_svg_modules():
    global _MODULES
    if _MODULES is not None:
        return _MODULES
    root = os.environ.get("SPARSE_VIDEO_GEN_ROOT")
    if not root:
        raise RuntimeError("SPARSE_VIDEO_GEN_ROOT is required for SVG-family attention.")
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "svg", "kernels", "build"))
    _install_rocm_import_stubs()
    _install_flashinfer_workspace_compat()

    from svg.models.wan import attention as svg_attention
    from svg.models.wan.attention import (
        WanAttn_EARAttn_Processor,
        WanAttn_SAPAttn_Processor,
        WanAttn_SVGAttn_Processor2_0,
        prepare_flexattention,
    )
    from svg.models.wan.utils import get_attention_mask, sparsity_to_width

    if torch.version.hip is not None:
        from svg.kmeans_utils import dynamic_block_sparse_fwd_triton

        from attention.svg_ear_rocm import (
            dynamic_block_sparse_prune_fwd_triton_rocm,
        )

        def dynamic_block_sparse_fwd_rocm(
            query,
            key,
            value,
            dynamic_map,
            query_cluster_sizes,
            key_cluster_sizes,
            **_kwargs,
        ):
            return dynamic_block_sparse_fwd_triton(
                query,
                key,
                value,
                dynamic_map,
                query_cluster_sizes,
                key_cluster_sizes,
            )

        svg_attention.dynamic_block_sparse_fwd_flashinfer = (
            dynamic_block_sparse_fwd_rocm
        )
        svg_attention.dynamic_block_sparse_prune_fwd_flashinfer = (
            dynamic_block_sparse_prune_fwd_triton_rocm
        )

    _MODULES = {
        "attention": svg_attention,
        "svg": WanAttn_SVGAttn_Processor2_0,
        "svg2": WanAttn_SAPAttn_Processor,
        "ear": WanAttn_EARAttn_Processor,
        "prepare_flexattention": prepare_flexattention,
        "get_attention_mask": get_attention_mask,
        "sparsity_to_width": sparsity_to_width,
    }
    return _MODULES


def _configure(backend, grid_size, query):
    device_key = (query.device.type, query.device.index)
    key = (backend, tuple(int(value) for value in grid_size), device_key)
    if key in _CONFIGURED:
        return _load_svg_modules()

    modules = _load_svg_modules()
    frames, height, width = (int(value) for value in grid_size)
    sequence = frames * height * width
    processor_type = modules[backend]
    processor_type.first_layers_fp = 0
    processor_type.first_times_fp = 1001
    processor_type.context_length = 0
    processor_type.num_frame = frames
    processor_type.frame_size = height * width

    if backend == "svg":
        sparsity = 0.3
        max_rows = min(10000, sequence)
        processor_type.num_sampled_rows = 64
        processor_type.sample_mse_max_row = max_rows
        processor_type.sparsity = sparsity
        processor_type.attention_masks = [
            modules["get_attention_mask"](
                name,
                max_rows,
                0,
                frames,
                height * width,
            )
            for name in ("spatial", "temporal")
        ]
        width_blocks = modules["sparsity_to_width"](
            sparsity, 0, frames, height * width
        )
        processor_type.block_mask = modules["prepare_flexattention"](
            query.size(0),
            query.size(2),
            query.size(3),
            query.dtype,
            query.device,
            0,
            0,
            frames,
            height * width,
            width_blocks,
            width_blocks,
        )
    else:
        processor_type.num_q_centroids = 300
        processor_type.num_k_centroids = 1000
        processor_type.top_p_kmeans = float(os.environ.get("SVG_TOP_P", "0.9"))
        processor_type.min_kc_ratio = 0.10
        processor_type.kmeans_iter_init = 50
        processor_type.kmeans_iter_step = 2
        processor_type.zero_step_kmeans_init = False
        processor_type.logging_file = os.environ.get("SVG_DENSITY_LOG") or None

    _CONFIGURED.add(key)
    return modules


def svg_family_attention(query, key, value, backend, layer_idx, grid_size):
    """Run SVG, SVG-2, or SVG-EAR on native [B, S, H, D] Wan QKV."""
    backend = {"svg_all": "svg", "svg2_all": "svg2", "ear_all": "ear"}[backend]
    modules = _configure(backend, grid_size, query)
    processor_key = (
        backend,
        int(layer_idx),
        tuple(int(value) for value in grid_size),
        query.device.type,
        query.device.index,
    )
    processor = _PROCESSORS.get(processor_key)
    if processor is None:
        processor = modules[backend](layer_idx=int(layer_idx))
        _PROCESSORS[processor_key] = processor

    output = processor.attention_core_logic(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        torch.zeros(1, device=query.device),
    )
    return output.transpose(1, 2).contiguous()


def reset_svg_family_state():
    for processor in _PROCESSORS.values():
        if hasattr(processor, "centroids_init"):
            processor.centroids_init = False
            processor.q_centroids = None
            processor.k_centroids = None
