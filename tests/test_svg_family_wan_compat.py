"""Tests for SVG-family runtime compatibility adapters."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import torch

from attention.svg_family_wan import _install_flashinfer_workspace_compat


class SVGFamilyWanCompatTests(unittest.TestCase):
    def test_new_flashinfer_wrapper_accepts_legacy_workspace_arguments(self) -> None:
        class Wrapper:
            def __init__(self, float_workspace_buffer, backend="auto"):
                del backend
                self.device = float_workspace_buffer.device
                self._float_workspace_buffer = float_workspace_buffer
                self._int_workspace_buffer = torch.empty(
                    0, dtype=torch.int32, device=self.device
                )
                self.reset_calls = []

            def reset_workspace_buffer(
                self, float_workspace_buffer, int_workspace_buffer
            ):
                self.reset_calls.append(
                    (float_workspace_buffer, int_workspace_buffer)
                )

        flashinfer = types.ModuleType("flashinfer")
        flashinfer.sparse = types.SimpleNamespace(
            VariableBlockSparseAttentionWrapper=Wrapper
        )
        with mock.patch.dict(sys.modules, {"flashinfer": flashinfer}):
            _install_flashinfer_workspace_compat()
            workspace = torch.empty(16)
            wrapper = Wrapper(workspace)
            wrapper.reset_workspace_buffer(
                float_workspace_buffer=workspace,
                int_workspace_buffer=wrapper._int_workspace_buffer,
                vector_sparse_indices_buffer=torch.empty(0, dtype=torch.uint8),
                vector_sparse_indptr_buffer=torch.empty(0, dtype=torch.int32),
            )

        self.assertEqual(len(wrapper.reset_calls), 1)
        self.assertEqual(wrapper._vector_sparse_indices_buffer.numel(), 0)
        self.assertEqual(wrapper._vector_sparse_indptr_buffer.numel(), 0)


if __name__ == "__main__":
    unittest.main()
