import contextlib
import unittest
from unittest import mock

import torch
from torch.nn.attention import SDPBackend

from attention import core


class SDPABackendSelectionTests(unittest.TestCase):
    def test_unset_backend_uses_pytorch_default_dispatch(self):
        sentinel = object()
        query = torch.zeros(1)
        with mock.patch.object(core, "_FORCED_SDPA_BACKEND", ""), mock.patch.object(
            core.F, "scaled_dot_product_attention", return_value=sentinel
        ) as sdpa:
            output = core._scaled_dot_product_attention(
                query, query, query, None, {"dropout_p": 0.0}, False
            )

        self.assertIs(output, sentinel)
        sdpa.assert_called_once_with(
            query, query, query, attn_mask=None, dropout_p=0.0
        )

    def test_flash_backend_uses_explicit_sdpa_context(self):
        sentinel = object()
        query = torch.zeros(1)
        with mock.patch.object(
            core, "_FORCED_SDPA_BACKEND", "flash"
        ), mock.patch(
            "torch.nn.attention.sdpa_kernel",
            return_value=contextlib.nullcontext(),
        ) as kernel, mock.patch.object(
            core.F, "scaled_dot_product_attention", return_value=sentinel
        ):
            output = core._scaled_dot_product_attention(
                query, query, query, None, {"dropout_p": 0.0}, False
            )

        self.assertIs(output, sentinel)
        kernel.assert_called_once_with(backends=[SDPBackend.FLASH_ATTENTION])

    def test_cudnn_backend_uses_explicit_sdpa_context(self):
        query = torch.zeros(1)
        with mock.patch.object(
            core, "_FORCED_SDPA_BACKEND", "cudnn"
        ), mock.patch(
            "torch.nn.attention.sdpa_kernel",
            return_value=contextlib.nullcontext(),
        ) as kernel, mock.patch.object(core.F, "scaled_dot_product_attention"):
            core._scaled_dot_product_attention(
                query, query, query, None, {"dropout_p": 0.0}, False
            )

        kernel.assert_called_once_with(backends=[SDPBackend.CUDNN_ATTENTION])


if __name__ == "__main__":
    unittest.main()
