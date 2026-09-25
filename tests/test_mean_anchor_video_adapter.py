"""CPU-only lifecycle and kernel-binding checks for current-frame mean seeds."""
import unittest
from types import SimpleNamespace
from unittest import mock

from analysis import mean_anchor_video_adapter as module
from tests.test_zero_anchor_video_adapter import MetadataOnlyTensor, RecordingKernel


class Tensor(MetadataOnlyTensor):
    def data_ptr(self):
        return id(self)

    def storage_offset(self):
        return 0

    def stride(self):
        values = []
        stride = 1
        for size in reversed(self.shape):
            values.append(stride)
            stride *= size
        return tuple(reversed(values))


class MeanVideoAdapterTests(unittest.TestCase):
    def setUp(self):
        self.q, self.k, self.v = [Tensor() for _ in range(3)]
        self.seeds = Tensor(module.SEED_SHAPE, "torch.uint8")
        self.kernel = RecordingKernel()
        self.merge_module = SimpleNamespace(**{module.KERNEL_NAME: self.kernel})
        self.forwarded = []
        self.result = object()
        self.kernel_args = (object(), object())
        self.kernel_kwargs = {"FP8_SCORE_DOT": False, "USE_TEMPORAL_ROW_MAP": False, "SPLIT_KV_ACCUM": True}
        self.builder = mock.Mock(side_effect=lambda q: Tensor((1, 21, 80, 40, 128), q.dtype, q.device))
        patcher = mock.patch.object(module, "current_frame_means", self.builder)
        patcher.start()
        self.addCleanup(patcher.stop)

    def merge(self, *args, **kwargs):
        self.forwarded.append((args, kwargs))
        getattr(self.merge_module, module.KERNEL_NAME)[(1, 2)](*self.kernel_args, **self.kernel_kwargs)
        return self.result

    def call(self, adapter, start=0, **kwargs):
        return adapter(self.q, self.k, self.v, self.seeds, 128 ** -.5,
                       head_start=start, head_count=8, **kwargs)

    def complete(self, adapter):
        adapter.begin_attention()
        for start in (0, 8, 16, 24, 32):
            self.assertIs(self.call(adapter, start, output=self.result), self.result)
        adapter.end_attention()

    def test_one_mean_per_attention_five_head_chunks_and_explicit_reset(self):
        adapter = module.MeanSeedMergeAdapter(self.merge, self.merge_module)
        self.assertIsNone(adapter.means)
        self.complete(adapter)
        first_means = adapter.means
        self.assertEqual(self.builder.call_count, 1)
        self.builder.assert_called_once_with(self.q)
        self.assertEqual((adapter.total_calls, adapter.kernel_calls, adapter.mean_computations,
                          adapter.completed_attention_calls), (5, 5, 1, 1))
        self.assertTrue(all(row[2]["mean_seed_ptr"] is first_means for row in self.kernel.calls))
        # The exact same Q object/data_ptr must NOT trigger cross-call reuse.
        self.complete(adapter)
        self.assertEqual(self.builder.call_count, 2)
        self.assertIsNot(adapter.means, first_means)
        self.assertEqual((adapter.kernel_calls, adapter.mean_computations, adapter.completed_attention_calls), (10, 2, 2))
        with self.assertRaises(AttributeError):
            adapter.means = self.result

    def test_original_inputs_and_options_are_not_changed(self):
        adapter = module.MeanSeedMergeAdapter(self.merge, self.merge_module)
        adapter.begin_attention()
        self.call(adapter, output=self.result, fp8_score_dot=False)
        args, kwargs = self.forwarded[0]
        self.assertTrue(all(a is b for a, b in zip(args[:4], (self.q, self.k, self.v, self.seeds))))
        self.assertEqual(kwargs, {"head_start": 0, "head_count": 8, "output": self.result, "fp8_score_dot": False})
        _, native_args, native_options = self.kernel.calls[0]
        self.assertTrue(all(a is b for a, b in zip(native_args, self.kernel_args)))
        self.assertEqual(native_options, self.kernel_kwargs | {"MEAN_SEED_ZERO": True, "mean_seed_ptr": adapter.means})
        self.assertIs(getattr(self.merge_module, module.KERNEL_NAME), self.kernel)

    def test_missing_lifecycle_reordered_chunks_and_changed_inputs_fail(self):
        adapter = module.MeanSeedMergeAdapter(self.merge, self.merge_module)
        with self.assertRaises(RuntimeError): self.call(adapter)
        with self.assertRaises(RuntimeError): adapter.end_attention()
        adapter.begin_attention()
        with self.assertRaises(RuntimeError): adapter.begin_attention()
        with self.assertRaises(ValueError): self.call(adapter, 8)
        with self.assertRaises(RuntimeError): adapter.end_attention()
        self.call(adapter)
        with self.assertRaises(ValueError): self.call(adapter, 0)
        self.q = Tensor()
        with self.assertRaisesRegex(ValueError, "storage changed"): self.call(adapter, 8)
        self.assertEqual(self.builder.call_count, 1)

    def test_restoration_on_exception_zero_or_multiple_kernel_launches(self):
        error = RuntimeError("original native failure")
        def broken(*args, **kwargs): raise error
        for original in (broken, lambda *a, **kw: self.result):
            adapter = module.MeanSeedMergeAdapter(original, self.merge_module)
            adapter.begin_attention()
            with self.assertRaises(RuntimeError): self.call(adapter)
            self.assertIs(getattr(self.merge_module, module.KERNEL_NAME), self.kernel)
        def multiple(*args, **kwargs):
            self.merge(*args, **kwargs)
            try: self.merge(*args, **kwargs)
            except RuntimeError: pass
            return self.result
        adapter = module.MeanSeedMergeAdapter(multiple, self.merge_module)
        adapter.begin_attention()
        with self.assertRaisesRegex(RuntimeError, "exactly one"): self.call(adapter)
        self.assertEqual(adapter.kernel_calls, 1)
        self.assertIs(getattr(self.merge_module, module.KERNEL_NAME), self.kernel)

    def test_no_fp8_rowmap_or_double_override_and_bad_mean_metadata_rejected(self):
        for kwargs in ({"fp8_score_dot": True}, {"scaled_fp8_kv": True},
                       {"temporal_query_row_map": object()}, {"mean_seed_ptr": object()}):
            adapter = module.MeanSeedMergeAdapter(self.merge, self.merge_module)
            adapter.begin_attention()
            with self.assertRaises(ValueError): self.call(adapter, **kwargs)
        self.assertEqual(self.builder.call_count, 0)
        self.builder.side_effect = lambda q: Tensor((1, 1, 80, 40, 128))
        adapter = module.MeanSeedMergeAdapter(self.merge, self.merge_module)
        adapter.begin_attention()
        with self.assertRaisesRegex(ValueError, "row mean"): self.call(adapter)
        self.assertEqual(adapter.mean_computations, 0)
        with self.assertRaises(TypeError): module.MeanSeedMergeAdapter(adapter, self.merge_module)


if __name__ == "__main__":
    unittest.main()
