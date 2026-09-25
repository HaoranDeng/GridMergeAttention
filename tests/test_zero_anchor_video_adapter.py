"""CPU metadata-only mocks for the isolated zero-query video merge adapter."""
import unittest
from types import SimpleNamespace

from analysis.zero_anchor_video_adapter import (
    GRID_SHAPE, KERNEL_NAME, SEED_SHAPE, ZeroSeedMergeAdapter,
)


class MetadataOnlyTensor:
    def __init__(self, shape=GRID_SHAPE, dtype="torch.bfloat16", device="cuda:0", contiguous=True):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.is_cuda = str(device).startswith("cuda")
        self.contiguous = contiguous
        self.allocations = []

    def new_zeros(self, shape):
        result = MetadataOnlyTensor(shape, self.dtype, self.device)
        self.allocations.append(result)
        return result

    def is_contiguous(self):
        return self.contiguous

    def __getitem__(self, key):
        raise AssertionError("adapter must not read tensor values")

    def __getattr__(self, name):
        raise AssertionError("adapter must not access tensor operation " + name)


class RecordingKernel:
    def __init__(self):
        self.calls = []
        self.error = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            if self.error is not None:
                raise self.error
            return "kernel-result"
        return launch


class ZeroVideoAdapterTests(unittest.TestCase):
    def setUp(self):
        self.q, self.k, self.v = [MetadataOnlyTensor() for _ in range(3)]
        self.seeds = MetadataOnlyTensor(SEED_SHAPE, "torch.uint8")
        self.kernel = RecordingKernel()
        self.module = SimpleNamespace(**{KERNEL_NAME: self.kernel})
        self.scale = 128 ** -.5
        self.original_calls = []
        self.kernel_args = (object(), object(), object(), object())
        self.kernel_kwargs = {"FP8_SCORE_DOT": False, "FP8_MERGE_K_DOT": False,
                              "FP8_MERGE_V_DOT": False, "SCALED_FP8_KV": False,
                              "USE_TEMPORAL_ROW_MAP": False, "SPLIT_KV_ACCUM": True,
                              "SCALE": self.scale, "num_warps": 4}
        self.result = object()

    def merge(self, *args, **kwargs):
        self.original_calls.append((args, kwargs))
        value = getattr(self.module, KERNEL_NAME)[(945, 8)](*self.kernel_args, **self.kernel_kwargs)
        self.assertEqual(value, "kernel-result")
        return self.result

    def call(self, adapter, **kwargs):
        return adapter(self.q, self.k, self.v, self.seeds, self.scale, **kwargs)

    def test_forwards_identical_inputs_and_kernel_arguments_then_restores(self):
        adapter = ZeroSeedMergeAdapter(self.merge, self.module)
        options = {"head_start": 0, "head_count": 8, "fp8_score_dot": False, "output": object()}
        returned = self.call(adapter, **options)
        self.assertIs(returned, self.result)
        forwarded, forwarded_options = self.original_calls[0]
        for actual, expected in zip(forwarded[:4], (self.q, self.k, self.v, self.seeds)):
            self.assertIs(actual, expected)
        self.assertEqual(forwarded[4], self.scale)
        self.assertEqual(forwarded_options, options)
        grid, args, kwargs = self.kernel.calls[0]
        self.assertEqual(grid, (945, 8))
        for actual, expected in zip(args, self.kernel_args):
            self.assertIs(actual, expected)
        self.assertEqual({k: v for k, v in kwargs.items() if k not in ("MEAN_SEED_ZERO", "mean_seed_ptr")},
                         self.kernel_kwargs)
        self.assertIs(kwargs["mean_seed_ptr"], self.q.allocations[0])
        self.assertIs(kwargs["MEAN_SEED_ZERO"], True)
        self.assertNotIn("MEAN_SEED_ZERO", self.kernel_kwargs)
        self.assertEqual((adapter.total_calls, adapter.kernel_calls), (1, 1))
        self.assertIs(getattr(self.module, KERNEL_NAME), self.kernel)

    def test_cached_zero_buffer_is_allocated_without_reading_q_values(self):
        adapter = ZeroSeedMergeAdapter(self.merge, self.module)
        self.assertIsNone(adapter.zeros)
        self.call(adapter)
        original_q = self.q
        self.q = MetadataOnlyTensor()  # different input values/storage, same metadata
        self.call(adapter, head_start=8)
        self.assertEqual(len(original_q.allocations), 1)
        self.assertEqual(len(self.q.allocations), 0)
        zeros = original_q.allocations[0]
        self.assertIs(adapter.zeros, zeros)
        with self.assertRaises(AttributeError):
            adapter.zeros = object()
        self.assertEqual(zeros.shape, (1, 21, 80, 40, 128))
        self.assertTrue(zeros.is_contiguous())
        self.assertIs(self.kernel.calls[0][2]["mean_seed_ptr"], self.kernel.calls[1][2]["mean_seed_ptr"])
        self.assertEqual((adapter.total_calls, adapter.kernel_calls), (2, 2))
        self.q, self.k, self.v = [MetadataOnlyTensor(device="cuda:1") for _ in range(3)]
        self.seeds = MetadataOnlyTensor(SEED_SHAPE, "torch.uint8", "cuda:1")
        self.call(adapter)
        self.assertEqual(len(self.q.allocations), 1)
        self.assertIsNot(self.kernel.calls[2][2]["mean_seed_ptr"], zeros)
        self.assertIs(adapter.zeros, self.q.allocations[0])

    def test_restoration_on_original_or_native_exception(self):
        original_error = RuntimeError("original merge failed")
        def broken(*args, **kwargs):
            raise original_error
        adapter = ZeroSeedMergeAdapter(broken, self.module)
        with self.assertRaises(RuntimeError) as caught:
            self.call(adapter)
        self.assertIs(caught.exception, original_error)
        self.assertIs(getattr(self.module, KERNEL_NAME), self.kernel)
        self.assertFalse(adapter._active)
        self.assertEqual((adapter.total_calls, adapter.kernel_calls), (1, 0))

        native_error = RuntimeError("native launch failed")
        self.kernel.error = native_error
        adapter = ZeroSeedMergeAdapter(self.merge, self.module)
        with self.assertRaises(RuntimeError) as caught:
            self.call(adapter)
        self.assertIs(caught.exception, native_error)
        self.assertIs(getattr(self.module, KERNEL_NAME), self.kernel)
        self.assertEqual((adapter.total_calls, adapter.kernel_calls), (1, 1))

    def test_requires_exactly_one_native_launch_even_if_extra_error_is_swallowed(self):
        adapter = ZeroSeedMergeAdapter(lambda *a, **k: self.result, self.module)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            self.call(adapter)
        self.assertIs(getattr(self.module, KERNEL_NAME), self.kernel)
        for swallow in (False, True):
            def multiple(*args, **kwargs):
                self.merge(*args, **kwargs)
                try:
                    self.merge(*args, **kwargs)
                except RuntimeError:
                    if not swallow:
                        raise
                return self.result
            adapter = ZeroSeedMergeAdapter(multiple, self.module)
            count = len(self.kernel.calls)
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                self.call(adapter)
            self.assertEqual(len(self.kernel.calls), count + 1)
            self.assertEqual((adapter.total_calls, adapter.kernel_calls), (1, 1))
            self.assertIs(getattr(self.module, KERNEL_NAME), self.kernel)

    def test_rejects_fp8_rowmaps_and_double_overrides_at_both_levels(self):
        adapter = ZeroSeedMergeAdapter(self.merge, self.module)
        for invalid in ({"fp8_score_dot": True}, {"compressed_fp8_kv": True},
                        {"fp8_merge_k_dot": True}, {"fp8_merge_v_dot": True},
                        {"temporal_query_row_map": object()}, {"MEAN_SEED_ZERO": False},
                        {"mean_seed_ptr": object()}):
            with self.assertRaises(ValueError):
                self.call(adapter, **invalid)
        self.assertEqual(adapter.total_calls, 0)
        for name, value in (("FP8_SCORE_DOT", True), ("SCALED_FP8_KV", True),
                            ("USE_TEMPORAL_ROW_MAP", True), ("MEAN_SEED_ZERO", False),
                            ("mean_seed_ptr", object())):
            original_options = self.kernel_kwargs
            self.kernel_kwargs = {**original_options, name: value}
            with self.assertRaises(ValueError):
                self.call(adapter)
            self.assertIs(getattr(self.module, KERNEL_NAME), self.kernel)
            self.kernel_kwargs = original_options
        self.assertEqual(adapter.kernel_calls, 0)
        with self.assertRaises(TypeError):
            ZeroSeedMergeAdapter(adapter, self.module)
        self.module._hard_c4_720p_temporal_shared_merge_kernel = RecordingKernel()
        with self.assertRaisesRegex(RuntimeError, "changed kernel"):
            self.call(adapter)

    def test_rejects_unsupported_metadata_without_allocating_or_mutating_inputs(self):
        adapter = ZeroSeedMergeAdapter(self.merge, self.module)
        for change in ({"shape": (1, 21, 45, 80, 4, 128)}, {"dtype": "torch.float16"},
                       {"device": "cpu"}):
            valid = self.q
            self.q = MetadataOnlyTensor(**change)
            with self.assertRaises(ValueError):
                self.call(adapter)
            self.assertEqual(self.q.allocations, [])
            self.q = valid
        original = self.seeds
        for tensor in (MetadataOnlyTensor(SEED_SHAPE, "torch.int32"),
                       MetadataOnlyTensor((1, 21, 45, 80, 40, 4), "torch.uint8"),
                       MetadataOnlyTensor(SEED_SHAPE, "torch.uint8", "cuda:1")):
            self.seeds = tensor
            with self.assertRaises(ValueError):
                self.call(adapter)
        self.seeds = original
        self.scale = 1.0
        with self.assertRaises(ValueError):
            self.call(adapter)
        self.assertEqual((adapter.total_calls, adapter.kernel_calls), (0, 0))
        self.assertEqual(self.q.allocations, [])


if __name__ == "__main__":
    unittest.main()
