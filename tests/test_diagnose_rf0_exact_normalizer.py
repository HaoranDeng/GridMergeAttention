"""CPU-only exact-normalizer oracle contracts; native GPU paths never run."""
import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/diagnose_rf0_exact_normalizer.py"
spec = importlib.util.spec_from_file_location("exact_norm_diagnostic", SOURCE)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
try:
    import torch
except ImportError:
    torch = None


class ExactNormalizerTests(unittest.TestCase):
    def test_phase_stream_chunks_cover_original_uniform16_phase_once(self):
        for chunk in (1, 4096, 10001):
            indices = [start + offset for start in range(0, diagnostic.N, chunk)
                       for offset in diagnostic.phase_offsets(start, min(start + chunk, diagnostic.N))]
            np.testing.assert_array_equal(indices, np.arange(0, diagnostic.N, 5))
            self.assertEqual(len(indices), diagnostic.PHASE_N)
        self.assertTrue(np.isin(np.arange(256) * 295, np.arange(0, diagnostic.N, 5)).all())

    def test_g256_units_common_factor_and_uniform_identity(self):
        for size in (diagnostic.N, diagnostic.PHASE_N):
            self.assertAlmostEqual(diagnostic.log2_g256_convention(math.log(size), size), 8)
        shifted = np.array([50., 55., 70.])
        for count in (diagnostic.N, diagnostic.PHASE_N):
            actual = diagnostic.log2_g256_convention(shifted, count)
            np.testing.assert_allclose(np.diff(actual), np.diff(shifted) / math.log(2))
            np.testing.assert_allclose((shifted - math.log(count / 256)) / math.log(2), actual)
        with self.assertRaises(ValueError):
            diagnostic.log2_g256_convention(1., 20)

    def test_constant_normalizer_offset_cancels_candidate_mass_weights(self):
        local = np.array([4., 7., 2., 5.])
        raw = np.array([10., 12., 9., 8.])
        def weights(log_mass):
            values = np.exp(log_mass - log_mass.max())
            return values / values.sum()
        for count in (diagnostic.N, diagnostic.PHASE_N):
            injected = diagnostic.log2_g256_convention(raw, count)
            np.testing.assert_allclose(weights(local - raw), weights(local - injected * math.log(2)))

    def test_unpatched_baseline_dispatch_is_fixed_pure_rf0_partition(self):
        observed = []
        sentinel = object()
        def full(*inputs, **options):
            observed.append((inputs, options))
            return SimpleNamespace(contiguous=lambda: sentinel)
        token = SimpleNamespace(_hard_c4_attention=full)
        self.assertIs(diagnostic.unpatched_attention(token, "q", "k", "v"), sentinel)
        inputs, options = observed[0]
        self.assertEqual(inputs, ("q", "k", "v"))
        self.assertEqual(options["seed_mode"], diagnostic.base.MODES["partition_c16"])
        self.assertIn("_j4_t1_rf0_u16_cg16_sq1_", options["seed_mode"])
        for key in ("exact_intra_frame_attention", "first_frame_full_attention", "exact_neighbor_window_columns",
                    "exact_query_rows", "full_attention_anchor_time_count", "full_attention_anchor_row_count"):
            self.assertFalse(options[key])
        self.assertEqual(options["head_chunk"], 8)
        self.assertEqual(options["query_outer_chunk"], 21)

    def test_normalizer_oracle_bypasses_original_and_preserves_received_arguments(self):
        class Tensor:
            device = "cpu"
            def __init__(self, shape, dtype):
                self.shape, self.dtype = shape, dtype
            def is_contiguous(self):
                return True
            def detach(self):
                return self
            def cpu(self):
                return self
            def clone(self):
                return self
        query, keys = Tensor((1, 1, 45, 80, 4, 128), "bf16"), Tensor((1, 21, 45, 80, 4, 128), "bf16")
        norm = Tensor(diagnostic.NORM_SHAPE, "fp32")
        fake = SimpleNamespace(float32="fp32", equal=lambda a, b: a is b,
                               isfinite=lambda value: SimpleNamespace(all=lambda: True))
        original_calls = []
        def original(q, k, **kwargs):
            original_calls.append((q, k, kwargs))
            return norm
        with patch.dict(sys.modules, {"torch": fake}):
            records = []
            adapter = diagnostic.normalizer_adapter(original, query, keys, records)
            self.assertIs(adapter(query, keys, scale=diagnostic.SCALE, global_probe_count=256), norm)
            self.assertEqual(len(original_calls), 1)
            self.assertTrue(records[0]["original_g256_called"])
            records = []
            oracle = diagnostic.normalizer_adapter(original, query, keys, records, replacement=norm)
            self.assertIs(oracle(query, keys, scale=diagnostic.SCALE, global_probe_count=256), norm)
            self.assertEqual(len(original_calls), 1)  # No unused G256 computation.
            self.assertFalse(records[0]["original_g256_called"])
            with self.assertRaises(ValueError):
                oracle(query, keys, scale=diagnostic.SCALE, global_probe_count=256)
            for kw in ({"scale": 1., "global_probe_count": 256}, {"scale": diagnostic.SCALE, "global_probe_count": 128}):
                bad = diagnostic.normalizer_adapter(original, query, keys, [], replacement=norm)
                with self.assertRaises(ValueError):
                    bad(query, keys, **kw)

    def test_seed_changes_are_relative_to_partition_not_hellinger(self):
        rows = np.arange(45).reshape(1, 1, 1, 45, 1, 1, 1)
        table = np.broadcast_to((rows + np.arange(4)) % 45, diagnostic.base.SEED_SHAPE).astype(np.uint8)
        baseline = table.copy()
        same_set = diagnostic.seed_changes(table[..., [0, 2, 1, 3]], baseline)
        self.assertEqual(same_set["reference"], "partition_g256")
        self.assertEqual(same_set["mean_shared_rows"], 4)
        self.assertEqual(same_set["ordered_tuple_changed_fraction"], 1)
        self.assertEqual(same_set["set_changed_fraction"], 0)
        self.assertNotIn("mean_shared_rows_vs_hellinger", same_set)

    def test_actual_import_hook_and_exactly_three_fixed_offpath_methods(self):
        production = ast.parse((ROOT / "attention/partition_mass_seed_triton.py").read_text(encoding="utf-8"))
        wrapper = next(n for n in production.body if isinstance(n, ast.FunctionDef) and n.name == "partition_mass_seed_rows_triton")
        imports = [n for n in ast.walk(wrapper) if isinstance(n, ast.ImportFrom)]
        self.assertTrue(any(n.module == "entropy_mass_seed_triton" and n.level == 1
                            and any(v.name == "global_probe_log_norm_triton" for v in n.names) for n in imports))
        self.assertEqual(diagnostic.METHODS, ("partition_g256", "partition_oracle_allkeys", "partition_oracle_phase0"))
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("scaled_dot_product_attention(", source)
        self.assertNotIn("partition_mass_medoid", source)
        self.assertIn('"eligible_candidate": False', source)
        self.assertIn('"mean_attention_ms": None', source)
        self.assertIn('"speedup_vs_sdpa": None', source)
        self.assertLess(source.index("baseline_output = unpatched_attention"), source.index('with patch.object(norm_module'))

    @unittest.skipIf(torch is None, "Torch CPU unavailable")
    def test_streamed_torch_partitions_and_leftmost_global_peak(self):
        generator = torch.Generator().manual_seed(3)
        query = torch.randn(2, 7, 8, generator=generator)
        keys = torch.randn(2, 19, 8, generator=generator)
        expected = torch.bmm(query, keys.transpose(1, 2)) * 0.5
        prior = torch.backends.cuda.matmul.allow_tf32
        raw, peaks = diagnostic.stream_normalizers(query, keys, query_chunk=3, key_chunk=4, scale=0.5)
        torch.testing.assert_close(raw[..., 0], expected.double().logsumexp(-1), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(raw[..., 1], expected[..., ::5].double().logsumexp(-1), atol=2e-6, rtol=2e-6)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, prior)
        torch.testing.assert_close(peaks["argmax_key_index"], expected.argmax(-1))
        torch.testing.assert_close(peaks["peak_mass_fp64"], (expected.max(-1).values.double() - raw[..., 0]).exp(), atol=2e-6, rtol=2e-6)
        _, ties = diagnostic.stream_normalizers(torch.zeros(1, 2, 3), torch.zeros(1, 13, 3), query_chunk=1, key_chunk=4)
        self.assertTrue(bool((ties["argmax_key_index"] == 0).all()))
        torch.testing.assert_close(ties["peak_mass_fp64"], torch.full((1, 2), 1 / 13, dtype=torch.float64))

    @unittest.skipIf(torch is None, "Torch CPU unavailable")
    def test_normalizer_query_head_native_mapping(self):
        raw = torch.arange(4 * 3600, dtype=torch.float64).reshape(4, 3600) * 0.001
        norm = diagnostic.native_layout(raw, diagnostic.N, "cpu")
        self.assertEqual(tuple(norm.shape), diagnostic.NORM_SHAPE)
        self.assertEqual(norm.dtype, torch.float32)
        self.assertTrue(norm.is_contiguous())
        for head, row, column in ((0, 0, 0), (3, 44, 79), (1, 12, 35)):
            expected = diagnostic.log2_g256_convention(raw[head, row * 80 + column], diagnostic.N)
            self.assertAlmostEqual(float(norm[0, 0, row, column, head]), float(expected), places=5)


if __name__ == "__main__":
    unittest.main()
