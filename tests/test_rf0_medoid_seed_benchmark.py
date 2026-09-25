"""CPU-only contracts for the isolated benchmark; not GPU semantic parity."""
import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/benchmark_rf0_medoid_seed_attention.py"
spec = importlib.util.spec_from_file_location("rf0_medoid_benchmark", SOURCE)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def table(heads=1):
    rows = np.arange(45, dtype=np.uint8).reshape(1, 1, 1, 45, 1, 1, 1)
    seeds = (rows + np.arange(4, dtype=np.uint8)) % 45
    return np.broadcast_to(seeds, (1, 1, 21, 45, 80, heads, 4)).copy()


class MedoidBenchmarkTests(unittest.TestCase):
    def test_fixed_scope_and_existing_controls(self):
        self.assertEqual(list(benchmark.METHODS), ["rf0", "partition_c16", "medoid_c16"])
        self.assertEqual(benchmark.METHODS["medoid_c16"], benchmark.base.MODES["partition_c16"])
        self.assertTrue(all("_t1_rf0_u16_" in mode for mode in benchmark.METHODS.values()))
        self.assertEqual((benchmark.ITERATIONS, benchmark.REPEATS, benchmark.WARMUP), (5, 4, 2))
        self.assertEqual(benchmark.PARITY_HEADS, (0, 10, 20, 30))
        self.assertEqual(len(benchmark.base.DEFAULT_CAPTURES), 2)

    def test_adapter_preserves_arguments_and_actual_return(self):
        calls, observations, actual = [], [], object()
        def selector(*inputs, **options):
            calls.append((inputs, options))
            return actual
        options = {"candidate_count": 16, "candidate_geometry_count": 16, "block_j": 4,
                   "probe_count": 16, "global_probe_count": 256, "gs_num_warps": 4,
                   "reuse_owner_logp": True, "output_dtype": "uint8"}
        wrapped = benchmark.selector_adapter(selector, lambda *args: observations.append(args))
        self.assertIs(wrapped("query", "key", .125, **options), actual)
        self.assertEqual(calls, [(("query", "key", .125), options)])
        self.assertIs(observations[0][0], actual)
        for change in ({"gs_num_warps": 8}, {"reuse_owner_logp": False}, {"global_normalizer": object()}):
            with self.assertRaises(AssertionError):
                wrapped("query", "key", .125, **(options | change))
        self.assertEqual(len(calls), 1)  # No repaired old dispatch can reach the kernel.

    def test_anchor_contract_is_removed_only_for_medoid(self):
        value = table()
        result = benchmark.validate_seed_array(value, 1, anchored=True)
        self.assertEqual(result["first_row_anchor_fraction"], 1.0)
        free = value[..., [1, 0, 2, 3]]
        self.assertEqual(benchmark.validate_seed_array(free, 1, anchored=False)["first_row_anchor_fraction"], 0.0)
        with self.assertRaises(AssertionError):
            benchmark.validate_seed_array(free, 1, anchored=True)
        for kind in ("range", "repeat", "dtype", "shape"):
            bad = value.copy()
            if kind == "range":
                bad.flat[0] = 45
            elif kind == "repeat":
                bad.flat[1] = bad.flat[0]
            elif kind == "dtype":
                bad = bad.astype(np.int32)
            else:
                bad = bad[..., :3]
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                benchmark.validate_seed_array(bad, 1, anchored=False)

    def test_exact_parity_no_aggregate_tolerance(self):
        actual = np.tile([0, 1, 2, 3], (1000, 1))
        expected = actual.copy()
        self.assertTrue(benchmark.parity_counts(actual, expected)["exact_table_equal"])
        expected[-1, [0, 1]] = expected[-1, [1, 0]]
        result = benchmark.parity_counts(actual, expected)
        self.assertFalse(result["exact_table_equal"])
        self.assertEqual(result["ordered_slot_mismatches"], 2)
        self.assertEqual(result["ordered_tuple_mismatches"], 1)
        self.assertEqual(result["first_seed_mismatches"], 1)
        self.assertEqual(result["set_mismatches"], 0)
        expected[-1, -1] = 4
        self.assertEqual(benchmark.parity_counts(actual, expected)["set_mismatches"], 1)

    def test_full_population_coordinates_match_native_table_order(self):
        index = np.arange(21 * 45 * 80 * 4)
        head, column, row, frame = benchmark.block_coordinates(index)
        np.testing.assert_array_equal((((frame * 45 + row) * 80 + column) * 4 + head), index)
        self.assertEqual(tuple(int(v[-1]) for v in (frame, row, column, head)), (20, 44, 79, 3))
        array = np.zeros((1, 1, 21, 45, 80, 40, 4), dtype=np.uint8)
        self.assertEqual(array[..., benchmark.PARITY_HEADS, :].shape, (1, 1, 21, 45, 80, 4, 4))

    def test_independent_reference_gate_precedes_timing_and_hashes_are_covered(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        parity = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "semantic_parity")
        event = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "Event")
        self.assertLess(parity.lineno, event.lineno)
        self.assertIn('if not record["semantic_parity"]["exact_table_equal"]:', source)
        self.assertIn("partition_mass_medoid_seeds_reference(logits, importance, return_diagnostics=True)", source)
        self.assertIn("native_in_bank = bool((native_match.sum(0) == 1).all())", source)
        self.assertIn("if native_in_bank else None", source)
        self.assertIn("known L33/head30 weakness is not covered", source)
        self.assertIn("Passing correctness/runtime cannot promote quality", source)
        self.assertIn("SDPBackend.FLASH_ATTENTION", source)
        self.assertIn("start.elapsed_time(end) / ITERATIONS", source)
        self.assertIn('"medoid_c16" else nullcontext()', source)
        self.assertNotIn("PARTITION_OVERRIDE", source)
        for path in benchmark.EXTRA_SOURCES:
            self.assertTrue((ROOT / path).is_file())
        self.assertEqual(benchmark.sidecar(Path("/tmp/result.json"), 1, "parity").name, "result_capture1_parity.npz")

    def test_same_capture_runtime_ratios_use_reused_helper(self):
        values = {name: benchmark.base.latency_stats([value] * 4)
                  for name, value in (("rf0", 10), ("partition_c16", 9), ("medoid_c16", 11), ("sdpa", 22))}
        result = benchmark.base.relative_timings(values)
        self.assertEqual(result["medoid_c16"]["mean_attention_ms"], 11)
        self.assertEqual(result["medoid_c16"]["speedup_vs_sdpa"], 2)
        self.assertEqual(result["medoid_c16"]["ratio_vs_rf0"], 1.1)


if __name__ == "__main__":
    unittest.main()
