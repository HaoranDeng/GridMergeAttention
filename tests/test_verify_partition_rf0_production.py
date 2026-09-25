"""CPU checks that parity verification cannot hide the historical dispatcher."""
import ast
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "analysis" / "verify_partition_rf0_production.py"
spec = importlib.util.spec_from_file_location("partition_rf0_verifier", SOURCE)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class PartitionRF0VerifierTests(unittest.TestCase):
    def test_target_protocol_is_fixed(self):
        self.assertIn("_c16_p16_g256_j4_t1_rf0_u16_cg16_", verifier.MODE)
        self.assertEqual(verifier.QKV_SHAPE, (1, 75600, 40, 128))
        self.assertEqual(verifier.SEED_SHAPE, (1, 1, 21, 45, 80, 40, 4))
        verifier.require_production_options(dict(verifier.EXPECTED_OPTIONS))

    def test_old_dispatcher_cannot_pass_via_later_override(self):
        old = dict(verifier.EXPECTED_OPTIONS, gs_num_warps=8)
        old.pop("reuse_owner_logp")
        with self.assertRaisesRegex(AssertionError, "Production port"):
            verifier.require_production_options(old)
        # The override would hide this error if applied before the assertion.
        verifier.require_production_options(old | verifier.benchmark.PARTITION_OVERRIDE)

    def test_each_missing_or_changed_required_option_fails(self):
        for key in verifier.EXPECTED_OPTIONS:
            with self.subTest(key=key):
                options = dict(verifier.EXPECTED_OPTIONS)
                options.pop(key)
                with self.assertRaises(AssertionError):
                    verifier.require_production_options(options)
        for extra in ("num_warps", "global_normalizer", "native_low_precision_dot"):
            with self.assertRaises(AssertionError):
                verifier.require_production_options(verifier.EXPECTED_OPTIONS | {extra: 8})
        with self.assertRaises(AssertionError):
            verifier.require_production_options(verifier.EXPECTED_OPTIONS | {"reuse_owner_logp": False})

    def test_observer_checks_before_override_and_returns_real_table(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        observer = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                        and node.name == "recording_selector")
        check = next(node for node in ast.walk(observer) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "require_production_options")
        override = next(node for node in observer.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "effective" for target in node.targets))
        self.assertLess(check.lineno, override.lineno)
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == "real_selector" for node in ast.walk(observer)))
        self.assertIsInstance(observer.body[-1], ast.Return)
        self.assertEqual(observer.body[-1].value.id, "table")

    def test_timing_is_explicitly_unavailable_and_reference_is_external_flash(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('"mean_attention_ms": None', source)
        self.assertIn('"speedup_vs_sdpa": None', source)
        self.assertIn("SDPBackend.FLASH_ATTENTION", source)
        self.assertIn('production = _attention(token_module, q, k, v)', source)
        self.assertIn('production_vs_override_seed_bitwise_equal', source)
        self.assertIn('unpatched_vs_observed_output_bitwise_equal', source)
        self.assertNotIn("torch.cuda.Event", source)


if __name__ == "__main__":
    unittest.main()
