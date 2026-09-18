import unittest
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "capacity_contract", ROOT / "attention" / "capacity_contract.py"
)
CAPACITY_CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPACITY_CONTRACT)
EXACT_MERGED_CAPACITY = CAPACITY_CONTRACT.EXACT_MERGED_CAPACITY
require_exact_merged_capacity = CAPACITY_CONTRACT.require_exact_merged_capacity


class ExactCapacityContractTests(unittest.TestCase):
    def test_c4_and_c8_are_complete_counts(self):
        self.assertEqual(EXACT_MERGED_CAPACITY, {4: 4, 8: 8})
        self.assertEqual(require_exact_merged_capacity(4), 4)
        self.assertEqual(require_exact_merged_capacity(8), 8)

    def test_c4_and_c8_reject_singletons(self):
        for capacity in (4, 8):
            with self.subTest(capacity=capacity):
                with self.assertRaisesRegex(ValueError, "singleton/SQ1/C\\+1"):
                    require_exact_merged_capacity(capacity, 1)

    def test_packaged_c4_and_c8_cannot_route_an_extra_token(self):
        c4 = (ROOT / "kernels/best/c4/operator.py").read_text(encoding="utf-8")
        self.assertIn("SINGLETON_TOP_R=0", c4)
        self.assertIn("SINGLETON_QUERY_COLUMN=False", c4)
        for relative in ("kernels/best/c8/gma_c8_native.py", "kernels/best/c8_r2/gma_c8_r2.py"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("require_exact_merged_capacity(8, 0)", source)

    def test_legacy_native_fisher_c4_entry_is_fail_closed(self):
        source = (ROOT / "native_fisher/operator.py").read_text(encoding="utf-8")
        self.assertIn("if capacity == 4", source)
        self.assertIn("C4+SQ1 is retired", source)

    def test_no_runnable_c4_or_c8_sq1_config(self):
        offenders = []
        for path in (ROOT / "configs").glob("*.yaml"):
            text = path.read_text(encoding="utf-8").lower()
            if "num_merge_tokens: 4" in text or "num_merge_tokens: 8" in text:
                if "sq1" in text or "singleton" in text:
                    offenders.append(path.name)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
