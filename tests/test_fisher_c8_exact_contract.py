import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "analysis/fisher_c8_exact.py"


class ExactFisherC8ContractTests(unittest.TestCase):
    def test_operator_declares_exact_eight_capacity(self):
        source = OPERATOR.read_text(encoding="utf-8")
        self.assertIn("require_exact_merged_capacity(8, 0)", source)
        self.assertIn('"merged_representatives": C8_REPRESENTATIVES', source)
        self.assertIn('"singleton_count": 0', source)
        self.assertNotIn("8 + 1", source)

    def test_fisher_selector_and_retained_c8_back_half_are_connected(self):
        source = OPERATOR.read_text(encoding="utf-8")
        self.assertIn("fisher_geometric_c8_seed_rows_triton", source)
        self.assertIn("NativeFisherSelector(", source)
        self.assertIn("NativeFusedFisherC8Selector", source)
        self.assertIn('"back_half": "kernels.best.c8_r2"', source)
        self.assertIn("MERGE_TILES", source)

    def test_route_receives_only_eight_merged_representatives(self):
        tree = ast.parse(OPERATOR.read_text(encoding="utf-8"))
        route_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "route"
        ]
        self.assertEqual(len(route_calls), 1)
        names = [arg.id for arg in route_calls[0].args if isinstance(arg, ast.Name)]
        self.assertEqual(names[:5], ["q", "output", "head", "scale"])
        rendered = ast.unparse(route_calls[0])
        self.assertIn("backend.mk", rendered)
        self.assertIn("backend.mv", rendered)
        self.assertNotIn(", k,", rendered)
        self.assertNotIn(", v,", rendered)

    def test_all_native_fisher_launchers_accept_and_dispatch_c8(self):
        selector = (ROOT / "native_fisher/selector.py").read_text(encoding="utf-8")
        self.assertIn("capacitynotin(4,8,10)", selector.replace(" ", ""))
        for name in (
            "fisher_gs.cu",
            "fisher_gs_distributed.cu",
            "fisher_gs_register_distribute.cu",
        ):
            source = (ROOT / "native_fisher" / name).read_text(encoding="utf-8")
            self.assertIn("c==8", source)
            self.assertIn("fisher_gs<8", source)

    def test_default_c8_selector_is_fp32_and_exact_capacity(self):
        operator = OPERATOR.read_text(encoding="utf-8")
        self.assertIn("fisher_c8_gs_register_halving.cu", operator)
        self.assertIn("fisher_c8_selector_fused_v2_dbuf.cu", operator)
        self.assertIn('selector_backend="fused"', operator)
        selector = (ROOT / "native_fisher/selector.py").read_text(encoding="utf-8")
        self.assertIn("capacity == 8 and layout == 'probe4'", selector)
        self.assertIn("source = 'fisher_c8_gs_register_halving.cu'", selector)
        source = (ROOT / "native_fisher/fisher_c8_gs_register_halving.cu").read_text(
            encoding="utf-8"
        )
        self.assertIn("seeds.numel() == int64_t(groups) * 8", source)
        self.assertIn("float residual[P][PR]", source)
        self.assertNotIn("double residual", source)


if __name__ == "__main__":
    unittest.main()
