"""No-GPU contract tests: execute actual parser/setter AST with metadata only."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

from analysis.quality_first_mode_contract import (
    BF16_MODES, JEFFREYS_OVERRIDE, method_contract,
    quality_first_token_merging_args, validate_token_merging_args,
)


ROOT = Path(__file__).resolve().parents[1]


def tree(relative):
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def run_nodes(nodes, scope):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<actual production AST>", "exec"), scope)


def production_suffixes(mode):
    function = next(node for node in tree("attention/token_merging.py").body
                    if isinstance(node, ast.FunctionDef) and node.name == "_hard_c4_attention")
    start = next(i for i, node in enumerate(function.body) if assigned(node, "requested_seed_mode"))
    end = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
               and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Name)
               and node.value.func.id == "_strip_hellinger_probe_policy")
    scope = {"seed_mode": mode, "re": re}
    run_nodes(function.body[start:end], scope)
    return scope


class QualityFirstModeContractTests(unittest.TestCase):
    def test_stable_sq1_methods_are_retired(self):
        from analysis.quality_first_mode_contract import STABLE_SQ1_METHODS, VIDEO_MODES
        self.assertEqual(STABLE_SQ1_METHODS, ())
        self.assertFalse(any("sq1" in method for method in VIDEO_MODES if "c10" not in method))

    def test_retired_stable_mass_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            method_contract("mass_stable_sq1_rf0")

    def test_exact_capacity_modes_remove_all_singleton_suffixes(self):
        node = next(node for node in tree("analysis/screen_native_seed_trajectory.py").body
                    if assigned(node, "MODES"))
        old = ast.literal_eval(node.value)
        for name, old_key in (("hellinger_rf0", "rf0"), ("mass_rf0", "mass_c32"),
                              ("jeffreys_rf0", "mass_c32")):
            with self.subTest(method=name):
                self.assertEqual(BF16_MODES[name], old[old_key])
                self.assertNotIn("sq1", BF16_MODES[name])
                self.assertNotIn("singleton", BF16_MODES[name])

    def test_actual_parser_disables_all_fp8_and_keeps_the_same_substrate(self):
        false_flags = (
            "fp8_score_dot", "fp8_merge_k_dot", "fp8_merge_v_dot",
            "compressed_fp8_kv", "scaled_fp8_kv", "direct_bulk_normalization",
            "reuse_partition_exp", "materialize_singleton", "split_merge",
            "mean_centered_routing", "gaussian_partition_routing", "packed_singleton_kv",
            "common_c_ablation", "temporal_simpson3",
        )
        for name, mode in BF16_MODES.items():
            with self.subTest(method=name):
                scope = production_suffixes(mode)
                for flag in false_flags:
                    self.assertIs(scope[flag], False, flag)
                for flag in ("split_kv_accum", "direct_native_seed_layout", "query_column_major"):
                    self.assertIs(scope[flag], False, flag)
                self.assertEqual(scope["merge_loop_stages"], 1)
                self.assertEqual(scope["merge_frame_group"], 1)
                self.assertEqual(scope["merge_col_group"], 1)
                self.assertEqual(scope["candidate_geometry_count"], 45)
                self.assertEqual(scope["singleton_residual"], 0)
                self.assertTrue(scope["seed_mode"].endswith("_t1_rf0_u16"))

    def test_directbulk_remains_a_separate_explicit_ablation(self):
        for mode in BF16_MODES.values():
            separate = mode + "_directbulk"
            self.assertTrue(production_suffixes(separate)["direct_bulk_normalization"])
            args = quality_first_token_merging_args("mass_rf0")
            args["seed_mode"] = separate
            with self.assertRaises(ValueError):
                validate_token_merging_args(args, "mass_rf0", dtype="bf16")

    def test_each_contract_requires_its_actual_selector_identity(self):
        for name in BF16_MODES:
            declaration = method_contract(name)
            self.assertEqual(validate_token_merging_args(
                quality_first_token_merging_args(name), name, dtype="bf16",
                selector_override=declaration["selector_override"],
            ), declaration)
        self.assertEqual(BF16_MODES["jeffreys_rf0"], BF16_MODES["mass_rf0"])
        for name, override in (("jeffreys_rf0", None), ("mass_rf0", JEFFREYS_OVERRIDE),
                               ("hellinger_rf0", JEFFREYS_OVERRIDE)):
            with self.subTest(method=name), self.assertRaises(ValueError):
                validate_token_merging_args(quality_first_token_merging_args(name), name,
                                           dtype="bf16", selector_override=override)

    def test_no_learned_hybrid_unknown_or_dtype_override(self):
        invalid = {
            "enable": False, "enabled_layers": [0, 1], "head_chunk": 4,
            "exact_intra_frame_attention": True, "first_frame_full_attention": True,
            "exact_neighbor_window_columns": 1, "exact_query_rows": 1,
            "full_attention_anchor_time_count": 1, "full_attention_anchor_row_count": 1,
            "full_attention_anchor_width_count": 1, "full_attention_anchor_scale": 2.0,
            "full_attention_anchor_layers": [0], "dense_head_schedule": [[30]] * 40,
            "singleton_layer_schedule": [1] * 40, "q_init": "learned",
            "seed_query_proj_checkpoint": "projector.pt", "new_option": False,
            "f_tied": True, "seed": False,
        }
        for key, value in invalid.items():
            with self.subTest(option=key), self.assertRaises(ValueError):
                validate_token_merging_args(
                    quality_first_token_merging_args("mass_rf0") | {key: value},
                    "mass_rf0", dtype="bf16",
                )
        for dtype in ("fp16", "fp32", "fp8", None):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                validate_token_merging_args(quality_first_token_merging_args("mass_rf0"),
                                           "mass_rf0", dtype=dtype)

    def test_explicit_all_layers_and_neutral_options_accepted(self):
        args = quality_first_token_merging_args("mass_rf0") | {
            "enabled_layers": list(range(40)), "exact_intra_frame_attention": False,
            "dense_head_schedule": None, "full_attention_anchor_scale": 1.0,
        }
        self.assertTrue(validate_token_merging_args(args, "mass_rf0", dtype="bf16")["all_layer_operator"])
        for layers in ([], list(range(39)), [0] * 40, list(range(1, 41)), [False] + list(range(1, 40))):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                validate_token_merging_args(args | {"enabled_layers": layers}, "mass_rf0", dtype="bf16")

    def test_actual_model_setter_applies_bf16_mode_to_all_40_layers(self):
        setter = next(node for node in ast.walk(tree("wan/modules/model.py"))
                      if isinstance(node, ast.FunctionDef) and node.name == "token_merging_args"
                      and len(node.args.args) == 2)
        setter.decorator_list = []
        scope = {}
        run_nodes([setter], scope)
        for name in BF16_MODES:
            model = SimpleNamespace(blocks=[SimpleNamespace(self_attn=SimpleNamespace()) for _ in range(40)])
            scope["token_merging_args"](model, quality_first_token_merging_args(name))
            for index, block in enumerate(model.blocks):
                with self.subTest(method=name, layer=index):
                    attention = block.self_attn
                    self.assertIs(attention.enable_token_merging, True)
                    self.assertEqual(attention.token_merging_seed_mode, BF16_MODES[name])
                    self.assertEqual(attention.token_merging_dense_heads, ())
                    for suffix in ("exact_intra_frame_attention", "exact_neighbor_window_columns",
                                   "exact_query_rows", "first_frame_full_attention",
                                   "full_attention_anchor_time_count", "full_attention_anchor_row_count",
                                   "full_attention_anchor_width_count"):
                        self.assertFalse(getattr(attention, "token_merging_" + suffix))

    def test_returned_declarations_are_independent_and_unknown_method_rejected(self):
        declaration = method_contract("mass_rf0")
        declaration["seed_mode"] = "changed"
        self.assertEqual(method_contract("mass_rf0")["seed_mode"], BF16_MODES["mass_rf0"])
        args = quality_first_token_merging_args("mass_rf0")
        args["enable"] = False
        self.assertIs(quality_first_token_merging_args("mass_rf0")["enable"], True)
        for name in ("hellinger_rf10", "default", "mass_rf0_directbulk"):
            with self.assertRaises(ValueError):
                method_contract(name)

    def test_sq1_video_candidate_is_removed(self):
        from analysis.quality_first_mode_contract import VIDEO_MODES
        self.assertNotIn("sq1_partition_rf0", BF16_MODES)
        self.assertNotIn("sq1_partition_rf0", VIDEO_MODES)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            method_contract("sq1_partition_rf0")

    def test_full_grid_fisher_changes_only_candidate_population(self):
        from analysis.quality_first_mode_contract import VIDEO_MODES
        mode=VIDEO_MODES['fisher_full45_rf0']
        self.assertEqual(mode,VIDEO_MODES['fisher_geometric_rf0'].replace('_c32_','_c45_'))
        self.assertNotIn('fisher_full45_rf0',BF16_MODES)
        declaration=method_contract('fisher_full45_rf0')
        validate_token_merging_args(quality_first_token_merging_args('fisher_full45_rf0'),
            'fisher_full45_rf0',dtype='bf16',selector_override=declaration['selector_override'])
        self.assertEqual(declaration['bulk_representatives'],4)
        self.assertEqual(declaration['router_frame'],0)

    def test_expected_fisher_declares_actual_j1_and_unchanged_merge_substrate(self):
        from analysis.quality_first_mode_contract import VIDEO_MODES
        name = "fisher_expected_rf0"
        self.assertEqual(VIDEO_MODES[name], BF16_MODES["mass_rf0"].replace("_j4_", "_j1_"))
        self.assertNotIn(name, BF16_MODES)
        declaration = method_contract(name)
        self.assertEqual(declaration["selector_override"],
                         "attention.expected_fisher_seed_triton.fisher_expected_seed_rows_triton")
        self.assertNotIn("synthetic_seed", declaration)
        self.assertEqual(declaration["router_frame"], 0)
        self.assertEqual(declaration["bulk_representatives"], 4)
        self.assertTrue(declaration["all_layer_operator"])
        self.assertTrue(declaration["full_compressed_route"])
        validate_token_merging_args(quality_first_token_merging_args(name), name,
                                   dtype="bf16", selector_override=declaration["selector_override"])
        old = production_suffixes(BF16_MODES["mass_rf0"])
        new = production_suffixes(VIDEO_MODES[name])
        self.assertEqual(new["seed_mode"].replace("_j1_", "_j4_"), old["seed_mode"])
        for field in old:
            if field not in ("requested_seed_mode", "seed_mode"):
                if isinstance(old[field], re.Match):
                    self.assertIsInstance(new[field], re.Match)
                    self.assertEqual(new[field].group(0), old[field].group(0), field)
                    self.assertEqual(new[field].groups(), old[field].groups(), field)
                    self.assertEqual(new[field].span(), old[field].span(), field)
                else:
                    self.assertEqual(new[field], old[field], field)
        bad = quality_first_token_merging_args(name)
        bad["seed_mode"] = bad["seed_mode"].replace("_j1_", "_j4_")
        with self.assertRaises(ValueError):
            validate_token_merging_args(bad, name, dtype="bf16",
                                       selector_override=declaration["selector_override"])

    def test_anchor_gap_guard_keeps_j4_incumbent_and_four_real_merge(self):
        from analysis.quality_first_mode_contract import VIDEO_MODES
        name = "anchor_gap_guard_rf0"
        self.assertEqual(VIDEO_MODES[name], BF16_MODES["mass_rf0"])
        self.assertNotIn(name, BF16_MODES)
        declaration = method_contract(name)
        self.assertEqual(declaration["selector_override"],
                         "attention.anchor_gap_guard_seed_triton.anchor_gap_guard_seed_rows_triton")
        self.assertNotIn("synthetic_seed", declaration)
        self.assertEqual(declaration["router_frame"], 0)
        self.assertEqual(declaration["bulk_representatives"], 4)
        self.assertTrue(declaration["all_layer_operator"])
        self.assertTrue(declaration["full_compressed_route"])
        validate_token_merging_args(quality_first_token_merging_args(name), name,
                                   dtype="bf16", selector_override=declaration["selector_override"])

    def test_phase_average_is_rf0_selection_only(self):
        from analysis.quality_first_mode_contract import VIDEO_MODES
        name = "phase_averaged_fisher_rf0"
        declaration = method_contract(name)
        self.assertEqual(VIDEO_MODES[name], BF16_MODES["mass_rf0"])
        self.assertEqual(declaration["selector_override"],
                         "attention.phase_averaged_seed.phase_averaged_fisher_seed_rows")
        self.assertEqual(declaration["router_frame"], 0)
        self.assertEqual(declaration["bulk_representatives"], 4)
        self.assertNotIn("synthetic_seed", declaration)
        self.assertIn("FP64 transform then BF16", declaration["selection_reference"])
        self.assertIn("original current-frame real rows", declaration["merge_queries"])
        validate_token_merging_args(quality_first_token_merging_args(name), name,
                                   dtype="bf16", selector_override=declaration["selector_override"])


if __name__ == "__main__":
    unittest.main()
