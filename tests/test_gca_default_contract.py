"""CPU-only contracts for the exact-capacity Fisher C4 generation default.

Read YAML and execute isolated production AST fragments, never importing
generate, Wan, Torch, or Triton. These check configuration and dispatch policy,
not numerical kernel parity, video quality, or GPU performance.
"""

import argparse
import ast
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/token_merging_wan_720p_fisher_c4_exact_capacity_fewstep.yaml"
DEFAULT_MODE = "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_rf0_u16_cg45"


def source_tree(relative):
    path = ROOT / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def execute_nodes(nodes, scope):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<production AST>", "exec"), scope)


def config_document():
    return yaml.safe_load((ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8"))


def parse_cli(*arguments):
    node = next(node for node in source_tree("generate.py").body
                if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    scope = {"argparse": argparse}
    execute_nodes([node], scope)
    with mock.patch.object(sys, "argv", ["generate.py", *arguments]):
        return scope["parse_args"]()


class GCADefaultContractTests(unittest.TestCase):
    def test_cli_defaults_to_original_mass_gs_rf0_and_canonical_prompts(self):
        args = parse_cli("--checkpoint", "unused.pt")
        self.assertEqual(args.config, DEFAULT_CONFIG)
        self.assertTrue((ROOT / args.config).is_file())
        self.assertEqual(args.prompt_path, "prompts/MovieGenVideoBench_extended.txt")
        self.assertEqual((args.seed, args.num_videos, args.num_samples), (0, 20, 1))
        self.assertEqual(args.dtype, "bf16")
        self.assertFalse(args.allow_noncanonical_prompts)

    def test_cli_uses_gca_name_and_keeps_explicit_historical_config_override(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            parse_cli("--help")
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("Grid Compressed Attention (GCA)", output.getvalue())
        legacy = "configs/token_merging_wan_hard_c4_triton_fewstep.yaml"
        self.assertTrue((ROOT / legacy).is_file())
        self.assertEqual(parse_cli("--checkpoint", "unused.pt", "--config", legacy).config, legacy)

    def test_yaml_freezes_exact_c4_without_singleton(self):
        self.assertEqual(config_document()["token_merging_args"], {
            "enable": True,
            "f_tied": 1,
            "h_reduce": 1,
            "w_reduce": 1,
            "layout": "hard_route",
            "query_outer_chunk": 21,
            "head_chunk": 8,
            "num_merge_tokens": 4,
            "seed_mode": DEFAULT_MODE,
            "seed": 0,
            "hard_route_backend": "hard_c4",
        })
        self.assertNotIn("sq1", config_document()["token_merging_args"]["seed_mode"].lower())

    def test_yaml_keeps_measured_720p_four_step_model_and_no_extra_overrides(self):
        document = config_document()
        self.assertEqual(set(document), {
            "image_or_video_shape", "denoising_step_list", "warp_denoising_step",
            "model_kwargs", "token_merging_args",
        })
        self.assertEqual(document["image_or_video_shape"], [1, 21, 16, 90, 160])
        self.assertEqual(document["denoising_step_list"], [1000, 750, 500, 250])
        self.assertIs(document["warp_denoising_step"], True)
        self.assertEqual(document["model_kwargs"], {
            "model_name": "Wan2.1-T2V-14B-StepDistill-CfgDistill",
            "load_pretrained_weights": False,
            "timestep_shift": 5.0,
        })

    def test_default_cli_enables_no_learned_or_hybrid_override(self):
        args = parse_cli("--checkpoint", "unused.pt")
        for name in (
            "seed_query_proj_checkpoint", "gma_output_calibration_checkpoint",
            "token_merging_enabled_layers", "token_merging_num_merge_tokens",
            "token_merging_full_attention_anchor_time_count",
            "token_merging_full_attention_anchor_row_count",
            "token_merging_full_attention_anchor_width_count",
            "token_merging_full_attention_anchor_scale",
            "token_merging_exact_intra_frame_attention",
            "token_merging_exact_neighbor_window_columns",
        ):
            with self.subTest(argument=name):
                self.assertIsNone(getattr(args, name))

    def test_production_model_setter_applies_same_operator_to_all_40_layers(self):
        setter = next(node for node in ast.walk(source_tree("wan/modules/model.py"))
                      if isinstance(node, ast.FunctionDef) and node.name == "token_merging_args"
                      and len(node.args.args) == 2)
        setter.decorator_list = []
        scope = {}
        execute_nodes([setter], scope)
        model = SimpleNamespace(blocks=[SimpleNamespace(self_attn=SimpleNamespace()) for _ in range(40)])
        scope["token_merging_args"](model, config_document()["token_merging_args"])
        for index, block in enumerate(model.blocks):
            with self.subTest(layer=index):
                attention = block.self_attn
                self.assertIs(attention.enable_token_merging, True)
                self.assertEqual(attention.token_merging_seed_mode, DEFAULT_MODE)
                self.assertEqual(attention.token_merging_hard_route_backend, "hard_c4")
                self.assertEqual(attention.token_merging_dense_heads, ())
                for suffix in (
                    "exact_intra_frame_attention", "exact_neighbor_window_columns",
                    "exact_query_rows", "first_frame_full_attention",
                    "full_attention_anchor_time_count", "full_attention_anchor_row_count",
                    "full_attention_anchor_width_count",
                ):
                    self.assertFalse(getattr(attention, "token_merging_" + suffix))

    def test_original_normalizer_precision_and_selector_defaults_stay_opt_out(self):
        tree = source_tree("attention/entropy_mass_seed_triton.py")
        for name in ("global_probe_log_norm_triton", "entropy_mass_seed_rows_triton"):
            node = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == name)
            defaults = {arg.arg: value for arg, value in zip(node.args.kwonlyargs, node.args.kw_defaults)}
            for option in ("native_low_precision_dot", "tf32_low_precision_dot"):
                with self.subTest(function=name, option=option):
                    self.assertIs(ast.literal_eval(defaults[option]), False)
            if name == "entropy_mass_seed_rows_triton":
                self.assertIs(ast.literal_eval(defaults["singleton_complement"]), False)
                self.assertEqual(ast.literal_eval(defaults["weight_mode"]), "mass")
                self.assertEqual(ast.literal_eval(defaults["gs_num_warps"]), 4)
                self.assertEqual(ast.literal_eval(defaults["norm_block_q"]), 32)
                self.assertEqual(ast.literal_eval(defaults["norm_num_warps"]), 8)

    def test_mass_dispatch_passes_no_optional_precision_or_partition_overrides(self):
        tree = source_tree("attention/token_merging.py")
        branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and any(isinstance(child, ast.Assign)
                              and any(isinstance(target, ast.Name) and target.id == "entropy_selector"
                                      for target in child.targets) for child in node.body))
        capture = mock.Mock(return_value="captured")
        scope = {
            "entropy_mass_rule": "mass_span", "requested_seed_mode": DEFAULT_MODE,
            "entropy_mass_seed_rows_triton": capture,
            "joint_frame_mass_seed_rows_triton": mock.Mock(),
            "partition_mass_seed_rows_triton": mock.Mock(),
            "router_query": object(), "router_key": object(), "sm_scale": 1.0,
            "candidates": 32, "candidate_geometry_count": 45, "block_j": 4,
            "probes": 16, "entropy_global_probes": 256,
            "h": 45, "w": 80, "query_blocks": SimpleNamespace(size=lambda axis: 21),
            "torch": SimpleNamespace(uint8="metadata-only-uint8"),
        }
        execute_nodes(branch.body, scope)
        capture.assert_called_once_with(
            scope["router_query"], scope["router_key"], 1.0,
            candidate_count=32, candidate_geometry_count=45, block_j=4,
            probe_count=16, global_probe_count=256, output_dtype="metadata-only-uint8",
            gs_num_warps=4,
        )
        scope["joint_frame_mass_seed_rows_triton"].assert_not_called()
        scope["partition_mass_seed_rows_triton"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
