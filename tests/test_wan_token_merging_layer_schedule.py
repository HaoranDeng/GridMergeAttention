"""Tests for layer-selective token-merging schedules in native Wan."""

from __future__ import annotations

import unittest

import tempfile

import torch

from attention.gma_output_calibration import (
    apply_query_conditioned_gma_calibration,
)
from generate import (
    load_gma_output_calibrators,
    load_seed_query_projectors,
    parse_enabled_layers,
)
from wan.modules.model import WanModel


def _tiny_wan(num_layers: int = 6) -> WanModel:
    return WanModel(
        model_type="t2v",
        patch_size=(1, 1, 1),
        text_len=4,
        in_dim=2,
        dim=24,
        ffn_dim=48,
        freq_dim=8,
        text_dim=16,
        out_dim=2,
        num_heads=2,
        num_layers=num_layers,
        cross_attn_norm=False,
    )


class WanTokenMergingLayerScheduleTests(unittest.TestCase):
    def test_gma_output_calibrator_is_folded_exactly(self) -> None:
        model = _tiny_wan(num_layers=3)
        heads = model.blocks[0].self_attn.num_heads
        dim = model.blocks[0].self_attn.head_dim
        transforms = {}
        expected_weights = []
        for layer_idx, block in enumerate(model.blocks):
            transform = torch.eye(dim).repeat(heads, 1, 1)
            transform[:, 0, 0] = 1.0 + 0.01 * (layer_idx + 1)
            transforms[str(layer_idx)] = transform
            weight = block.self_attn.o.weight.detach().clone().view(
                block.self_attn.dim, heads, dim
            )
            expected_weights.append(
                torch.einsum("ohd,hed->ohe", weight, transform).reshape_as(
                    block.self_attn.o.weight
                )
            )

        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save({"layer_transforms": transforms}, checkpoint.name)
            load_gma_output_calibrators(model, checkpoint.name)

        for block, expected in zip(model.blocks, expected_weights):
            torch.testing.assert_close(block.self_attn.o.weight, expected)

    def test_timestep_gma_output_calibrator_selects_folded_weight(self) -> None:
        model = _tiny_wan(num_layers=2)
        heads = model.blocks[0].self_attn.num_heads
        dim = model.blocks[0].self_attn.head_dim
        timestep_transforms = {}
        expected = {}
        for timestep, scale in ((1000, 1.1), (500, 0.9)):
            transforms = {}
            expected[timestep] = []
            for layer_idx, block in enumerate(model.blocks):
                transform = torch.eye(dim).repeat(heads, 1, 1)
                transform[:, 0, 0] = scale + 0.01 * layer_idx
                transforms[str(layer_idx)] = transform
                weight = block.self_attn.o.weight.detach().clone().view(
                    block.self_attn.dim, heads, dim
                )
                expected[timestep].append(
                    torch.einsum("ohd,hed->ohe", weight, transform).reshape_as(
                        block.self_attn.o.weight
                    )
                )
            timestep_transforms[str(timestep)] = transforms

        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save(
                {"timestep_layer_transforms": timestep_transforms}, checkpoint.name
            )
            load_gma_output_calibrators(model, checkpoint.name)

        for layer_idx, block in enumerate(model.blocks):
            self.assertIsNotNone(block.self_attn.gma_output_calibration_weights)
            for timestep in expected:
                torch.testing.assert_close(
                    block.self_attn.gma_output_calibration_weights[timestep],
                    expected[timestep][layer_idx],
                )

    def test_query_conditioned_gma_calibrator_loads_exactly(self) -> None:
        model = _tiny_wan(num_layers=2)
        heads = model.blocks[0].self_attn.num_heads
        dim = model.blocks[0].self_attn.head_dim
        gma_transforms = {}
        query_transforms = {}
        biases = {}
        for timestep, scale in ((1000, 1.1), (500, 0.9)):
            gma_transforms[str(timestep)] = {}
            query_transforms[str(timestep)] = {}
            biases[str(timestep)] = {}
            for layer_idx in range(2):
                gma_transforms[str(timestep)][str(layer_idx)] = (
                    torch.eye(dim).repeat(heads, 1, 1) * scale
                )
                query_transforms[str(timestep)][str(layer_idx)] = (
                    torch.eye(dim).repeat(heads, 1, 1) * (0.1 + layer_idx)
                )
                biases[str(timestep)][str(layer_idx)] = torch.full(
                    (heads, dim), 0.01 * (layer_idx + 1)
                )

        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save(
                {
                    "timestep_layer_transforms": gma_transforms,
                    "timestep_layer_query_transforms": query_transforms,
                    "timestep_layer_biases": biases,
                },
                checkpoint.name,
            )
            load_gma_output_calibrators(model, checkpoint.name)

        for layer_idx, block in enumerate(model.blocks):
            parameters = block.self_attn.gma_query_calibration_parameters
            self.assertEqual(set(parameters), {1000, 500})
            for timestep in parameters:
                expected = (
                    gma_transforms[str(timestep)][str(layer_idx)],
                    query_transforms[str(timestep)][str(layer_idx)],
                    biases[str(timestep)][str(layer_idx)],
                )
                for actual, wanted in zip(parameters[timestep], expected):
                    torch.testing.assert_close(actual, wanted)

    def test_query_conditioned_gma_calibration_matches_equation(self) -> None:
        generator = torch.Generator().manual_seed(7)
        approximate = torch.randn(1, 5, 2, 3, generator=generator)
        query = torch.randn(1, 5, 2, 3, generator=generator)
        gma_transform = torch.randn(2, 3, 3, generator=generator)
        query_transform = torch.randn(2, 3, 3, generator=generator)
        bias = torch.randn(2, 3, generator=generator)
        expected = (
            torch.einsum("bshd,hde->bshe", approximate, gma_transform)
            + torch.einsum("bshd,hde->bshe", query, query_transform)
            + bias[None, None]
        )
        actual = apply_query_conditioned_gma_calibration(
            approximate,
            query,
            (gma_transform, query_transform, bias),
        )
        torch.testing.assert_close(actual, expected)

    def test_full_seed_query_projectors_load_per_layer(self) -> None:
        model = _tiny_wan(num_layers=3)
        layer_states = {
            str(index): {
                name: tensor.detach().clone()
                for name, tensor in block.self_attn.q.state_dict().items()
            }
            for index, block in enumerate(model.blocks)
        }
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save({"layer_state_dicts": layer_states}, checkpoint.name)
            load_seed_query_projectors(model, checkpoint.name)

        for block in model.blocks:
            self.assertIsNotNone(block.self_attn.seed_query_proj)
            torch.testing.assert_close(
                block.self_attn.seed_query_proj.weight,
                block.self_attn.q.weight,
            )

    def test_cli_layer_list_parser(self) -> None:
        self.assertEqual(parse_enabled_layers("0, 4,19"), [0, 4, 19])
        with self.assertRaisesRegex(ValueError, "without empty entries"):
            parse_enabled_layers("0,,4")

    def test_omitted_schedule_preserves_all_layer_behavior(self) -> None:
        model = _tiny_wan()
        model.token_merging_args = {"enable": True}

        self.assertEqual(
            [block.self_attn.enable_token_merging for block in model.blocks],
            [True] * 6,
        )

    def test_enabled_layers_select_only_requested_zero_based_layers(self) -> None:
        model = _tiny_wan()
        model.token_merging_args = {
            "enable": True,
            "enabled_layers": [1, 3, 5],
            "attention_backend": "sdpa",
        }

        self.assertEqual(
            [block.self_attn.enable_token_merging for block in model.blocks],
            [False, True, False, True, False, True],
        )
        self.assertEqual(
            [block.self_attn.layer_idx for block in model.blocks],
            list(range(6)),
        )
        self.assertTrue(
            all(block.self_attn.attention_backend == "sdpa" for block in model.blocks)
        )

    def test_global_disable_still_disables_scheduled_layers(self) -> None:
        model = _tiny_wan()
        model.token_merging_args = {
            "enable": False,
            "enabled_layers": [1, 3, 5],
        }

        self.assertFalse(
            any(block.self_attn.enable_token_merging for block in model.blocks)
        )

    def test_full_attention_anchor_grid_is_propagated_to_every_layer(self) -> None:
        model = _tiny_wan()
        model.token_merging_args = {
            "enable": True,
            "full_attention_anchor_time_count": 3,
            "full_attention_anchor_row_count": 5,
            "full_attention_anchor_width_count": 7,
            "full_attention_anchor_scale": 0.5,
        }

        self.assertTrue(
            all(
                block.self_attn.token_merging_full_attention_anchor_time_count == 3
                and block.self_attn.token_merging_full_attention_anchor_row_count == 5
                and block.self_attn.token_merging_full_attention_anchor_width_count
                == 7
                and block.self_attn.token_merging_full_attention_anchor_scale == 0.5
                for block in model.blocks
            )
        )

    def test_full_attention_anchor_grid_can_target_sensitive_layers(self) -> None:
        model = _tiny_wan()
        model.token_merging_args = {
            "enable": True,
            "full_attention_anchor_time_count": 4,
            "full_attention_anchor_row_count": 6,
            "full_attention_anchor_width_count": 7,
            "full_attention_anchor_scale": 0.75,
            "full_attention_anchor_layers": [0, 1, 3],
        }

        self.assertEqual(
            [
                block.self_attn.token_merging_full_attention_anchor_time_count
                for block in model.blocks
            ],
            [4, 4, 0, 4, 0, 0],
        )
        self.assertEqual(
            [
                block.self_attn.token_merging_full_attention_anchor_row_count
                for block in model.blocks
            ],
            [6, 6, 0, 6, 0, 0],
        )
        self.assertEqual(
            [
                block.self_attn.token_merging_full_attention_anchor_width_count
                for block in model.blocks
            ],
            [7, 7, 0, 7, 0, 0],
        )
        self.assertTrue(
            all(block.self_attn.enable_token_merging for block in model.blocks)
        )

    def test_exact_intra_frame_is_propagated_to_every_enabled_layer(self) -> None:
        model = _tiny_wan()
        model.token_merging_args = {
            "enable": True,
            "exact_intra_frame_attention": True,
            "exact_neighbor_window_columns": 30,
        }

        self.assertTrue(
            all(
                block.self_attn.enable_token_merging
                and block.self_attn.token_merging_exact_intra_frame_attention
                and block.self_attn.token_merging_exact_neighbor_window_columns == 30
                for block in model.blocks
            )
        )

    def test_exact_query_budget_is_propagated_to_every_enabled_layer(self) -> None:
        model = _tiny_wan()
        model.token_merging_args = {
            "enable": True,
            "exact_query_rows": 3,
        }

        self.assertTrue(
            all(
                block.self_attn.enable_token_merging
                and block.self_attn.token_merging_exact_query_rows == 3
                for block in model.blocks
            )
        )

    def test_singleton_rate_schedule_maps_budgets_to_layer_modes(self) -> None:
        model = _tiny_wan()
        model.token_merging_args = {
            "enable": True,
            "seed_mode": "singleton_leverage_layerwise",
            "singleton_layer_schedule": [0, 20, 32, 12, 16, 6],
        }

        self.assertEqual(
            [block.self_attn.token_merging_seed_mode for block in model.blocks],
            [
                "local_with_ith",
                "singleton_leverage_top20",
                "singleton_leverage_top32",
                "singleton_leverage_top12",
                "singleton_leverage_top16",
                "singleton_leverage_top6",
            ],
        )

    def test_singleton_rate_schedule_validates_length_and_budget(self) -> None:
        model = _tiny_wan()
        with self.assertRaisesRegex(ValueError, "one budget per layer"):
            model.token_merging_args = {
                "enable": True,
                "seed_mode": "singleton_leverage_layerwise",
                "singleton_layer_schedule": [0, 4],
            }
        with self.assertRaisesRegex(ValueError, "supports only"):
            model.token_merging_args = {
                "enable": True,
                "seed_mode": "singleton_leverage_layerwise",
                "singleton_layer_schedule": [0, 4, 8, 12, 16, 10],
            }

    def test_schedule_rejects_duplicate_or_out_of_range_layers(self) -> None:
        model = _tiny_wan()
        with self.assertRaisesRegex(ValueError, "duplicates"):
            model.token_merging_args = {
                "enable": True,
                "enabled_layers": [1, 1],
            }
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            model.token_merging_args = {
                "enable": True,
                "enabled_layers": [6],
            }
        with self.assertRaisesRegex(ValueError, "duplicates"):
            model.token_merging_args = {
                "enable": True,
                "full_attention_anchor_layers": [0, 0],
            }
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            model.token_merging_args = {
                "enable": True,
                "full_attention_anchor_layers": [6],
            }


if __name__ == "__main__":
    unittest.main()
