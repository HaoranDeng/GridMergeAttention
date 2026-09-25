import unittest

import torch
from einops import rearrange

from analysis.probe_cluster_mass_calibration_real_qkv import (
    anchor_residuals,
    cluster_log_partitions,
    compressed_cluster_logits,
    compressed_output,
    corrected_logits,
    coupled_cumulant_outputs,
    isotropic_quadratic_residual,
    mass_kl,
    nearest_residual,
    oracle_cluster_value_scale_outputs,
)
from attention.hard_route_pytorch import (
    hard_route_assignment_pytorch,
    hard_route_merge_kv_pytorch,
)


class ClusterMassCalibrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.query = torch.randn(1, 1, 3, 2, 1, 4)
        self.key = torch.randn(1, 1, 2, 4, 1, 4)
        self.value = torch.randn_like(self.key)
        self.scale = 0.5
        self.assignment = torch.zeros(1, 1, 1, 1, 2, 2, 2, 4, dtype=torch.bool)
        self.assignment[..., 0, :2] = True
        self.assignment[..., 1, 2:] = True
        self.seeds = torch.randn(1, 1, 1, 2, 2, 2, 1, 4)

    def test_cluster_log_partition_matches_direct_enumeration(self):
        actual = cluster_log_partitions(
            self.query, self.key, self.assignment, self.scale, row_chunk=2
        )
        scores = torch.einsum(
            "baijhd,bfklhd->bhafkjil", self.query, self.key
        ) * self.scale
        expected = torch.stack(
            (
                scores[..., :2].logsumexp(-1),
                scores[..., 2:].logsumexp(-1),
            ),
            dim=-1,
        )
        torch.testing.assert_close(actual, expected)

    def test_all_query_anchors_recover_exact_cluster_mass(self):
        assignment = hard_route_assignment_pytorch(self.seeds, self.key, self.scale)
        merged_key, _, routing_bias = hard_route_merge_kv_pytorch(
            self.seeds, self.key, self.value, self.scale
        )
        approximate = compressed_cluster_logits(
            self.query, merged_key, routing_bias, self.scale
        )
        exact = cluster_log_partitions(
            self.query, self.key, assignment, self.scale
        )
        points = rearrange(self.query, "b a i j h d -> b h a j i d")
        residual = anchor_residuals(
            points,
            self.key,
            assignment,
            merged_key,
            routing_bias,
            self.scale,
        )
        prediction, _ = nearest_residual(points, points, residual)
        calibrated = corrected_logits(approximate, prediction)
        torch.testing.assert_close(calibrated, exact)
        self.assertLess(float(mass_kl(exact, calibrated).abs().max()), 1e-6)

    def test_isotropic_quadratic_recovers_radial_gap(self):
        points = torch.tensor([[[[[[0.0], [1.0], [2.0], [3.0]]]]]])
        centers = points[..., [0, 2, 3], :]
        seeds = torch.tensor([[[[[[[[1.0]]]]]]]])
        coefficient = 2.5
        anchor_gap = coefficient * (
            centers - 1.0
        ).square().mean(dim=-1, keepdim=True)
        predicted = isotropic_quadratic_residual(
            points, centers, anchor_gap, seeds
        )
        expected = coefficient * (points - 1.0).square().mean(
            dim=-1, keepdim=True
        )
        torch.testing.assert_close(predicted, expected)

    def test_exact_cluster_reconstruction_matches_dense_attention(self):
        assignment = hard_route_assignment_pytorch(self.seeds, self.key, self.scale)
        merged_key, merged_value, routing_bias = hard_route_merge_kv_pytorch(
            self.seeds, self.key, self.value, self.scale
        )
        approximate = compressed_cluster_logits(
            self.query, merged_key, routing_bias, self.scale
        )
        outputs = oracle_cluster_value_scale_outputs(
            self.query,
            self.key,
            self.value,
            assignment,
            merged_value,
            approximate,
            self.scale,
            row_chunk=2,
        )
        scores = torch.einsum(
            "baijhd,bfklhd->bhajifkl", self.query, self.key
        ).float() * self.scale
        probability = torch.softmax(scores.flatten(-3, -1), dim=-1)
        values = rearrange(self.value, "b f k l h d -> b h (f k l) d").float()
        dense = torch.einsum("bhajit,bhtd->bhajid", probability, values)
        dense = rearrange(dense, "b h a j i d -> b a i j h d")
        torch.testing.assert_close(
            outputs["oracle_exact_cluster_reconstruction"], dense, atol=2e-6, rtol=2e-6
        )

    def test_radial_cumulant_matches_first_order_at_seed(self):
        torch.manual_seed(11)
        seed = torch.randn(1, 1, 1, 1, 1, 1, 1, 4)
        direction = torch.randn(1, 1, 1, 1, 1, 4)
        query = seed[:, :, 0, 0, :, 0].unsqueeze(2) + 1e-2 * direction
        key = torch.randn(1, 1, 1, 5, 1, 4)
        value = torch.randn_like(key)
        assignment = torch.ones(1, 1, 1, 1, 1, 1, 1, 5, dtype=torch.bool)
        merged_key, merged_value, routing_bias = hard_route_merge_kv_pytorch(
            seed, key, value, self.scale
        )
        approximate = compressed_cluster_logits(
            query, merged_key, routing_bias, self.scale
        )
        exact = cluster_log_partitions(query, key, assignment, self.scale)
        outputs = coupled_cumulant_outputs(
            query,
            key,
            value,
            seed,
            assignment,
            merged_key,
            merged_value,
            approximate,
            exact,
            self.scale,
            row_chunk=1,
        )

        scores = torch.einsum("baijhd,bfklhd->bhajifkl", query, key) * self.scale
        probability = torch.softmax(scores.flatten(-3, -1), dim=-1)
        values = rearrange(value, "b f k l h d -> b h (f k l) d")
        dense = torch.einsum("bhajit,bhtd->bhajid", probability, values)
        dense = rearrange(dense, "b h a j i d -> b a i j h d")
        baseline = compressed_output(approximate, merged_value)
        baseline_error = (baseline - dense).square().mean()
        radial = outputs["cumulant_radial_value_first_order"]
        radial_error = (radial - dense).square().mean()
        self.assertLess(float(radial_error), float(baseline_error))

        direction_value = rearrange(
            merged_value, "b a f j k c h d -> b a (f k c) j h d"
        ).squeeze(2)
        denominator = direction_value.square().sum(dim=-1)

        def projected_scale(output):
            return (output * direction_value).sum(dim=-1) / denominator

        baseline_projection_error = (
            projected_scale(baseline) - projected_scale(dense)
        ).square().mean()
        radial_projection_error = (
            projected_scale(radial) - projected_scale(dense)
        ).square().mean()
        self.assertLess(
            float(radial_projection_error),
            float(baseline_projection_error) * 0.05,
        )


if __name__ == "__main__":
    unittest.main()
