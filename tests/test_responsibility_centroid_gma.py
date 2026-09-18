import unittest

import torch
from einops import rearrange

from attention.hard_route_pytorch import (
    hard_route_merge_kv_pytorch,
    hard_route_responsibility_centroid_from_seeds_pytorch,
    responsibility_centroids_pytorch,
)


class ResponsibilityCentroidGmaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.query = torch.randn(1, 1, 5, 2, 1, 8)
        self.key = torch.randn(1, 2, 3, 6, 1, 8)
        self.value = torch.randn_like(self.key)
        self.prototypes = torch.randn(1, 1, 2, 3, 2, 4, 1, 8)
        self.scale = self.query.size(-1) ** -0.5

    def test_closed_form_centroid_matches_weighted_variance_identity(self):
        merged_key, _, routing_bias = hard_route_merge_kv_pytorch(
            self.prototypes, self.key, self.value, self.scale
        )
        centers, gain = responsibility_centroids_pytorch(
            self.query,
            merged_key,
            routing_bias,
            self.prototypes,
            self.scale,
        )

        query_h = rearrange(
            self.query, "b a i j h d -> b h a j i d"
        ).float()
        key_h = rearrange(
            merged_key, "b a f j k c h d -> b h a j (f k c) d"
        ).float()
        bias_h = rearrange(
            routing_bias, "b h a f k j c -> b h a j (f k c)"
        ).float()
        probability = torch.softmax(
            torch.einsum("bhajid,bhajtd->bhajit", query_h, key_h)
            * self.scale
            - bias_h.unsqueeze(-2),
            dim=-1,
        )
        prototypes_h = rearrange(
            self.prototypes, "b a f k j c h d -> b h a j (f k c) d"
        ).float()
        centers_h = rearrange(
            centers, "b a f k j c h d -> b h a j (f k c) d"
        ).float()
        old_objective = (
            probability.unsqueeze(-1)
            * (query_h.unsqueeze(-2) - prototypes_h.unsqueeze(-3)).square()
        ).sum(dim=(-3, -1))
        new_objective = (
            probability.unsqueeze(-1)
            * (query_h.unsqueeze(-2) - centers_h.unsqueeze(-3)).square()
        ).sum(dim=(-3, -1))
        gain_h = rearrange(
            gain, "b a f k j c h -> b h a j (f k c)"
        )

        torch.testing.assert_close(
            old_objective - new_objective, gain_h, rtol=2e-5, atol=2e-5
        )
        self.assertTrue(torch.all(gain >= -1e-6))

    def test_budgeted_reference_returns_finite_output(self):
        output = hard_route_responsibility_centroid_from_seeds_pytorch(
            self.query,
            self.key,
            self.value,
            self.prototypes,
            self.scale,
            top_r=2,
        )
        self.assertEqual(output.shape, self.query.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_budget_must_not_exceed_cluster_count(self):
        with self.assertRaisesRegex(ValueError, "top_r"):
            hard_route_responsibility_centroid_from_seeds_pytorch(
                self.query,
                self.key,
                self.value,
                self.prototypes,
                self.scale,
                top_r=5,
            )


if __name__ == "__main__":
    unittest.main()
