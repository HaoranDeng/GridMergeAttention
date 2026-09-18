import unittest

import torch
from einops import rearrange

from analysis.probe_hierarchical_split_gma_real_qkv import (
    hard_c4_binary_children,
    refined_representatives,
)


class HierarchicalSplitGmaTest(unittest.TestCase):
    def test_refinement_monotonically_tightens_cluster_mass(self):
        torch.manual_seed(17)
        scale = 0.5
        seed = torch.randn(1, 1, 1, 1, 2, 1, 1, 4)
        query = torch.randn(1, 1, 5, 2, 1, 4)
        key = torch.randn(1, 1, 1, 8, 1, 4)
        value = torch.randn_like(key)
        query_h = rearrange(query, "b a i j h d -> b h a j i d").float()

        def approximate_log_mass(representatives, bias):
            representatives = rearrange(
                representatives, "b h a f k j c d -> b h a j (f k c) d"
            )
            bias = rearrange(bias, "b h a f k j c -> b h a j (f k c)")
            logits = torch.einsum(
                "bhajid,bhajtd->bhajit", query_h, representatives
            )
            return (logits * scale - bias.unsqueeze(-2)).logsumexp(dim=-1)

        exact_scores = torch.einsum(
            "baijhd,bfklhd->bhajifkl", query, key
        ).float() * scale
        exact_log_mass = exact_scores.flatten(-3, -1).logsumexp(dim=-1)
        for split_mode in ("median", "mean"):
            children = hard_c4_binary_children(
                seed, key, value, scale, split_mode=split_mode
            )
            parent_key = rearrange(
                children["parent_key"], "b a f j k c h d -> b h a f k j c d"
            ).float()
            refined_key, _, refined_bias = refined_representatives(children, 1)
            parent_log_mass = approximate_log_mass(
                parent_key, children["parent_bias"]
            )
            refined_log_mass = approximate_log_mass(refined_key, refined_bias)
            self.assertTrue(torch.all(parent_log_mass <= refined_log_mass + 1e-6))
            self.assertTrue(torch.all(refined_log_mass <= exact_log_mass + 1e-6))

    def test_refinement_is_exact_at_expansion_seed(self):
        torch.manual_seed(23)
        scale = 0.5
        seed = torch.randn(1, 1, 1, 1, 1, 1, 1, 4)
        query = seed[:, :, 0, 0, :, 0].unsqueeze(2)
        key = torch.randn(1, 1, 1, 8, 1, 4)
        value = torch.randn_like(key)
        children = hard_c4_binary_children(seed, key, value, scale)
        refined_key, refined_value, refined_bias = refined_representatives(children, 1)
        query_h = rearrange(query, "b a i j h d -> b h a j i d").float()
        key_flat = rearrange(
            refined_key, "b h a f k j c d -> b h a j (f k c) d"
        )
        value_flat = rearrange(
            refined_value, "b h a f k j c d -> b h a j (f k c) d"
        )
        bias_flat = rearrange(
            refined_bias, "b h a f k j c -> b h a j (f k c)"
        )
        probability = torch.softmax(
            torch.einsum("bhajid,bhajtd->bhajit", query_h, key_flat) * scale
            - bias_flat.unsqueeze(-2),
            dim=-1,
        )
        refined = torch.einsum("bhajit,bhajtd->bhajid", probability, value_flat)

        exact_scores = torch.einsum(
            "baijhd,bfklhd->bhajifkl", query, key
        ).float() * scale
        exact_probability = torch.softmax(exact_scores.flatten(-3, -1), dim=-1)
        exact_value = rearrange(value, "b f k l h d -> b h (f k l) d").float()
        exact = torch.einsum("bhajit,bhtd->bhajid", exact_probability, exact_value)
        torch.testing.assert_close(refined, exact, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
