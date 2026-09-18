import unittest

import torch

from analysis.probe_multiresidual_basis_real_qkv import (
    PAIRINGS,
    _collapse_pairing,
)


class MultiresidualBasisTests(unittest.TestCase):
    def test_pair_collapse_matches_partition_at_mean_seed_anchor(self):
        torch.manual_seed(0)
        shape = (1, 1, 2, 3, 2, 4, 2, 5)
        key = torch.randn(shape)
        value = torch.randn(shape)
        seeds = torch.randn(1, 1, 2, 2, 3, 4, 2, 5)
        bias = -torch.rand(1, 2, 1, 2, 2, 3, 4)
        scale = 5**-0.5

        collapsed_key, _, collapsed_bias = _collapse_pairing(
            key, value, bias, seeds, PAIRINGS[0], scale
        )
        seed = seeds.permute(0, 1, 2, 4, 3, 5, 6, 7)
        aligned_bias = bias.permute(0, 2, 3, 5, 4, 6, 1)
        collapsed_bias = collapsed_bias.permute(0, 2, 3, 5, 4, 6, 1)

        for group, children in enumerate(PAIRINGS[0]):
            anchor = seed[..., list(children), :, :].mean(dim=5)
            child_logit = (
                key[..., list(children), :, :]
                * anchor.unsqueeze(5)
            ).sum(dim=-1) * scale - aligned_bias[..., list(children), :]
            expected = torch.logsumexp(child_logit, dim=5)
            actual = (
                collapsed_key[..., group, :, :] * anchor
            ).sum(dim=-1) * scale - collapsed_bias[..., group, :]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_expanded_probability_fusion_matches_separate_outputs(self):
        torch.manual_seed(1)
        tokens, clusters, dim = 7, 4, 5
        value = torch.randn(tokens, clusters, dim)
        parent_probability = torch.softmax(torch.randn(tokens, clusters).flatten(), 0).view(
            tokens, clusters
        )
        coefficients = torch.tensor([0.3, -0.2, 0.1])
        separate = torch.einsum("tc,tcd->d", parent_probability, value)
        effective = (1.0 + coefficients.sum()) * parent_probability

        for coefficient, pairing in zip(coefficients, PAIRINGS):
            pair_probability = torch.softmax(torch.randn(tokens, 2).flatten(), 0).view(
                tokens, 2
            )
            child_weight = torch.zeros(tokens, clusters)
            coarse = torch.zeros(dim)
            for group, children in enumerate(pairing):
                weight = torch.softmax(torch.randn(tokens, 2), dim=-1)
                child_weight[:, list(children)] = weight
                pair_value = (
                    weight.unsqueeze(-1) * value[:, list(children)]
                ).sum(dim=1)
                coarse += torch.einsum(
                    "t,td->d", pair_probability[:, group], pair_value
                )
            separate += coefficient * (
                torch.einsum("tc,tcd->d", parent_probability, value) - coarse
            )
            expanded = torch.zeros_like(parent_probability)
            for group, children in enumerate(pairing):
                expanded[:, list(children)] = (
                    pair_probability[:, group : group + 1]
                    * child_weight[:, list(children)]
                )
            effective -= coefficient * expanded

        fused = torch.einsum("tc,tcd->d", effective, value)
        torch.testing.assert_close(fused, separate, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
