import math
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from analysis.tangent_advantage_seed import (
    METHODS, _probe_banks, select, tangent_advantage_features,
    tangent_advantage_indices,
)


def scalar_features(z, anchor):
    """Independent target LSE minus seed tangent, with explicit s/i loops."""
    normalizer = np.logaddexp.reduce(z, axis=-1)
    p = np.exp(z - normalizer[:, None])
    entropy = -(p * np.log(p)).sum(-1)
    distance = np.array([
        [normalizer[i] - (p[s] @ z[i] + entropy[s]) for i in range(len(z))]
        for s in range(len(z))
    ])
    advantage = np.maximum(np.exp(-distance) - np.exp(-distance[anchor]), 0)
    advantage[anchor] = 0
    return advantage, distance


def scalar_picks(z, anchor):
    residual, _ = scalar_features(z, anchor)
    chosen = [anchor]
    for _ in range(3):
        scores = np.array([row @ row for row in residual])
        scores[chosen] = -np.inf
        pick = int(scores.argmax())
        chosen.append(pick)
        basis = residual[pick] / math.sqrt(residual[pick] @ residual[pick] + 1e-12)
        residual -= np.array([row @ basis for row in residual])[:, None] * basis
    return chosen


@unittest.skipIf(torch is None, "PyTorch numerical checks require Torch")
class TangentAdvantageTests(unittest.TestCase):
    def test_exact_identity_axes_and_positive_gain(self):
        z = np.random.default_rng(519).normal(size=(5, 8, 7)) * 2
        anchor = np.arange(5)
        feature, distance = tangent_advantage_features(torch.tensor(z), anchor)
        for bank in range(5):
            expected, exact = scalar_features(z[bank], anchor[bank])
            np.testing.assert_allclose(distance[bank], exact, atol=1e-13, rtol=1e-12)
            np.testing.assert_allclose(feature[bank], expected, atol=1e-13, rtol=1e-12)
            self.assertGreater(np.max(np.abs(exact - exact.T)), .01)
            # Exact gain of the two-tangent envelope, target by target.
            coverage = np.exp(-exact)
            envelope = np.maximum(coverage, coverage[anchor[bank]])
            np.testing.assert_allclose(
                feature[bank], envelope - coverage[anchor[bank]], atol=1e-13
            )
        self.assertTrue(bool((feature >= 0).all() & (feature <= 1).all()))
        torch.testing.assert_close(feature[torch.arange(5), torch.tensor(anchor)],
                                   torch.zeros(5, 8, dtype=torch.float64))

    def test_gs_matches_independent_scalar(self):
        z = np.random.default_rng(520).normal(size=(9, 11, 6)) * 3
        anchor = np.arange(9)
        result = tangent_advantage_indices(torch.tensor(z), anchor).numpy()
        expected = [scalar_picks(zz, int(a)) for zz, a in zip(z, anchor)]
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(np.all(np.diff(np.sort(result, axis=-1), axis=-1) > 0))

    def test_zero_gain_ties_remain_distinct(self):
        z = torch.zeros(2, 7, 5, dtype=torch.float64)
        feature, distance = tangent_advantage_features(z, [3, 5])
        self.assertEqual(torch.count_nonzero(feature).item(), 0)
        self.assertEqual(torch.count_nonzero(distance).item(), 0)
        self.assertEqual(tangent_advantage_indices(z, [3, 5]).tolist(),
                         [[3, 0, 1, 2], [5, 0, 1, 2]])

    def test_per_query_logit_shift_invariance(self):
        z = torch.randn(3, 8, 9, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(521))
        shifted = z + 7 * torch.arange(8)[None, :, None]
        torch.testing.assert_close(tangent_advantage_features(z, [0, 2, 7])[0],
                                   tangent_advantage_features(shifted, [0, 2, 7])[0])
        self.assertEqual(tangent_advantage_indices(z, [0, 2, 7]).tolist(),
                         tangent_advantage_indices(shifted, [0, 2, 7]).tolist())

    def test_rf0_gather_and_table_layout(self):
        # Expanded tensors make independent token addressing visible cheaply.
        position = torch.arange(75600, dtype=torch.float32)
        qbase = ((position // 3600) + (position % 45) / 64).bfloat16()
        kbase = ((position // 80) / 1024 + (position % 80) / 128).bfloat16()
        q = qbase[None, :, None, None].expand(1, 75600, 2, 128)
        k = kbase[None, :, None, None].expand_as(q)
        table = torch.arange(45)[:, None].add(torch.arange(32)[None]).remainder(45)
        coordinates = [(0, 0, 0, 0), (20, 44, 79, 1), (8, 17, 31, 0)]
        ids = torch.tensor([((f * 45 + r) * 80 + c) * 2 + h
                            for f, r, c, h in coordinates])
        logits, real, row = _probe_banks(q, k, ids, table)
        for bank, (f, r, c, h) in enumerate(coordinates):
            self.assertEqual(row[bank].item(), r)
            for candidate in (0, 17, 31):
                for probe in (0, 7, 15):
                    rr = table[r, candidate].item()
                    expected = torch.dot(q[0, rr * 80 + c, h].float(),
                        k[0, f * 3600 + r * 80 + probe * 5, h].float()).double()
                    expected *= 128 ** -.5
                    torch.testing.assert_close(logits[bank, candidate, probe], expected)
            self.assertEqual(real[bank].tolist(), table[r].tolist())
        # Run the public loop with lightweight fake logits/pivots to verify
        # reshape ordering and REAL row mapping without a full CPU selector.
        def fake_banks(q, k, ids, table):
            rr = (ids // (2 * 80)) % 45
            return torch.zeros(len(ids), 32, 1), table[rr].long(), rr
        def fake_indices(z, anchor):
            return torch.stack([anchor, (anchor + 1) % 32,
                                (anchor + 2) % 32, (anchor + 3) % 32], -1)
        with patch('attention.adaptive_seed_pytorch.multiscale_candidate_table',
                   return_value=(table, torch.zeros(45, dtype=torch.long))), \
             patch('analysis.tangent_advantage_seed._probe_banks', side_effect=fake_banks), \
             patch('analysis.tangent_advantage_seed.tangent_advantage_indices', side_effect=fake_indices):
            output = select(q, k)[METHODS[0]]
        self.assertEqual(tuple(output.shape), (1, 1, 21, 45, 80, 2, 4))
        self.assertEqual(output.dtype, torch.uint8)
        for f, r, c, h in coordinates:
            self.assertEqual(output[0, 0, f, r, c, h].tolist(), table[r, :4].tolist())

    def test_input_contract(self):
        for z, anchor in ((torch.zeros(1, 3, 4), [0]),
                          (torch.zeros(1, 5, 4), [5]),
                          (torch.zeros(1, 5, 4), [0, 1]),
                          (torch.full((1, 5, 4), float('nan')), [0])):
            with self.assertRaises(ValueError): tangent_advantage_features(z, anchor)
        with self.assertRaises(ValueError): select(torch.zeros(2, 3), torch.zeros(2, 3))


if __name__ == '__main__':
    unittest.main()
