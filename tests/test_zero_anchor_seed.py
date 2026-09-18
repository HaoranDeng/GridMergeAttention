"""CPU-only independent checks for the isolated zero-query seed prototype."""
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from analysis.zero_anchor_seed import select, zero_indices, zero_queries


def numpy_pivots(z):
    """Independent scalar-bank oracle; no production feature/helper calls."""
    z = np.asarray(z, dtype=np.float64)
    residual = (z - np.mean(z, axis=1, keepdims=True)) / np.sqrt(z.shape[1])
    chosen = []
    for _ in range(3):
        norms = np.einsum("ij,ij->i", residual, residual)
        norms[chosen] = -np.inf
        pick = int(np.argmax(norms))
        chosen.append(pick)
        basis = residual[pick] / np.sqrt(np.dot(residual[pick], residual[pick]) + 1e-12)
        residual = residual - np.outer(residual @ basis, basis)
    return chosen


def numpy_candidates(row):
    """Independent reproduction of the fixed C32/g45 geometry contract."""
    chosen = {row}
    chosen.update(range(max(0, row - 11), min(45, row + 12)))
    chosen.update(round(index * 44 / 10) for index in range(11))
    while len(chosen) < 32:
        remaining = [item for item in range(45) if item not in chosen]
        chosen.add(max(remaining, key=lambda item: (min(abs(item - old) for old in chosen), -item)))
    return np.array(sorted(chosen), dtype=np.int64)


class ZeroAnchorMathTests(unittest.TestCase):
    def test_exact_uniform_gap_and_local_curvature(self):
        rng = np.random.default_rng(211)
        direction = rng.normal(size=16)
        direction -= direction.mean()
        uniform = np.full(16, 1 / 16)
        for epsilon in (0.2, 0.01, 0.001):
            z = epsilon * direction
            logp = z - np.logaddexp.reduce(z)
            kl = uniform @ (np.log(uniform) - logp)
            gap = np.logaddexp.reduce(z) - z.mean() - np.log(len(z))
            self.assertAlmostEqual(kl, gap, places=14)
        quadratic = np.dot(direction, direction) / 32
        epsilon = 1e-3
        z = epsilon * direction
        gap = np.logaddexp.reduce(z) - z.mean() - np.log(len(z))
        self.assertLess(abs(gap / epsilon ** 2 - quadratic), 3e-4)

    def test_zero_seed_conditional_uniform_after_singleton(self):
        # The zero seed competes with normalized real-query responses.
        logits = np.array([[0., 0., 0., 0., 0., 0.], [3., -3., -3., -3., -3., -3.],
                           [-3., 3., -3., -3., -3., -3.], [-3., -3., 3., -3., -3., -3.]])
        logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        owner = logp.argmax(axis=0)
        cell = np.flatnonzero((owner == 0) & (np.arange(6) != 5))
        self.assertEqual(cell.tolist(), [3, 4])
        weights = np.exp(logp[0, cell]); weights /= weights.sum()
        np.testing.assert_allclose(weights, np.full(len(cell), 1 / len(cell)))
        self.assertAlmostEqual(float(-(weights * np.log(weights)).sum()), np.log(len(cell)))


@unittest.skipIf(torch is None, "optional PyTorch CPU tests")
class ZeroAnchorTorchTests(unittest.TestCase):
    def test_independent_scalar_gs_and_constant_shift(self):
        z = np.random.default_rng(12).normal(size=(7, 32, 16))
        actual = zero_indices(torch.from_numpy(z)).numpy()
        expected = np.array([numpy_pivots(bank) for bank in z])
        np.testing.assert_array_equal(actual, expected)
        shifted = z + np.arange(32)[None, :, None] * 8
        np.testing.assert_array_equal(zero_indices(torch.from_numpy(shifted)).numpy(), actual)
        self.assertTrue(all(len(set(row)) == 3 for row in actual))

    def test_stable_ties_and_no_forced_real_anchor(self):
        np.testing.assert_array_equal(zero_indices(torch.zeros(2, 32, 16)).numpy(), [[0, 1, 2]] * 2)
        z = torch.zeros(1, 5, 16, dtype=torch.float64)
        z[0, 4, 0] = 10
        self.assertEqual(zero_indices(z)[0, 0].item(), 4)

    def test_full_standard_layout_rf0_only_and_zero_tensor(self):
        generator = torch.Generator().manual_seed(215)
        q = torch.full((1, 75600, 4, 128), float("nan"), dtype=torch.bfloat16)
        q[:, :3600] = torch.randn((1, 3600, 4, 128), generator=generator).to(torch.bfloat16)
        k = torch.randn(q.shape, generator=generator).to(torch.bfloat16)
        result = select(q, k, bank_chunk=3072)
        self.assertEqual(set(result), {"zero_fisher32"})
        rows = result["zero_fisher32"]
        self.assertEqual(rows.shape, (1, 1, 21, 45, 80, 4, 4))
        self.assertEqual(rows.dtype, torch.uint8)
        self.assertTrue(torch.all(rows < 45).item())
        self.assertTrue(torch.all(rows[..., 1:].sort(dim=-1).values.diff(dim=-1) > 0).item())
        expected_dummy = torch.arange(45)[None, None, None, :, None, None].expand(1, 1, 21, 45, 80, 4)
        torch.testing.assert_close(rows[..., 0].long(), expected_dummy)
        for frame, row, column, head in ((0, 0, 0, 0), (0, 44, 79, 3), (7, 13, 32, 1),
                                         (20, 0, 79, 2), (20, 44, 0, 3)):
            candidates = numpy_candidates(row)
            qq = q[0, candidates * 80 + column, head].float().numpy()
            kk = k[0, frame * 3600 + row * 80 + np.arange(0, 80, 5), head].float().numpy()
            z = (qq @ kk.T).astype(np.float64) / np.sqrt(128)
            expected = candidates[numpy_pivots(z)]
            np.testing.assert_array_equal(rows[0, 0, frame, row, column, head, 1:].numpy(), expected)
        zeros = zero_queries(q)
        self.assertEqual(zeros.shape, (1, 21, 80, 4, 128))
        self.assertEqual(zeros.dtype, torch.bfloat16)
        self.assertTrue(zeros.is_contiguous())
        self.assertEqual(torch.count_nonzero(zeros).item(), 0)
        self.assertTrue(torch.isnan(q[:, 3600:]).all().item())

    def test_input_guards(self):
        with self.assertRaises(ValueError): zero_indices(torch.zeros(1, 2, 16))
        with self.assertRaises(ValueError): zero_indices(torch.full((1, 3, 16), float("nan")))
        with self.assertRaises(ValueError): zero_queries(torch.zeros(1, 3600, 1, 128, dtype=torch.bfloat16))
        q = torch.zeros(1, 75600, 1, 128, dtype=torch.bfloat16)
        with self.assertRaises(ValueError): select(q.float(), q.float())
        with self.assertRaises(ValueError): select(q, q, bank_chunk=0)


if __name__ == "__main__":
    unittest.main()
