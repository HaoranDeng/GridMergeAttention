"""CPU-only independent arithmetic and full-layout RF0 selection tests."""
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from analysis.barycentric_medoid_seed import medoid_indices, select


def log_probabilities(z):
    maximum = np.max(z, axis=-1, keepdims=True)
    return z - maximum - np.log(np.exp(z - maximum).sum(axis=-1, keepdims=True))


def numpy_medoid(z):
    """Independent scalar oracle, including an explicit all-pairs KL cost."""
    z = np.asarray(z, dtype=np.float64)
    logp = log_probabilities(z)
    p = np.exp(logp)
    # Directly minimize average_i KL(p_s || p_i), not the implementation's
    # barycenter shortcut. These costs differ only by a seed-independent term.
    direct = np.array([np.mean([np.dot(p[s], logp[s] - logp[i]) for i in range(len(z))])
                       for s in range(len(z))])
    first = int(np.argmin(direct))
    rho = p[first]
    delta = z - z[first]
    residual = (delta - np.outer(delta @ rho, np.ones(z.shape[1]))) * np.sqrt(rho)
    chosen = [first]
    for _ in range(3):
        score = np.einsum("ij,ij->i", residual, residual)
        score[chosen] = -np.inf
        pick = int(np.argmax(score))
        chosen.append(pick)
        basis = residual[pick] / np.sqrt(np.dot(residual[pick], residual[pick]) + 1e-12)
        residual -= np.outer(residual @ basis, basis)
    return np.array(chosen), direct


def numpy_candidates(row):
    chosen = {row}
    chosen.update(range(max(0, row - 11), min(45, row + 12)))
    chosen.update(round(index * 44 / 10) for index in range(11))
    while len(chosen) < 32:
        remaining = [item for item in range(45) if item not in chosen]
        chosen.add(max(remaining, key=lambda item: (min(abs(item - old) for old in chosen), -item)))
    return np.array(sorted(chosen), dtype=np.int64)


class BarycentricMedoidMathTests(unittest.TestCase):
    def test_reverse_kl_barycenter_identity_and_mean_query(self):
        rng = np.random.default_rng(188)
        q, k = rng.normal(size=(9, 8)), rng.normal(size=(7, 8))
        z = q @ k.T / np.sqrt(8)
        mean = q.mean(axis=0) @ k.T / np.sqrt(8)
        np.testing.assert_allclose(mean, z.mean(axis=0), atol=1e-15)
        logp, log_center = log_probabilities(z), log_probabilities(mean)
        p = np.exp(logp)
        _, direct = numpy_medoid(z)
        medoid_cost = np.sum(p * (logp - log_center), axis=-1)
        constant = np.mean(np.logaddexp.reduce(z, axis=-1)) - np.logaddexp.reduce(mean)
        np.testing.assert_allclose(direct, medoid_cost + constant, atol=1e-14)
        # The mean-query distribution is the unrestricted one-cell optimum.
        center = np.exp(log_center)
        optimum = np.mean([np.dot(center, log_center - row) for row in logp])
        self.assertAlmostEqual(optimum, constant, places=14)
        self.assertGreaterEqual(float(direct.min()) + 1e-14, optimum)


@unittest.skipIf(torch is None, "optional PyTorch CPU tests")
class BarycentricMedoidTorchTests(unittest.TestCase):
    def test_batched_first_cost_and_all_three_gs_steps(self):
        z = np.random.default_rng(190).normal(size=(6, 32, 16))
        actual, costs = medoid_indices(torch.from_numpy(z), return_costs=True)
        expected = np.array([numpy_medoid(bank)[0] for bank in z])
        np.testing.assert_array_equal(actual.numpy(), expected)
        lp = log_probabilities(z)
        lrho = log_probabilities(z.mean(axis=1))
        expected_cost = np.sum(np.exp(lp) * (lp - lrho[:, None]), axis=-1)
        np.testing.assert_allclose(costs.numpy(), expected_cost, rtol=1e-12, atol=1e-14)
        self.assertEqual(costs.dtype, torch.float64)
        self.assertTrue(all(len(set(row)) == 4 for row in actual.tolist()))
        self.assertTrue(any(first != 0 for first in actual[:, 0].tolist()))

    def test_reverse_not_forward_kl_and_shift_invariance(self):
        z = np.random.default_rng(221).normal(size=(12, 9, 7)) * 2
        indices = medoid_indices(torch.from_numpy(z))
        lp = log_probabilities(z)
        center_lp = log_probabilities(z.mean(axis=1))
        reverse = np.sum(np.exp(lp) * (lp - center_lp[:, None]), axis=-1).argmin(axis=-1)
        forward = np.sum(np.exp(center_lp[:, None]) * (center_lp[:, None] - lp), axis=-1).argmin(axis=-1)
        self.assertTrue(np.any(reverse != forward))
        np.testing.assert_array_equal(indices[:, 0].numpy(), reverse)
        shifted = z + np.arange(9)[None, :, None] * 5
        np.testing.assert_array_equal(medoid_indices(torch.from_numpy(shifted)).numpy(), indices.numpy())

    def test_uniform_ties_extreme_logits_and_distinctness(self):
        np.testing.assert_array_equal(medoid_indices(torch.zeros(3, 32, 16)).numpy(), [[0, 1, 2, 3]] * 3)
        z = torch.tensor([[[1000., -1000., 0.], [-1000., 1000., 0.],
                           [0., 0., 0.], [0., -1000., 1000.], [0., 0., -1000.]]])
        selected, costs = medoid_indices(z, return_costs=True)
        self.assertTrue(torch.isfinite(costs).all().item())
        self.assertEqual(len(set(selected[0].tolist())), 4)
        expected, _ = numpy_medoid(z[0].double().numpy())
        np.testing.assert_array_equal(selected[0].numpy(), expected)

    def test_full_standard_layout_non_rf0_nan_and_independent_banks(self):
        generator = torch.Generator().manual_seed(234)
        q = torch.full((1, 75600, 4, 128), float("nan"), dtype=torch.bfloat16)
        q[:, :3600] = torch.randn((1, 3600, 4, 128), generator=generator).to(torch.bfloat16)
        k = torch.randn(q.shape, generator=generator).to(torch.bfloat16)
        result = select(q, k, bank_chunk=3072)
        self.assertEqual(set(result), {"barycentric_medoid32"})
        rows = result["barycentric_medoid32"]
        self.assertEqual(rows.shape, (1, 1, 21, 45, 80, 4, 4))
        self.assertEqual(rows.dtype, torch.uint8)
        self.assertTrue(torch.all(rows < 45).item())
        self.assertTrue(torch.all(rows.sort(dim=-1).values.diff(dim=-1) > 0).item())
        matched = torch.arange(45)[None, None, None, :, None, None]
        self.assertTrue(torch.any(rows[..., 0] != matched).item())
        for frame, row, column, head in ((0, 0, 0, 0), (0, 44, 79, 3), (7, 13, 32, 1),
                                         (20, 0, 79, 2), (20, 44, 0, 3)):
            candidates = numpy_candidates(row)
            qq = q[0, candidates * 80 + column, head].float().numpy()
            kk = k[0, frame * 3600 + row * 80 + np.arange(0, 80, 5), head].float().numpy()
            logits = (qq @ kk.T).astype(np.float64) / np.sqrt(128)
            expected, _ = numpy_medoid(logits)
            np.testing.assert_array_equal(rows[0, 0, frame, row, column, head].numpy(), candidates[expected])
        self.assertTrue(torch.isnan(q[:, 3600:]).all().item())

    def test_input_guards(self):
        with self.assertRaises(ValueError): medoid_indices(torch.zeros(1, 3, 16))
        with self.assertRaises(ValueError): medoid_indices(torch.full((1, 4, 16), float("nan")))
        with self.assertRaises(ValueError): medoid_indices(torch.ones(1, 4, 16, dtype=torch.long))
        q = torch.zeros(1, 75600, 1, 128, dtype=torch.bfloat16)
        with self.assertRaises(ValueError): select(q.float(), q.float())
        with self.assertRaises(ValueError): select(q, q, bank_chunk=False)
        with self.assertRaises(ValueError): select(q[:, :3600], q[:, :3600])


if __name__ == "__main__":
    unittest.main()
