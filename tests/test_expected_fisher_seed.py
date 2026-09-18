"""Independent stacked-feature oracle for the expected-Fisher Gram reference."""
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from analysis.expected_fisher_seed import (
    METHOD, expected_fisher_gram, expected_fisher_gram_numpy,
    expected_fisher_indices, expected_fisher_indices_numpy,
    expected_fisher_metric, expected_fisher_metric_numpy, select,
)


def stacked_oracle(z, anchor):
    """Explicit independent R*P features; no averaged-Hessian/Gram updates."""
    bank = np.asarray(z, dtype=np.float64)
    p = np.exp(bank - np.logaddexp.reduce(bank, axis=-1, keepdims=True))
    delta = bank - bank[anchor]
    feature = np.array([
        np.concatenate([np.sqrt(prob / len(bank)) * (d - prob @ d) for prob in p])
        for d in delta
    ])
    residual = feature.copy()
    chosen = [int(anchor)]
    scores = []
    for _ in range(3):
        norm = np.einsum("ij,ij->i", residual, residual)
        scores.append(norm.copy())
        norm[chosen] = -np.inf
        pick = int(norm.argmax())
        chosen.append(pick)
        basis = residual[pick] / np.sqrt(residual[pick] @ residual[pick] + 1e-12)
        residual -= np.outer(residual @ basis, basis)
    return np.asarray(chosen), feature, np.asarray(scores)


class ExpectedFisherNumpyTests(unittest.TestCase):
    def test_safeguarded_gram_update_matches_feature_gs_when_epsilon_matters(self):
        # Feature squared norms are comparable to 1e-12: a plain Cholesky
        # downdate by 1/(d+eps) is not the same operation in this regime.
        z = np.random.default_rng(782).normal(size=(24, 32, 16)) * 1e-6
        anchor = np.arange(24) * 3 % 32
        actual = expected_fisher_indices_numpy(z, anchor)
        for bank, first, result in zip(z, anchor, actual):
            np.testing.assert_array_equal(result, stacked_oracle(bank, first)[0])

    def test_gram_and_pivots_match_independent_stacked_features(self):
        rng = np.random.default_rng(9607)
        z = rng.normal(size=(20, 32, 16)) * 2.5
        anchors = np.arange(20) * 7 % 32
        gram = expected_fisher_gram_numpy(z, anchors)
        selected = expected_fisher_indices_numpy(z, anchors)
        for n, anchor in enumerate(anchors):
            rows, feature, scores = stacked_oracle(z[n], anchor)
            np.testing.assert_allclose(gram[n], feature @ feature.T, atol=2e-12, rtol=2e-12)
            np.testing.assert_array_equal(selected[n], rows)
            self.assertEqual(rows[0], anchor)
            self.assertEqual(len(set(rows)), 4)

    def test_psd_nullspace_and_distinction_from_fisher_of_mixture(self):
        z = np.random.default_rng(296).normal(size=(5, 32, 16)) * 3
        p = np.exp(z - np.logaddexp.reduce(z, axis=-1, keepdims=True))
        metric = expected_fisher_metric_numpy(z)
        self.assertGreaterEqual(np.linalg.eigvalsh(metric).min(), -2e-15)
        np.testing.assert_allclose(metric.sum(-1), 0, atol=1e-15)
        for n in range(len(z)):
            mean = p[n].mean(0)
            mixture_fisher = np.diag(mean) - np.outer(mean, mean)
            centered = p[n] - mean
            covariance = centered.T @ centered / 32
            np.testing.assert_allclose(mixture_fisher - metric[n], covariance, atol=1e-15)
            self.assertGreater(np.linalg.norm(covariance), 0.01)
            self.assertGreaterEqual(np.linalg.eigvalsh(mixture_fisher - metric[n]).min(), -2e-15)

    def test_constant_logit_shifts_preserve_metric_gram_and_pivots(self):
        rng = np.random.default_rng(114)
        z = rng.normal(size=(6, 32, 16))
        shifted = z + rng.integers(-32, 32, size=(6, 32, 1))
        anchor = np.array([0, 3, 12, 21, 28, 31])
        np.testing.assert_allclose(expected_fisher_metric_numpy(z),
                                   expected_fisher_metric_numpy(shifted), atol=2e-15)
        np.testing.assert_allclose(expected_fisher_gram_numpy(z, anchor),
                                   expected_fisher_gram_numpy(shifted, anchor), atol=3e-13)
        np.testing.assert_array_equal(expected_fisher_indices_numpy(z, anchor),
                                      expected_fisher_indices_numpy(shifted, anchor))

    def test_identical_profiles_reduce_to_single_fisher_and_ties_are_distinct(self):
        profile = np.array([0., 1., -1., 2.])
        z = np.tile(profile, (2, 32, 1)) + np.arange(32)[None, :, None] * 4
        p = np.exp(profile - np.logaddexp.reduce(profile))
        expected = np.diag(p) - np.outer(p, p)
        np.testing.assert_allclose(expected_fisher_metric_numpy(z),
                                   np.broadcast_to(expected, (2, 4, 4)), atol=2e-16)
        np.testing.assert_array_equal(expected_fisher_gram_numpy(z, [31, 0]), 0)
        np.testing.assert_array_equal(expected_fisher_indices_numpy(z, [31, 0]),
                                      [[31, 0, 1, 2], [0, 1, 2, 3]])

    def test_single_probe_saturated_and_rank_deficient_inputs_are_finite(self):
        z = np.arange(32, dtype=float)[None, :, None]
        np.testing.assert_array_equal(expected_fisher_indices_numpy(z, [12]), [[12, 0, 1, 2]])
        saturated = np.full((1, 32, 16), -1000.)
        saturated[:, :, 0] = 1000.
        np.testing.assert_array_equal(expected_fisher_metric_numpy(saturated), 0)
        np.testing.assert_array_equal(expected_fisher_indices_numpy(saturated, [7]), [[7, 0, 1, 2]])
        rank_one = np.arange(32.)[None, :, None] * np.array([1., -1.])[None, None]
        actual = expected_fisher_indices_numpy(rank_one, [0])
        np.testing.assert_array_equal(actual, expected_fisher_indices_numpy(rank_one, [0]))
        self.assertEqual(len(set(actual[0])), 4)
        self.assertTrue(np.isfinite(expected_fisher_gram_numpy(rank_one, [0])).all())

    def test_input_validation(self):
        for z, a in ((np.zeros((1, 3, 16)), [0]), (np.zeros((1, 32, 0)), [0]),
                     (np.zeros((1, 32, 16)), [32]), (np.zeros((1, 32, 16)), [1.5]),
                     (np.zeros((1, 32, 16)), [True]), (np.zeros((1, 32, 16)), [0, 1]),
                     (np.full((1, 32, 16), np.nan), [0])):
            with self.assertRaises(ValueError): expected_fisher_indices_numpy(z, a)
        with self.assertRaises(TypeError):
            expected_fisher_indices_numpy(np.zeros((1, 32, 16), dtype=int), [0])


@unittest.skipIf(torch is None, "actual PyTorch CPU optional in local runtime")
class ExpectedFisherTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_torch_fp64_matches_numpy_and_stacked_oracle(self):
        z = np.random.default_rng(131).normal(size=(15, 32, 16)) * 3
        anchor = np.arange(15) * 5 % 32
        tz = torch.tensor(z, dtype=torch.float64)
        metric = expected_fisher_metric(tz)
        gram = expected_fisher_gram(tz, anchor)
        selected = expected_fisher_indices(tz, anchor)
        self.assertEqual(metric.dtype, torch.float64)
        self.assertEqual(selected.dtype, torch.int64)
        np.testing.assert_allclose(metric.numpy(), expected_fisher_metric_numpy(z), atol=2e-15)
        np.testing.assert_allclose(gram.numpy(), expected_fisher_gram_numpy(z, anchor), atol=2e-12)
        for bank, first, got in zip(z, anchor, selected):
            np.testing.assert_array_equal(got.numpy(), stacked_oracle(bank, first)[0])
        tiny = z * 1e-6
        selected = expected_fisher_indices(torch.from_numpy(tiny), anchor)
        for bank, first, got in zip(tiny, anchor, selected):
            np.testing.assert_array_equal(got.numpy(), stacked_oracle(bank, first)[0])

    def test_degenerate_ties_and_finite_validation(self):
        z = torch.zeros((2, 32, 16))
        self.assertEqual(expected_fisher_indices(z, [31, 0]).tolist(), [[31, 0, 1, 2], [0, 1, 2, 3]])
        for a in ([32, 0], [0., 1.], [True, False], [0]):
            with self.assertRaises(ValueError): expected_fisher_indices(z, a)
        with self.assertRaises(TypeError): expected_fisher_indices(z.long(), [0, 0])
        with self.assertRaises(ValueError): expected_fisher_indices(z * float("nan"), [0, 0])

    def test_full_row_table_is_rf0_only_and_maps_real_anchor_rows(self):
        from attention.adaptive_seed_pytorch import multiscale_candidate_table

        generator = torch.Generator().manual_seed(619)
        # Integer BF16 vectors make FP32 dot products exact. Non-RF0 Q is NaN.
        q = torch.randint(-3, 4, (2, 7200, 2, 8), generator=generator).to(torch.bfloat16)
        k = torch.randint(-3, 4, (2, 7200, 2, 8), generator=generator).to(torch.bfloat16)
        q[:, 3600:] = float("nan")
        output = select(q, k, bank_chunk=417)[METHOD]
        self.assertEqual(tuple(output.shape), (2, 1, 2, 45, 80, 2, 4))
        self.assertEqual(output.dtype, torch.uint8)
        self.assertTrue(bool((output[..., 0] == torch.arange(45)[None, None, None, :, None, None]).all()))
        self.assertTrue(bool((output.long().sort(-1).values.diff(dim=-1) > 0).all()))
        table, anchors = multiscale_candidate_table(45, 32, "cpu", geometry_count=45)
        for batch, frame, row, column, head in ((0, 0, 0, 0, 0), (1, 1, 44, 79, 1), (1, 0, 22, 7, 0)):
            real = table[row].long()
            qq = q[batch, real * 80 + column, head].float()
            kk = k[batch, frame * 3600 + row * 80 + torch.arange(0, 80, 5), head].float()
            z = (qq @ kk.T).double().numpy() * 8 ** -0.5
            selected = stacked_oracle(z, int(anchors[row]))[0]
            self.assertEqual(output[batch, 0, frame, row, column, head].tolist(), real[selected].tolist())

    def test_qkv_contract_and_batch_chunk_validation(self):
        q = torch.zeros((1, 3600, 1, 3), dtype=torch.bfloat16)
        for kw in ({"bank_chunk": 0}, {"bank_chunk": True}):
            with self.assertRaises(ValueError): select(q, q, **kw)
        with self.assertRaises(ValueError): select(q, q.float())
        with self.assertRaises(ValueError): select(q[:, :-1], q)

    def test_standard_twenty_one_frame_shape_without_reading_later_q(self):
        generator = torch.Generator().manual_seed(762)
        q = torch.randn((1, 75600, 1, 3), generator=generator).to(torch.bfloat16)
        k = torch.randn((1, 75600, 1, 3), generator=generator).to(torch.bfloat16)
        q[:, 3600:] = float("nan")
        actual = select(q, k)[METHOD]
        self.assertEqual(tuple(actual.shape), (1, 1, 21, 45, 80, 1, 4))
        self.assertTrue(bool((actual[..., 0] == torch.arange(45)[None, None, None, :, None, None]).all()))
        self.assertTrue(bool((actual.long().sort(-1).values.diff(dim=-1) > 0).all()))


if __name__ == "__main__":
    unittest.main()
