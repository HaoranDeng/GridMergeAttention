import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from analysis.joint_column_fisher_seed import joint_indices, select


def explicit_features(z, anchor):
    """Independent NumPy construction; z is [J,R,P]."""
    mean = z.mean(axis=1)
    rho = np.exp(mean - np.logaddexp.reduce(mean, axis=-1, keepdims=True))
    delta = z - z[:, anchor:anchor + 1]
    return np.sqrt(rho[:, None]) * (delta - (delta * rho[:, None]).sum(-1, keepdims=True))


def explicit_gs(z, anchor, *, separate=False):
    feature = explicit_features(z, anchor)
    residual = feature.copy() if separate else feature.transpose(1, 0, 2).reshape(z.shape[1], -1)
    selected = [int(anchor)]
    for step in range(3):
        scores = (residual * residual).sum(axis=(0, 2) if separate else 1)
        scores[selected] = -np.inf
        pick = int(scores.argmax())
        selected.append(pick)
        if step < 2:
            if separate:
                basis = residual[:, pick] / np.sqrt((residual[:, pick] ** 2).sum(-1, keepdims=True) + 1e-12)
                residual -= (residual * basis[:, None]).sum(-1, keepdims=True) * basis[:, None]
            else:
                basis = residual[pick] / np.sqrt(residual[pick] @ residual[pick] + 1e-12)
                residual -= (residual @ basis)[:, None] * basis
    return selected


@unittest.skipIf(torch is None, "PyTorch is required for numerical CPU tests")
class JointColumnFisherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_matches_explicit_concatenation_for_width2_and_width4(self):
        rng = np.random.default_rng(418)
        for width in (2, 4):
            z = rng.normal(size=(8, width, 11, 7)) * 2
            anchors = np.arange(8)
            actual = joint_indices(torch.tensor(z), anchors)
            expected = [explicit_gs(bank, a) for bank, a in zip(z, anchors)]
            np.testing.assert_array_equal(actual.numpy(), expected)
            self.assertEqual(actual.dtype, torch.int64)
            self.assertEqual(actual[:, 0].tolist(), anchors.tolist())
            self.assertTrue((actual.sort(-1).values.diff(dim=-1) > 0).all())

    def test_joint_projection_differs_from_separate_column_projections(self):
        # Each column is one-dimensional, but their joint response has rank 2.
        amplitudes = np.array([[0, 3, -3, 2, -2, .1, -.1],
                               [0, 3, -3, -2, 2, .1, -.1]], dtype=float)
        z = np.stack((amplitudes, -amplitudes), axis=-1)
        joint = joint_indices(torch.tensor(z[None]), [0])[0].tolist()
        separate = explicit_gs(z, 0, separate=True)
        self.assertEqual(joint, explicit_gs(z, 0))
        self.assertEqual(joint[:3], [0, 1, 3])
        self.assertEqual(separate[:3], [0, 1, 2])
        self.assertNotEqual(joint, separate)

    def test_geometric_curvature_is_per_column(self):
        z = np.random.default_rng(62).normal(size=(4, 9, 5))
        z += np.arange(4)[:, None, None] * np.array([2., -1., 3., 0., -2.])
        feature = explicit_features(z, 2)
        # The block-diagonal Fisher metric equals the concatenated norm.
        expected_norm = []
        for row in range(z.shape[1]):
            value = 0.
            for column in range(4):
                mean = z[column].mean(0)
                p = np.exp(mean - np.logaddexp.reduce(mean))
                metric = np.diag(p) - np.outer(p, p)
                delta = z[column, row] - z[column, 2]
                value += delta @ metric @ delta
            expected_norm.append(value)
        np.testing.assert_allclose((feature * feature).sum(axis=(0, 2)), expected_norm, atol=1e-12)
        self.assertEqual(joint_indices(torch.tensor(z[None]), [2])[0].tolist(), explicit_gs(z, 2))

    def test_offsets_column_permutation_and_float32_promotion(self):
        generator = torch.Generator().manual_seed(47)
        z = torch.randn(5, 4, 10, 6, dtype=torch.float64, generator=generator)
        anchors = [0, 1, 2, 3, 4]
        base = joint_indices(z, anchors)
        offset = torch.randn(5, 4, 10, 1, dtype=torch.float64, generator=generator) * 4
        torch.testing.assert_close(base, joint_indices(z + offset, anchors))
        torch.testing.assert_close(base, joint_indices(z[:, [2, 0, 3, 1]], anchors))
        torch.testing.assert_close(joint_indices(z.float(), anchors), joint_indices(z.float().double(), anchors))

    def test_zero_features_fill_distinct_slots_in_table_order(self):
        for width in (2, 4):
            actual = joint_indices(torch.zeros(2, width, 6, 4), [3, 0])
            self.assertEqual(actual.tolist(), [[3, 0, 1, 2], [0, 1, 2, 3]])

    def test_complete_table_layout_rf0_scope_and_real_row_mapping(self):
        from attention.adaptive_seed_pytorch import multiscale_candidate_table

        generator = torch.Generator().manual_seed(803)
        q = torch.randn(2, 7200, 1, 3, generator=generator).to(torch.bfloat16)
        k = torch.randn(2, 7200, 1, 3, generator=generator).to(torch.bfloat16)
        # A selector that reads query frame 1 would now fail its finite check.
        q[:, 3600:] = float("nan")
        table, anchors = multiscale_candidate_table(45, 32, "cpu", geometry_count=45)
        for width in (2, 4):
            actual = select(q, k, joint_width=width, bank_chunk=127)[f"fisher_joint{width}"]
            self.assertEqual(tuple(actual.shape), (2, 1, 2, 45, 80, 1, 4))
            self.assertEqual(actual.dtype, torch.uint8)
            self.assertEqual(actual.device.type, "cpu")
            self.assertTrue((actual[..., 0] == torch.arange(45)[None, None, None, :, None, None]).all())
            self.assertTrue((actual.long().sort(-1).values.diff(dim=-1) > 0).all())
            grouped = actual.reshape(2, 1, 2, 45, 80 // width, width, 1, 4)
            self.assertTrue((grouped == grouped[:, :, :, :, :, :1]).all())
            for batch, frame, row, group in ((0, 0, 0, 0), (1, 0, 22, 7), (1, 1, 44, 80 // width - 1)):
                real = table[row].long()
                columns = torch.arange(group * width, (group + 1) * width)
                keys = k[batch, frame * 3600 + row * 80 + torch.arange(0, 80, 5), 0].float()
                logits = []
                for column in columns:
                    queries = q[batch, real * 80 + column, 0].float()
                    logits.append((queries @ keys.T).double().numpy() * 3 ** -.5)
                slots = explicit_gs(np.stack(logits), int(anchors[row]))
                self.assertEqual(actual[batch, 0, frame, row, group * width, 0].tolist(), real[slots].tolist())
            # Cross a different set of chunk boundaries without changing semantics.
            repeated = select(q, k, joint_width=width, bank_chunk=1024)[f"fisher_joint{width}"]
            torch.testing.assert_close(actual, repeated)

    def test_invalid_inputs(self):
        for z, anchor in ((torch.zeros(1, 3, 5, 4), [0]),
                          (torch.zeros(1, 4, 3, 4), [0]),
                          (torch.zeros(1, 4, 5, 4), [5]),
                          (torch.zeros(1, 4, 5, 4), [1.5]),
                          (torch.full((1, 4, 5, 4), float("nan")), [0])):
            with self.assertRaises(ValueError):
                joint_indices(z, anchor)
        q = torch.zeros(1, 3600, 1, 2)
        for kwargs in ({"joint_width": 3}, {"bank_chunk": 0}):
            with self.assertRaises(ValueError):
                select(q, q, **kwargs)
        with self.assertRaises(ValueError):
            select(q[:, :-1], q)
        with self.assertRaises(ValueError):
            select(q, q.double())


if __name__ == "__main__":
    unittest.main()
