import builtins
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from analysis.anchor_fullgrid_seed import METHODS, anchor_features, anchor_indices, select


def numpy_oracle(z, anchor):
    """Independent scalar categorical Hessian, embedding and anchored GS."""
    z = np.asarray(z, dtype=np.float64)
    rho = np.exp(z[anchor] - np.logaddexp.reduce(z[anchor]))
    delta = z - z[anchor]
    metric = np.diag(rho) - np.outer(rho, rho)
    feature = np.array([np.sqrt(rho) * (d - rho @ d) for d in delta])
    np.testing.assert_allclose(np.sum(feature ** 2, axis=-1),
                               np.einsum("ni,ij,nj->n", delta, metric, delta), atol=1e-11)
    residual = feature.copy()
    selected = [int(anchor)]
    for _ in range(3):
        score = np.array([r @ r for r in residual])
        score[selected] = -np.inf
        pick = int(np.argmax(score))
        selected.append(pick)
        basis = residual[pick] / np.sqrt(residual[pick] @ residual[pick] + 1e-12)
        residual -= np.outer(residual @ basis, basis)
    return selected, feature, rho


@unittest.skipIf(torch is None, "PyTorch is required for numerical CPU tests")
class AnchorFullgridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_numpy_oracle_and_fp64_geometry(self):
        z = np.random.default_rng(423).normal(size=(12, 45, 16)) * 2
        anchors = np.arange(12) * 4 % 45
        slots = anchor_indices(torch.tensor(z), anchors)
        feature, rho = anchor_features(torch.tensor(z), anchors)
        for bank, anchor in enumerate(anchors):
            expected, xf, rf = numpy_oracle(z[bank], anchor)
            self.assertEqual(slots[bank].tolist(), expected)
            np.testing.assert_allclose(feature[bank].numpy(), xf, atol=1e-13)
            np.testing.assert_allclose(rho[bank].numpy(), rf, atol=1e-14)
        self.assertEqual(feature.dtype, torch.float64)
        self.assertTrue((slots.sort(-1).values.diff(dim=-1) > 0).all())

    def test_expansion_keeps_anchor_curvature_and_shared_row_features_fixed(self):
        from attention.adaptive_seed_pytorch import multiscale_candidate_table

        table, anchors = multiscale_candidate_table(45, 32, "cpu", geometry_count=45)
        z = torch.randn(45, 45, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(61))
        subset = z.gather(1, table.long()[:, :, None].expand(-1, -1, 16))
        full_feature, full_rho = anchor_features(z, torch.arange(45))
        subset_feature, subset_rho = anchor_features(subset, anchors)
        torch.testing.assert_close(subset_rho, full_rho, rtol=0, atol=0)
        torch.testing.assert_close(subset_feature,
            full_feature.gather(1, table.long()[:, :, None].expand(-1, -1, 16)), rtol=0, atol=0)
        # Modify only the extra candidates: they must never move the metric.
        modified = z.clone()
        for r in range(45):
            extra = [i for i in range(45) if i not in table[r].tolist()]
            modified[r, extra] *= 30
        torch.testing.assert_close(anchor_features(modified, torch.arange(45))[1], full_rho, rtol=0, atol=0)

    def test_shifts_degeneracy_and_distinct_tie_filling(self):
        self.assertEqual(anchor_indices(torch.zeros(2, 45, 16), [44, 0]).tolist(),
                         [[44, 0, 1, 2], [0, 1, 2, 3]])
        z = torch.randn(4, 45, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(711))
        shifted = z + torch.arange(45)[None, :, None] * 4
        torch.testing.assert_close(anchor_indices(z, [0, 8, 22, 44]),
                                   anchor_indices(shifted, [0, 8, 22, 44]))

    def test_complete_output_shape_rf0_only_and_row_mapping(self):
        from attention.adaptive_seed_pytorch import multiscale_candidate_table

        generator = torch.Generator().manual_seed(231)
        q = torch.randn(1, 7200, 1, 3, generator=generator).to(torch.bfloat16)
        k = torch.randn(1, 75600, 1, 3, generator=generator).to(torch.bfloat16)
        q[:, 3600:] = float("nan")
        actual = select(q, k, bank_chunk=1024)
        table, anchors = multiscale_candidate_table(45, 32, "cpu", geometry_count=45)
        for method in METHODS:
            result = actual[method]
            self.assertEqual(tuple(result.shape), (1, 1, 21, 45, 80, 1, 4))
            self.assertEqual(result.dtype, torch.uint8)
            self.assertTrue((result[..., 0] == torch.arange(45)[None, None, None, :, None, None]).all())
            self.assertTrue((result.long().sort(-1).values.diff(dim=-1) > 0).all())
            self.assertTrue((result < 45).all())
            for frame, row, column in ((0, 0, 0), (1, 22, 7), (20, 44, 79)):
                real = table[row].long() if method.endswith("32") else torch.arange(45)
                anchor = int(anchors[row]) if method.endswith("32") else row
                qq = q[0, real * 80 + column, 0].float()
                kk = k[0, frame * 3600 + row * 80 + torch.arange(0, 80, 5), 0].float()
                z = (qq @ kk.T).double().numpy() * 3 ** -0.5
                slots = numpy_oracle(z, anchor)[0]
                self.assertEqual(result[0, 0, frame, row, column, 0].tolist(), real[slots].tolist())

    def test_c32_matches_existing_secant_anchor_first_chunk(self):
        from analysis import screen_fisher_seeds_bf16 as existing
        from attention.adaptive_seed_pytorch import multiscale_candidate_table

        generator = torch.Generator().manual_seed(826)
        q = torch.randn(1, 75600, 1, 128, generator=generator).to(torch.bfloat16)
        k = torch.randn(1, 75600, 1, 128, generator=generator).to(torch.bfloat16)
        # Execute the unchanged existing selector for its first 2048 banks;
        # discard its unfilled tail. This bounds CPU work, not a video gate.
        def first_chunk_only(*args):
            if len(args) == 3 and args[0] == 0 and args[2] == 2048:
                return [0]
            return builtins.range(*args)
        with patch.object(existing, "range", first_chunk_only, create=True):
            control = existing.select_population(q, k, secant=True)["fisher_anchor"].reshape(-1, 4)[:2048]
        table, anchors = multiscale_candidate_table(45, 32, "cpu", geometry_count=45)
        ids = torch.arange(2048)
        row, column = ids // 80, ids % 80
        real = table[row].long()
        qq = q[0, real * 80 + column[:, None], 0].float()
        kk = k[0, row[:, None] * 80 + torch.arange(0, 80, 5), 0].float()
        logits = torch.bmm(qq, kk.transpose(1, 2)).double() * 128 ** -0.5
        actual = real.gather(1, anchor_indices(logits, anchors[row])).to(torch.uint8)
        torch.testing.assert_close(actual, control, rtol=0, atol=0)

    def test_invalid_inputs(self):
        for z, anchor in ((torch.zeros(1, 3, 16), [0]),
                          (torch.zeros(1, 45, 16), [45]),
                          (torch.zeros(1, 45, 16), [1.5]),
                          (torch.full((1, 45, 16), float("nan")), [0])):
            with self.assertRaises(ValueError):
                anchor_indices(z, anchor)
        q = torch.zeros(1, 3600, 1, 3, dtype=torch.bfloat16)
        for kwargs in ({"bank_chunk": 0}, {"bank_chunk": True}):
            with self.assertRaises(ValueError):
                select(q, q, **kwargs)
        with self.assertRaises(ValueError):
            select(q, q.float())
        with self.assertRaises(ValueError):
            select(q[:, :-1], q)


if __name__ == "__main__":
    unittest.main()
