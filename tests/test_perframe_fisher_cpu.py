import unittest

import numpy as np

from analysis.screen_perframe_fisher_cpu import select_from_logits


class PerframeFisherTests(unittest.TestCase):
    def test_current_frame_information_changes_selection(self):
        rng = np.random.default_rng(602)
        z0 = rng.normal(size=(30, 32, 16))
        z1 = rng.normal(size=(30, 32, 16)) * 2
        anchor = np.arange(30) % 32
        for geometry in ("fisher", "uniform"):
            shared = select_from_logits(z0, anchor, geometry)
            current = select_from_logits(z1, anchor, geometry)
            np.testing.assert_array_equal(shared, select_from_logits(z0.copy(), anchor, geometry))
            np.testing.assert_array_equal(current[:, 0], anchor)
            self.assertTrue(np.all(np.diff(np.sort(current, axis=-1), axis=-1) > 0))
            self.assertGreater(np.any(shared != current, axis=1).mean(), 0.9)

    def test_scalar_fisher_and_uniform_gs(self):
        rng = np.random.default_rng(95)
        z = rng.normal(size=(7, 11, 6)) * 3
        anchors = np.arange(7)
        for geometry in ("fisher", "uniform"):
            expected = []
            for bank in range(len(z)):
                ref = z[bank].mean(0)
                rho = np.exp(ref - np.logaddexp.reduce(ref)) if geometry == "fisher" else np.full(6, 1 / 6)
                delta = z[bank] - z[bank, anchors[bank]]
                residual = (delta - (delta @ rho)[:, None]) * np.sqrt(rho)
                chosen = [int(anchors[bank])]
                for _ in range(3):
                    scores = np.array([r @ r for r in residual])
                    scores[chosen] = -np.inf
                    pick = int(scores.argmax())
                    chosen.append(pick)
                    basis = residual[pick] / np.sqrt(residual[pick] @ residual[pick] + 1e-12)
                    residual = np.array([r - (r @ basis) * basis for r in residual])
                expected.append(chosen)
            np.testing.assert_array_equal(select_from_logits(z, anchors, geometry), expected)
            offsets = rng.normal(size=(7, 11, 1)) * 10
            np.testing.assert_array_equal(select_from_logits(z + offsets, anchors, geometry), expected)


if __name__ == "__main__":
    unittest.main()
