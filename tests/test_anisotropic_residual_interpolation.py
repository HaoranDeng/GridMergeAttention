import math
import unittest

import torch

from analysis.probe_anisotropic_residual_anchors_real_qkv import (
    continuous_axis_allocation,
    interpolate_anchor_residual_3d,
)
from attention.token_merging import _uniform_anchor_indices


class AnisotropicResidualInterpolationTest(unittest.TestCase):
    def test_reconstructs_trilinear_field_exactly(self):
        frames, rows, width = 7, 9, 11
        time = torch.arange(frames).view(1, frames, 1, 1, 1, 1)
        row = torch.arange(rows).view(1, 1, rows, 1, 1, 1)
        column = torch.arange(width).view(1, 1, 1, width, 1, 1)
        field = 1.5 * time - 0.75 * row + 0.25 * column + 3.0
        time_anchors = _uniform_anchor_indices(frames, 3, field.device)
        row_anchors = _uniform_anchor_indices(rows, 4, field.device)
        width_anchors = _uniform_anchor_indices(width, 5, field.device)
        sampled = (
            field.index_select(1, time_anchors)
            .index_select(2, row_anchors)
            .index_select(3, width_anchors)
        )

        reconstructed = interpolate_anchor_residual_3d(
            sampled,
            time_anchors,
            row_anchors,
            width_anchors,
            frames=frames,
            rows=rows,
            width=width,
        )

        torch.testing.assert_close(reconstructed, field)

    def test_continuous_solution_preserves_budget_and_curvature_ratios(self):
        coefficients = {"time": 1.0, "row": 4.0, "width": 9.0}
        allocation = continuous_axis_allocation(coefficients, 3600)

        self.assertAlmostEqual(math.prod(allocation.values()), 3600.0)
        self.assertAlmostEqual(allocation["row"] / allocation["time"], 2.0)
        self.assertAlmostEqual(allocation["width"] / allocation["time"], 3.0)


if __name__ == "__main__":
    unittest.main()
