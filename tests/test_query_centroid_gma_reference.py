import unittest

import torch

from analysis.probe_query_centroid_gma_real_qkv import (
    farthest_point_medoids,
    global_centroid_seed_queries,
    lloyd_centroids,
    local_centroid_seed_queries,
    theoretical_flops,
)


class QueryCentroidGmaReferenceTest(unittest.TestCase):
    def test_lloyd_update_does_not_increase_fit_error(self):
        points = torch.tensor([[[0.0], [1.0], [8.0], [9.0]]])
        initial = torch.tensor([[[0.0], [1.0]]])
        before = (points.unsqueeze(-2) - initial.unsqueeze(-3)).square().amin(-2).mean()
        centers = lloyd_centroids(points, initial, steps=2)
        after = (points.unsqueeze(-2) - centers.unsqueeze(-3)).square().amin(-2).mean()
        self.assertLessEqual(float(after), float(before))

    def test_farthest_medoids_are_input_points(self):
        points = torch.arange(12, dtype=torch.float32).view(1, 6, 2)
        centers = farthest_point_medoids(points, clusters=4)
        self.assertEqual(centers.shape, (1, 4, 2))
        for center in centers[0]:
            self.assertTrue(torch.any(torch.all(points[0] == center, dim=-1)))

    def test_seed_shapes_match_hard_c4_contract(self):
        query = torch.randn(1, 1, 9, 5, 2, 8)
        global_seed = global_centroid_seed_queries(
            query, 3, 9, "farthest", 1, "euclidean", False
        )
        local_seed = local_centroid_seed_queries(
            query, 3, 9, 7, 1, "euclidean", False
        )
        local_c5_seed = local_centroid_seed_queries(
            query, 3, 9, 7, 1, "euclidean", False, clusters=5
        )
        expected = (1, 1, 3, 9, 5, 4, 2, 8)
        self.assertEqual(global_seed.shape, expected)
        self.assertEqual(local_seed.shape, expected)
        self.assertEqual(local_c5_seed.shape, (1, 1, 3, 9, 5, 5, 2, 8))

    def test_centroid_variants_fit_svg_budget(self):
        flops = theoretical_flops(0.37906453310046345)
        self.assertTrue(flops["global_passes_limit"])
        self.assertTrue(flops["local_passes_limit"])
        self.assertTrue(flops["local_c5_passes_limit"])


if __name__ == "__main__":
    unittest.main()
