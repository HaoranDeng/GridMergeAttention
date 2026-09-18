import unittest

from attention.conditioned_gridmerge_tail_cuda import (
    conditioned_gridmerge_flop_ratio,
)


class ConditionedGridMergeTailTests(unittest.TestCase):
    def test_flop_ratio_counts_exact_tail_and_representative_work(self):
        ratio = conditioned_gridmerge_flop_ratio(
            sequence=75_600,
            query_clusters=300,
            key_clusters=1_000,
            exact_density=0.2,
        )
        expected = 0.2 + 1_000 / 75_600 + 3 * 300 / (2 * 75_600)
        self.assertAlmostEqual(ratio, expected)


if __name__ == "__main__":
    unittest.main()
