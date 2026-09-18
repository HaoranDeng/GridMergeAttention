import itertools
import unittest

import torch

from analysis.evaluate_fixed_qblock_seed_oracle import (
    C4,
    COLS,
    FRAMES,
    ROWS,
    _key_block_indices,
    _repeat_over_key_blocks,
    _select_proposal_per_column,
    _set_agreement,
    exact_c4_pmedian,
)


class FixedQBlockSeedOracleTests(unittest.TestCase):
    def test_exact_pmedian_matches_bruteforce(self) -> None:
        points = torch.arange(ROWS, dtype=torch.float32)
        distance = (points[:, None] - points[None, :]).abs()[None]
        combinations = torch.tensor(
            list(itertools.combinations(range(ROWS), C4)), dtype=torch.long
        )
        selected, cost = exact_c4_pmedian(
            distance, combinations, combo_chunk=997
        )
        brute_cost = distance[0, :, combinations].amin(-1).mean(0)
        self.assertAlmostEqual(float(cost[0]), float(brute_cost.min()), places=6)
        self.assertEqual(
            selected[0].tolist(), combinations[brute_cost.argmin()].tolist()
        )

    def test_repeat_over_key_blocks_is_fixed(self) -> None:
        rows = torch.arange(COLS * C4).reshape(COLS, C4) % ROWS
        table = _repeat_over_key_blocks(rows)
        self.assertEqual(table.shape, (COLS, FRAMES * ROWS, C4))
        self.assertTrue(torch.equal(table[:, 0], rows))
        self.assertTrue(torch.equal(table[:, -1], rows))

    def test_set_agreement(self) -> None:
        left = torch.tensor([[0, 4, 8, 12], [1, 2, 3, 4]])
        right = torch.tensor([[12, 8, 20, 24], [1, 2, 3, 4]])
        value = _set_agreement(left, right)
        self.assertAlmostEqual(value["mean_overlap_of_4"], 3.0)
        self.assertAlmostEqual(value["exact_set_fraction"], 0.5)

    def test_key_block_indices(self) -> None:
        self.assertEqual(_key_block_indices([0, 2], [1, 3]), [1, 3, 91, 93])

    def test_select_proposal_per_column(self) -> None:
        proposals = {
            "a": torch.zeros(COLS, C4, dtype=torch.long),
            "b": torch.ones(COLS, C4, dtype=torch.long),
        }
        a_cost = torch.arange(COLS, dtype=torch.float32)
        b_cost = torch.arange(COLS - 1, -1, -1, dtype=torch.float32)
        selected, counts = _select_proposal_per_column(
            proposals, {"a": a_cost, "b": b_cost}
        )
        self.assertEqual(counts, {"a": COLS // 2, "b": COLS // 2})
        self.assertTrue((selected[: COLS // 2] == 0).all())
        self.assertTrue((selected[COLS // 2 :] == 1).all())


if __name__ == "__main__":
    unittest.main()
