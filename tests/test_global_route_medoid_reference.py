"""CPU-only independent reconstruction and enumeration, no QKV or GPU."""

import unittest

import numpy as np

from analysis.global_route_medoid_reference import (
    global_quadrature_state, refine_global_route_medoids,
)
from analysis.sq1_conditional_medoid_reference import uniform16_sq1_support


def lsm(x):
    y = x - np.max(x, axis=-1, keepdims=True)
    return y - np.log(np.exp(y).sum(axis=-1, keepdims=True))


def raw_oracle(z, seeds, mu, sq):
    """Raw measure entropy plus explicit effective attention, not helper KL."""
    banks, targets, probes = z.shape
    local_logp = lsm(z + np.log(mu))
    global_logp = lsm((z + np.log(mu)).transpose(1, 0, 2).reshape(targets, -1))
    weights = np.zeros((banks, 5, probes))
    bounds = np.full((banks, targets, 5), -np.inf)
    owners = np.empty((banks, probes), dtype=int)
    for b in range(banks):
        owners[b] = np.argmax(local_logp[b, seeds[b]], axis=0)
        owners[b, sq] = 4
        for c in range(4):
            mask = owners[b] == c
            if not mask.any():
                continue
            logw = lsm(z[b, seeds[b, c], mask] + np.log(mu[mask]))
            w = np.exp(logw)
            weights[b, c, mask] = w
            measure_entropy = np.sum(w * (np.log(mu[mask]) - logw))
            bounds[b, :, c] = z[b][:, mask] @ w + measure_entropy
        weights[b, 4, sq] = 1.0
        bounds[b, :, 4] = z[b, :, sq]
    global_log_beta = lsm(bounds.transpose(1, 0, 2).reshape(targets, -1))
    global_beta = np.exp(global_log_beta).reshape(targets, banks, 5).transpose(1, 0, 2)
    effective = np.einsum("bic,bcp->ibp", global_beta, weights).reshape(targets, -1)
    local_beta = np.exp(lsm(bounds))
    local_effective = np.einsum("bic,bcp->bip", local_beta, weights)
    return {"owners": owners, "weights": weights, "bounds": bounds,
            "global_beta": global_beta, "local_beta": local_beta,
            "global_logp": global_logp, "local_logp": local_logp,
            "global_gap": direct_kl(effective, global_logp),
            "local_gap": direct_kl(local_effective, local_logp)}


def direct_kl(a, logp):
    term = np.zeros_like(a)
    positive = a > 0
    term[positive] = a[positive] * (np.log(a[positive]) - logp[positive])
    return term.sum(axis=-1)


def fixed_surrogate(z, initial, final, weighting):
    if weighting == "global":
        a = np.einsum("bic,bcp->ibp", initial["global_beta"], final["weights"])
        return float(direct_kl(a.reshape(45, -1), initial["global_logp"]).mean())
    a = np.einsum("bic,bcp->bip", initial["local_beta"], final["weights"])
    return float(direct_kl(a, initial["local_logp"]).mean())


def fixture(banks=3, probes=17):
    rng = np.random.default_rng(61421)
    z = rng.normal(scale=0.45, size=(banks, 45, probes))
    for b in range(banks):
        for row in range(45):
            z[b, row, :16] += np.where(np.arange(16) // 4 == row % 4, 4., -4.)
            z[b, row] += 3 * np.cos(row / 7. + b * 1.9)
        for row in range(4):
            z[b, row, row * 4] += 2.0
    tables = np.stack([np.roll(np.arange(32), b * 7) for b in range(banks)])
    seeds = np.tile(np.arange(4), (banks, 1))
    return z, tables, seeds


def run(z, tables, seeds, mode="global", column=17, order=None):
    columns = uniform16_sq1_support(column)["probe_columns"]
    if order is not None:
        columns = columns[order]
    return refine_global_route_medoids(z, tables, seeds, columns, column, weighting=mode)


class GlobalRouteMedoidTests(unittest.TestCase):
    def test_global_state_matches_independent_full_reconstruction(self):
        z, tables, seeds = fixture()
        result = run(z, tables, seeds)
        for actual in (result["initial"], result["final"]):
            oracle = raw_oracle(z, actual["seed_rows"], result["measures"], result["singleton_index"])
            np.testing.assert_array_equal(actual["owners"], oracle["owners"])
            np.testing.assert_allclose(actual["weights"], oracle["weights"], atol=2e-14)
            np.testing.assert_allclose(np.exp(actual["global_log_beta"]), oracle["global_beta"], atol=2e-14)
            np.testing.assert_allclose(actual["global_gap"], oracle["global_gap"], atol=3e-14, rtol=3e-13)
        self.assertLess(result["new_mean_global_gap"], result["old_mean_global_gap"])

    def test_every_candidate_matches_frozen_demand_brute_enumeration(self):
        z, tables, seeds = fixture(2)
        for mode in ("global", "bank_local"):
            result = run(z, tables, seeds, mode)
            initial = raw_oracle(z, seeds, result["measures"], result["singleton_index"])
            demand = initial["global_beta" if mode == "global" else "local_beta"]
            selected = seeds.copy()
            for step in result["steps"]:
                b, c = step["bank"], step["cell"]
                mask = initial["owners"][b] == c
                alpha = demand[b, :, c] / demand[b, :, c].sum()
                np.testing.assert_allclose(np.exp(step["log_alpha"]), alpha, atol=2e-14)
                # Independent explicit trial ownership, not inequality helper.
                feasible = np.zeros(32, dtype=bool)
                loss = np.zeros(32)
                conditionals = lsm(z[b][:, mask] + np.log(result["measures"][mask]))
                for slot, row in enumerate(tables[b]):
                    trial = selected.copy()
                    trial[b, c] = row
                    if len(set(trial[b])) == 4:
                        owner = raw_oracle(z, trial, result["measures"], result["singleton_index"])["owners"]
                        feasible[slot] = np.array_equal(owner, initial["owners"])
                    w = np.exp(conditionals[row])
                    loss[slot] = sum(alpha[i] * np.sum(w * (conditionals[row] - conditionals[i]))
                                     for i in range(45))
                np.testing.assert_array_equal(step["feasible_candidates"], feasible)
                old = step["old_slot"]
                np.testing.assert_allclose(step["candidate_conditional_kl"] - step["candidate_conditional_kl"][old],
                                           loss - loss[old], atol=4e-14, rtol=3e-12)
                best = int(np.argmin(np.where(feasible, loss, np.inf)))
                chosen = best if loss[best] < loss[old] else old
                self.assertEqual(step["chosen_slot"], chosen)
                selected[b, c] = tables[b, chosen]
                np.testing.assert_array_equal(step["after_rows"], selected[b])
            np.testing.assert_array_equal(selected, result["seed_rows"])

    def test_surrogate_gain_matches_independent_frozen_effective_attention(self):
        z, tables, seeds = fixture()
        for mode in ("global", "bank_local"):
            result = run(z, tables, seeds, mode)
            initial = raw_oracle(z, seeds, result["measures"], result["singleton_index"])
            final = raw_oracle(z, result["seed_rows"], result["measures"], result["singleton_index"])
            before = fixed_surrogate(z, initial, initial, mode)
            after = fixed_surrogate(z, initial, final, mode)
            self.assertAlmostEqual(before - after, result["fixed_beta_surrogate_improvement"], places=13)
            self.assertGreater(result["fixed_beta_surrogate_improvement"], 0)
            self.assertTrue(all(s["normalized_surrogate_improvement"] >= 0 for s in result["steps"]))
            optimized = final["global_gap"].mean() if mode == "global" else final["local_gap"].mean()
            self.assertLessEqual(optimized, after + 3e-14)
            self.assertLessEqual(after, before + 3e-14)
            expected = result["initial"]["global_log_beta" if mode == "global" else "local_log_beta"]
            np.testing.assert_array_equal(result["frozen_log_beta"], expected)
            for step in result["steps"]:
                np.testing.assert_array_equal(step["frozen_log_demand"], expected[step["bank"], :, step["cell"]])

    def test_bank_shifts_change_global_but_not_local_demand(self):
        z, tables, seeds = fixture(2)
        shifted = z.copy()
        shifted[0] += 100.
        first = run(z, tables, seeds)
        second = run(shifted, tables, seeds)
        self.assertGreater(np.max(np.abs(np.exp(first["frozen_log_beta"]) - np.exp(second["frozen_log_beta"]))), .1)
        np.testing.assert_allclose(first["initial"]["local_log_beta"], second["initial"]["local_log_beta"], atol=3e-13)
        local1, local2 = run(z, tables, seeds, "bank_local"), run(shifted, tables, seeds, "bank_local")
        np.testing.assert_array_equal(local1["seed_rows"], local2["seed_rows"])

    def test_global_query_gauge_invariance_and_one_bank_identity(self):
        z, tables, seeds = fixture()
        z = np.round(z * 16) / 16
        shifted = z + ((np.arange(45) - 22) * 128)[None, :, None]
        a, b = run(z, tables, seeds), run(shifted, tables, seeds)
        for key in ("seed_rows", "frozen_log_beta", "old_global_gap", "new_global_gap"):
            np.testing.assert_array_equal(a[key], b[key])
        a, b = run(z[:1], tables[:1], seeds[:1]), run(z[:1], tables[:1], seeds[:1], "bank_local")
        for key in ("seed_rows", "frozen_log_beta", "old_global_gap", "new_global_gap"):
            np.testing.assert_array_equal(a[key], b[key])

    def test_extremely_small_finite_bank_demand_is_not_skipped(self):
        z, tables, seeds = fixture(2)
        z[1] -= 2000.
        result = run(z, tables, seeds)
        tiny = [s for s in result["steps"] if s["bank"] == 1]
        self.assertTrue(all(s["active"] for s in tiny))
        self.assertTrue(all(np.all(np.exp(s["frozen_log_demand"]) == 0) for s in tiny))
        self.assertTrue(any(s["accepted"] for s in tiny))
        for step in tiny:
            self.assertAlmostEqual(np.exp(step["log_alpha"]).sum(), 1., places=12)
            if step["accepted"]:
                self.assertTrue(np.isfinite(step["log_surrogate_improvement"]))
                self.assertLess(step["log_surrogate_improvement"], -1000)

    def test_all45_targets_include_rows_not_in_candidate_table(self):
        z, tables, seeds = fixture(2)
        altered = z.copy()
        altered[:, 32:, :16] += np.tile([3., -3., 2., -2.], 4)
        a, b = run(z, tables, seeds), run(altered, tables, seeds)
        np.testing.assert_array_equal(z[:, :32], altered[:, :32])
        self.assertFalse(np.allclose(a["steps"][0]["candidate_conditional_kl"],
                                    b["steps"][0]["candidate_conditional_kl"]))
        self.assertEqual(a["old_global_gap"].shape, (45,))

    def test_physical_rows_table_order_ownership_and_inputs_unchanged(self):
        z, tables, seeds = fixture(2)
        originals = [x.copy() for x in (z, tables, seeds)]
        result = run(z, tables, seeds)
        for actual, expected in zip((z, tables, seeds), originals):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(result["seed_rows"][:, 0], seeds[:, 0])
        np.testing.assert_array_equal(result["initial"]["owners"], result["final"]["owners"])
        for b in range(2):
            self.assertEqual(len(set(result["seed_rows"][b])), 4)
            self.assertTrue(np.isin(result["seed_rows"][b], tables[b]).all())
        for step in result["steps"]:
            self.assertEqual(step["after_rows"][step["cell"]], tables[step["bank"], step["chosen_slot"]])

    def test_empty_cells_exact_ties_and_sq_support_permutation(self):
        z, tables, seeds = fixture(2)
        result = run(np.zeros_like(z), tables, seeds)
        np.testing.assert_array_equal(result["seed_rows"], seeds)
        self.assertTrue(all(not s["active"] for s in result["steps"]))
        self.assertEqual(result["fixed_beta_surrogate_improvement"], 0.)
        for row in range(45):
            z[:, row, :16] = np.where(np.arange(16) // 4 == row % 4, 4., -4.)
            z[:, row, 16] = 0
        seeds = np.tile([12, 21, 30, 19], (2, 1))
        tied = run(z, tables, seeds)
        np.testing.assert_array_equal(tied["seed_rows"], seeds)
        z, tables, seeds = fixture(2, 16)
        order = np.arange(16)[::-1]
        a, b = run(z, tables, seeds, column=15), run(z[:, :, order], tables, seeds, column=15, order=order)
        np.testing.assert_array_equal(a["seed_rows"], b["seed_rows"])
        np.testing.assert_allclose(a["new_global_gap"], b["new_global_gap"], atol=3e-14)
        self.assertEqual(a["measures"][3], 1.)
        self.assertEqual(a["measures"][0], 79. / 15.)

    def test_new_minimizer_ties_follow_candidate_table_not_physical_row(self):
        z = np.zeros((1, 45, 17))
        for row in range(45):
            z[0, row, :16] = np.where(np.arange(16) // 4 == row % 4, 4., -4.)
        z[0, 1, 4] += 2.
        # Feasible improved cell1 profiles are identical; reversed table order
        # must prefer physical row29 rather than the smallest such real row5.
        tables = np.arange(32)[None, ::-1].copy()
        seeds = np.array([[0, 1, 2, 3]])
        result = run(z, tables, seeds)
        first = result["steps"][0]
        self.assertTrue(first["accepted"])
        self.assertEqual(first["chosen_slot"], 2)
        self.assertEqual(result["seed_rows"][0, 1], 29)

    def test_old_singleton_owner_may_change_but_bulk_owners_do_not(self):
        z = np.zeros((1, 45, 17))
        for row in range(45):
            z[0, row, :16] = np.where(np.arange(16) // 4 == row % 4, 4., -4.)
        z[0, 1, 4] += 2.
        tables, seeds = np.arange(32)[None], np.array([[0, 1, 2, 3]])
        z[:, 1, 16] = 6.
        z[:, 0, 16] = 2.
        result = run(z, tables, seeds)
        sq = result["singleton_index"]
        self.assertEqual(result["initial"]["owners_before_singleton"][0, sq], 1)
        self.assertNotEqual(result["final"]["owners_before_singleton"][0, sq], 1)
        np.testing.assert_array_equal(result["initial"]["owners"], result["final"]["owners"])
        self.assertEqual(result["final"]["owners"][0, sq], 4)

    def test_public_state_api_and_fail_closed_contract(self):
        z, tables, seeds = fixture(2)
        columns = uniform16_sq1_support(17)["probe_columns"]
        state = global_quadrature_state(z, seeds, columns, 17)
        np.testing.assert_array_equal(state["global_gap"], run(z, tables, seeds)["old_global_gap"])
        args = [z, tables, seeds, columns, 17]
        for index, bad in ((0, z[:, :32]), (0, z.astype(int)), (0, z.astype(complex)),
                           (1, tables.astype(float)), (1, np.zeros((2, 32), int)),
                           (1, tables[:, :31]), (2, np.tile([0, 1, 2, 44], (2, 1))),
                           (2, np.tile([0, 1, 2, 2], (2, 1))), (2, seeds.astype(float)),
                           (3, columns.astype(float)), (3, np.zeros(17, int)),
                           (4, True), (4, 80)):
            actual = list(args)
            actual[index] = bad
            with self.assertRaises((ValueError, TypeError)):
                refine_global_route_medoids(*actual)
        for value in (np.nan, np.inf):
            bad = z.copy()
            bad[0, 0, 0] = value
            with self.assertRaises(ValueError): run(bad, tables, seeds)
        with self.assertRaises(ValueError): run(z, tables, seeds, mode="refresh")


if __name__ == "__main__":
    unittest.main()
