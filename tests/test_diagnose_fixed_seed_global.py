"""CPU-only contracts for the fixed-native-seed global diagnostic.

The independent reference uses Python scalar loops, not the implementation's
batched products/reductions. No CUDA operation, capture, or model is required.
"""
from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


SOURCE = Path(__file__).resolve().parents[1] / "analysis" / "diagnose_fixed_seed_global.py"


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def logsumexp(values):
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(x - peak) for x in values))


def softmax(values):
    norm = logsumexp(values)
    return [math.exp(x - norm) for x in values]


def scalar_reference(queries, seeds, keys, values, singleton, scale):
    """Full expanded distribution from independently constructed hard cells."""
    nb, nk, nc = len(keys), len(keys[0]), len(seeds[0])
    dim = len(values[0][0])
    compressed_k, compressed_v, entropies, cell_weights, owners = [], [], [], [], []
    for b in range(nb):
        logits = [[dot(s, key) * scale for key in keys[b]] for s in seeds[b]]
        profiles = [[x - logsumexp(row) for x in row] for row in logits]
        owner = [max(range(nc), key=lambda c: profiles[c][j]) for j in range(nk)]
        owners.append(owner)
        weights, mk, mv, entropy = [], [], [], []
        for c in range(nc):
            members = [j for j in range(nk) if owner[j] == c and j != singleton]
            w = [0.0] * nk
            if members:
                local = softmax([logits[c][j] for j in members])
                for j, mass in zip(members, local):
                    w[j] = mass
                entropy.append(-math.fsum(t * math.log(t) for t in local))
            else:
                entropy.append(-math.inf)
            weights.append(w)
            mk.append([math.fsum(w[j] * keys[b][j][d] for j in range(nk)) for d in range(len(keys[b][0]))])
            mv.append([math.fsum(w[j] * values[b][j][d] for j in range(nk)) for d in range(dim)])
        weights.append([float(j == singleton) for j in range(nk)])
        mk.append(keys[b][singleton])
        mv.append(values[b][singleton])
        entropy.append(0.0)
        compressed_k.append(mk)
        compressed_v.append(mv)
        entropies.append(entropy)
        cell_weights.append(weights)
    records = []
    for q in queries:
        z = [[dot(q, key) * scale for key in block] for block in keys]
        route = [[dot(q, compressed_k[b][c]) * scale + entropies[b][c]
                  for c in range(nc + 1)] for b in range(nb)]
        rho = softmax([logsumexp(row) for row in z])
        approx_rho = softmax([logsumexp(row) for row in route])
        p = [softmax(row) for row in z]
        alpha = [softmax(row) for row in route]
        p_hat = [[math.fsum(alpha[b][c] * cell_weights[b][c][j] for c in range(nc + 1))
                  for j in range(nk)] for b in range(nb)]
        block_dense = [[math.fsum(p[b][j] * values[b][j][d] for j in range(nk))
                        for d in range(dim)] for b in range(nb)]
        block_ideal = [[math.fsum(p_hat[b][j] * values[b][j][d] for j in range(nk))
                        for d in range(dim)] for b in range(nb)]
        dense = [math.fsum(rho[b] * block_dense[b][d] for b in range(nb)) for d in range(dim)]
        ideal = [math.fsum(approx_rho[b] * block_ideal[b][d] for b in range(nb)) for d in range(dim)]
        within = [math.fsum(rho[b] * (block_ideal[b][d] - block_dense[b][d]) for b in range(nb)) for d in range(dim)]
        reweight = [math.fsum((approx_rho[b] - rho[b]) * block_ideal[b][d] for b in range(nb)) for d in range(dim)]
        records.append(dict(dense=dense, ideal=ideal, within=within, reweight=reweight,
                            rho=rho, approx_rho=approx_rho, p=p, p_hat=p_hat,
                            eps=[logsumexp(z[b]) - logsumexp(route[b]) for b in range(nb)],
                            gap=logsumexp([x for row in z for x in row]) - logsumexp([x for row in route for x in row]),
                            block_dense=block_dense, block_ideal=block_ideal))
    return owners, cell_weights, records


@unittest.skipIf(torch is None, "Torch is unavailable; these tests require CPU Torch only")
class FixedSeedGlobalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("fixed_seed_global_under_test", SOURCE)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.cuda_was_initialized = torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_initialized() != cls.cuda_was_initialized:
            raise AssertionError("CPU contract tests unexpectedly initialized CUDA")

    def tensor(self, data):
        return torch.tensor(data, dtype=torch.float64, device="cpu")

    def fixture(self):
        q = self.tensor([[0.4, -0.7, 0.2], [1.1, 0.3, -0.9], [-0.5, 0.8, 0.6]])
        seeds = self.tensor([[[0.7, -0.3, 0.1], [-0.2, 0.8, 0.6], [0.7, -0.3, 0.1]],
                             [[-0.4, 0.2, 1.0], [0.9, -0.6, 0.3], [-0.4, 0.2, 1.0]]])
        k = self.tensor([[[1.0, 0.2, -0.1], [-0.4, 0.9, 0.7], [0.2, -0.8, 1.2], [-0.6, -0.3, 0.4]],
                         [[-0.3, 0.5, 1.1], [0.8, -0.2, -0.4], [0.6, 1.0, 0.3], [-1.1, 0.4, -0.2]]])
        v = self.tensor([[[0.1, 1.0, -0.4], [-0.9, 0.3, 0.8], [1.4, -0.2, 0.5], [0.6, 0.7, -1.0]],
                         [[-0.5, 1.2, 0.6], [0.3, -0.8, 1.1], [-1.3, 0.4, 0.2], [0.9, -0.5, -0.7]]])
        return q, seeds, k, v, 1, 0.73

    def assertClose(self, actual, expected, atol=3e-13):
        torch.testing.assert_close(actual, self.tensor(expected), rtol=3e-12, atol=atol)

    def test_requires_true_fp64_inputs(self):
        q, seeds, k, v, singleton, scale = self.fixture()
        for index in range(3):
            inputs = [seeds, k, v]
            inputs[index] = inputs[index].float()
            with self.assertRaisesRegex(ValueError, "FP64 inputs required"):
                self.module.ideal_merge(*inputs, singleton, scale)

    def test_ownership_normalizes_full_support_before_sq1(self):
        seeds = self.tensor([[[10, 2, 0], [0, 1, 0]]])
        k = torch.eye(3, dtype=torch.float64, device="cpu")[None]
        merged = self.module.ideal_merge(seeds, k, k, 0, 1.0)
        # Raw logits assign key1 to seed0; full-support probabilities assign it
        # to seed1. Removing singleton before ownership also changes that owner.
        self.assertEqual(merged[5].tolist(), [[0, 1, 1]])
        self.assertFalse(bool(merged[3][0, 0]))
        self.assertEqual(merged[4][0, 0].tolist(), [0, 0, 0])
        self.assertAlmostEqual(float(merged[4][0, 1, 1]), math.exp(1) / (math.exp(1) + 1), places=14)

    def test_ties_empty_cells_and_singleton_original_kv(self):
        q, seeds, k, v, singleton, scale = self.fixture()
        mk, mv, entropy, valid, weights, owner = self.module.ideal_merge(seeds, k, v, singleton, scale)
        self.assertTrue(bool((owner < 2).all()))  # duplicate seed2 loses to seed0
        self.assertTrue(bool((~valid[:, 2]).all()))
        self.assertTrue(bool((weights[:, 2] == 0).all()))
        self.assertTrue(bool((weights[:, :, singleton] == 0).all()))
        self.assertClose(mk[:, -1], k[:, singleton].tolist())
        self.assertClose(mv[:, -1], v[:, singleton].tolist())
        self.assertTrue(bool((entropy[:, -1] == 0).all()))
        self.assertTrue(bool(valid[:, -1].all()))
        self.assertTrue(bool(torch.isfinite(mk).all() and torch.isfinite(mv).all() and torch.isfinite(entropy).all()))

    def test_merge_matches_independent_scalar_cells(self):
        q, seeds, k, v, singleton, scale = self.fixture()
        owners, weights, _ = scalar_reference(q.tolist(), seeds.tolist(), k.tolist(), v.tolist(), singleton, scale)
        actual = self.module.ideal_merge(seeds, k, v, singleton, scale)
        self.assertEqual(actual[5].tolist(), owners)
        self.assertClose(actual[4], [row[:-1] for row in weights])

    def test_dense_reference_is_full_flattened_attention(self):
        q, seeds, k, v, singleton, scale = self.fixture()
        actual = self.module.block_reference(q, k, v, scale)
        _, _, reference = scalar_reference(q.tolist(), seeds.tolist(), k.tolist(), v.tolist(), singleton, scale)
        self.assertClose(actual[0], [r["dense"] for r in reference])
        self.assertClose(actual[2], [r["rho"] for r in reference])
        self.assertClose(actual[3], [r["block_dense"] for r in reference])

    def test_global_route_and_signed_error_decomposition(self):
        q, seeds, k, v, singleton, scale = self.fixture()
        _, _, reference = scalar_reference(q.tolist(), seeds.tolist(), k.tolist(), v.tolist(), singleton, scale)
        merged = self.module.ideal_merge(seeds, k, v, singleton, scale)
        summary, arrays = self.module.route_decomposition(q, merged, self.module.block_reference(q, k, v, scale), scale)
        for key in ("ideal", "within", "reweight"):
            self.assertClose(arrays[key], [r[key] for r in reference])
        n = q.numel()
        within = math.fsum(x*x for r in reference for x in r["within"]) / n
        reweight = math.fsum(x*x for r in reference for x in r["reweight"]) / n
        cross = 2 * math.fsum(dot(r["within"], r["reweight"]) for r in reference) / n
        local = math.fsum(r["rho"][b] * math.fsum((a-bv)**2 for a, bv in zip(r["block_ideal"][b], r["block_dense"][b]))
                             for r in reference for b in range(k.shape[0])) / n
        diagonal = math.fsum(r["rho"][b]**2 * math.fsum((a-bv)**2 for a, bv in zip(r["block_ideal"][b], r["block_dense"][b]))
                                for r in reference for b in range(k.shape[0])) / n
        for name, value in dict(within_mse=within, reweight_mse=reweight, cross_term=cross,
                                weighted_local_conditional_mse=local, within_diagonal_mse=diagonal,
                                within_cross_block_term=within-diagonal).items():
            self.assertAlmostEqual(summary[name], value, places=13)
        self.assertAlmostEqual(summary["ideal_mse"], within + reweight + cross, places=13)
        self.assertNotAlmostEqual(summary["within_mse"], summary["weighted_local_conditional_mse"], places=8)

    def test_global_kl_entropy_lower_bound_and_jeffreys_identity(self):
        q, seeds, k, v, singleton, scale = self.fixture()
        _, _, reference = scalar_reference(q.tolist(), seeds.tolist(), k.tolist(), v.tolist(), singleton, scale)
        merged = self.module.ideal_merge(seeds, k, v, singleton, scale)
        _, arrays = self.module.route_decomposition(q, merged, self.module.block_reference(q, k, v, scale), scale)
        for i, row in enumerate(reference):
            dense = [row["rho"][b] * p for b in range(k.shape[0]) for p in row["p"][b]]
            approximate = [row["approx_rho"][b] * p for b in range(k.shape[0]) for p in row["p_hat"][b]]
            reverse = math.fsum(a * math.log(a/p) for a, p in zip(approximate, dense))
            forward = math.fsum(p * math.log(p/a) for a, p in zip(approximate, dense))
            gamma = 0.0
            for b in range(k.shape[0]):
                p, a = row["p"][b], row["p_hat"][b]
                local_reverse = math.fsum(x * math.log(x/y) for x, y in zip(a, p))
                local_forward = math.fsum(y * math.log(y/x) for x, y in zip(a, p))
                self.assertGreaterEqual(row["eps"][b], -1e-13)
                self.assertAlmostEqual(local_reverse, row["eps"][b], places=13)
                gamma += row["rho"][b] * (local_reverse + local_forward)
            self.assertAlmostEqual(math.fsum(approximate), 1.0, places=13)
            self.assertAlmostEqual(reverse, float(arrays["global_gap"][i]), places=13)
            self.assertAlmostEqual(reverse + forward, gamma, places=13)

    def test_uniform_keys_entropy_restores_exact_counting(self):
        q, seeds, k, v, singleton, scale = self.fixture()
        k.zero_()
        merged = self.module.ideal_merge(seeds, k, v, singleton, scale)
        summary, arrays = self.module.route_decomposition(q, merged, self.module.block_reference(q, k, v, scale), scale)
        self.assertAlmostEqual(float(merged[2][0, 0]), math.log(k.shape[1] - 1), places=14)
        self.assertLess(summary["ideal_mse"], 1e-28)
        self.assertLess(abs(summary["global_reverse_kl_mean"]), 1e-13)

    def test_worker_rf10_shares_row_ids_not_query_vectors(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "worker")
        assignments = []
        for target in ("seed_rows", "seed_q"):
            matches = [node for node in ast.walk(worker) if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == target for t in node.targets)]
            self.assertEqual(len(matches), 1, f"unexpected {target} extraction contract")
            assignments.extend(matches)
        # Preserve real 21*45 block layout, but only two vector coordinates.
        table = (np.arange(21 * 45 * 3 * 4).reshape(1, 1, 21, 45, 3, 1, 4) % 45).astype(np.uint8)
        qgrid = torch.arange(11 * 45 * 3 * 2, dtype=torch.float64, device="cpu").reshape(11, 45, 3, 1, 2)
        context = dict(torch=torch, np=np, tables={"hellinger_rf10": table}, name="hellinger_rf10",
                       q=torch.zeros(1, device="cpu"), qgrid=qgrid, col=2, slot=0, rf=10)
        code = compile(ast.fix_missing_locations(ast.Module(body=assignments, type_ignores=[])), str(SOURCE), "exec")
        exec(code, context)
        expected_rows = table[0, 0, :, :, 2, 0].reshape(945, 4).astype(np.int64)
        self.assertTrue(np.array_equal(context["seed_rows"].numpy(), expected_rows))
        self.assertClose(context["seed_q"], qgrid[0, :, 2, 0][torch.from_numpy(expected_rows)].tolist())
        self.assertFalse(torch.equal(context["seed_q"], qgrid[10, :, 2, 0][torch.from_numpy(expected_rows)]))


if __name__ == "__main__":
    unittest.main()
