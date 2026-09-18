"""CPU synthetic checks for the actual Torch-screen selector's arithmetic.

The first eight tests require only NumPy and the standard library, executing
the small selector AST through a NumPy adapter. Additional real PyTorch CPU
tests skip when Torch is unavailable. No attention package, Triton, or GPU
initialization is needed. This is not compiled-kernel validation.
"""

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_functions(path, names, scope):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("requested test functions were not found")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


class CPUArray(np.ndarray):
    @property
    def device(self):
        return "cpu"


def _reference(profile, log_mass, anchor, count=4):
    """Independent scalar/batch/frame-loop reference, not the vectorized code."""
    output = np.empty(anchor.shape + (count,), dtype=np.int64)
    for batch_index in np.ndindex(anchor.shape):
        residual = profile[(slice(None),) + batch_index].copy()
        mass = log_mass[(slice(None),) + batch_index]
        rho = np.exp(mass - mass.max())
        selected = []
        for step in range(count):
            if step == 0:
                picked = int(anchor[batch_index])
            else:
                score = sum(rho[t] * np.square(residual[t]).sum(-1)
                            for t in range(residual.shape[0]))
                score[selected] = -np.inf
                picked = int(score.argmax())
            selected.append(picked)
            for frame in range(residual.shape[0]):
                basis = residual[frame, picked].copy()
                basis /= np.sqrt(np.dot(basis, basis) + 1e-12)
                residual[frame] -= np.outer(residual[frame] @ basis, basis)
        output[batch_index] = selected
    return output


class JointFrameMassGSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        adapter = SimpleNamespace(
            exp=np.exp, bool=np.bool_, inf=np.inf,
            amax=lambda x, dim, keepdim=False: np.max(x, axis=dim, keepdims=keepdim),
            sum=lambda x, dim, keepdim=False: np.sum(x, axis=dim, keepdims=keepdim),
            arange=lambda count, device=None: np.arange(count),
            zeros_like=lambda x, dtype=None: np.zeros_like(x, dtype=dtype),
            where=np.where,
            argmax=lambda x, dim: np.argmax(x, axis=dim),
            broadcast_to=np.broadcast_to,
            gather=lambda x, dim, index: np.take_along_axis(x, index, axis=dim),
            rsqrt=lambda x: 1 / np.sqrt(x),
            stack=lambda items, dim: np.stack(items, axis=dim),
        )
        cls.scope = _load_functions(
            ROOT / "analysis/screen_partition_population_seeds.py",
            {"_independent_frame_mass_gs", "_joint_mass_variant_config"},
            {"torch": adapter, "JOINT_MASS_VARIANTS": (
                "gs_c16_256", "gs_joint2_c16_256", "gs_joint2_c32_256")},
        )

    def select(self, profile, mass, anchor, count=4):
        return self.scope["_independent_frame_mass_gs"](
            np.asarray(profile).view(CPUArray), np.asarray(mass), np.asarray(anchor), count
        )

    def test_fixed_variant_definitions(self):
        config = self.scope["_joint_mass_variant_config"]
        self.assertEqual(config("gs_c16_256", 21), (16, 16, [10]))
        self.assertEqual(config("gs_joint2_c16_256", 21), (16, 16, [5, 15]))
        self.assertEqual(config("gs_joint2_c32_256", 21), (32, 45, [5, 15]))
        with self.assertRaises(ValueError):
            config("gs_joint2_c16_256", 1)

    def test_actual_c16_candidate_bank_retains_distinct_rows_and_anchor(self):
        scope = _load_functions(
            ROOT / "attention/adaptive_seed_pytorch.py", {"multiscale_candidate_table"},
            {"torch": SimpleNamespace(device=str, int32=np.int32,
                                      tensor=lambda x, **kw: np.array(x, dtype=kw["dtype"])),
             "_MULTISCALE_TABLE_CACHE": {}},
        )
        bank, anchors = scope["multiscale_candidate_table"](45, 16, "cpu", geometry_count=16)
        self.assertEqual(bank.shape, (45, 16))
        np.testing.assert_array_equal(bank[np.arange(45), anchors], np.arange(45))
        self.assertTrue(np.all(np.diff(bank, axis=-1) > 0))
        self.assertTrue(np.all((bank >= 0) & (bank < 45)))

    def test_matches_independent_reference_on_batched_synthetic_profiles(self):
        rng = np.random.default_rng(418)
        for frames in (1, 2, 3):
            for _ in range(8):
                profile = rng.normal(size=(frames, 2, 3, 7, 5))
                profile -= profile.mean(-1, keepdims=True)
                mass = rng.normal(size=profile.shape[:-1]) * 2
                anchor = rng.integers(0, 7, size=(2, 3))
                original = profile.copy()
                selected = self.select(profile, mass, anchor)
                np.testing.assert_array_equal(selected, _reference(profile, mass, anchor))
                np.testing.assert_array_equal(selected[..., 0], anchor)
                self.assertTrue(np.all(np.diff(np.sort(selected, axis=-1), axis=-1) > 0))
                np.testing.assert_array_equal(profile, original)

    def test_single_frame_and_duplicated_frame_equivalence(self):
        rng = np.random.default_rng(991)
        profile = rng.normal(size=(1, 4, 16, 8))
        profile -= profile.mean(-1, keepdims=True)
        mass = rng.normal(size=profile.shape[:-1])
        anchor = np.array([0, 4, 8, 15])
        single = self.select(profile, mass, anchor)
        duplicated = self.select(np.repeat(profile, 2, axis=0), np.repeat(mass, 2, axis=0), anchor)
        np.testing.assert_array_equal(single, duplicated)

    def test_frame_permutation_and_common_mass_shift_invariance(self):
        rng = np.random.default_rng(77)
        profile = rng.normal(size=(2, 3, 8, 6))
        mass = rng.normal(size=profile.shape[:-1])
        anchor = np.array([0, 2, 7])
        expected = self.select(profile, mass, anchor)
        np.testing.assert_array_equal(self.select(profile[::-1], mass[::-1], anchor), expected)
        np.testing.assert_array_equal(self.select(profile, mass + 4000, anchor), expected)

    def test_common_frame_candidate_mass_scale_preserves_frame_importance(self):
        profile = np.zeros((2, 1, 3, 2))
        profile[0, 0, 1] = [1, -1]
        profile[1, 0, 2] = [2, -2]
        mass = np.zeros((2, 1, 3))
        mass[0] = 10
        anchor = np.array([0])
        self.assertEqual(self.select(profile, mass, anchor, 2)[0, 1], 1)
        per_frame_normalized = mass - mass.max(-1, keepdims=True)
        self.assertEqual(self.select(profile, per_frame_normalized, anchor, 2)[0, 1], 2)

    def test_independent_projection_is_not_flattened_feature_projection(self):
        profile = np.array([
            [[[1., -1., 0.], [2., -2., 0.], [1., 1., -2.]]],
            [[[0., 1., -1.], [0., -2., 2.], [2., -1., -1.]]],
        ])
        mass = np.zeros((2, 1, 3))
        self.assertEqual(self.select(profile, mass, np.array([0]), 2)[0, 1], 2)
        flat = profile[:, 0].transpose(1, 0, 2).reshape(3, -1)
        basis = flat[0] / np.sqrt(np.dot(flat[0], flat[0]) + 1e-12)
        residual = flat - np.outer(flat @ basis, basis)
        score = np.square(residual).sum(-1)
        score[0] = -np.inf
        self.assertEqual(score.argmax(), 1)

    def test_zero_profiles_use_distinct_left_ties_after_fixed_anchor(self):
        selected = self.select(np.zeros((2, 1, 6, 4)), np.zeros((2, 1, 6)), np.array([3]))
        np.testing.assert_array_equal(selected, [[3, 0, 1, 2]])


class RealTorchCPUJointFrameMassGSTests(unittest.TestCase):
    """Actual CPU Torch indexing/arithmetic, never importing GPU attention."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest("optional PyTorch CPU checks: Torch unavailable") from exc
        cls.torch = torch
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(previous_threads, 2))
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        cls.names = ("gs_c16_256", "gs_joint2_c16_256", "gs_joint2_c32_256")
        cls.scope = {
            "torch": torch, "JOINT_MASS_VARIANTS": cls.names,
            "_MULTISCALE_TABLE_CACHE": {}, "C4": 4,
            "NEG_INF": torch.finfo(torch.float32).min,
        }
        _load_functions(
            ROOT / "attention/adaptive_seed_pytorch.py",
            {"multiscale_candidate_table"}, cls.scope,
        )
        _load_functions(
            ROOT / "analysis/sweep_simple_weighted_span_selectors.py",
            {"_gather_candidates", "_project_out", "_greedy_span"}, cls.scope,
        )
        _load_functions(
            ROOT / "analysis/screen_partition_population_seeds.py",
            {"_joint_mass_variant_config", "_independent_frame_mass_gs", "_joint_mass_tables"},
            cls.scope,
        )
        rng = np.random.default_rng(160215)
        cls.q_numpy = rng.normal(size=(1, 21, 45, 16, 4)).astype(np.float32)
        cls.k_numpy = rng.normal(size=(1, 21, 45, 16, 4)).astype(np.float32)
        with torch.inference_mode(), redirect_stdout(io.StringIO()):
            cls.tables = cls.scope["_joint_mass_tables"](
                torch.from_numpy(cls.q_numpy), torch.from_numpy(cls.k_numpy),
                cls.names, chunk=189, probes=16,
            )

    def test_real_torch_helper_matches_independent_numpy_reference(self):
        torch = self.torch
        rng = np.random.default_rng(2701)
        for frames in (1, 2, 3):
            for dtype in (np.float32, np.float64):
                with self.subTest(frames=frames, dtype=dtype):
                    profile = rng.normal(size=(frames, 2, 3, 7, 5)).astype(dtype)
                    profile -= profile.mean(-1, keepdims=True)
                    mass = rng.normal(size=profile.shape[:-1]).astype(dtype)
                    anchor = rng.integers(0, 7, size=(2, 3), dtype=np.int64)
                    tensor_profile = torch.from_numpy(profile.copy())
                    with torch.inference_mode():
                        selected = self.scope["_independent_frame_mass_gs"](
                            tensor_profile, torch.from_numpy(mass), torch.from_numpy(anchor)
                        )
                    self.assertEqual(selected.device.type, "cpu")
                    np.testing.assert_array_equal(selected.numpy(), _reference(profile, mass, anchor))
                    np.testing.assert_array_equal(tensor_profile.numpy(), profile)

    def test_real_torch_full_tables_shape_anchor_distinct_rows_and_bank(self):
        torch = self.torch
        for name, table in self.tables.items():
            with self.subTest(variant=name):
                self.assertEqual(tuple(table.shape), (1, 1, 21, 45, 16, 1, 4))
                self.assertEqual(table.device.type, "cpu")
                self.assertEqual(table.dtype, torch.uint8)
                array = table.numpy()[0, 0, :, :, :, 0, :]
                expected_anchor = np.broadcast_to(np.arange(45)[None, :, None], (21, 45, 16))
                np.testing.assert_array_equal(array[..., 0], expected_anchor)
                self.assertTrue(np.all(np.diff(np.sort(array, axis=-1), axis=-1) > 0))
                count, geometry, _ = self.scope["_joint_mass_variant_config"](name, 21)
                bank, _ = self.scope["multiscale_candidate_table"](
                    45, count, "cpu", geometry_count=geometry
                )
                for row in range(45):
                    self.assertTrue(np.isin(array[:, row], bank[row].numpy()).all())

    def test_real_torch_selected_blocks_match_manual_numpy_qk_and_seed_tuples(self):
        def logsumexp(values):
            maximum = values.max(-1, keepdims=True)
            return (maximum + np.log(np.exp(values - maximum).sum(-1, keepdims=True)))[..., 0]

        # Includes boundaries and different key frames, rows, and columns.
        points = ((0, 0, 0), (0, 17, 7), (10, 44, 15), (20, 7, 3))
        full_key = self.k_numpy[0].reshape(-1, 4).astype(np.float64)
        global_positions = np.arange(256) * (len(full_key) // 256)
        for name in self.names:
            count, geometry, frames = self.scope["_joint_mass_variant_config"](name, 21)
            bank, anchors = self.scope["multiscale_candidate_table"](
                45, count, "cpu", geometry_count=geometry
            )
            for key_frame, key_row, column in points:
                with self.subTest(variant=name, point=(key_frame, key_row, column)):
                    rows = bank[key_row].numpy().astype(np.int64)
                    query = self.q_numpy[0, frames][:, rows, column].astype(np.float64)
                    local_key = self.k_numpy[0, key_frame, key_row].astype(np.float64)
                    # W16/P16 means every local key column is a probe.
                    logits = query @ local_key.T * 0.5
                    global_logits = query @ full_key[global_positions].T * 0.5
                    local_mass = logsumexp(logits)
                    mass = local_mass - logsumexp(global_logits)
                    profile = np.sqrt(np.exp(logits - local_mass[..., None]))
                    profile -= profile.mean(-1, keepdims=True)
                    selected = _reference(
                        profile[:, None], mass[:, None],
                        np.array([int(anchors[key_row])]),
                    )[0]
                    expected_rows = rows[selected]
                    actual_rows = self.tables[name][0, 0, key_frame, key_row, column, 0].numpy()
                    np.testing.assert_array_equal(actual_rows, expected_rows)


if __name__ == "__main__":
    unittest.main()
