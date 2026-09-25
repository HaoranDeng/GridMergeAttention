"""CPU-only analytic and dispatch contracts; no native/GPU quality claims."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None


SOURCE = Path(__file__).resolve().parents[1] / "attention" / "phase_averaged_seed.py"
SPEC = importlib.util.spec_from_file_location("phase_averaged_seed_under_test", SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def complex_rotation(q, position):
    """Independent complex-exponential forward rotation, including absolute t."""
    output = np.array(q, dtype=np.float64, copy=True)
    complex_pairs = q[..., :44:2] + 1j * q[..., 1:44:2]
    frequency = np.exp(-np.log(10000.0) * np.arange(22) / 22.0)
    rotated = complex_pairs * np.exp(1j * position * frequency)
    output[..., :44:2] = rotated.real
    output[..., 1:44:2] = rotated.imag
    return output


def log_softmax(x):
    return x - np.logaddexp.reduce(x, axis=-1, keepdims=True)


class PhaseNumpyTests(unittest.TestCase):
    def setUp(self):
        self.q = np.random.default_rng(6021).normal(size=(3, 128))

    def test_t1_is_identity_and_does_not_alias_or_modify(self):
        self.q[0, 0] = -0.0
        original = self.q.copy()
        actual = module.phase_averaged_query_numpy(self.q, 1)
        np.testing.assert_array_equal(actual.view(np.uint64), original.view(np.uint64))
        self.assertFalse(np.shares_memory(actual, self.q))
        np.testing.assert_array_equal(self.q, original)

    def test_explicit_rotations_agree_with_independent_complex_oracle(self):
        original = self.q.copy()
        for frames in (2, 7, 21):
            expected = np.mean([complex_rotation(self.q, a) for a in range(frames)], axis=0)
            actual = module.phase_averaged_query_numpy(self.q, frames)
            np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
            np.testing.assert_array_equal(actual[..., 44:], self.q[..., 44:])
            self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(self.q, original)

    def test_positive_rotation_is_not_negative_rotation(self):
        q = np.zeros((1, 128), dtype=np.float64)
        q[0, 0] = 1
        actual = module.phase_averaged_query_numpy(q, 2)
        self.assertAlmostEqual(actual[0, 1], np.sin(1.0) / 2)
        negative = (q + complex_rotation(q, -1)) / 2
        self.assertGreater(abs(actual[0, 1] - negative[0, 1]), 0.8)

    def test_nonzero_absolute_start_requires_only_relative_rotation(self):
        start, frames = 13, 21
        roped = complex_rotation(self.q, start)
        actual = module.phase_averaged_query_numpy(roped, frames)
        expected = np.mean([complex_rotation(self.q, start + a)
                            for a in range(frames)], axis=0)
        np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=3e-14)
        double_start = np.mean([complex_rotation(roped, start + a)
                                for a in range(frames)], axis=0)
        self.assertGreater(np.max(np.abs(actual - double_start)), 0.1)

    def test_reverse_kl_virtual_barycenter_identity_and_direction(self):
        rng = np.random.default_rng(3268)
        q = rng.normal(size=128)
        keys = rng.normal(size=(17, 128))
        virtual = np.stack([complex_rotation(q, a) for a in range(21)])
        virtual_logp = log_softmax(virtual @ keys.T * 0.2)
        mean_logp = virtual_logp.mean(axis=0)
        optimum_logp = log_softmax(mean_logp)
        effective = module.phase_averaged_query_numpy(q, 21)
        actual_logp = log_softmax(effective @ keys.T * 0.2)
        np.testing.assert_allclose(actual_logp, optimum_logp, atol=2e-14, rtol=2e-14)
        optimum = np.exp(optimum_logp)
        minimum = np.mean(np.sum(optimum * (optimum_logp - virtual_logp), axis=-1))
        for _ in range(8):
            trial_logp = log_softmax(rng.normal(size=17))
            trial = np.exp(trial_logp)
            objective = np.mean(np.sum(trial * (trial_logp - virtual_logp), axis=-1))
            divergence = np.sum(trial * (trial_logp - optimum_logp))
            self.assertAlmostEqual(objective - minimum, divergence, places=12)
        # The opposite KL direction has an arithmetic-probability barycenter.
        self.assertGreater(np.max(np.abs(optimum - np.exp(virtual_logp).mean(axis=0))), 1e-3)

    def test_input_validation(self):
        for frames in (0, -1):
            with self.assertRaises(ValueError): module.phase_averaged_query_numpy(self.q, frames)
        for frames in (True, 2.0, "21"):
            with self.assertRaises(TypeError): module.phase_averaged_query_numpy(self.q, frames)
        with self.assertRaises(ValueError): module.phase_averaged_query_numpy(np.zeros(44), 21)
        for dtype in (np.complex128, np.int32, np.bool_):
            with self.assertRaises(TypeError):
                module.phase_averaged_query_numpy(np.zeros(128, dtype=dtype), 21)


class PhaseDispatchTests(unittest.TestCase):
    def tensor(self, frames=1):
        return SimpleNamespace(ndim=6, shape=(1, frames, 45, 80, 2, 128))

    def test_wrapper_preserves_exact_real_seed_object_and_all_options(self):
        q, k, effective = self.tensor(), self.tensor(21), object()
        seeds = np.broadcast_to(np.array([17, 2, 33, 4], dtype=np.uint8),
                                (1, 1, 21, 45, 80, 2, 4))
        options = dict(candidate_count=32, candidate_geometry_count=45, probe_count=16,
                       block_j=4, score_audit=object(), output_dtype=object())
        selector = mock.Mock(return_value=seeds)
        with mock.patch.object(module, "phase_averaged_query", return_value=effective) as transform:
            with mock.patch.object(module, "_anchor_selector", return_value=selector):
                actual = module.phase_averaged_fisher_seed_rows(q, k, 128 ** -0.5, **options)
        self.assertIs(actual, seeds)
        transform.assert_called_once_with(q, 21)
        selector.assert_called_once_with(effective, k, 128 ** -0.5, **options)
        np.testing.assert_array_equal(actual[0, 0, 0, 0, 0, 0], [17, 2, 33, 4])

    def test_t_comes_only_from_key_frame_shape_and_k_is_not_read(self):
        selector = mock.Mock(return_value=object())
        with mock.patch.object(module, "phase_averaged_query", return_value=object()) as transform:
            with mock.patch.object(module, "_anchor_selector", return_value=selector):
                for frames in (1, 21):
                    q, k = self.tensor(), self.tensor(frames)
                    module.phase_averaged_fisher_seed_rows(q, k)
                    transform.assert_called_with(q, frames)
                    self.assertIs(selector.call_args.args[1], k)
        for k in (self.tensor(0), SimpleNamespace(ndim=5, shape=(1, 21, 45, 80, 128)),
                  SimpleNamespace(ndim=6, shape=(2, 21, 45, 80, 2, 128))):
            with self.assertRaises(ValueError): module.phase_averaged_fisher_seed_rows(self.tensor(), k)

    def test_source_has_only_analytic_coefficient_cache_and_no_virtual_q_axis(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        cached = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and any(isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                          and d.func.id == "lru_cache" for d in node.decorator_list)]
        self.assertEqual([node.name for node in cached], ["_torch_phase_coefficients"])
        self.assertEqual([arg.arg for arg in cached[0].args.args],
                         ["temporal_frames", "device_type", "device_index"])
        transform = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name == "phase_averaged_query")
        self.assertFalse(any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(transform)))
        forbidden = {"expand", "repeat", "stack", "cat", "matmul", "bmm", "einsum"}
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in forbidden
                             for n in ast.walk(transform)))


@unittest.skipIf(torch is None, "Torch is not installed; these CPU tests have not run")
class PhaseTorchTests(unittest.TestCase):
    def query(self, heads=1):
        generator = torch.Generator(device="cpu").manual_seed(731)
        return torch.randn((1, 1, 45, 80, heads, 128), generator=generator).to(torch.bfloat16)

    def test_t1_bitwise_identity_and_independent_storage(self):
        q = self.query()
        q.flatten()[0] = -0.0
        original = q.clone()
        actual = module.phase_averaged_query(q, 1)
        self.assertTrue(torch.equal(actual.view(torch.int16), q.view(torch.int16)))
        self.assertNotEqual(actual.data_ptr(), q.data_ptr())
        self.assertTrue(torch.equal(q.view(torch.int16), original.view(torch.int16)))
        self.assertTrue(actual.is_contiguous())

    def test_numpy_explicit_mean_and_spatial_bits(self):
        q = self.query()
        # Include spatial signed zeros, infinities and NaN payloads: no spatial arithmetic.
        spatial_bits = q.view(torch.int16)[..., 44:]
        spatial_bits[..., :4] = torch.tensor([-32768, 32640, -128, 32705], dtype=torch.int16)
        original = q.clone()
        actual = module.phase_averaged_query(q, 21)
        expected = torch.from_numpy(module.phase_averaged_query_numpy(q.float().numpy(), 21)).to(torch.bfloat16)
        self.assertTrue(torch.equal(actual[..., :44], expected[..., :44]))
        self.assertTrue(torch.equal(actual.view(torch.int16)[..., 44:], spatial_bits))
        self.assertTrue(torch.equal(q.view(torch.int16), original.view(torch.int16)))
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(actual.is_contiguous())

    def test_noncontiguous_input_is_supported_without_mutation(self):
        q = self.query(heads=2)[..., ::2, :]
        self.assertFalse(q.is_contiguous())
        original = q.clone()
        actual = module.phase_averaged_query(q, 7)
        expected = module.phase_averaged_query(q.contiguous(), 7)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(q, original))
        self.assertTrue(actual.is_contiguous())

    def test_fp64_analytic_coefficients_and_cache_never_reuse_query(self):
        module._torch_phase_coefficients.cache_clear()
        q = self.query()
        first = module.phase_averaged_query(q, 21)
        cache = module._torch_phase_coefficients.cache_info()
        self.assertEqual((cache.misses, cache.hits), (1, 0))
        q.zero_()
        second = module.phase_averaged_query(q, 21)
        self.assertEqual(torch.count_nonzero(second).item(), 0)
        self.assertGreater(torch.count_nonzero(first).item(), 0)
        self.assertEqual(module._torch_phase_coefficients.cache_info().hits, 1)
        cosine, sine = module._torch_phase_coefficients(21, "cpu", None)
        self.assertEqual(cosine.dtype, torch.float64)
        self.assertEqual(sine.dtype, torch.float64)
        self.assertEqual(cosine.shape, (22,))
        expected = np.mean(np.exp(1j * np.arange(21)[:, None]
                                  * 10000.0 ** (-np.arange(22) / 22.0)), axis=0)
        np.testing.assert_allclose(cosine.numpy(), expected.real, atol=3e-15, rtol=3e-15)
        np.testing.assert_allclose(sine.numpy(), expected.imag, atol=3e-15, rtol=3e-15)

    def test_strict_bf16_rf0_full_shape_and_positive_frame_contract(self):
        q = self.query()
        with self.assertRaises(TypeError): module.phase_averaged_query(q.float(), 21)
        with self.assertRaises(TypeError): module.phase_averaged_query(q.half(), 21)
        with self.assertRaises(TypeError): module.phase_averaged_query(q, True)
        with self.assertRaises(ValueError): module.phase_averaged_query(q, 0)
        for invalid in (q[:, :, :44], q[..., :44], q.expand(1, 2, 45, 80, 1, 128)):
            with self.assertRaises(ValueError): module.phase_averaged_query(invalid, 21)


if __name__ == "__main__":
    unittest.main()
