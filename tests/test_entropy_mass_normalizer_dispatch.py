"""Reproducible dispatch tests with only Python's standard library.

Run with ``python -m unittest tests.test_entropy_mass_normalizer_dispatch -v``.
The actual production wrapper AST is executed against metadata-only tensors
and recording launchers; Torch, Triton, CUDA, and attention package imports are
never needed. These tests verify API/launch decisions, not numerical parity or
compiled GPU behavior. Use benchmark_mass_normalizer_precision.py for those.
"""

import ast
import inspect
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "attention" / "entropy_mass_seed_triton.py"
PUBLIC = ("global_probe_log_norm_triton", "entropy_mass_seed_rows_triton")
FLAGS = ("native_low_precision_dot", "tf32_low_precision_dot")
DTYPES = ("bf16", "fp16", "fp32")


class MetadataTensor:
    def __init__(self, frames, dtype="bf16"):
        self.shape = (1, frames, 45, 80, 40, 128)
        self.dtype = dtype
        self.ndim = 6
        self.device = "cuda:metadata-only"
        self.is_cuda = True

    def contiguous(self):
        return self


class RecordingKernel:
    def __init__(self, name, launches):
        self.name = name
        self.launches = launches

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append(SimpleNamespace(
                name=self.name, grid=grid, args=args, kwargs=kwargs,
            ))
        return launch


class NormalizerDispatchTests(unittest.TestCase):
    def setUp(self):
        self.launches = []
        self.allocations = []

        def empty(shape, *, device, dtype):
            tensor = SimpleNamespace(shape=tuple(shape), device=device, dtype=dtype)
            self.allocations.append(tensor)
            return tensor

        # An isolated package name avoids importing or replacing attention's
        # real package/modules when this file runs in a larger test suite.
        package_name = "_entropy_mass_dispatch_test"
        package = ModuleType(package_name)
        package.__path__ = []
        candidates = ModuleType(package_name + ".adaptive_seed_pytorch")
        candidates.multiscale_candidate_table = mock.Mock(
            return_value=(object(), object())
        )
        modules = mock.patch.dict(sys.modules, {
            package_name: package,
            candidates.__name__: candidates,
        })
        modules.start()
        self.addCleanup(modules.stop)
        self.scope = {
            "__package__": package_name,
            "torch": SimpleNamespace(
                bfloat16="bf16", float16="fp16", float32="fp32",
                uint8="u8", int32="i32", empty=empty,
            ),
            "triton": SimpleNamespace(
                cdiv=lambda x, y: (x + y - 1) // y,
                next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
            ),
            "_global_probe_log_norm_kernel": RecordingKernel("norm", self.launches),
            "_entropy_mass_seed_kernel": RecordingKernel("seed", self.launches),
        }
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        names = {"_check_inputs", *PUBLIC}
        functions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.assertEqual({node.name for node in functions}, names)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), self.scope)

    def call(self, public, dtype="bf16", **options):
        self.launches.clear()
        self.allocations.clear()
        return self.scope[public](MetadataTensor(1, dtype), MetadataTensor(21, dtype), **options)

    def assert_flags(self, native=False, tf32=False):
        self.assertEqual(self.launches[0].name, "norm")
        self.assertIs(self.launches[0].kwargs["NATIVE_LOW_PRECISION_DOT"], native)
        self.assertIs(self.launches[0].kwargs["TF32_LOW_PRECISION_DOT"], tf32)

    def test_both_public_flags_default_off(self):
        for public in PUBLIC:
            for flag in FLAGS:
                with self.subTest(public=public, flag=flag):
                    self.assertIs(inspect.signature(self.scope[public]).parameters[flag].default, False)

    def test_default_dispatch_stays_on_original_path_for_every_dtype(self):
        for public in PUBLIC:
            for dtype in DTYPES:
                with self.subTest(public=public, dtype=dtype):
                    self.call(public, dtype)
                    self.assert_flags()
                    self.assertEqual(self.launches[0].kwargs["num_stages"], 1)
                    self.assertEqual(self.launches[0].kwargs["num_warps"], 8)

    def test_opt_in_dispatches_only_requested_path_for_low_precision(self):
        for public in PUBLIC:
            for dtype in ("bf16", "fp16"):
                for flag in FLAGS:
                    with self.subTest(public=public, dtype=dtype, flag=flag):
                        self.call(public, dtype, **{flag: True})
                        self.assert_flags(native=flag == FLAGS[0], tf32=flag == FLAGS[1])

    def test_fp32_opt_in_preserves_original_launch_configuration(self):
        for public in PUBLIC:
            self.call(public, "fp32")
            original = [(launch.name, launch.grid, launch.kwargs) for launch in self.launches]
            for flag in FLAGS:
                with self.subTest(public=public, flag=flag):
                    self.call(public, "fp32", **{flag: True})
                    self.assert_flags()
                    self.assertEqual(
                        [(launch.name, launch.grid, launch.kwargs) for launch in self.launches],
                        original,
                    )

    def test_conflicting_flags_rejected_before_allocation_or_launch(self):
        for public in PUBLIC:
            for dtype in DTYPES:
                with self.subTest(public=public, dtype=dtype):
                    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                        self.call(public, dtype, **dict.fromkeys(FLAGS, True))
                    self.assertEqual(self.launches, [])
                    self.assertEqual(self.allocations, [])

    def test_zscore_rejects_conflicting_flags_even_without_normalizer(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.call(PUBLIC[1], weight_mode="zscore", **dict.fromkeys(FLAGS, True))
        self.assertEqual(self.launches, [])
        self.assertEqual(self.allocations, [])

    def test_zscore_skips_global_normalizer(self):
        for options in ({}, {FLAGS[0]: True}, {FLAGS[1]: True}):
            with self.subTest(options=options):
                self.call(PUBLIC[1], weight_mode="zscore", **options)
                self.assertEqual([launch.name for launch in self.launches], ["seed"])
                self.assertIs(self.launches[0].kwargs["MASS_WEIGHT"], False)

    def test_output_metadata_and_normalizer_handoff(self):
        norm = self.call(PUBLIC[0])
        self.assertEqual(norm.shape, (1, 1, 45, 80, 40))
        self.assertEqual(norm.dtype, "fp32")
        self.assertIs(self.launches[0].args[2], norm)
        seeds = self.call(PUBLIC[1])
        self.assertEqual(seeds.shape, (1, 1, 21, 45, 80, 40, 4))
        self.assertEqual(seeds.dtype, "u8")
        self.assertEqual([launch.name for launch in self.launches], ["norm", "seed"])
        self.assertIs(self.launches[1].args[2], self.launches[0].args[2])
        self.assertEqual(self.launches[1].kwargs["num_warps"], 4)

    def test_precision_options_do_not_change_seed_kernel_arguments(self):
        self.call(PUBLIC[1])
        original = self.launches[1].kwargs
        for flag in FLAGS:
            with self.subTest(flag=flag):
                self.call(PUBLIC[1], **{flag: True})
                self.assertEqual(self.launches[1].kwargs, original)

    def test_mass_wrapper_forwards_normalizer_settings(self):
        for flag in FLAGS:
            with self.subTest(flag=flag):
                self.call(PUBLIC[1], scale=0.25, global_probe_count=128,
                          norm_block_q=64, norm_num_warps=4, **{flag: True})
                norm = self.launches[0]
                self.assertEqual(norm.grid, (57, 40))
                for name, value in {"PROBES": 128, "BLOCK_Q": 64, "SCALE": 0.25,
                                    "num_warps": 4, "num_stages": 1}.items():
                    self.assertEqual(norm.kwargs[name], value)

    def test_direct_normalizer_keeps_explicit_execution_settings(self):
        self.call(PUBLIC[0], block_q=16, num_warps=4, num_stages=2)
        self.assertEqual(self.launches[0].grid, (225, 40))
        self.assertEqual(self.launches[0].kwargs["num_warps"], 4)
        self.assertEqual(self.launches[0].kwargs["num_stages"], 2)

    def test_unsupported_or_mismatched_dtype_rejected(self):
        for public in PUBLIC:
            for q_dtype, k_dtype in (("fp64", "fp64"), ("bf16", "fp16")):
                with self.subTest(public=public, q_dtype=q_dtype, k_dtype=k_dtype):
                    with self.assertRaisesRegex(TypeError, "same float16, bfloat16, or float32"):
                        self.scope[public](MetadataTensor(1, q_dtype), MetadataTensor(21, k_dtype))
                    self.assertEqual(self.launches, [])
                    self.assertEqual(self.allocations, [])

    def test_missing_triton_fails_before_launch(self):
        self.scope["triton"] = None
        for public in PUBLIC:
            with self.subTest(public=public):
                with self.assertRaisesRegex(RuntimeError, "Triton is required"):
                    self.call(public)
                self.assertEqual(self.launches, [])

    def test_singleton_complement_default_off_and_only_changes_seed_flag(self):
        parameter = inspect.signature(self.scope[PUBLIC[1]]).parameters["singleton_complement"]
        self.assertIs(parameter.default, False)
        original_output = self.call(PUBLIC[1])
        original = [(launch.grid, dict(launch.kwargs)) for launch in self.launches]
        self.assertIs(original[1][1]["SINGLETON_COMPLEMENT"], False)
        complement_output = self.call(PUBLIC[1], singleton_complement=True)
        self.assertEqual([launch.name for launch in self.launches], ["norm", "seed"])
        self.assertEqual((self.launches[0].grid, self.launches[0].kwargs), original[0])
        expected_seed_kwargs = {**original[1][1], "SINGLETON_COMPLEMENT": True}
        self.assertEqual((self.launches[1].grid, self.launches[1].kwargs),
                         (original[1][0], expected_seed_kwargs))
        self.assertEqual(complement_output.shape, original_output.shape)
        self.assertEqual(complement_output.dtype, original_output.dtype)

    def test_singleton_complement_keeps_precision_dtype_gates(self):
        for dtype in DTYPES:
            for flag in FLAGS:
                with self.subTest(dtype=dtype, flag=flag):
                    self.call(PUBLIC[1], dtype, singleton_complement=True, **{flag: True})
                    self.assert_flags(native=flag == FLAGS[0] and dtype != "fp32",
                                      tf32=flag == FLAGS[1] and dtype != "fp32")
                    self.assertIs(self.launches[1].kwargs["SINGLETON_COMPLEMENT"], True)

    def test_singleton_complement_rejects_zero_support_geometry(self):
        for probes in (0, 1, 81):
            with self.subTest(probes=probes):
                with self.assertRaisesRegex(ValueError, "2 <= probe_count <= columns"):
                    self.call(PUBLIC[1], singleton_complement=True, probe_count=probes)
                self.assertEqual(self.launches, [])
                self.assertEqual(self.allocations, [])


def _singleton_mask_expression():
    """Scalarize the actual Triton mask AST; no tensor package is imported."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    kernel = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_entropy_mass_seed_kernel")
    assignment = next(
        node for node in ast.walk(kernel) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "local_probe_valid"
                for target in node.targets)
    )

    class Scalarize(ast.NodeTransformer):
        def visit_Subscript(self, node):
            names = {"probe_valid": "probe_valid", "probe_columns": "probe_column",
                     "columns": "query_column"}
            if isinstance(node.value, ast.Name) and node.value.id in names:
                return ast.copy_location(ast.Name(id=names[node.value.id], ctx=ast.Load()), node)
            return self.generic_visit(node)

    expression = ast.Expression(body=Scalarize().visit(assignment.value))
    return compile(ast.fix_missing_locations(expression), str(SOURCE), "eval")


class SingletonComplementMaskTests(unittest.TestCase):
    def setUp(self):
        self.expression = _singleton_mask_expression()

    def valid(self, query_column, slot):
        return eval(self.expression, {"__builtins__": {}}, {
            "query_column": query_column, "probe_column": slot * 80 // 16,
            "probe_valid": slot < 16,
        })

    def test_all_80_columns_retain_15_or_16_distinct_valid_probes(self):
        for column in range(80):
            with self.subTest(column=column):
                mask = [self.valid(column, slot) for slot in range(32)]
                self.assertEqual(sum(mask), 15 if column % 5 == 0 else 16)
                self.assertFalse(any(mask[16:]))
                self.assertFalse(any(keep and slot * 5 == column
                                     for slot, keep in enumerate(mask)))

    def test_cpu_reference_centering_keeps_excluded_residual_zero(self):
        # Reference arithmetic validates the intended mask/normalization
        # geometry. It does not execute Triton or claim GPU numerical parity.
        for column in range(80):
            mask = [self.valid(column, slot) for slot in range(16)]
            logits = [1000.0 if slot * 5 == column else math.sin(slot) * 3.0
                      for slot in range(16)]
            maximum = max(value for value, keep in zip(logits, mask) if keep)
            exponential = [math.exp(value - maximum) if keep else 0.0
                           for value, keep in zip(logits, mask)]
            denominator = sum(exponential)
            profile = [math.sqrt(value / denominator) for value in exponential]
            mean = sum(profile) / sum(mask)
            residual = [value - mean if keep else 0.0
                        for value, keep in zip(profile, mask)]
            with self.subTest(column=column):
                self.assertTrue(math.isfinite(maximum + math.log(denominator)))
                self.assertAlmostEqual(sum(residual), 0.0, places=12)
                self.assertTrue(all(value == 0.0 for value, keep in zip(residual, mask) if not keep))


if __name__ == "__main__":
    unittest.main()
