"""CPU-only checks of the actual joint-frame wrapper AST and index mapping.

No Torch/Triton imports or GPU calls. These are dispatch/index tests, not
compiled-kernel or numerical-parity validation.
"""

import ast
import inspect
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "attention" / "joint_frame_mass_seed_triton.py"
NORMALIZER = "joint_frame_global_probe_log_norm_triton"
SELECTOR = "joint_frame_mass_seed_rows_triton"


class MetadataTensor:
    def __init__(self, frames, dtype="bf16", batch=1):
        self.shape = (batch, frames, 45, 80, 40, 128)
        self.ndim, self.dtype = 6, dtype
        self.device, self.is_cuda = "cuda:metadata-only", True

    def contiguous(self):
        return self


class RecordingKernel:
    def __init__(self, name, launches):
        self.name, self.launches = name, launches

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append(SimpleNamespace(name=self.name, grid=grid, args=args, kwargs=kwargs))
        return launch


class JointFrameDispatchTests(unittest.TestCase):
    def setUp(self):
        self.launches, self.allocations = [], []

        def empty(shape, *, device, dtype):
            tensor = SimpleNamespace(shape=tuple(shape), device=device, dtype=dtype)
            self.allocations.append(tensor)
            return tensor

        package_name = "_joint_frame_dispatch_test"
        package = ModuleType(package_name)
        package.__path__ = []
        candidates = ModuleType(package_name + ".adaptive_seed_pytorch")
        self.candidate_table = mock.Mock(return_value=(object(), object()))
        candidates.multiscale_candidate_table = self.candidate_table
        normalizer_module = ModuleType(package_name + ".entropy_mass_seed_triton")
        normalizer_module._global_probe_log_norm_kernel = RecordingKernel("norm", self.launches)
        modules = mock.patch.dict(sys.modules, {
            package_name: package, candidates.__name__: candidates,
            normalizer_module.__name__: normalizer_module,
        })
        modules.start()
        self.addCleanup(modules.stop)
        self.scope = {
            "__package__": package_name, "math": math,
            "torch": SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32",
                                     uint8="u8", int32="i32", empty=empty),
            "triton": SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y,
                                      next_power_of_2=lambda x: 1 << (x - 1).bit_length()),
            "_joint_frame_mass_seed_kernel": RecordingKernel("seed", self.launches),
        }
        self.tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        names = {"_check_joint_inputs", NORMALIZER, SELECTOR}
        nodes = [node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in nodes}, names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), self.scope)

    def call(self, function=SELECTOR, dtype="bf16", batch=1, **kwargs):
        self.launches.clear()
        self.allocations.clear()
        return self.scope[function](MetadataTensor(2, dtype, batch), MetadataTensor(21, dtype, batch), **kwargs)

    def test_one_normalizer_launch_covers_both_frames_and_batches(self):
        output = self.call(NORMALIZER, batch=2)
        self.assertEqual(output.shape, (2, 2, 45, 80, 40))
        self.assertEqual(output.dtype, "fp32")
        self.assertEqual(len(self.launches), 1)
        launch = self.launches[0]
        self.assertEqual(launch.grid, (225, 80))
        self.assertEqual(launch.kwargs["QUERY_TOKENS"], 7200)
        self.assertEqual(launch.kwargs["KEY_TOKENS"], 75600)
        self.assertEqual(launch.kwargs["PROBES"], 256)

    def test_default_c16_shared_output_and_normalizer_handoff(self):
        output = self.call()
        self.assertEqual(output.shape, (1, 1, 21, 45, 80, 40, 4))
        self.assertEqual(output.dtype, "u8")
        self.assertEqual([launch.name for launch in self.launches], ["norm", "seed"])
        norm, seed = self.launches
        self.assertIs(seed.args[2], norm.args[2])
        self.assertEqual(seed.grid, (900, 40))
        self.assertEqual(seed.kwargs["BLOCK_ROWS"], 16)
        self.assertEqual(seed.kwargs["BLOCK_J"], 4)
        self.assertEqual(seed.kwargs["num_warps"], 4)
        self.candidate_table.assert_called_once_with(45, 16, "cuda:metadata-only", geometry_count=16)

    def test_c32_diagnostic_and_execution_settings(self):
        output = self.call(candidate_count=32, candidate_geometry_count=45, block_j=2,
                           gs_num_warps=8, gs_num_stages=2, norm_block_q=64,
                           norm_num_warps=4, norm_num_stages=2, output_dtype="i32")
        self.assertEqual(output.dtype, "i32")
        norm, seed = self.launches
        self.assertEqual(norm.grid, (113, 40))
        self.assertEqual((norm.kwargs["num_warps"], norm.kwargs["num_stages"]), (4, 2))
        self.assertEqual(seed.grid, (1800, 40))
        self.assertEqual((seed.kwargs["BLOCK_ROWS"], seed.kwargs["num_warps"], seed.kwargs["num_stages"]), (32, 8, 2))
        self.candidate_table.assert_called_once_with(45, 32, "cuda:metadata-only", geometry_count=45)

    def test_native_flag_is_off_by_default_and_dtype_gated(self):
        for function in (NORMALIZER, SELECTOR):
            self.assertIs(inspect.signature(self.scope[function]).parameters["native_low_precision_dot"].default, False)
            for dtype in ("bf16", "fp16", "fp32"):
                self.call(function, dtype=dtype)
                self.assertIs(self.launches[0].kwargs["NATIVE_LOW_PRECISION_DOT"], False)
                old_seed = dict(self.launches[-1].kwargs) if function == SELECTOR else None
                self.call(function, dtype=dtype, native_low_precision_dot=True)
                self.assertIs(self.launches[0].kwargs["NATIVE_LOW_PRECISION_DOT"], dtype != "fp32")
                self.assertIs(self.launches[0].kwargs["TF32_LOW_PRECISION_DOT"], False)
                if old_seed is not None:
                    self.assertEqual(self.launches[-1].kwargs, old_seed)

    def test_invalid_geometry_probe_and_scale_fail_before_launch(self):
        cases = [dict(candidate_count=32), dict(candidate_geometry_count=45),
                 dict(probe_count=15), dict(global_probe_count=128),
                 dict(block_j=3), dict(gs_num_warps=2), dict(norm_num_stages=0),
                 dict(scale=float("nan"))]
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.call(**options)
                self.assertEqual(self.launches, [])
                self.assertEqual(self.allocations, [])

    def test_requires_exactly_two_router_frames(self):
        for frames in (1, 3):
            for function in (NORMALIZER, SELECTOR):
                with self.assertRaisesRegex(ValueError, "exactly two"):
                    self.scope[function](MetadataTensor(frames), MetadataTensor(21))
        self.assertEqual(self.launches, [])

    def test_actual_packed_index_expressions_keep_frame_and_candidate_axes(self):
        kernel = next(node for node in ast.walk(self.tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "_joint_frame_mass_seed_kernel")
        names = ("candidate_slot", "frame_slot", "columns", "query_token")
        expressions = {}
        for node in kernel.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in names:
                    expressions[name] = compile(ast.Expression(node.value), str(SOURCE), "eval")
        for candidates in (16, 32):
            for block_j in (1, 2, 4, 8):
                seen = set()
                for flat in range(block_j * 2 * candidates):
                    values = dict(flat=flat, BLOCK_ROWS=candidates, BLOCK_J=block_j,
                                  group=2, batch=1, ROWS=45, COLS=80)
                    for name in names:
                        if name == "query_token":
                            values["candidate_rows"] = (3 * values["candidate_slot"] + 7) % 45
                        values[name] = eval(expressions[name], {"__builtins__": {}}, values)
                    decoded = values["query_token"] - 2 * 45 * 80
                    frame, spatial = divmod(decoded, 45 * 80)
                    row, column = divmod(spatial, 80)
                    self.assertEqual(frame, values["frame_slot"])
                    self.assertEqual((row, column), (values["candidate_rows"], values["columns"]))
                    seen.add((column, frame, values["candidate_slot"]))
                self.assertEqual(len(seen), block_j * 2 * candidates)


if __name__ == "__main__":
    unittest.main()
