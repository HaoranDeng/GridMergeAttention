"""CPU regression tests for the narrowly promoted RF0 partition launch.

Small AST-policy tests run without Torch. Full actual-dispatch tests reuse
the existing real Torch CPU fixture and capture kwargs before any kernel.
This validates launch routing only, not numerical GPU parity or performance.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

HELPERS = Path(__file__).with_name("test_router_frame_policy.py")
spec = importlib.util.spec_from_file_location("router_policy_helpers", HELPERS)
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)
TARGET_MODE = router._mode("partition_c16")


def actual_launch_policy(mode, *, frames=21, height=45, width=80, rule="partition_mass"):
    """Evaluate the actual production launch dictionary/guard, not a copy."""
    tree = ast.parse(router.TOKEN_SOURCE.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_hard_c4_attention")
    parent = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                  and any(isinstance(child, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "entropy_launch_options"
                                  for target in child.targets) for child in node.body))
    index = next(index for index, node in enumerate(parent.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "entropy_launch_options"
                         for target in node.targets))
    nodes = parent.body[index:index + 2]
    if not isinstance(nodes[1], ast.If):
        raise AssertionError("expected launch dictionary followed by its precise-scope guard")
    scope = {"requested_seed_mode": mode, "entropy_mass_rule": rule, "h": height, "w": width,
             "query_blocks": SimpleNamespace(size=lambda dim: frames if dim == 1 else None)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(router.TOKEN_SOURCE), "exec"), scope)
    return scope["entropy_launch_options"]


class PartitionRF0LaunchPolicyTests(unittest.TestCase):
    def test_exact_target_and_active_config_get_measured_options(self):
        self.assertEqual(TARGET_MODE, router._config_mode(router.ACTIVE_CONFIGS["partition_c16"]))
        self.assertEqual(actual_launch_policy(TARGET_MODE), {"gs_num_warps": 4, "reuse_owner_logp": True})

    def test_every_neighboring_legacy_mode_keeps_default_options(self):
        alternatives = [
            TARGET_MODE.replace("_rf0_", "_rf10_"), TARGET_MODE.replace("_rf0_", "_"),
            TARGET_MODE.replace("_u16_", "_"), TARGET_MODE.replace("_u16_", "_a8u8_"),
            TARGET_MODE.replace("_c16_", "_c32_").replace("_cg16_", "_cg45_"),
            TARGET_MODE.replace("_j4_", "_j2_"), TARGET_MODE.replace("_p16_", "_p32_"),
            TARGET_MODE.replace("_g256_", "_g128_"), TARGET_MODE.replace("_cg16", "_cg15"),
            TARGET_MODE + "_legacy",
        ]
        for mode in alternatives:
            with self.subTest(mode=mode):
                self.assertEqual(actual_launch_policy(mode), {"gs_num_warps": 8})

    def test_noncanonical_grid_does_not_receive_promoted_launch(self):
        for shape in (dict(frames=20), dict(height=30), dict(width=52)):
            with self.subTest(shape=shape):
                self.assertEqual(actual_launch_policy(TARGET_MODE, **shape), {"gs_num_warps": 8})

    def test_mass_and_joint_keep_four_warps_without_partition_only_kwarg(self):
        for name in ("mass_c16", "mass_c32"):
            self.assertEqual(actual_launch_policy(router._mode(name), rule="mass_span"), {"gs_num_warps": 4})
        mode = TARGET_MODE.replace("entropy_partition_mass", "entropy_joint_mass_span")
        self.assertEqual(actual_launch_policy(mode, rule="joint_mass_span"), {"gs_num_warps": 4})
        # Even an inconsistent future parser result cannot leak this option
        # into a selector whose API does not support owner-logp reuse.
        for rule in ("mass_span", "joint_mass_span"):
            self.assertEqual(actual_launch_policy(TARGET_MODE, rule=rule), {"gs_num_warps": 4})


@unittest.skipIf(router.torch is None, "Torch CPU required for actual dispatcher kwargs tests")
class PartitionRF0DispatcherTests(unittest.TestCase):
    setUp = router.RouterFrameDispatchTests.setUp
    dispatch = router.RouterFrameDispatchTests.dispatch

    def assert_partition(self, mode, *, warps, reuse=False, frame=0):
        symbol, query, _, kwargs = self.dispatch(mode)
        self.assertEqual(symbol, "partition_mass_seed_rows_triton")
        self.assertEqual(router.torch.unique(query).tolist(), [float(frame)])
        self.assertEqual(kwargs["gs_num_warps"], warps)
        if reuse:
            self.assertIs(kwargs["reuse_owner_logp"], True)
        else:
            # Omit the new keyword entirely to preserve the legacy call API.
            self.assertNotIn("reuse_owner_logp", kwargs)
        return kwargs

    def test_actual_target_config_dispatches_four_warps_with_owner_reuse(self):
        mode = router._config_mode(router.ACTIVE_CONFIGS["partition_c16"])
        kwargs = self.assert_partition(mode, warps=4, reuse=True)
        for key, value in {"candidate_count": 16, "candidate_geometry_count": 16,
                           "probe_count": 16, "global_probe_count": 256, "block_j": 4}.items():
            self.assertEqual(kwargs[key], value)

    def test_rf10_and_implicit_legacy_keep_eight_warps(self):
        for suffix in ("_rf10", ""):
            with self.subTest(suffix=suffix):
                self.assert_partition(router._mode("partition_c16", suffix), warps=8, frame=10)

    def test_c32_j2_and_nonpromoted_tail_remain_unchanged(self):
        for mode in (TARGET_MODE.replace("_c16_", "_c32_").replace("_cg16_", "_cg45_"),
                     TARGET_MODE.replace("_j4_", "_j2_"), TARGET_MODE.removesuffix("_ml2"),
                     TARGET_MODE.replace("_u16_", "_")):
            with self.subTest(mode=mode):
                self.assert_partition(mode, warps=8)

    def test_mass_and_hellinger_receive_no_partition_only_keyword(self):
        for name in ("rf0", "mass_c32", "mass_c16"):
            for suffix in ("_rf0", "_rf10", ""):
                with self.subTest(name=name, suffix=suffix):
                    symbol, _, _, kwargs = self.dispatch(router._mode(name, suffix))
                    self.assertNotEqual(symbol, "partition_mass_seed_rows_triton")
                    self.assertEqual(kwargs["gs_num_warps"], 4)
                    self.assertNotIn("reuse_owner_logp", kwargs)


if __name__ == "__main__":
    unittest.main()
