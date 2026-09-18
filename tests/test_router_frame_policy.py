"""CPU regressions for explicit RF0 without changing historical T1 semantics.

The numerical tests execute the actual full dispatcher and parsers extracted
by AST, with real Torch CPU frame-ID tensors. A stub captures the selected
query immediately before any GPU kernel. No attention/Triton module import,
CUDA initialization, or kernel compilation is required. Only these numerical
tests skip when Torch is unavailable; configuration policy checks still run.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
TOKEN_SOURCE = ROOT / "attention" / "token_merging.py"
ADAPTIVE_SOURCE = ROOT / "attention" / "adaptive_seed_pytorch.py"
STREAM_SOURCE = ROOT / "analysis" / "screen_native_seed_trajectory.py"
TAIL = "_u16_cg{geometry}"
CORES = {
    "rf0": ("hellinger_span_alpha0p2_c32_p16_a2_j4_t1", 32, 45, 4),
    "mass_c32": ("entropy_mass_span_c32_p16_g256_j4_t1", 32, 45, 4),
    "mass_c16": ("entropy_mass_span_c16_p16_g256_j4_t1", 16, 16, 4),
    "partition_c16": ("entropy_partition_mass_c16_p16_g256_j4_t1", 16, 16, 4),
}
ACTIVE_CONFIGS = {
    "rf0": "token_merging_wan_720p_fisher_c4_exact_capacity_fewstep.yaml",
    "mass_c32": "token_merging_wan_720p_entropy_mass_span_rf0_fewstep.yaml",
    "mass_c16": "token_merging_wan_720p_entropy_mass_span_c16_rf0_fewstep.yaml",
    "partition_c16": "token_merging_wan_720p_entropy_partition_mass_c16_rf0_fewstep.yaml",
}


def _mode(name, router_suffix="_rf0"):
    core, _, geometry, _ = CORES[name]
    return core + router_suffix + TAIL.format(geometry=geometry)


def _config_mode(filename):
    source = (ROOT / "configs" / filename).read_text(encoding="utf-8")
    matches = re.findall(r"^\s*seed_mode:\s*([^\r\n#]+)", source, re.MULTILINE)
    if len(matches) != 1:
        raise AssertionError(f"expected one seed_mode in {filename}, found {len(matches)}")
    return matches[0].strip().strip("\"'")


def _stream_modes():
    tree = ast.parse(STREAM_SOURCE.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "MODES" for target in node.targets)]
    if len(nodes) != 1:
        raise AssertionError("the active stream must have exactly one literal MODES definition")
    return ast.literal_eval(nodes[0].value)


def _load_functions(path, names, scope):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"missing actual source functions in {path}: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)


class ActiveRF0PolicyTests(unittest.TestCase):
    def test_active_configs_explicitly_freeze_rf0_uniform16_and_banks(self):
        for name, filename in ACTIVE_CONFIGS.items():
            with self.subTest(name=name, filename=filename):
                self.assertEqual(_config_mode(filename), _mode(name))

    def test_active_stream_matches_configs_without_implicit_or_joint_router(self):
        modes = _stream_modes()
        self.assertEqual(set(modes), set(CORES))
        for name in CORES:
            with self.subTest(name=name):
                self.assertEqual(modes[name], _mode(name))
                self.assertEqual(modes[name], _config_mode(ACTIVE_CONFIGS[name]))
                self.assertNotIn("_lyrc", modes[name])
                self.assertNotIn("_lpf", modes[name])
                self.assertNotIn("joint", modes[name])


class SelectorCaptured(Exception):
    pass


@unittest.skipIf(torch is None, "Torch CPU is required for actual tensor dispatcher tests")
class RouterFrameDispatchTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.scope = {"torch": torch, "re": re, "HARD_C4_BACKEND": "hard_c4"}
        _load_functions(ADAPTIVE_SOURCE, {"parse_hellinger_span_seed_mode"}, self.scope)
        _load_functions(TOKEN_SOURCE, {
            "_strip_hellinger_probe_policy", "_use_uniform_hellinger_probes",
            "_hard_c4_video_to_blocks", "_hard_c4_attention",
        }, self.scope)
        for symbol in ("hellinger_span_seed_rows_triton", "entropy_mass_seed_rows_triton",
                       "partition_mass_seed_rows_triton", "joint_frame_mass_seed_rows_triton"):
            def capture(query, key, *args, _symbol=symbol, **kwargs):
                self.calls.append((_symbol, query, key, kwargs))
                raise SelectorCaptured
            self.scope[symbol] = capture
        # Four real CPU heads suffice to verify routing and per-head dispatch.
        frames = torch.arange(21, dtype=torch.float32, device="cpu").view(1, 21, 1, 1, 1, 1)
        self.query = frames.expand(1, 21, 45, 80, 4, 1).reshape(1, 75600, 4, 1)
        self.key = self.query + 100
        self.value = self.query + 200

    def dispatch(self, mode):
        self.calls.clear()
        with self.assertRaises(SelectorCaptured):
            self.scope["_hard_c4_attention"](
                self.query, self.key, self.value,
                h=45, w=80, h_reduce=1, w_reduce=1, sm_scale=1.0,
                query_outer_chunk=21, head_chunk=8, hard_route_backend="hard_c4",
                exact_intra_frame_attention=False, exact_neighbor_window_columns=0,
                exact_query_rows=0, first_frame_full_attention=False,
                full_attention_anchor_time_count=0, full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0, seed_mode=mode)
        self.assertEqual(len(self.calls), 1)
        return self.calls[0]

    def assert_router(self, name, mode, expected_frame):
        symbol, query, key, kwargs = self.dispatch(mode)
        expected_symbol = ("hellinger_span_seed_rows_triton" if name == "rf0" else
                           "partition_mass_seed_rows_triton" if name == "partition_c16" else
                           "entropy_mass_seed_rows_triton")
        self.assertEqual(symbol, expected_symbol)
        self.assertEqual(tuple(query.shape), (1, 1, 45, 80, 4, 1))
        self.assertEqual(torch.unique(query).tolist(), [float(expected_frame)])
        self.assertTrue(torch.equal(key, self.key.reshape(1, 21, 45, 80, 4, 1)))
        _, candidates, geometry, block_j = CORES[name]
        self.assertEqual(kwargs["candidate_count"], candidates)
        self.assertEqual(kwargs["candidate_geometry_count"], geometry)
        self.assertEqual(kwargs["block_j"], block_j)
        if name == "rf0":
            self.assertEqual((kwargs["top_probes"], kwargs["uniform_probes"]), (0, 16))
        else:
            self.assertEqual((kwargs["probe_count"], kwargs["global_probe_count"]), (16, 256))

    def test_explicit_rf0_selects_original_first_frame_for_all_active_methods(self):
        for name in CORES:
            with self.subTest(name=name):
                self.assert_router(name, _mode(name, "_rf0"), 0)

    def test_explicit_rf10_remains_an_exact_historical_override(self):
        for name in CORES:
            with self.subTest(name=name):
                self.assert_router(name, _mode(name, "_rf10"), 10)

    def test_legacy_t1_without_rf_marker_still_means_frame10(self):
        for name in CORES:
            with self.subTest(name=name):
                self.assert_router(name, _mode(name, ""), 10)

    def test_actual_active_config_and_stream_strings_dispatch_to_rf0(self):
        modes = _stream_modes()
        for name, filename in ACTIVE_CONFIGS.items():
            with self.subTest(name=name, source="config"):
                self.assert_router(name, _config_mode(filename), 0)
            with self.subTest(name=name, source="stream"):
                self.assert_router(name, modes[name], 0)

    def test_multiframe_legacy_coverage_is_unchanged(self):
        for count, expected in ((3, [3, 10, 17]), (4, [0, 7, 13, 20])):
            with self.subTest(count=count):
                mode = _mode("rf0", "").replace("_t1_", f"_t{count}_")
                _, query, _, _ = self.dispatch(mode)
                self.assertEqual(query[0, :, 0, 0, 0, 0].tolist(), expected)

    def test_count_zero_keeps_all_frames_with_existing_head_sharing(self):
        # g2 enters the batched routing branch when temporal_frame_count=0.
        mode = _mode("rf0", "").replace("_j4_t1_", "_j4_g2_t0_")
        _, query, _, _ = self.dispatch(mode)
        self.assertEqual(query[0, :, 0, 0, 0, 0].tolist(), list(range(21)))
        self.assertEqual(query.shape[4], 2)

    def test_explicit_router_rejects_multiframe_modes_before_selector(self):
        for count in (0, 3, 4):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "requires t1"):
                    self.dispatch(_mode("rf0").replace("_t1_", f"_t{count}_"))
                self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
