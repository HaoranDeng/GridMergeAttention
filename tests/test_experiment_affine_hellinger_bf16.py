"""CPU-only contracts for the isolated monitored affine BF16 output screen."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from analysis import experiment_affine_hellinger_bf16 as experiment
from tests.test_quality_first_mode_contract import production_suffixes


def selector_kwargs(method):
    common = dict(candidate_count=32, candidate_geometry_count=45, block_j=4,
                  gs_num_warps=4, output_dtype="torch.uint8")
    if method in ("hellinger_rf0", "affine_rf0"):
        common.update(alpha=.2, top_probes=0, uniform_probes=16, probe_anchor_count=2,
                      group_tile=16, selection_mode="gs", force_local_minus_one=False)
    else:
        common.update(probe_count=16, global_probe_count=256)
    return common


def merge_kwargs():
    return dict(head_start=0, head_count=4, top_r=1, selector="query_column",
        compressed_fp8_kv=False, scaled_fp8_kv=False, mean_centered_routing=False,
        gaussian_partition_routing=False, reuse_partition_exp=False,
        direct_bulk_normalization=False, fp8_score_dot=False, fp8_merge_k_dot=False,
        fp8_merge_v_dot=False, query_frame_group=21, split_merge=False,
        materialize_singleton=False, query_column_major=True,
        query_column_secondary_frame_stride=1, query_column_primary_row_mod4_omit2=False,
        packed_singleton_kv=False, seed_rows_native_layout=True, merge_frame_group=1,
        merge_col_group=1, merge_loop_stages=2, merge_disable_licm=False,
        split_kv_accum=True, output=object())


class AffineExperimentTests(unittest.TestCase):
    def test_all_modes_actual_parser_is_bf16_same_merge_substrate(self):
        self.assertEqual(set(experiment.MODES), {"hellinger_rf0", "mass_rf0", "jeffreys_rf0", "affine_rf0"})
        self.assertEqual(experiment.MODES["affine_rf0"], experiment.MODES["hellinger_rf0"])
        for mode in experiment.MODES.values():
            parsed = production_suffixes(mode)
            self.assertNotIn("fp8", mode)
            for flag in ("fp8_score_dot", "fp8_merge_k_dot", "fp8_merge_v_dot", "compressed_fp8_kv",
                         "scaled_fp8_kv", "direct_bulk_normalization", "mean_centered_routing"):
                self.assertIs(parsed[flag], False)
            self.assertEqual(parsed["singleton_residual"], 1)
            self.assertTrue(parsed["split_kv_accum"])
            self.assertTrue(parsed["direct_native_seed_layout"])

    def test_affine_adapter_accepts_only_exact_hf_kwargs_and_drops_alpha(self):
        implementation = Mock(return_value="seeds")
        package = ModuleType("attention")
        package.__path__ = []
        module = ModuleType("attention.affine_hellinger_seed_triton")
        module.affine_hellinger_seed_rows_triton = implementation
        with patch.dict(sys.modules, {"attention": package, module.__name__: module}):
            q, k = object(), object()
            self.assertEqual(experiment.affine_selector_adapter(q, k, 128 ** -.5,
                             **selector_kwargs("affine_rf0")), "seeds")
            implementation.assert_called_once_with(q, k, 128 ** -.5, candidate_count=32,
                candidate_geometry_count=45, block_j=4, probe_count=16,
                output_dtype="torch.uint8", gs_num_warps=4)
            for update in ({"alpha": .3}, {"top_probes": 8}, {"force_local_minus_one": True},
                           {"candidate_count": True}, {"unexpected": 0}):
                with self.assertRaises(ValueError):
                    experiment.affine_selector_adapter(q, k, 128 ** -.5,
                        **(selector_kwargs("affine_rf0") | update))
            self.assertEqual(implementation.call_count, 1)

    def test_affine_context_substitutes_hf_symbol_temporarily_only(self):
        hf, mass = object(), object()
        token = SimpleNamespace(hellinger_span_seed_rows_triton=hf, entropy_mass_seed_rows_triton=mass)
        with experiment.candidate_context(token, "affine_rf0"):
            self.assertIs(token.hellinger_span_seed_rows_triton, experiment.affine_selector_adapter)
            self.assertIs(token.entropy_mass_seed_rows_triton, mass)
        self.assertIs(token.hellinger_span_seed_rows_triton, hf)

    def test_actual_selector_and_merge_options_fail_closed(self):
        for method in experiment.MODES:
            self.assertEqual(experiment.validate_selector_kwargs(selector_kwargs(method), method), selector_kwargs(method))
        expected = merge_kwargs()
        self.assertEqual(experiment.validate_merge_kwargs(expected), {k: v for k, v in expected.items() if k != "output"})
        for update in ({"fp8_score_dot": True}, {"fp8_merge_k_dot": True}, {"fp8_merge_v_dot": True},
                       {"compressed_fp8_kv": True}, {"direct_bulk_normalization": True},
                       {"head_count": 3}, {"query_frame_group": 1}, {"new_route": False}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                experiment.validate_merge_kwargs(expected | update)

    def test_observed_complete_call_sees_real_seed_and_one_bf16_merge(self):
        class Tensor:
            dtype = "bf16"

            def __init__(self, name, shape):
                self.name, self.shape = name, tuple(shape)

            def reshape(self, *shape):
                return Tensor(self.name, shape)

            def __getitem__(self, item):
                shape = list(self.shape)
                shape[1] = 1
                return Tensor(self.name, shape)

            def contiguous(self):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.array([0, 1, 2, 3], dtype=np.uint8)

        q, k, v = [Tensor(name, (1, 75600, 4, 128)) for name in ("q", "k", "v")]
        seed = Tensor("seed", (1, 1, 21, 45, 80, 4, 4))
        output = Tensor("output", (1, 75600, 4, 128))
        fake_torch = ModuleType("torch")
        fake_torch.bfloat16 = "bf16"
        fake_torch.equal = lambda first, second: (first.name, first.shape) == (second.name, second.shape)
        token = SimpleNamespace(hellinger_span_seed_rows_triton=Mock(return_value=seed))
        real_merge = Mock(return_value=None)
        setattr(token, experiment.MERGE_SYMBOL, real_merge)

        def hard(q, k, v, **kwargs):
            grids = [x.reshape(1, 21, 45, 80, 4, 128) for x in (q, k, v)]
            seeds = token.hellinger_span_seed_rows_triton(grids[0][:, :1], grids[1], 128 ** -.5,
                                                        **selector_kwargs("hellinger_rf0"))
            getattr(token, experiment.MERGE_SYMBOL)(*grids, seeds, 128 ** -.5, **merge_kwargs())
            return output

        token._hard_c4_attention = hard
        with patch.dict(sys.modules, {"torch": fake_torch}):
            actual, table, dispatch = experiment.observed_attention(token, q, k, v, "hellinger_rf0")
        self.assertIs(actual, output)
        self.assertEqual(table.tolist(), [0, 1, 2, 3])
        self.assertIs(dispatch["merge_route"]["fp8_score_dot"], False)
        self.assertEqual(real_merge.call_count, 1)
        self.assertIs(getattr(token, experiment.MERGE_SYMBOL), real_merge)

    def test_replication_checks_full_frame_head_sse_not_only_pooled_mse(self):
        measured = dict(numel=38707200, squared_error_sum=4., mse=1e-4,
            frame_head_squared_error_sum=[[1., 1.], [1., 1.]],
            frame_mse=[1e-4, 1e-4], head_mse=[1e-4, 1e-4],
            rf0_mse=1e-4, other20_frames_mse=1e-4)
        kwargs = selector_kwargs("hellinger_rf0")
        old = dict(mode=experiment.MODES["hellinger_rf0"], metrics=measured, received=kwargs)
        table = np.arange(8).reshape(2, 4)
        self.assertTrue(experiment.baseline_replication(measured, kwargs, table, old, table)["valid"])
        modified = copy.deepcopy(measured)
        modified["frame_head_squared_error_sum"][0] = [1.1, .9]
        self.assertFalse(experiment.baseline_replication(modified, kwargs, table, old, table)["valid"])
        self.assertFalse(experiment.baseline_replication(measured, kwargs, table, old, table + 1)["valid"])
        self.assertFalse(experiment.baseline_replication(measured, kwargs | {"alpha": .3}, table, old, table)["valid"])

    def test_internal_worker_cannot_skip_parent_identity_check(self):
        args = SimpleNamespace(output=Path("not-used"))
        startup = dict(parent_pid=123, physical_gpu=2, compute_processes=[])
        environment = dict(GCA_AFFINE_SCREEN_PARENT_PID="123", CUDA_VISIBLE_DEVICES="2", WAN_SDPA_BACKEND="flash")
        with patch.object(Path, "read_text", return_value=json.dumps(startup)):
            experiment.validate_worker_parent(args, environment, 123)
            for changed in ({}, environment | {"GCA_AFFINE_SCREEN_PARENT_PID": "124"},
                            environment | {"CUDA_VISIBLE_DEVICES": "1"},
                            environment | {"TOKEN_MERGING_EXIT_AFTER_QKV_SAVE": "1"}):
                with self.assertRaises(ValueError):
                    experiment.validate_worker_parent(args, changed, 123)

    def test_parent_busy_gpu_never_spawns_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(gpu=0, output=Path(directory) / "not-created")
            state = dict(uuid="fake-id", memory_mib=2000, utilization=2)
            with patch.object(experiment, "gpu_state", return_value=state), \
                 patch.object(experiment, "gpu_processes", return_value=[]), \
                 patch.object(experiment.subprocess, "Popen") as launch:
                with self.assertRaises(ValueError):
                    experiment.parent(args)
                launch.assert_not_called()
                self.assertFalse(args.output.exists())

    def test_parent_rechecks_exclusivity_immediately_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(gpu=0, output=Path(directory) / "new", capture_root=Path("capture"),
                                   oracle_root=Path("oracle"), native_root=Path("native"), baseline_root=Path("baseline"))
            idle = dict(uuid="fake-id", memory_mib=0, utilization=0)
            busy = idle | {"memory_mib": 4000}
            with patch.object(experiment, "gpu_state", side_effect=[idle, busy]), \
                 patch.object(experiment, "gpu_processes", return_value=[]), \
                 patch.object(experiment.subprocess, "Popen") as launch:
                with self.assertRaises(ValueError):
                    experiment.parent(args)
                launch.assert_not_called()

    def test_terminal_requires_all_prompts_controls_methods_and_parity(self):
        row = {"complete": True, "native_parity": {"valid": True},
               "metrics": {name: {} for name in experiment.MODES},
               "selectors": {name: {} for name in experiment.MODES},
               "baseline_replication": {name: {"valid": True} for name in experiment.CONTROLS}}
        result = {"valid": True, "complete": True, "l33": [copy.deepcopy(row) for _ in range(3)],
                  "source_freeze_verified_at_end": True}
        self.assertTrue(experiment.results_complete(result))
        for kind in ("parity", "control", "method", "source"):
            broken = copy.deepcopy(result)
            if kind == "parity":
                broken["l33"][1]["native_parity"]["valid"] = False
            elif kind == "control":
                broken["l33"][1]["baseline_replication"]["mass_rf0"]["valid"] = False
            elif kind == "method":
                del broken["l33"][1]["metrics"]["affine_rf0"]
            else:
                broken["source_freeze_verified_at_end"] = False
            self.assertFalse(experiment.results_complete(broken))


if __name__ == "__main__":
    unittest.main()
