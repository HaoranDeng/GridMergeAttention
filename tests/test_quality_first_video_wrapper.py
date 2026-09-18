"""CPU-only generation-wrapper contracts using metadata stand-ins, no Torch."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis import run_quality_first_videos as wrapper
from analysis.quality_first_mode_contract import VIDEO_MODES as BF16_MODES, quality_first_token_merging_args


class Tensor:
    def __init__(self, shape=wrapper.SHAPE, stride=(387072000, 5120, 128, 1),
                 pointer=101, offset=0, dtype="torch.bfloat16", device="cuda:0"):
        self.shape, self._stride = tuple(shape), tuple(stride)
        self._pointer, self._offset = pointer, offset
        self.dtype, self.device = dtype, device

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self._offset

    def data_ptr(self):
        return self._pointer

    @classmethod
    def from_metadata(cls, metadata):
        return cls(metadata["shape"], metadata["stride"], metadata["data_ptr"],
                   metadata["storage_offset"], metadata["dtype"], metadata["device"])


def layer(method, index):
    attributes = {"token_merging_" + key: value for key, value in quality_first_token_merging_args(method).items()
                  if key != "enable"}
    attributes.update({"token_merging_" + key: False for key in wrapper.HYBRID_FIELDS})
    return SimpleNamespace(**attributes, layer_idx=index, enable_token_merging=True,
                           token_merging_dense_heads=(), token_merging_full_attention_anchor_scale=1.0,
                           token_merging_q_init=None, seed_query_proj=None,
                           gma_output_calibration_weights=None, gma_query_calibration_parameters=None)


def core_inputs(method):
    return (Tensor(pointer=101), Tensor(pointer=201), Tensor(pointer=301), 1, 1, 1, 45, 80), {
        "seed_mode": BF16_MODES[method], "seed_query": None, "dense_heads": (),
    }


def selector_inputs(audit):
    args = (Tensor.from_metadata(wrapper.grid_metadata(audit.qkv[0], rf0=True)),
            Tensor.from_metadata(wrapper.grid_metadata(audit.qkv[1])), 128 ** -0.5)
    kwargs = {"candidate_count": 45 if "_c45_" in BF16_MODES[audit.method] else 32,
              "candidate_geometry_count": 45,
              "block_j": 1 if audit.method == "fisher_expected_rf0" else 4,
              "gs_num_warps": 4, "output_dtype": "torch.uint8"}
    if audit.declaration["selector_symbol"] == "hellinger_span_seed_rows_triton":
        kwargs.update(alpha=0.2, top_probes=0, uniform_probes=16, probe_anchor_count=2,
                      group_tile=16, selection_mode="gs", force_local_minus_one=False)
    else:
        kwargs.update(probe_count=16, global_probe_count=256)
    return args, kwargs


def merge_inputs(audit, seed):
    args = tuple(Tensor.from_metadata(wrapper.grid_metadata(meta)) for meta in audit.qkv) + (seed, 128 ** -0.5)
    # Execute the actual production merge call AST with a metadata capture.
    tree = ast.parse((wrapper.ROOT / "attention/token_merging.py").read_text(encoding="utf-8"))
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == wrapper.MERGE_SYMBOL
                and any(keyword.arg == "seed_rows_native_layout" for keyword in node.keywords))
    values = {
        "output": object(), "h_start": audit.merge_calls % 5 * 8, "h_end": audit.merge_calls % 5 * 8 + 8,
        "singleton_residual": 1, "singleton_selector": "query_column", "query_frame_group": 21,
        "query_column_major": True, "query_column_secondary_frame_stride": 1,
        "direct_native_seed_layout": True, "merge_frame_group": 1, "merge_col_group": 1,
        "merge_loop_stages": 1 if audit.method in wrapper.C10_METHODS else 2, "merge_disable_licm": False, "split_kv_accum": True,
    }
    for keyword in call.keywords:
        for node in ast.walk(keyword.value):
            if isinstance(node, ast.Name):
                values.setdefault(node.id, False)
    kwargs = {keyword.arg: eval(compile(ast.Expression(keyword.value), "<production merge>", "eval"), values)
              for keyword in call.keywords}
    return args, kwargs


def one_call(audit):
    audit.enter_layer(layer(audit.method, audit.calls % 40), (21, 45, 80))
    audit.enter_core(*core_inputs(audit.method))
    audit.enter_selector(*selector_inputs(audit))
    count = audit.declaration["bulk_representatives"]
    shape = wrapper.SEED_SHAPE[:-1] + (count,)
    strides = (302400*count, 302400*count, 14400*count, 320*count, 4*count, count, 1)
    seed = Tensor(shape, strides, pointer=401, dtype="torch.uint8")
    audit.leave_selector(seed)
    args, kwargs = merge_inputs(audit, seed)
    chunks = wrapper.axis_chunks(audit.method)
    for index in range(chunks):
        kwargs["head_start"] = index * (40 // chunks)
        kwargs["head_count"] = 40 // chunks
        audit.enter_merge(args, kwargs)
        if audit.method in wrapper.SYNTHETIC_METHODS:
            audit.synthetic_seed_kernel_calls += 1
        if audit.method in wrapper.STABLE_SQ1_METHODS:
            audit.stable_merge_kernel_calls += 1
            audit.stable_merge_wrapper_calls += 1
    if audit.method in wrapper.MEAN_METHODS:
        audit.mean_seed_constructions += 1
    audit.leave_layer()


class QualityFirstVideoWrapperTests(unittest.TestCase):
    def test_tail_range_keeps_global_prompt_indices(self):
        provenance = {"planned": {"num_videos": 20, "prompt_start": 15}, "config": "config.json"}
        self.assertEqual(wrapper.planned_prompt_indices(provenance), [15, 16, 17, 18, 19])
        argv = wrapper.generation_argv(Path("run"), provenance, Path("checkpoint"))
        self.assertEqual(argv[argv.index("--num_videos") + 1], "20")
        self.assertEqual(argv[argv.index("--prompt_start") + 1], "15")
        tree = ast.parse((wrapper.ROOT / "generate.py").read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "indexed_prompts" for t in node.targets))
        scope = {"prompts": [f"prompt-{i}" for i in range(20)], "args": SimpleNamespace(prompt_start=15)}
        selected = eval(compile(ast.Expression(assignment.value), "<prompt selection>", "eval"), scope)
        self.assertEqual(selected, [(i, f"prompt-{i}") for i in range(15, 20)])
        self.assertEqual(wrapper.planned_prompt_indices({"planned": {"num_videos": 15}}), list(range(15)))
        for start in (-1, 20, True, 1.5):
            with self.subTest(start=start), self.assertRaises(ValueError):
                wrapper.planned_prompt_indices({"planned": {"num_videos": 20, "prompt_start": start}})

    def test_stable_sq1_runner_is_retired(self):
        from analysis import launch_quality_first_videos as launcher
        from analysis.quality_first_mode_contract import STABLE_SQ1_METHODS
        self.assertIs(wrapper.STABLE_SQ1_METHODS, STABLE_SQ1_METHODS)
        self.assertIs(launcher.worker.STABLE_SQ1_METHODS, STABLE_SQ1_METHODS)
        self.assertEqual(STABLE_SQ1_METHODS, ())
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            wrapper.DispatchAudit("mass_stable_sq1_rf0")

    def test_stable_requires_actual_kernel_counts_and_retains_real_merge(self):
        for method in wrapper.STABLE_SQ1_METHODS:
            audit = wrapper.DispatchAudit(method)
            one_call(audit)
            self.assertEqual((audit.calls, audit.selector_calls, audit.merge_calls,
                              audit.stable_merge_kernel_calls, audit.stable_merge_wrapper_calls), (1, 1, 5, 5, 5))
            self.assertEqual(audit.synthetic_seed_kernel_calls, 0)
            self.assertEqual(audit.mean_seed_constructions, 0)
            audit.calls = 0
            audit.stable_merge_kernel_calls = 4
            with self.assertRaisesRegex(ValueError, "stable SQ1"):
                audit.leave_layer()

    def test_stable_complete_requires_16000_kernel_calls_and_immutable_run_artifact(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "stable_kernel.py"
            source.write_text("# explicit test generated artifact\n")
            for method in wrapper.STABLE_SQ1_METHODS:
                audit = wrapper.DispatchAudit(method, root / "dispatch_audit.json")
                audit.generated_stable_kernel = dict(path=str(source), bytes=source.stat().st_size,
                                                     sha256=wrapper.digest(source))
                audit.videos, audit.calls, audit.core_calls, audit.selector_calls = 20, 3200, 3200, 3200
                audit.layers = [80] * 40
                audit.merge_calls = audit.stable_merge_kernel_calls = audit.stable_merge_wrapper_calls = 16000
                self.assertTrue(audit.save(True)["complete"])
                for name in ("core_calls", "selector_calls", "merge_calls", "stable_merge_kernel_calls", "stable_merge_wrapper_calls"):
                    before = getattr(audit, name)
                    setattr(audit, name, before - 1)
                    with self.assertRaises(ValueError): audit.save(True)
                    setattr(audit, name, before)
                audit.generated_stable_kernel["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "source changed"): audit.save(True)

    def test_config_and_argv_are_fixed_and_do_not_enable_resume_or_learning(self):
        for method in BF16_MODES:
            config = wrapper.config_document(method)
            self.assertEqual(config["token_merging_args"], quality_first_token_merging_args(method))
            self.assertEqual(config["denoising_step_list"], [1000, 750, 500, 250])
            argv = wrapper.generation_argv(Path("stage"), {"config": "configs/frozen.json", "planned": {"num_videos": 20}}, "weights.pt")
            for name, value in (("--num_videos", "20"), ("--num_samples", "1"), ("--batch_size", "1"),
                                ("--seed", "0"), ("--dtype", "bf16")):
                self.assertEqual(argv[argv.index(name) + 1], value)
            for forbidden in ("--skip_existing", "--use_ema", "--allow_noncanonical_prompts",
                              "--seed_query_proj_checkpoint", "--gma_output_calibration_checkpoint"):
                self.assertNotIn(forbidden, argv)

    def test_fp8_15_video_count_and_precision_contract(self):
        method = wrapper.FP8_C10_METHOD
        audit = wrapper.DispatchAudit(method, num_videos=15)
        self.assertTrue(audit.declaration["scaled_fp8_kv"])
        self.assertEqual(wrapper.quality_first_token_merging_args(method),
                         wrapper.quality_first_token_merging_args("fisher_geometric_c10_rf0"))
        argv = wrapper.generation_argv(Path("stage"),
                {"config": "frozen.json", "planned": {"num_videos": 15}}, "weights.pt")
        self.assertEqual(argv[argv.index("--num_videos") + 1], "15")
        audit.videos = 15
        audit.calls = audit.core_calls = audit.selector_calls = 2400
        audit.merge_calls = 24000
        audit.layers = [60] * 40
        self.assertTrue(audit.save(True)["complete"])
        audit.merge_calls -= 1
        with self.assertRaises(ValueError):
            audit.save(True)
        for n in (0, 21, True):
            with self.assertRaises(ValueError):
                wrapper.DispatchAudit(method, num_videos=n)

    def test_all_methods_accept_actual_merge_dispatch_metadata(self):
        for method in BF16_MODES:
            if method in wrapper.AXIS_METHODS:
                # Its inner-core bridge is tested separately; it must never
                # pretend the old full80 selector/merge wrapper executed.
                continue
            audit = wrapper.DispatchAudit(method)
            one_call(audit)
            self.assertEqual((audit.calls, audit.core_calls, audit.selector_calls, audit.merge_calls), (1, 1, 1, wrapper.axis_chunks(method)))
            self.assertEqual(audit.layers, [1] + [0] * 39)

    def test_selector_rejects_rf10_clone_offset_stride_and_changed_settings(self):
        audit = wrapper.DispatchAudit("mass_rf0")
        audit.enter_layer(layer("mass_rf0", 0), (21, 45, 80))
        audit.enter_core(*core_inputs("mass_rf0"))
        args, kwargs = selector_inputs(audit)
        for field, bad in (("_pointer", 999), ("_offset", 184320000), ("shape", wrapper.GRID),
                           ("_stride", (1, 2, 3, 4, 5, 6)), ("dtype", "torch.float16")):
            q = Tensor.from_metadata(wrapper.tensor_metadata(args[0]))
            setattr(q, field, bad)
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.enter_selector((q, args[1], args[2]), kwargs)
        for key, value in (("probe_count", 80), ("global_probe_count", 128), ("score_audit", object())):
            with self.subTest(option=key), self.assertRaises(ValueError):
                audit.enter_selector(args, kwargs | {key: value})

    def test_anchor_fullgrid_accepts_45_and_rejects_32_at_actual_dispatch(self):
        audit = wrapper.DispatchAudit("fisher_anchor_full45_rf0")
        audit.enter_layer(layer(audit.method, 0), (21,45,80))
        audit.enter_core(*core_inputs(audit.method))
        args, kwargs = selector_inputs(audit)
        self.assertEqual(kwargs["candidate_count"],45)
        with self.assertRaises(ValueError):
            audit.enter_selector(args,kwargs | {"candidate_count":32})
        audit.enter_selector(args,kwargs)

    def test_synthetic_method_requires_actual_zero_kernel_counter(self):
        audit = wrapper.DispatchAudit("zero_fisher_rf0")
        one_call(audit)
        self.assertEqual(audit.synthetic_seed_kernel_calls,5)
        self.assertEqual(audit.declaration["synthetic_seed"],"analytic_zero_slot0_plus_three_real_rf0_rows")
        audit.synthetic_seed_kernel_calls=0
        audit.calls=0
        with self.assertRaisesRegex(ValueError,"analytic seed"):
            audit.leave_layer()

    def test_expected_fisher_requires_observed_j1_and_no_synthetic_merge(self):
        audit = wrapper.DispatchAudit("fisher_expected_rf0")
        audit.enter_layer(layer(audit.method, 0), (21, 45, 80))
        audit.enter_core(*core_inputs(audit.method))
        args, kwargs = selector_inputs(audit)
        self.assertEqual(kwargs["block_j"], 1)
        for wrong in (2, 4, 8):
            with self.assertRaises(ValueError):
                audit.enter_selector(args, kwargs | {"block_j": wrong})
        audit.enter_selector(args, kwargs)
        self.assertNotIn(audit.method, wrapper.SYNTHETIC_METHODS)
        self.assertNotIn("synthetic_seed", audit.declaration)
        self.assertEqual(audit.declaration["selector_override"],
                         "attention.expected_fisher_seed_triton.fisher_expected_seed_rows_triton")

    def test_mean_method_requires_fresh_once_per_attention_construction(self):
        for method in wrapper.MEAN_METHODS:
            audit = wrapper.DispatchAudit(method)
            one_call(audit)
            self.assertEqual(audit.synthetic_seed_kernel_calls,5)
            self.assertEqual(audit.mean_seed_constructions,1)
            self.assertIn("current_frame_row_mean",audit.declaration["synthetic_seed"])
            audit.calls=0
            audit.mean_seed_constructions=0
            with self.assertRaisesRegex(ValueError,"current-frame mean"):
                audit.leave_layer()

    def test_phase_average_observes_original_rf0_and_real_merge(self):
        audit = wrapper.DispatchAudit("phase_averaged_fisher_rf0")
        one_call(audit)
        self.assertEqual(audit.calls, 1)
        self.assertEqual(audit.selector_calls, 1)
        self.assertEqual(audit.merge_calls, 5)
        self.assertEqual(audit.synthetic_seed_kernel_calls, 0)
        self.assertEqual(audit.mean_seed_constructions, 0)
        self.assertNotIn(audit.method, wrapper.SYNTHETIC_METHODS)
        self.assertEqual(audit.declaration["selector_override"],
                         "attention.phase_averaged_seed.phase_averaged_fisher_seed_rows")

    def test_no_missing_layer_or_trained_hybrid_path(self):
        for attribute, value in (("layer_idx", 1), ("enable_token_merging", False),
                                 ("seed_query_proj", object()), ("gma_output_calibration_weights", object()),
                                 ("token_merging_dense_heads", (30,)),
                                 ("token_merging_exact_intra_frame_attention", True),
                                 ("token_merging_seed_mode", "local_with_ith")):
            module = layer("mass_rf0", 0)
            setattr(module, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                wrapper.DispatchAudit("mass_rf0").enter_layer(module, (21, 45, 80))

    def test_count_gates_and_merge_precision_guard(self):
        audit = wrapper.DispatchAudit("mass_rf0")
        audit.enter_layer(layer("mass_rf0", 0), (21, 45, 80))
        audit.enter_core(*core_inputs("mass_rf0"))
        audit.enter_selector(*selector_inputs(audit))
        seed = Tensor(wrapper.SEED_SHAPE, dtype="torch.uint8")
        audit.leave_selector(seed)
        args, kwargs = merge_inputs(audit, seed)
        for key, value in (("fp8_score_dot", True), ("direct_bulk_normalization", True),
                           ("top_r", 2), ("head_start", 8), ("selector", "mass")):
            with self.subTest(option=key), self.assertRaises(ValueError):
                audit.enter_merge(args, kwargs | {key: value})
        with self.assertRaises(ValueError):
            audit.leave_layer()
        with self.assertRaises(ValueError):
            audit.save(True)

    def test_complete_counters_need_all20videos_all40layers(self):
        audit = wrapper.DispatchAudit("mass_rf0")
        for video in range(20):
            audit.calls = audit.core_calls = audit.selector_calls = (video + 1) * 160
            audit.merge_calls = audit.calls * 5
            audit.layers = [(video + 1) * 4] * 40
            audit.finish_video({"attention_calls": 160, "full_attention_calls": 160})
        result = audit.save(True)
        self.assertTrue(result["complete"])
        self.assertEqual(result["merge_route_head_chunk_calls"], 16000)
        self.assertEqual(result["per_layer_calls"], [80] * 40)

    def test_prepare_is_cpu_only_bounded_and_rejects_changed_frozen_source(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            checkpoint = root / "weights.pt"
            checkpoint.write_bytes(b"fake external checkpoint")
            names = [*wrapper.OWN_SOURCES, wrapper.PROMPTS, "generate.py"]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fake CPU-only test source\n", encoding="utf-8")
            stage = Path(temporary) / "run"
            assets = {"checkpoint": wrapper.checkpoint_identity(checkpoint), "fake": "metadata-only test"}
            with patch.object(wrapper, "source_names", return_value=names), \
                 patch.object(wrapper.subprocess, "check_output", return_value="abc123\n"), \
                 patch.object(wrapper, "asset_preflight", return_value=assets), \
                 patch.object(wrapper.subprocess, "run") as preflight:
                provenance = wrapper.prepare(stage, "mass_rf0", checkpoint, root=root)
            preflight.assert_called_once()
            self.assertIn("verify_prompt_contract.py", preflight.call_args.args[0][1])
            self.assertFalse((stage / "videos").exists())
            with patch.object(wrapper, "asset_preflight", return_value=assets):
                self.assertEqual(wrapper.verify_snapshot(stage, "mass_rf0", checkpoint), provenance)
            with self.assertRaises(ValueError):
                wrapper.prepare(stage, "mass_rf0", checkpoint, root=root)
            (stage / "source/generate.py").write_text("changed", encoding="utf-8")
            with patch.object(wrapper, "asset_preflight", return_value=assets), self.assertRaises(ValueError):
                wrapper.verify_snapshot(stage, "mass_rf0", checkpoint)

    def test_asset_preflight_requires_matching_absolute_complete_assets(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            model = root / wrapper.MODEL_NAME
            for name in (*wrapper.ASSET_FILES, "distill_model.pt"):
                path = model / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            checkpoint = model / "distill_model.pt"
            with patch.object(wrapper, "CANONICAL_CHECKPOINT_BYTES", 7):
                assets = wrapper.asset_preflight(checkpoint, {"WAN_MODEL_ROOT": str(root)})
                self.assertEqual(len(assets["files"]), 7)
                self.assertTrue(all("sha256" in item for item in assets["files"] if not item["path"].endswith(".pth")))
                for environment in ({}, {"WAN_MODEL_ROOT": "relative"},
                                    {"WAN_MODEL_ROOT": str(root), "GMA_OUTPUT_CALIBRATION_CAPTURE_PATH": "capture.pt"}):
                    with self.assertRaises(ValueError):
                        wrapper.asset_preflight(checkpoint, environment)
                (model / "config.json").unlink()
                with self.assertRaises(ValueError):
                    wrapper.asset_preflight(checkpoint, {"WAN_MODEL_ROOT": str(root)})

    def test_capture_environment_is_rejected_before_model_import(self):
        for name in wrapper.CAPTURE_ENVIRONMENT:
            with self.subTest(variable=name), self.assertRaises(ValueError):
                wrapper.check_environment({name: "enabled"})
        wrapper.check_environment({})

    def test_metadata_helpers_have_no_tensor_compute_or_gpu_sync(self):
        tree = ast.parse(Path(wrapper.__file__).read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in {"tensor_metadata", "grid_metadata"}]
        selected += [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DispatchAudit"]
        prohibited = {"cpu", "numpy", "item", "equal", "synchronize", "sum", "clone", "to", "reshape"}
        for node in selected:
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                    self.assertNotIn(call.func.attr, prohibited)

    def test_generation_cannot_execute_without_parent_single_gpu_contract(self):
        with self.assertRaises(ValueError):
            wrapper.execute("unused", "mass_rf0", "unused.pt", parent_purity_managed=False)
        for visible in ("", "0,1", "-1"):
            with patch.dict(wrapper.os.environ, {"CUDA_VISIBLE_DEVICES": visible}), self.assertRaises(ValueError):
                wrapper.execute("unused", "mass_rf0", "unused.pt", parent_purity_managed=True)


if __name__ == "__main__":
    unittest.main()
