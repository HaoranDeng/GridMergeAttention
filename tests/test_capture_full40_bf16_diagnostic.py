"""CPU protocol and actual storage-AST checks; no model, Torch, or GPU."""
import ast
import copy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from analysis import capture_full40_bf16_diagnostic as capture

SOURCE = Path(capture.__file__)


def config():
    return {"image_or_video_shape": [1, 21, 16, 90, 160], "denoising_step_list": [1000, 750, 500, 250],
            "warp_denoising_step": True, "token_merging_args": {"enable": False},
            "model_kwargs": {"model_name": "Wan2.1-T2V-14B-StepDistill-CfgDistill",
                             "timestep_shift": 5.0, "load_pretrained_weights": False}}


def records():
    return [capture.call_record(p, c, c % 40, c // 40, capture.ACTUAL_TIMESTEPS[c // 40])
            for p in range(3) for c in range(160)]


class Full40CaptureTests(unittest.TestCase):
    def test_exact_480_calls_24_disjoint_files_and_full_storage_budget(self):
        rows = records()
        files = capture.validate_complete(rows)
        self.assertEqual(len(rows), 480)
        self.assertEqual(len(files), 24)
        self.assertEqual(len(set(files)), 24)
        self.assertEqual(files[0], "prompt000_call000_layer0.pt")
        self.assertEqual(files[-1], "prompt002_call153_layer33.pt")
        self.assertEqual(capture.HEADS, tuple(range(40)))
        self.assertEqual(capture.TARGET_CALLS, (0, 33, 40, 73, 80, 113, 120, 153))
        self.assertEqual(capture.BYTES_PER_CAPTURE, 3096576000)
        self.assertEqual(capture.RAW_CAPTURE_BYTES, 74317824000)
        for prompt in range(3):
            for step in range(4):
                selected = [row for row in rows if row["prompt_index"] == prompt
                            and row["step_index"] == step and row["capture_file"]]
                self.assertEqual([row["layer_index"] for row in selected], [0, 33])

    def test_missing_duplicate_reordered_short_or_wrong_step_is_rejected(self):
        source = records()
        for bad in (source[:-1], source + source[-1:], source[:120] + source[160:280] + source[320:440],
                    source[:384], [source[1], source[0]] + source[2:]):
            with self.assertRaises(ValueError):
                capture.validate_complete(bad)
        bad = copy.deepcopy(source)
        bad[40]["capture_file"] = "prompt000_call040_layer33.pt"
        with self.assertRaises(ValueError):
            capture.validate_complete(bad)
        for args in ((3, 33, 33, 0, 1000.), (0, 160, 0, 4, 0), (0, 33, 32, 0, 1000.),
                     (0, 73, 33, 0, 1000.), (0, 73, 33, 1, 750.)):
            with self.assertRaises(ValueError):
                capture.call_record(*args)

    def test_config_rejects_candidate_hybrid_wrong_model_and_schedule(self):
        capture.validate_config(config())
        for key, value in (("causal", True), ("denoising_step_list", [1000, 750, 500]),
                           ("image_or_video_shape", [1, 21, 16, 60, 104]), ("warp_denoising_step", False),
                           ("token_merging_args", {"enable": True}),
                           ("token_merging_args", {"enable": False, "attention_backend": "svg_all"})):
            with self.assertRaises(ValueError):
                capture.validate_config(config() | {key: value})
        for key, value in (("model_name", "Wan2.1-T2V-1.3B"), ("timestep_shift", 1.), ("load_pretrained_weights", True)):
            bad = config()
            bad["model_kwargs"][key] = value
            with self.assertRaises(ValueError):
                capture.validate_config(bad)

    def test_canonical_prompt_and_frozen_source_identity(self):
        path = capture.ROOT / "prompts/MovieGenVideoBench_extended.txt"
        prompts = capture.load_nonempty_prompts(path)
        capture.validate_canonical_prompt_contract(path, prompts)
        manifest = capture.build_prompt_manifest(path, prompts, selected_count=3, seed=0, num_samples=1,
                                                 vbench_filenames=False, allow_noncanonical=False)
        self.assertEqual(manifest["seed"], 0)
        self.assertEqual([row["index"] for row in manifest["prompts"]], [0, 1, 2])
        self.assertEqual(manifest["prompt_file_sha256"], "50293c2717c8f2df5773154d4680c74e636ecb08eb4fe13cdce7357d0ab9d121")
        with self.assertRaises(ValueError):
            capture.validate_canonical_prompt_contract(path, list(reversed(prompts)))
        hashes = capture.source_hashes()
        self.assertEqual(hashes["analysis/screen_native_seed_trajectory.py"], capture.FROZEN_STREAM_SHA256)
        self.assertEqual(hashes["analysis/capture_full40_bf16_diagnostic.py"], hashlib.sha256(SOURCE.read_bytes()).hexdigest())

    def test_parent_guard_before_any_model_or_gpu_import(self):
        capture.require_parent({"FULL40_CAPTURE_PARENT": "1245"}, 1245)
        for env, pid in (({}, 1245), ({"FULL40_CAPTURE_PARENT": "1244"}, 1245),
                         ({"FULL40_CAPTURE_PARENT": "1"}, 1)):
            with self.assertRaises(RuntimeError):
                capture.require_parent(env, pid)
        with mock.patch.object(capture, "require_parent", side_effect=RuntimeError("no parent")), \
                mock.patch.object(capture, "check_capture_environment") as checked:
            with self.assertRaisesRegex(RuntimeError, "no parent"):
                capture.main()
            checked.assert_not_called()
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        for name in ("main", "capture"):
            body = next(node.body for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.unparse(body[0]), "require_parent()")
        self.assertFalse(any(isinstance(node, ast.Import) and any(item.name == "torch" for item in node.names)
                             for node in tree.body))

    def test_storage_ast_preserves_all_original_bf16_tensors_and_dense_result(self):
        text = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(text)
        hook = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "attention")
        self.assertEqual(ast.unparse(hook.body[0]), "dense = original_attention(q, k, v, *inputs, **options)")
        returns = [node for node in ast.walk(hook) if isinstance(node, ast.Return)]
        self.assertTrue(all(isinstance(node.value, ast.Name) and node.value.id == "dense" for node in returns))
        tensors = next(node.value for node in hook.body if isinstance(node, ast.If)
                       for node in node.body if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "tensors")
        self.assertEqual(ast.unparse(tensors.value), "value.detach().contiguous().cpu()")
        self.assertIn('zip(("q", "k", "v", "dense_output"), (q, k, v, dense))', text)
        for excluded in ("index_select", "head_index", "torch.float16", "np.savez", "_hard_c4_attention", "write_video"):
            self.assertNotIn(excluded, text)
        self.assertIn('value.dtype != torch.bfloat16', text.replace('x.dtype', 'value.dtype'))
        self.assertIn('torch.isfinite(value).all()', text)
        self.assertIn('destination.open("xb")', text)
        self.assertIn('torch.save(tensors | {"metadata": metadata}, stream)', text)
        self.assertIn('receipt = fingerprint(destination)', text)
        self.assertIn('payload["checkpoint_weights"] != "raw_state_dict"', text)
        self.assertIn('state["call"] != 160 or state["generator_calls"] != 4', text)

    def test_disk_budget_checks_full24_and_remaining_payload_not_four_heads(self):
        required = capture.RAW_CAPTURE_BYTES + capture.DISK_RESERVE_BYTES
        self.assertEqual(capture.require_disk_budget(required), required)
        with self.assertRaises(RuntimeError):
            capture.require_disk_budget(required - 1)
        one = capture.BYTES_PER_CAPTURE + capture.DISK_RESERVE_BYTES
        self.assertEqual(capture.require_disk_budget(one, remaining_captures=1), one)
        for invalid in (-1, 25, True, 1.5):
            with self.assertRaises(ValueError):
                capture.require_disk_budget(required, remaining_captures=invalid)

    def test_postsave_checksum_and_concurrent_file_change_are_checked(self):
        payload = b"original BF16 tensor storage receipt"
        before = SimpleNamespace(st_size=len(payload), st_mtime_ns=123)
        path = SimpleNamespace(stat=mock.Mock(return_value=before),
            open=mock.Mock(side_effect=lambda mode: io.BytesIO(payload)),
            resolve=lambda: "mock_capture.pt")
        with mock.patch.object(capture, "Path", side_effect=lambda value: value):
            receipt = capture.fingerprint(path)
        self.assertEqual(receipt["bytes"], len(payload))
        self.assertEqual(receipt["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(receipt["mtime_ns"], 123)
        path.stat = mock.Mock(side_effect=[before, SimpleNamespace(st_size=len(payload), st_mtime_ns=124)])
        with mock.patch.object(capture, "Path", side_effect=lambda value: value), \
                self.assertRaisesRegex(RuntimeError, "changed"):
            capture.fingerprint(path)


if __name__ == "__main__":
    unittest.main()
