"""Small CPU-only L33 capture contracts; these do not execute a pipeline/GPU."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/capture_rf0_layer33_diagnostic.py"
spec = importlib.util.spec_from_file_location("capture_layer33", SOURCE)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def config():
    return {"image_or_video_shape": [1, 21, 16, 90, 160], "denoising_step_list": [1000, 750, 500, 250],
            "warp_denoising_step": True, "token_merging_args": {"enable": False},
            "model_kwargs": {"model_name": "Wan2.1-T2V-14B-StepDistill-CfgDistill",
                             "timestep_shift": 5.0, "load_pretrained_weights": False}}


def records():
    return [capture.call_record(p, c, c % 40, c // 40, capture.ACTUAL_TIMESTEPS[c // 40])
            for p in range(3) for c in range(160)]


class Layer33CaptureTests(unittest.TestCase):
    def test_exact_twelve_call_map_and_payload_size(self):
        files = capture.validate_complete(records())
        self.assertEqual(len(files), 12)
        self.assertEqual(len(set(files)), 12)
        self.assertEqual(files[0], "prompt000_call033_layer33.pt")
        self.assertEqual(files[-1], "prompt002_call153_layer33.pt")
        self.assertEqual(capture.HEADS, (0, 10, 20, 30))
        self.assertEqual(capture.RAW_CAPTURE_BYTES, 3715891200)

    def test_missing_duplicate_reordered_or_three_step_trajectory_is_rejected(self):
        source = records()
        for bad in (source[:-1], source + source[-1:], source[:120] + source[160:280] + source[320:440],
                    [source[1], source[0]] + source[2:]):
            with self.assertRaises(ValueError):
                capture.validate_complete(bad)
        bad = copy.deepcopy(source)
        bad[33]["capture_file"] = None
        with self.assertRaises(ValueError):
            capture.validate_complete(bad)

    def test_real_step_layer_and_prompt_guards(self):
        for args in ((3, 33, 33, 0, 1000.0), (0, 160, 0, 4, 0), (0, 33, 32, 0, 1000.0),
                     (0, 73, 33, 0, 1000.0), (0, 73, 33, 1, 750.0)):
            with self.assertRaises(ValueError):
                capture.call_record(*args)
        self.assertEqual(capture.call_record(2, 113, 33, 2, 833.3333129882812)["step_index"], 2)

    def test_config_rejects_hybrid_wrong_model_or_schedule(self):
        capture.validate_config(config())
        mutations = [("causal", True), ("denoising_step_list", [1000, 750, 500]),
                     ("image_or_video_shape", [1, 21, 16, 60, 104]), ("warp_denoising_step", False),
                     ("token_merging_args", {"enable": True}),
                     ("token_merging_args", {"enable": False, "attention_backend": "svg_all"})]
        for key, value in mutations:
            with self.subTest(key=key), self.assertRaises(ValueError):
                capture.validate_config(config() | {key: value})
        for key, value in (("model_name", "Wan2.1-T2V-1.3B"), ("timestep_shift", 1.0), ("load_pretrained_weights", True)):
            bad = config()
            bad["model_kwargs"][key] = value
            with self.assertRaises(ValueError):
                capture.validate_config(bad)

    def test_canonical_manifest_and_frozen_stream_hash(self):
        path = ROOT / "prompts/MovieGenVideoBench_extended.txt"
        prompts = capture.load_nonempty_prompts(path)
        capture.validate_canonical_prompt_contract(path, prompts)
        manifest = capture.build_prompt_manifest(path, prompts, selected_count=3, seed=0, num_samples=1,
                                                 vbench_filenames=False, allow_noncanonical=False)
        self.assertEqual(manifest["selected_prompt_count"], 3)
        self.assertEqual(manifest["seed"], 0)
        self.assertEqual([p["index"] for p in manifest["prompts"]], [0, 1, 2])
        stream = ROOT / "analysis/screen_native_seed_trajectory.py"
        self.assertEqual(hashlib.sha256(stream.read_bytes()).hexdigest(), capture.FROZEN_STREAM_SHA256)

    def test_original_dense_pipeline_bf16_and_no_selection(self):
        text = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(text)
        self.assertNotIn("denoising_step_list[:-1]", text)
        self.assertNotIn("torch.float16", text)
        self.assertNotIn("_hard_c4_attention", text)
        self.assertNotIn("partition_mass_seed_rows_triton", text)
        self.assertIn("video = pipeline.inference(noise=noise, text_prompts=[prompts[index]])", text)
        self.assertIn("set_seed(index)", text)
        self.assertIn('state["generator_calls"] != 4', text)
        self.assertIn('state["call"] != 160', text)
        self.assertIn('zip(("q", "k", "v", "dense_output"), (q, k, v, dense))', text)
        self.assertIn('destination.open("xb")', text)
        self.assertIn('payload["checkpoint_weights"] != "raw_state_dict"', text)
        hook = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "attention")
        returns = [n for n in ast.walk(hook) if isinstance(n, ast.Return)]
        self.assertTrue(all(isinstance(n.value, ast.Name) and n.value.id == "dense" for n in returns))
        self.assertIsInstance(hook.body[0], ast.Assign)
        self.assertEqual(hook.body[0].value.func.id, "original_attention")


if __name__ == "__main__":
    unittest.main()
