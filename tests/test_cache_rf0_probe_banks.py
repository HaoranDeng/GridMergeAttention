"""Small CPU NumPy/hash tests; no Torch import, model loading, or CUDA."""
import copy
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "analysis" / "cache_rf0_probe_banks.py"
SPEC = importlib.util.spec_from_file_location("cache_rf0_probe_banks", SOURCE)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


class ProbeBankCacheTests(unittest.TestCase):
    def test_bank_count_order_and_identity(self):
        coordinates = cache.bank_coordinates()
        self.assertEqual(coordinates.shape, (600, 4))
        self.assertEqual(len(np.unique(coordinates, axis=0)), 600)
        for index, expected in ((0, [0, 0, 0, 0]), (4, [0, 0, 0, 44]),
                                (5, [0, 0, 10, 0]), (15, [0, 1, 0, 0]),
                                (150, [10, 0, 0, 0]), (599, [30, 76, 20, 44])):
            self.assertEqual(coordinates[index].tolist(), expected)

    def test_flat_indices_use_rf0_row_column_and_original_key_frame(self):
        coordinates = np.asarray([[30, 76, 20, 44], [0, 1, 10, 11]])
        query, key = cache.flat_indices(coordinates)
        self.assertEqual(query.shape, (2, 45))
        self.assertEqual(key.shape, (2, 80))
        self.assertEqual(query[0, [0, -1]].tolist(), [76, 3596])
        self.assertEqual(key[0, [0, -1]].tolist(), [75520, 75599])
        self.assertEqual(key[1, [0, -1]].tolist(), [36880, 36959])

    def test_invalid_coordinates_fail(self):
        for bad in ([[1, 0, 0, 0]], [[0, 80, 0, 0]], [[0, 0, 21, 0]],
                    [[0, 0, 0, -1]], [[0., 1., 0., 0.]], [1, 2, 3]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                cache.flat_indices(np.asarray(bad))

    def test_normalizer_layout_and_natural_log_conversion(self):
        raw = np.arange(4 * 3600 * 2, dtype=np.float64).reshape(4, 3600, 2)
        native = np.arange(3600 * 4, dtype=np.float32).reshape(1, 1, 45, 80, 4)
        coordinates = np.asarray([[30, 76, 20, 44], [0, 1, 10, 11]])
        exact, sampled = cache.normalizer_arrays(raw, native, coordinates)
        self.assertEqual(exact.shape, (2, 45))
        self.assertEqual(exact[0, 2], raw[3, 236, 0])
        self.assertEqual(exact[1, 0], raw[0, 1, 0])
        self.assertEqual(sampled[0, 2], float(native[0, 0, 2, 76, 3]) * np.log(2))
        self.assertEqual(sampled.dtype, np.float64)

    def test_nonfinite_or_wrong_dtype_normalizers_fail(self):
        raw = np.zeros((4, 3600, 2), np.float64)
        native = np.zeros((1, 1, 45, 80, 4), np.float32)
        coordinates = cache.bank_coordinates()[:1]
        with self.assertRaises(ValueError):
            cache.normalizer_arrays(raw.astype(np.float32), native, coordinates)
        raw[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            cache.normalizer_arrays(raw, native, coordinates)

    def test_fingerprint_uses_bytes_not_remote_path_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.pt"
            path.write_bytes(b"small fixture")
            identity = cache.fingerprint(path)
            receipt = identity | {"path": "/remote/another/location/input.pt"}
            self.assertEqual(cache.verify_fingerprint(path, receipt), identity)
            with self.assertRaisesRegex(ValueError, "audited input"):
                cache.verify_fingerprint(path, receipt, "0" * 64)
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                cache.verify_fingerprint(path, receipt | {"bytes": 1})
            path.write_bytes(b"small fixturE")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                cache.verify_fingerprint(path, receipt)

    def test_output_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            cache.write_json_new(path, {"valid": True})
            with self.assertRaises(FileExistsError):
                cache.write_json_new(path, {"valid": False})
            self.assertEqual(cache.read_json(path), {"valid": True})

    def test_cpu_environment_requires_explicit_hidden_cuda_and_bounded_threads(self):
        for value in ("", "-1"):
            cache.check_cpu_environment({"CUDA_VISIBLE_DEVICES": value}, 4)
        for environment, threads in (({}, 4), ({"CUDA_VISIBLE_DEVICES": "0"}, 4),
                                     ({"CUDA_VISIBLE_DEVICES": ""}, 5), ({"CUDA_VISIBLE_DEVICES": ""}, 0)):
            with self.subTest(environment=environment, threads=threads), self.assertRaises(ValueError):
                cache.check_cpu_environment(environment, threads)

    def test_cached_metadata_rejects_cross_prompt_frame_and_source_substitution(self):
        job = {"metadata": {"capture_file": "prompt000_call033_layer33.pt", "prompt_index": 0},
               "capture": {"sha256": "a" * 64}}
        oracle = {"normalizer_convention": {"partition_g256": "original"},
                  "evaluation_provenance": {"source_sha256": {"source.py": "b" * 64}}}
        metadata = {"capture_file": job["metadata"]["capture_file"], "capture_sha256": "a" * 64,
                    "prompt_index": 0, "step_index": 0, "router_frame": 0, "heads": list(cache.HEADS),
                    "normalizer_layout": [1, 1, 45, 80, 4], "raw_natural_layout": [4, 3600, 2],
                    "raw_natural_columns": ["allkeys", "phase0"], "eligible_candidate": False,
                    "normalizer_convention": oracle["normalizer_convention"],
                    "evaluation_source_sha256": oracle["evaluation_provenance"]["source_sha256"]}
        cache.validate_cache_metadata(metadata, job, oracle)
        for field, value in (("prompt_index", 1), ("router_frame", 10), ("capture_sha256", "c" * 64),
                             ("heads", [0, 1, 2, 3]), ("evaluation_source_sha256", {})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                cache.validate_cache_metadata(metadata | {field: value}, job, oracle)

    def test_capture_tensor_names_match_frozen_producer_contract(self):
        producer = SOURCE.with_name("capture_rf0_layer33_diagnostic.py")
        if not producer.is_file():
            self.skipTest("producer source is optional in an isolated exporter staging directory")
        tree = ast.parse(producer.read_text(encoding="utf-8"))
        capture_names = [ast.literal_eval(node.args[0]) for node in ast.walk(tree)
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                         and node.func.id == "zip" and node.args and isinstance(node.args[0], ast.Tuple)
                         and ast.literal_eval(node.args[0]) == ("q", "k", "v", "dense_output")]
        self.assertEqual(capture_names, [cache.CAPTURE_TENSORS])

    def test_capture_metadata_accepts_dense_output_but_rejects_aliases_and_unknown_fields(self):
        item = {"prompt_index": 0, "scale": cache.SCALE, "bytes": 1234}
        metadata = {"prompt_index": 0, "scale": cache.SCALE}
        payload = {name: object() for name in cache.CAPTURE_TENSORS} | {"metadata": metadata}
        cache.validate_capture_metadata(payload, item)
        renamed = dict(payload)
        renamed["dense"] = renamed.pop("dense_output")
        for altered in (renamed, payload | {"unknown": 1},
                        payload | {"metadata": metadata | {"prompt_index": 1}},
                        payload | {"metadata": metadata | {"unknown": 1}}):
            with self.subTest(keys=list(altered)), self.assertRaises(ValueError):
                cache.validate_capture_metadata(altered, item)

    def test_real_prompt_manifest_and_fixed_oracle_identities_match(self):
        root = Path("C:/GMA/experiments/rf0_layer33_dense_capture_20260904n/captures")
        if not root.is_dir():
            self.skipTest("local audited artifact is optional for helper tests")
        path = root / "prompt_manifest.json"
        self.assertEqual(cache.fingerprint(path)["sha256"], cache.PROMPT_MANIFEST_SHA)
        prompt = cache.read_json(path)
        cache.validate_prompt_manifest(prompt)
        altered = copy.deepcopy(prompt)
        altered["prompts"][1]["text"] += " changed"
        with self.assertRaises(ValueError):
            cache.validate_prompt_manifest(altered)


if __name__ == "__main__":
    unittest.main()
