"""Keep requested EMA flags distinct from the state dictionary actually loaded."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tests._support import import_module_cpu_safely


class CheckpointProvenanceTests(unittest.TestCase):
    def test_raw_state_dict_ignores_ema_flag(self):
        generate = import_module_cpu_safely("generate")
        state = {"patch_embedding.weight": object()}
        for requested in (False, True):
            pipeline = SimpleNamespace(generator=mock.Mock())
            with mock.patch.object(generate.torch, "load", return_value=state):
                key = generate.load_generator_checkpoint(pipeline, "unused.pt", requested)
            self.assertEqual(key, "raw_state_dict")
            pipeline.generator.model.load_state_dict.assert_called_once_with(state)

    def test_named_weight_selection_reports_actual_key(self):
        generate = import_module_cpu_safely("generate")
        state = {"generator": {"model.weight": object()},
                 "generator_ema": {"model.weight": object()}}
        for requested, expected in ((False, "generator"), (True, "generator_ema")):
            pipeline = SimpleNamespace(generator=mock.Mock())
            with mock.patch.object(generate.torch, "load", return_value=state):
                key = generate.load_generator_checkpoint(pipeline, "unused.pt", requested)
            self.assertEqual(key, expected)
            pipeline.generator.load_state_dict.assert_called_once_with(state[expected])

    def test_ema_request_falls_back_to_available_generator(self):
        generate = import_module_cpu_safely("generate")
        state = {"generator": {"model.weight": object()}}
        pipeline = SimpleNamespace(generator=mock.Mock())
        with mock.patch.object(generate.torch, "load", return_value=state):
            key = generate.load_generator_checkpoint(pipeline, "unused.pt", True)
        self.assertEqual(key, "generator")

    def test_runtime_records_requested_and_effective_weights(self):
        generate = import_module_cpu_safely("generate")
        args = SimpleNamespace(config="config.yaml", checkpoint="weights.pt",
                               use_ema=True, loaded_checkpoint_weights="raw_state_dict")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            with mock.patch.object(generate, "summarize_runtime", return_value={}):
                generate.write_runtime(path, [], args, {})
            payload = json.loads(path.read_text())
        self.assertIs(payload["use_ema_requested"], True)
        self.assertEqual(payload["checkpoint_weights"], "raw_state_dict")


if __name__ == "__main__":
    unittest.main()
