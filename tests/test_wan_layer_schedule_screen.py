"""Tests for the multi-schedule Wan screening CLI."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from analysis.screen_wan_layer_schedules import (
    load_runtime_records,
    main,
    parse_schedule,
)


class WanLayerScheduleScreenTests(unittest.TestCase):
    def test_parse_schedule(self) -> None:
        self.assertEqual(
            parse_schedule("candidate_1=0, 4,19"),
            ("candidate_1", [0, 4, 19]),
        )

    def test_parse_schedule_rejects_bad_name_or_empty_layer(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_schedule("bad name=0,1")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_schedule("candidate=0,,1")

    def test_load_runtime_records_only_when_resuming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_path = Path(directory) / "runtime.json"
            runtime_path.write_text(
                json.dumps({"records": [{"name": "candidate"}]}),
                encoding="utf-8",
            )
            self.assertEqual(load_runtime_records(runtime_path, resume=False), [])
            self.assertEqual(
                load_runtime_records(runtime_path, resume=True),
                [{"name": "candidate"}],
            )

    def test_dense_name_rejects_invalid_or_duplicate_name(self) -> None:
        base_args = SimpleNamespace(
            schedule=[("candidate", [0])],
            dense_name="candidate",
        )
        with mock.patch(
            "analysis.screen_wan_layer_schedules.parse_args", return_value=base_args
        ):
            with self.assertRaisesRegex(ValueError, "unique"):
                main()

        base_args.dense_name = "bad name"
        with mock.patch(
            "analysis.screen_wan_layer_schedules.parse_args", return_value=base_args
        ):
            with self.assertRaisesRegex(ValueError, "dense name"):
                main()


if __name__ == "__main__":
    unittest.main()
