"""CPU-only parent guard tests: every GPU/process call is mocked."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

from analysis import launch_quality_first_videos as launcher


STATE = {"index": 3, "name": "NVIDIA H200", "uuid": "GPU-test-one",
         "memory_mib": 1, "utilization_percent": 0}


class FakeProcess:
    def __init__(self, running=True, complete_after=2):
        self.pid = 12345
        self.returncode = None if running else 0
        self.polls = 0
        self.complete_after = complete_after
        self.terminated = self.killed = False

    def poll(self):
        self.polls += 1
        if self.polls >= self.complete_after and not self.terminated:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def fake_prepare(run_dir, method, checkpoint, *, num_videos=20, prompt_start=0):
    run_dir.mkdir()
    (run_dir / "source/analysis").mkdir(parents=True)
    (run_dir / "source/analysis/run_quality_first_videos.py").write_text("# frozen fixture\n")
    (run_dir / "source_provenance.json").write_text(json.dumps({"files": []}))


class QualityFirstVideoLauncherTests(TestCase):
    def test_account_devices_count_same_uid_devices_not_pids(self):
        rows = "GPU-a, 11\nGPU-a, 12\nGPU-b, 13\nGPU-c, 14\nGPU-gone, 15\n"
        owners = {11: 71, 12: 71, 13: 8, 14: 71}

        def stat(path):
            pid = int(path.name)
            if pid not in owners:
                raise FileNotFoundError(str(path))
            return SimpleNamespace(st_uid=owners[pid])

        with patch.object(launcher, "smi", return_value=rows), \
             patch.object(launcher.Path, "stat", new=stat), \
             patch.object(launcher.os, "getuid", return_value=71, create=True):
            self.assertEqual(launcher.account_devices(), ["GPU-a", "GPU-c"])
        with patch.object(launcher, "smi", return_value="GPU-a, 11"), \
             patch.object(launcher.Path, "stat", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                launcher.account_devices()

    def test_host_caps_are_scai6_two_scai7_one_and_unknown_rejected(self):
        for name, count in (("scai6", 2), ("scai6.local", 2), ("scai7", 1)):
            with self.subTest(host=name), patch.object(launcher.socket, "gethostname", return_value=name):
                self.assertEqual(launcher.account_limit(), count)
        with patch.object(launcher.socket, "gethostname", return_value="other-host"):
            with self.assertRaises(ValueError):
                launcher.account_limit()

    def test_gpu_parser_and_clean_single_h200_only(self):
        with patch.object(launcher, "smi", return_value="3, NVIDIA H200, GPU-abc, 1, 0"):
            self.assertEqual(launcher.gpu_state(3)["uuid"], "GPU-abc")
        with patch.object(launcher, "smi", return_value="3, H200, GPU-a, 1, 0\n4, H200, GPU-b, 1, 0"):
            with self.assertRaises(ValueError):
                launcher.gpu_state(3)
        for changes in ({"name": "NVIDIA A100"}, {"memory_mib": 101}, {"utilization_percent": 1},
                        {"index": 4}, {"uuid": "unidentified"}):
            with patch.object(launcher, "gpu_state", return_value=STATE | changes), \
                 patch.object(launcher, "gpu_processes", return_value=[]), self.assertRaises(ValueError):
                launcher.require_clean(3)
        with patch.object(launcher, "gpu_state", return_value=STATE), \
             patch.object(launcher, "gpu_processes", return_value=[{"pid": 777}]), self.assertRaises(ValueError):
            launcher.require_clean(3)

    def test_dirty_initial_gpu_does_not_prepare_spawn_or_sleep(self):
        with TemporaryDirectory() as directory:
            stage = Path(directory) / "never-created"
            with patch.object(launcher, "gpu_state", return_value=STATE | {"utilization_percent": 1}), \
                 patch.object(launcher, "gpu_processes", return_value=[]), \
                 patch.object(launcher.worker, "prepare") as prepare, \
                 patch.object(launcher.subprocess, "Popen") as spawn, \
                 patch.object(launcher.time, "sleep") as sleep, self.assertRaises(ValueError):
                launcher.launch(3, stage, "mass_rf0", "unused.pt")
            prepare.assert_not_called()
            spawn.assert_not_called()
            sleep.assert_not_called()
            self.assertFalse(stage.exists())

    def test_no_resume_even_when_gpu_clean(self):
        with TemporaryDirectory() as directory, patch.object(launcher, "require_clean") as query:
            with self.assertRaises(ValueError):
                launcher.launch(3, directory, "mass_rf0", "unused.pt")
            query.assert_not_called()

    def test_environment_is_one_uuid_and_uses_isolated_caches(self):
        environment = launcher.child_environment("stage", "GPU-one", {"LD_LIBRARY_PATH": "existing", "CUDA_VISIBLE_DEVICES": "0,1",
            "FLASHINFER_WORKSPACE_BASE": "/local2/validated-flashinfer-cuda129"})
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-one")
        self.assertEqual(environment["OMP_NUM_THREADS"], "4")
        self.assertEqual(environment["TOKEN_MERGING_FORCE_TORCH_RMSNORM"], "1")
        self.assertTrue(environment["LD_LIBRARY_PATH"].endswith("existing"))
        self.assertTrue(Path(environment["TRITON_CACHE_DIR"]).is_absolute())
        self.assertIn("python_cache", environment["PYTHONPYCACHEPREFIX"])
        self.assertEqual(environment["FLASHINFER_WORKSPACE_BASE"], str(Path("stage").resolve() / "flashinfer_workspace"))
        self.assertEqual(environment["FLASHINFER_CACHE_DIR"], str(Path("stage").resolve() / "flashinfer_cache"))

    def run_fake(self, stage, process, *, processes=None, completion=True, second_clean=True, interrupt=False,
                 account_states=None, limit=2):
        clean = {"utc": "fixed", "gpu": STATE, "compute_processes": []}
        checks = [clean, clean if second_clean else ValueError("GPU became occupied")]
        with patch.object(launcher, "require_clean", side_effect=checks), \
             patch.object(launcher, "account_limit", return_value=limit), \
             patch.object(launcher, "account_devices", side_effect=account_states, return_value=[]), \
             patch.object(launcher.worker, "asset_preflight", return_value={}), \
             patch.object(launcher.worker, "prepare", side_effect=fake_prepare) as prepare, \
             patch.object(launcher.worker, "verify_snapshot", return_value={}), \
             patch.object(launcher.subprocess, "Popen", return_value=process) as spawn, \
             patch.object(launcher, "gpu_processes", side_effect=processes or [[], [], []]), \
             patch.object(launcher, "verify_completion", side_effect=None if completion else ValueError("missing exit0"), return_value={"audit": "verified"}), \
             patch.object(launcher.time, "sleep", side_effect=KeyboardInterrupt("test signal") if interrupt else None) as sleep:
            result = launcher.launch(3, stage, "mass_rf0", str(stage.parent / "weights.pt"))
            return result, prepare, spawn, sleep

    def test_success_freezes_worker_and_writes_complete_receipts(self):
        with TemporaryDirectory() as directory:
            stage = Path(directory) / "run"
            result, prepare, spawn, sleep = self.run_fake(stage, FakeProcess(),
                account_states=[["GPU-other"], ["GPU-other"], ["GPU-other", STATE["uuid"]], ["GPU-other", STATE["uuid"]]])
            self.assertTrue(result["valid"])
            prepare.assert_called_once()
            command = spawn.call_args.args[0]
            self.assertEqual(Path(command[2]), stage / "source/analysis/run_quality_first_videos.py")
            self.assertIn("--execute-prepared", command)
            self.assertEqual(spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], STATE["uuid"])
            self.assertEqual(sleep.call_args.args, (5,))
            for name in ("startup.json", "process.json", "purity.log", "gpu_monitor.jsonl", "completion.json"):
                self.assertTrue((stage / name).exists())
            self.assertEqual((stage / "purity.log").stat().st_size, 0)
            self.assertTrue((stage / "source" / launcher.LAUNCHER_SOURCE).exists())
            self.assertEqual(result["account_gpu_limit"], 2)
            startup = json.loads((stage / "startup.json").read_text())
            self.assertEqual(startup["initial_account_devices"], ["GPU-other"])
            self.assertEqual(startup["immediate_account_devices"], ["GPU-other"])
            monitor = [json.loads(line) for line in (stage / "gpu_monitor.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["account_devices"] == ["GPU-other", STATE["uuid"]] for row in monitor))

    def test_initial_account_cap_refuses_before_prepare_or_spawn(self):
        for limit, occupied in ((2, ["GPU-a", "GPU-b"]), (1, ["GPU-a"])):
            with self.subTest(limit=limit), TemporaryDirectory() as directory:
                stage = Path(directory) / "run"
                process = FakeProcess()
                with self.assertRaisesRegex(ValueError, "cap has no remaining slot"):
                    self.run_fake(stage, process, account_states=[occupied], limit=limit)
                self.assertFalse(stage.exists())
                self.assertEqual(process.polls, 0)
                self.assertFalse(process.terminated)

    def test_account_cap_reached_before_spawn_or_during_monitor_is_invalid(self):
        for where in ("before_spawn", "monitor"):
            with self.subTest(where=where), TemporaryDirectory() as directory:
                stage = Path(directory) / "run"
                process = FakeProcess(complete_after=999)
                occupied = [[], ["GPU-a", "GPU-b"]] if where == "before_spawn" else [[], [], [STATE["uuid"], "GPU-a", "GPU-b"]]
                with self.assertRaises(ValueError):
                    self.run_fake(stage, process, account_states=occupied)
                receipt = json.loads((stage / "completion.json").read_text())
                self.assertFalse(receipt["valid"])
                self.assertIn("account GPU cap", receipt["failure"])
                if where == "before_spawn":
                    self.assertIsNone(receipt["worker_pid"])
                    self.assertFalse(process.terminated)
                else:
                    self.assertEqual(receipt["worker_pid"], process.pid)
                    self.assertTrue(process.terminated)
                    self.assertFalse(process.killed)

    def test_foreign_process_stops_only_own_child_and_invalidates(self):
        with TemporaryDirectory() as directory:
            stage = Path(directory) / "run"
            process = FakeProcess()
            with self.assertRaises(ValueError):
                self.run_fake(stage, process, processes=[[{"pid": 99999, "process_name": "foreign"}]])
            self.assertTrue(process.terminated)
            receipt = json.loads((stage / "completion.json").read_text())
            self.assertFalse(receipt["valid"])
            self.assertEqual(receipt["foreign"][0]["pid"], 99999)
            self.assertEqual(receipt["worker_pid"], process.pid)

    def test_monitor_failure_and_signal_stop_child_and_leave_invalid_receipt(self):
        for cause in ("query_error", "signal"):
            with self.subTest(cause=cause), TemporaryDirectory() as directory:
                stage = Path(directory) / "run"
                process = FakeProcess(complete_after=999)
                with self.assertRaises(ValueError):
                    self.run_fake(stage, process,
                                  processes=[RuntimeError("smi unavailable")] if cause == "query_error" else None,
                                  interrupt=cause == "signal")
                self.assertTrue(process.terminated)
                self.assertFalse(json.loads((stage / "completion.json").read_text())["valid"])

    def test_dirty_second_preflight_does_not_spawn(self):
        with TemporaryDirectory() as directory:
            stage = Path(directory) / "run"
            with self.assertRaises(ValueError):
                self.run_fake(stage, FakeProcess(), second_clean=False)
            receipt = json.loads((stage / "completion.json").read_text())
            self.assertIsNone(receipt["worker_pid"])
            self.assertFalse(receipt["valid"])

    def test_worker_exit0_without_complete_dispatch_is_invalid(self):
        with TemporaryDirectory() as directory:
            stage = Path(directory) / "run"
            with self.assertRaises(ValueError):
                self.run_fake(stage, FakeProcess(), completion=False)
            self.assertFalse(json.loads((stage / "completion.json").read_text())["valid"])

    def test_stubborn_own_child_is_killed_after_bounded_terminate(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("own worker", 15), -9]
        launcher.stop_own_child(process)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_count, 2)

    def stable_completion_fixture(self, path):
        (path / "generation.exitcode").write_text("0\n")
        (path / "stable_kernel.py").write_text("# generated stable kernel fixture\n")
        (path / "source_provenance.json").write_text("{}\n")
        source_sha = launcher.worker.digest(path / "stable_kernel.py")
        audit = dict(complete=True, videos_observed=20, self_attention_calls=3200, core_calls=3200,
            selector_calls=3200, merge_route_head_chunk_calls=16000, per_layer_calls=[80] * 40,
            declaration={"fixture": True}, error=None, stable_merge_kernel_calls=16000,
            callable_identities={"stable_merge_kernel": {"path": str((path / "stable_kernel.py").resolve()), "sha256": source_sha}})
        (path / "dispatch_audit.json").write_text(json.dumps(audit))
        (path / "runtime.json").write_text(json.dumps(dict(checkpoint_weights="raw_state_dict", records=[
            dict(prompt_index=index, attention_calls=160, full_attention_calls=160) for index in range(20)])))
        return audit, source_sha

    def test_stable_completion_missing_counter_fails(self):
        with TemporaryDirectory() as directory, \
             patch.object(launcher.worker, "STABLE_SQ1_METHODS", ("stable-fixture",)), \
             patch.object(launcher, "method_contract", return_value={"fixture": True}), \
             patch.object(launcher.worker, "verify_snapshot", return_value={"planned": {"num_videos": 20}}):
            path = Path(directory)
            audit, source_sha = self.stable_completion_fixture(path)
            self.assertEqual(launcher.verify_completion(path, "stable-fixture", "unused.pt")["stable_kernel_sha256"], source_sha)
            del audit["stable_merge_kernel_calls"]
            (path / "dispatch_audit.json").write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "stable_merge_kernel_calls"):
                launcher.verify_completion(path, "stable-fixture", "unused.pt")

    def test_stable_completion_wrong_generated_source_hash_fails(self):
        with TemporaryDirectory() as directory, \
             patch.object(launcher.worker, "STABLE_SQ1_METHODS", ("stable-fixture",)), \
             patch.object(launcher, "method_contract", return_value={"fixture": True}), \
             patch.object(launcher.worker, "verify_snapshot", return_value={"planned": {"num_videos": 20}}):
            path = Path(directory)
            audit, _ = self.stable_completion_fixture(path)
            audit["callable_identities"]["stable_merge_kernel"]["sha256"] = "0" * 64
            (path / "dispatch_audit.json").write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "stable merge generated source changed"):
                launcher.verify_completion(path, "stable-fixture", "unused.pt")


if __name__ == "__main__":
    main()
