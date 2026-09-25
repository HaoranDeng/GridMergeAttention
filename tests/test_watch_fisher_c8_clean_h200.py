import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/watch_fisher_c8_clean_h200.py"


def load_module():
    spec = importlib.util.spec_from_file_location("watch_fisher_c8_clean_h200", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WatchFisherC8Tests(unittest.TestCase):
    def test_contract_keeps_one_gpu_and_exact_c8_benchmark(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("benchmark_fisher_c8_exact.py", source)
        self.assertIn("stable-checks", source)
        self.assertIn("foreign GPU process", source)
        self.assertNotIn("sbatch", source)
        self.assertNotIn("srun", source)

    def test_only_contention_is_retryable(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.json"
            self.assertTrue(module.retryable_failure(missing))
            contaminated = root / "contaminated.json"
            contaminated.write_text(json.dumps({
                "monitor_errors": ["foreign GPU process", "foreign GPU process"]
            }), encoding="utf-8")
            self.assertTrue(module.retryable_failure(contaminated))
            fatal = root / "fatal.json"
            fatal.write_text(json.dumps({
                "error": "compile failed", "monitor_errors": []
            }), encoding="utf-8")
            self.assertFalse(module.retryable_failure(fatal))


if __name__ == "__main__":
    unittest.main()
