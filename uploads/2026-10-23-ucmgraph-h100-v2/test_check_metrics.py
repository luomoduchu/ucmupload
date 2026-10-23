"""Regression checks with synthetic snapshots; no GPU or vLLM required."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LEGACY = "ucm:cache_load_backend_shards_total"
GRAPH = "ucm:cache_transfer_graph_backend_shards_total"
BATCH = "ucm:cache_transfer_graph_backend_batches_total"


class CheckMetricsTest(unittest.TestCase):
    def check(self, mode, before, after, message=None, completed=16):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, values in (("before", before), ("after", after)):
                (root / name).write_text(
                    "".join(f'{key}{{worker_rank="0"}} {value}\n' for key, value in values.items())
                )
            (root / "result").write_text(json.dumps({"num_prompts": 16, "completed": completed}))
            run = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("check_metrics.py")),
                 mode, str(root / "before"), str(root / "after"), str(root / "result")],
                capture_output=True, text=True,
            )
            if message is None:
                self.assertEqual(run.returncode, 0, run.stderr)
            else:
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(message, run.stderr)

    def test_graph_on_legacy_zero(self):
        self.check("on", {LEGACY: 0, GRAPH: 10, BATCH: 1},
                   {LEGACY: 0, GRAPH: 100, BATCH: 4})

    def test_graph_on_legacy_absent(self):
        self.check("on", {}, {GRAPH: 90, BATCH: 3})

    def test_off(self):
        self.check("off", {LEGACY: 10}, {LEGACY: 100})

    def test_missing_metric(self):
        self.check("on", {}, {LEGACY: 100}, "Required metric not exported")

    def test_zero_increment(self):
        self.check("off", {LEGACY: 10}, {LEGACY: 10}, "no backend load increment")

    def test_missing_graph_batches(self):
        self.check("on", {}, {GRAPH: 90}, "Graph path was not exercised")

    def test_off_with_graph_activity(self):
        self.check("off", {}, {LEGACY: 90, GRAPH: 90, BATCH: 3}, "Expected graph OFF")

    def test_reset(self):
        self.check("off", {LEGACY: 100}, {LEGACY: 10}, "Counters reset")

    def test_failed_requests(self):
        self.check("off", {}, {LEGACY: 90}, "Benchmark requests failed", completed=15)

    def test_backend_error(self):
        self.check("on", {}, {GRAPH: 90, BATCH: 3,
                   "ucm:cache_backend_load_wait_errors_total": 1}, "UCM errors")


if __name__ == "__main__":
    unittest.main()
