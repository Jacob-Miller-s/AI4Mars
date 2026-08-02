import tempfile
import unittest
import sys
from pathlib import Path

from src.research_console.demo import run_smoke_training
from src.research_console.run_store import RunReader
from src.research_console.schema import RunStatus


class ResearchConsoleDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.tmp.name) / "outputs" / "runs"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_cpu_smoke_training_produces_non_benchmark_run_and_assets(self) -> None:
        logger = run_smoke_training(self.runs_root, run_id="synthetic-test-run", epochs=1)
        reader = RunReader(logger.run_dir)

        metadata = reader.metadata()
        summary = reader.summary()
        metric_events = reader.metrics()

        self.assertEqual(metadata.status, RunStatus.COMPLETED)
        self.assertFalse(metadata.provenance.protocol.valid)
        self.assertIn("synthetic_demo_not_research_benchmark", metadata.provenance.protocol.failed_gates)
        self.assertEqual(metadata.environment.python, sys.version)
        self.assertEqual(summary.status, RunStatus.COMPLETED)
        self.assertTrue(any(event["event_type"] == "batch" for event in metric_events))
        self.assertTrue(any(event["event_type"] == "epoch" for event in metric_events))
        self.assertTrue((logger.run_dir / "artifacts" / "prediction_index.jsonl").exists())
        self.assertTrue((logger.run_dir / "checkpoints" / "best_synthetic.pth").exists())

    def test_smoke_training_records_crash_without_leaking_traceback_path(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Intentional synthetic"):
            run_smoke_training(
                self.runs_root,
                run_id="synthetic-failure-run",
                epochs=2,
                fail_after_epoch=1,
            )

        reader = RunReader(self.runs_root / "synthetic-failure-run")
        self.assertEqual(reader.metadata().status, RunStatus.FAILED)
        self.assertEqual(reader.summary().status, RunStatus.FAILED)
        self.assertTrue((reader.run_dir / "artifacts" / "failure_traceback.txt").exists())


if __name__ == "__main__":
    unittest.main()