import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.research_console.run_store import RunLogger, RunReader, read_jsonl_tolerant
from src.research_console.schema import EpochMetrics, RunStatus
from tests.test_research_console_schema import valid_metadata


class RunStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.tmp.name) / "outputs" / "runs"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_logger_writes_live_readable_run_and_summary(self) -> None:
        logger = RunLogger(self.runs_root, valid_metadata())
        logger.start()
        logger.log_batch(
            epoch=1,
            batch=1,
            total_batches=4,
            loss=0.8,
            smoothed_loss=0.8,
            throughput_samples_per_second=3.5,
            eta_seconds=12.0,
        )
        logger.log_epoch(
            EpochMetrics(
                timestamp=datetime.now(timezone.utc),
                epoch=1,
                train_loss=0.8,
                val_loss=0.7,
                pixel_accuracy=0.75,
                mean_iou=0.5,
            )
        )
        summary = logger.finish()

        reader = RunReader(logger.run_dir)
        self.assertEqual(reader.metadata().status, RunStatus.COMPLETED)
        self.assertEqual(summary.best_epoch, 1)
        self.assertEqual(summary.best_validation_mean_iou, 0.5)
        self.assertEqual([event["event_type"] for event in reader.metrics()], ["batch", "epoch"])
        self.assertGreaterEqual(len(reader.system_metrics()), 1)

    def test_logger_persists_training_cuda_allocator_metrics(self) -> None:
        logger = RunLogger(self.runs_root, valid_metadata())
        logger.start()
        logger.log_batch(
            epoch=1,
            batch=1,
            total_batches=1,
            loss=0.8,
            gpu_memory_allocated_bytes=64,
            gpu_memory_reserved_bytes=128,
        )

        system_event = logger.reader.system_metrics()[-1]

        self.assertEqual(system_event["gpu_memory_allocated_bytes"], 64)
        self.assertEqual(system_event["gpu_memory_reserved_bytes"], 128)

    def test_reader_ignores_only_partial_final_jsonl_line(self) -> None:
        event_path = self.runs_root / "partial.jsonl"
        event_path.parent.mkdir(parents=True)
        event_path.write_text(json.dumps({"event_type": "epoch"}) + "\n" + '{"partial":', encoding="utf-8")

        events = read_jsonl_tolerant(event_path)

        self.assertEqual(events, [{"event_type": "epoch"}])

    def test_failed_run_captures_traceback_as_portable_artifact(self) -> None:
        logger = RunLogger(self.runs_root, valid_metadata())
        logger.start()
        summary = logger.fail(RuntimeError("synthetic crash"))

        self.assertEqual(summary.status, RunStatus.FAILED)
        self.assertEqual(summary.failure_reason, "synthetic crash")
        self.assertEqual(summary.traceback_artifact.path, "artifacts/failure_traceback.txt")
        self.assertTrue((logger.run_dir / "artifacts" / "failure_traceback.txt").exists())


if __name__ == "__main__":
    unittest.main()