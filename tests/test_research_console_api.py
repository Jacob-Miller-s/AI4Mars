import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.research_console.api import create_app
from src.research_console.run_store import RunLogger, append_jsonl
from src.research_console.schema import EpochMetrics, ProtocolRecord
from tests.test_research_console_schema import valid_metadata


class ResearchConsoleApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runs_root = self.root / "outputs" / "runs"
        self.client = TestClient(create_app(repo_root=self.root, runs_root=self.runs_root))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _create_run(self, run_id: str, *, valid: bool, manifest_hash: str) -> RunLogger:
        metadata = valid_metadata().model_copy(
            update={
                "run_id": run_id,
                "provenance": valid_metadata().provenance.model_copy(
                    update={
                        "dataset_manifest_sha256": manifest_hash,
                        "protocol": ProtocolRecord(valid=valid, failed_gates=[] if valid else ["split_overlap"]),
                    }
                ),
            }
        )
        logger = RunLogger(self.runs_root, metadata)
        logger.start()
        logger.log_epoch(
            EpochMetrics(
                timestamp=datetime.now(timezone.utc),
                epoch=1,
                train_loss=0.8,
                val_loss=0.7,
                pixel_accuracy=0.8,
                mean_iou=0.6 if valid else 0.99,
            )
        )
        logger.finish()
        return logger

    def test_overview_excludes_invalid_run_from_best_ranking(self) -> None:
        self._create_run("valid-run", valid=True, manifest_hash="a" * 64)
        self._create_run("invalid-run", valid=False, manifest_hash="a" * 64)

        response = self.client.get("/api/overview")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["expert_test_locked"])
        self.assertEqual(response.json()["best_protocol_valid_validation_run"]["run_id"], "valid-run")

    def test_compare_warns_on_incompatible_provenance(self) -> None:
        self._create_run("first-run", valid=True, manifest_hash="a" * 64)
        self._create_run("second-run", valid=True, manifest_hash="b" * 64)

        response = self.client.get("/api/compare", params=[("run_id", "first-run"), ("run_id", "second-run")])

        self.assertEqual(response.status_code, 200)
        self.assertIn("different dataset manifest hashes", " ".join(response.json()["warnings"]))

    def test_run_cards_expose_review_metadata(self) -> None:
        metadata = valid_metadata().model_copy(
            update={
                "run_id": "review-run",
                "hypothesis": "Class weighting improves big-rock recall.",
                "researcher_notes": "Inspect false positives before promotion.",
                "tags": ["weighted", "ablation"],
                "provenance": valid_metadata().provenance.model_copy(update={"random_seeds": {"torch": 17}}),
            }
        )
        logger = RunLogger(self.runs_root, metadata)
        logger.start()
        logger.finish()

        response = self.client.get("/api/runs")
        card = next(item for item in response.json()["runs"] if item["run_id"] == "review-run")

        self.assertEqual(card["hypothesis"], "Class weighting improves big-rock recall.")
        self.assertEqual(card["researcher_notes"], "Inspect false positives before promotion.")
        self.assertEqual(card["random_seeds"], {"torch": 17})

    def test_overview_refuses_cross_manifest_best_ranking(self) -> None:
        self._create_run("cohort-one", valid=True, manifest_hash="a" * 64)
        self._create_run("cohort-two", valid=True, manifest_hash="b" * 64)

        response = self.client.get("/api/overview")

        self.assertIsNone(response.json()["best_protocol_valid_validation_run"])
        self.assertIn("incompatible manifest/split cohorts", response.json()["ranking_warning"])

    def test_artifact_endpoint_blocks_path_escape(self) -> None:
        logger = self._create_run("asset-run", valid=True, manifest_hash="a" * 64)
        artifact = logger.run_dir / "artifacts" / "sample.txt"
        artifact.write_text("safe asset", encoding="utf-8")

        safe_response = self.client.get("/api/runs/asset-run/artifacts/artifacts/sample.txt")
        escaped_response = self.client.get("/api/runs/asset-run/artifacts/artifacts/../metadata.json")

        self.assertEqual(safe_response.status_code, 200)
        self.assertEqual(safe_response.text, "safe asset")
        self.assertEqual(escaped_response.status_code, 404)

    def test_health_degrades_without_gpu(self) -> None:
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertIn("gpu_available", response.json()["system"])

    def test_event_polling_returns_new_durable_events_without_reload(self) -> None:
        logger = self._create_run("event-run", valid=True, manifest_hash="a" * 64)
        initial = self.client.get("/api/runs/event-run/events")
        logger.log_batch(epoch=2, batch=1, total_batches=2, loss=0.5)
        updated = self.client.get("/api/runs/event-run/events", params={"after": initial.json()["next"]})

        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["events"][0]["event"]["event_type"], "batch")

    def test_samples_order_lowest_iou_and_highest_loss_or_uncertainty(self) -> None:
        logger = self._create_run("sample-run", valid=True, manifest_hash="a" * 64)
        index_path = logger.run_dir / "artifacts" / "prediction_index.jsonl"
        append_jsonl(index_path, {"sample_id": "low-iou", "split": "validation", "image_iou": 0.1, "loss": 0.2, "uncertainty": 0.4})
        append_jsonl(index_path, {"sample_id": "high-loss", "split": "train", "image_iou": 0.8, "loss": 0.9, "uncertainty": 0.1})
        append_jsonl(index_path, {"sample_id": "high-uncertainty", "split": "validation", "image_iou": 0.5, "loss": 0.4, "uncertainty": 0.95})

        by_iou = self.client.get("/api/runs/sample-run/samples", params={"sort_by": "image_iou"}).json()
        by_loss = self.client.get("/api/runs/sample-run/samples", params={"sort_by": "loss"}).json()
        by_uncertainty = self.client.get("/api/runs/sample-run/samples", params={"sort_by": "uncertainty"}).json()
        validation_page = self.client.get("/api/runs/sample-run/samples", params={"split": "validation", "offset": 1, "limit": 1}).json()

        self.assertEqual(by_iou["samples"][0]["sample_id"], "low-iou")
        self.assertEqual(by_loss["samples"][0]["sample_id"], "high-loss")
        self.assertEqual(by_uncertainty["samples"][0]["sample_id"], "high-uncertainty")
        self.assertEqual(by_iou["available_splits"], ["train", "validation"])
        self.assertEqual(validation_page["total"], 2)
        self.assertEqual(validation_page["samples"][0]["sample_id"], "high-uncertainty")

    def test_provenance_reports_group_overlap_as_failed_gate(self) -> None:
        manifests = self.root / "artifacts" / "manifests"
        splits = manifests / "splits"
        splits.mkdir(parents=True)
        header = "dataset_version,dataset_doi,stable_source_image_id,sequence_id,label_role,label_scheme,exclusion_reason,per_class_pixel_counts_json\n"
        row = '0.6,10.5281/zenodo.15995036,image-a,sequence-a,crowdsourced_train,NAV,,"{""0"": 4}"\n'
        (manifests / "ai4mars_dataset_manifest.csv").write_text(header + row, encoding="utf-8")
        split_header = "stable_source_image_id,sequence_id,label_role\n"
        (splits / "train_nav.csv").write_text(split_header + "image-a,sequence-a,crowdsourced_train\n", encoding="utf-8")
        (splits / "val_nav.csv").write_text(split_header + "image-a,sequence-a,crowdsourced_train\n", encoding="utf-8")
        (splits / "test_min1_100agree_nav.csv").write_text(
            split_header + "image-test,sequence-test,expert_gold_test\n",
            encoding="utf-8",
        )

        response = self.client.get("/api/provenance")

        self.assertEqual(response.status_code, 200)
        grouped_gate = next(gate for gate in response.json()["gates"] if gate["name"] == "grouped_split_isolation")
        self.assertFalse(grouped_gate["passed"])

    def test_legacy_artifact_is_visible_but_invalid_for_ranking(self) -> None:
        self._create_run("current-run", valid=True, manifest_hash="a" * 64)
        legacy = self.root / "artifacts" / "runs" / "old-run"
        legacy.mkdir(parents=True)
        (legacy / "config.json").write_text('{"model_name": "Unet/resnet34", "loss_name": "CE"}', encoding="utf-8")
        (legacy / "metrics.json").write_text('{"epoch": 3, "mean_iou": 0.99}', encoding="utf-8")

        response = self.client.get("/api/runs")

        legacy_card = next(card for card in response.json()["runs"] if card["run_id"] == "legacy-old-run")
        self.assertFalse(legacy_card["protocol_valid"])
        self.assertTrue(legacy_card["legacy"])


if __name__ == "__main__":
    unittest.main()