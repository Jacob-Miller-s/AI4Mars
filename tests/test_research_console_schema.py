import unittest

from pydantic import ValidationError

from src.research_console.schema import (
    ArtifactRef,
    ModelRecord,
    ProtocolRecord,
    ProvenanceRecord,
    RunMetadata,
    RunStatus,
    SplitRole,
    TrainingRecord,
)


def valid_metadata() -> RunMetadata:
    return RunMetadata(
        run_id="demo-001",
        experiment_name="Synthetic smoke training",
        status=RunStatus.RUNNING,
        provenance=ProvenanceRecord(
            dataset_name="AI4Mars",
            dataset_version="0.6",
            source_record="10.5281/zenodo.15995036",
            dataset_manifest_sha256="a" * 64,
            split_manifest_hashes={"train": "b" * 64, "val": "c" * 64},
            split_role=SplitRole.CROWDSOURCED_VALIDATION,
            protocol=ProtocolRecord(valid=True),
            git_commit="d" * 40,
            git_branch="feat/research-dashboard",
        ),
        model=ModelRecord(
            name="Unet",
            encoder="resnet34",
            pretrained_weights="imagenet",
            input_resolution=(256, 256),
        ),
        training=TrainingRecord(
            optimizer="AdamW",
            loss="CrossEntropyLoss(ignore_index=255)",
            batch_size=4,
            epochs=3,
        ),
        artifact_refs=[ArtifactRef(path="artifacts/curve.png", kind="figure")],
    )


class RunSchemaTests(unittest.TestCase):
    def test_metadata_serializes_portable_provenance(self) -> None:
        metadata = valid_metadata()

        payload = metadata.model_dump(mode="json")

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["provenance"]["split_role"], "crowdsourced_validation")
        self.assertEqual(payload["artifact_refs"][0]["path"], "artifacts/curve.png")

    def test_artifact_rejects_absolute_and_traversal_paths(self) -> None:
        for path in ("C:/outside.png", "/outside.png", "artifacts/../outside.png", "artifacts\\curve.png"):
            with self.subTest(path=path):
                with self.assertRaises(ValidationError):
                    ArtifactRef(path=path, kind="figure")


if __name__ == "__main__":
    unittest.main()