import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai4mars import paper_train
from ai4mars.cuda import cuda_architecture_is_supported, run_cuda_preflight


class _FakeTensor:
    def __init__(self, result: int = 8):
        self.result = result

    def sum(self):
        return self

    def item(self):
        return self.result


class CudaPreflightTests(unittest.TestCase):
    def _torch(self, *, architectures=("sm_60",), operation_error=None):
        cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "Tesla P100-PCIE-16GB",
            get_device_capability=lambda index: (6, 0),
            get_arch_list=lambda: list(architectures),
            synchronize=Mock(),
        )

        def ones(*args, **kwargs):
            if operation_error is not None:
                raise operation_error
            return _FakeTensor()

        return SimpleNamespace(
            __version__="2.7.1+cu126",
            version=SimpleNamespace(cuda="12.6"),
            cuda=cuda,
            ones=Mock(side_effect=ones),
        )

    def test_supported_p100_reports_versions_and_executes_cuda_reduction(self):
        torch_module = self._torch()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            report = run_cuda_preflight(torch_module)

        self.assertEqual(report.target_architecture, "sm_60")
        self.assertIn("Tesla P100-PCIE-16GB", output.getvalue())
        self.assertIn("Compute capability: 6.0 (sm_60)", output.getvalue())
        self.assertIn("PyTorch: 2.7.1+cu126", output.getvalue())
        self.assertIn("PyTorch CUDA: 12.6", output.getvalue())
        self.assertIn("Compiled CUDA architectures: sm_60", output.getvalue())
        torch_module.ones.assert_called_once_with(8, device="cuda:0")
        torch_module.cuda.synchronize.assert_called_once_with(0)

    def test_unsupported_p100_fails_after_operation_with_kaggle_remediation(self):
        torch_module = self._torch(architectures=("sm_70", "sm_80"))

        with self.assertRaisesRegex(RuntimeError, "does not include sm_60") as raised:
            run_cuda_preflight(torch_module)

        self.assertIn("requirements-kaggle.txt", str(raised.exception))
        self.assertIn("do not install requirements.txt", str(raised.exception))
        torch_module.ones.assert_called_once()

    def test_cuda_operation_failure_has_actionable_diagnostics(self):
        torch_module = self._torch(operation_error=RuntimeError("no kernel image"))

        with self.assertRaisesRegex(RuntimeError, "no kernel image") as raised:
            run_cuda_preflight(torch_module)

        self.assertIn("Compiled architectures: sm_60", str(raised.exception))
        self.assertIn("restart with a clean GPU session", str(raised.exception))

    def test_unavailable_cuda_fails_before_querying_device(self):
        torch_module = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

        with self.assertRaisesRegex(RuntimeError, "Attach a GPU accelerator"):
            run_cuda_preflight(torch_module)

    def test_ptx_architecture_supports_newer_compute_capability(self):
        self.assertTrue(cuda_architecture_is_supported((8, 0), ("compute_70",)))
        self.assertFalse(cuda_architecture_is_supported((6, 0), ("sm_70",)))
        self.assertIsNone(cuda_architecture_is_supported((6, 0), ()))

    def test_training_preflight_runs_before_outputs_or_dataset_audit(self):
        paths = SimpleNamespace(accelerator="cuda", ensure_writable_roots=Mock())
        config = {"runtime": {}, "data": {}, "training": {"mixed_precision": True}, "paper_model_spec": Mock()}

        with (
            patch.object(paper_train, "load_and_validate_config", return_value=config),
            patch.object(paper_train, "resolve_runtime_paths", return_value=paths),
            patch.object(paper_train, "run_cuda_preflight", side_effect=RuntimeError("incompatible CUDA")),
            patch.object(paper_train, "summarize_reproduction_manifests") as summarize,
            self.assertRaisesRegex(RuntimeError, "incompatible CUDA"),
        ):
            paper_train.run_training(SimpleNamespace(config="config.yaml"))

        paths.ensure_writable_roots.assert_not_called()
        summarize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
