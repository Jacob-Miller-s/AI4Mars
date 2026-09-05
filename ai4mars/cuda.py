"""CUDA runtime compatibility checks for GPU reproduction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


KAGGLE_P100_REMEDIATION = (
    "On Kaggle P100, restart with a clean GPU session and install "
    "requirements-kaggle.txt before importing torch; do not install requirements.txt."
)


@dataclass(frozen=True)
class CudaPreflightReport:
    gpu_model: str
    compute_capability: tuple[int, int]
    pytorch_version: str
    pytorch_cuda_version: str | None
    compiled_architectures: tuple[str, ...]

    @property
    def target_architecture(self) -> str:
        return f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"


def cuda_architecture_is_supported(
    compute_capability: tuple[int, int],
    compiled_architectures: Sequence[str],
) -> bool | None:
    """Return wheel support for a capability, or None when no build list is exposed."""
    if not compiled_architectures:
        return None
    target = compute_capability[0] * 10 + compute_capability[1]
    if f"sm_{target}" in compiled_architectures:
        return True
    ptx_targets = [
        int(architecture.removeprefix("compute_"))
        for architecture in compiled_architectures
        if architecture.removeprefix("compute_").isdigit() and architecture.startswith("compute_")
    ]
    return any(ptx_target <= target for ptx_target in ptx_targets)


def format_cuda_preflight(report: CudaPreflightReport) -> str:
    architectures = ", ".join(report.compiled_architectures) or "not reported"
    capability = ".".join(str(value) for value in report.compute_capability)
    return (
        "CUDA preflight:\n"
        f"  GPU model: {report.gpu_model}\n"
        f"  Compute capability: {capability} ({report.target_architecture})\n"
        f"  PyTorch: {report.pytorch_version}\n"
        f"  PyTorch CUDA: {report.pytorch_cuda_version or 'none'}\n"
        f"  Compiled CUDA architectures: {architectures}"
    )


def run_cuda_preflight(torch_module: Any = torch, *, device_index: int = 0) -> CudaPreflightReport:
    """Report the CUDA contract and prove that a real CUDA reduction executes."""
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA preflight failed: torch.cuda.is_available() is false. "
            "Attach a GPU accelerator or select --device cpu."
        )

    report = CudaPreflightReport(
        gpu_model=str(torch_module.cuda.get_device_name(device_index)),
        compute_capability=tuple(torch_module.cuda.get_device_capability(device_index)),
        pytorch_version=str(torch_module.__version__),
        pytorch_cuda_version=torch_module.version.cuda,
        compiled_architectures=tuple(torch_module.cuda.get_arch_list()),
    )
    print(format_cuda_preflight(report), flush=True)

    operation_error: Exception | None = None
    try:
        result = torch_module.ones(8, device=f"cuda:{device_index}").sum().item()
        torch_module.cuda.synchronize(device_index)
        if result != 8:
            raise RuntimeError(f"unexpected CUDA reduction result {result!r}")
    except Exception as error:
        operation_error = error

    architecture_supported = cuda_architecture_is_supported(
        report.compute_capability,
        report.compiled_architectures,
    )
    if architecture_supported is False or operation_error is not None:
        reason = (
            f"the wheel does not include {report.target_architecture}"
            if architecture_supported is False
            else f"a CUDA tensor reduction could not execute ({operation_error})"
        )
        raise RuntimeError(
            f"CUDA preflight failed for {report.gpu_model}: {reason}. "
            f"Compiled architectures: {', '.join(report.compiled_architectures) or 'not reported'}. "
            f"{KAGGLE_P100_REMEDIATION}"
        ) from operation_error

    return report
