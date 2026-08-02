"""Low-overhead local host telemetry with optional NVML GPU support."""

from __future__ import annotations

from datetime import datetime, timezone

import psutil

from .schema import SystemMetrics


class SystemTelemetrySampler:
    """Collect host metrics without importing PyTorch or initializing CUDA."""

    def __init__(self) -> None:
        self._nvml = None
        self._gpu_handle = None
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None
            self._gpu_handle = None

    def collect(
        self,
        *,
        gpu_memory_allocated_bytes: int | None = None,
        gpu_memory_reserved_bytes: int | None = None,
    ) -> SystemMetrics:
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        payload = {
            "timestamp": datetime.now(timezone.utc),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": memory.percent,
            "ram_used_bytes": memory.used,
            "disk_read_bytes": disk_io.read_bytes if disk_io else None,
            "disk_write_bytes": disk_io.write_bytes if disk_io else None,
            "gpu_memory_allocated_bytes": gpu_memory_allocated_bytes,
            "gpu_memory_reserved_bytes": gpu_memory_reserved_bytes,
            "gpu_available": False,
        }
        if self._nvml is not None and self._gpu_handle is not None:
            try:
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_memory = self._nvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                payload.update(
                    {
                        "gpu_utilization_percent": utilization.gpu,
                        "gpu_temperature_celsius": self._nvml.nvmlDeviceGetTemperature(
                            self._gpu_handle,
                            self._nvml.NVML_TEMPERATURE_GPU,
                        ),
                        "gpu_memory_used_bytes": gpu_memory.used,
                        "gpu_memory_total_bytes": gpu_memory.total,
                        "gpu_available": True,
                    }
                )
            except Exception:
                # A driver reset or unavailable NVML must not interrupt training.
                pass
        return SystemMetrics(**payload)