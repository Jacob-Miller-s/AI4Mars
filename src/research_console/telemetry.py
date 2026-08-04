"""Low-overhead local host telemetry with optional GPU support."""

from datetime import datetime, timezone

import psutil

from .schema import SystemMetrics


class SystemTelemetrySampler:
    """Collect host metrics without importing PyTorch or initializing CUDA."""

    def collect(
        self,
        *,
        gpu_memory_allocated_bytes: int | None = None,
        gpu_memory_reserved_bytes: int | None = None,
    ) -> SystemMetrics:
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        return SystemMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=psutil.cpu_percent(interval=None),
            ram_percent=memory.percent,
            ram_used_bytes=memory.used,
            disk_read_bytes=disk_io.read_bytes if disk_io else None,
            disk_write_bytes=disk_io.write_bytes if disk_io else None,
            gpu_memory_allocated_bytes=gpu_memory_allocated_bytes,
            gpu_memory_reserved_bytes=gpu_memory_reserved_bytes,
        )