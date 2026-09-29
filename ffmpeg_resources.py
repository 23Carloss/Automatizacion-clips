"""Shared CPU controls for FFmpeg encoding processes."""
from __future__ import annotations

import asyncio
import os
import subprocess
from contextlib import asynccontextmanager
from typing import AsyncIterator


class FfmpegCpuLimiter:
    """Prevent independent clip jobs from saturating the CPU at the same time."""

    def __init__(self, max_concurrent_encodes: int = 1) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent_encodes)

    @asynccontextmanager
    async def encoding_slot(self) -> AsyncIterator[None]:
        async with self._semaphore:
            yield


def low_priority_process_kwargs(enabled: bool) -> dict[str, int]:
    """Return safe subprocess flags for lower-priority encoders on Windows."""
    if enabled and os.name == "nt":
        return {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS}
    return {}
