"""Resolve Twitch with Streamlink and emit low-rate OpenCV frames via FFmpeg."""
from __future__ import annotations

import asyncio
import contextlib
import time
from typing import AsyncIterator

import numpy as np
from streamlink import Streamlink

from config import Settings


class StreamMonitor:
    """Keeps a small live segment cache; it never downloads an entire stream."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.source_url: str | None = None
        self._frame_process: asyncio.subprocess.Process | None = None
        self._cache_process: asyncio.subprocess.Process | None = None
        self._cache_prune_task: asyncio.Task[None] | None = None

    def resolve_stream(self) -> str:
        session = Streamlink()
        streams = session.streams(self.settings.twitch_url)
        stream = streams.get(self.settings.twitch_quality) or streams.get("best")
        if stream is None:
            raise RuntimeError("No playable Twitch stream was returned by Streamlink.")
        # HLSStream exposes to_url; Streamlink owns Twitch access/resolution.
        return stream.to_url()

    async def start(self) -> None:
        self.settings.prepare_directories()
        self.source_url = await asyncio.to_thread(self.resolve_stream)
        if self.settings.source_kind == "live":
            await self._start_rolling_cache()

    def _frame_timestamp(self, index: int) -> float:
        """Use arrival time for live HLS and timeline position for seekable VODs.

        Twitch HLS normally arrives several seconds behind wall clock. Deriving
        live timestamps from process start incorrectly labels that transport
        latency as OCR backlog and can cause every keyframe to be discarded.
        Arrival time also matches the rolling segments' filesystem timestamps.
        """
        if self.settings.source_kind == "live":
            return time.time()
        return self.settings.vod_start_offset_seconds + index / self.settings.sample_fps

    async def _start_rolling_cache(self) -> None:
        pattern = str(self.settings.cache_dir / "live_%08d.ts")
        command = [
            self.settings.ffmpeg_binary, "-hide_banner", "-loglevel", "warning", "-y", "-i", self.source_url,
            "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-f", "segment", "-segment_time", "1",
            "-segment_format", "mpegts", "-reset_timestamps", "1", pattern,
        ]
        self._cache_process = await asyncio.create_subprocess_exec(*command)
        self._cache_prune_task = asyncio.create_task(
            self._prune_cache_loop(), name="live-cache-pruner"
        )

    async def _prune_cache_loop(self) -> None:
        """Bound disk usage even when no visual events are detected."""
        try:
            while self._cache_process and self._cache_process.returncode is None:
                cutoff = time.time() - self.settings.buffer_seconds
                for segment in self.settings.cache_dir.glob("live_*.ts"):
                    try:
                        if segment.stat().st_mtime < cutoff:
                            segment.unlink(missing_ok=True)
                    except (FileNotFoundError, PermissionError):
                        continue
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            raise

    async def frames(self) -> AsyncIterator[tuple[np.ndarray, float]]:
        if not self.source_url:
            raise RuntimeError("Call start() before requesting frames.")
        fps = str(self.settings.sample_fps)
        command = [
            self.settings.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-i", self.source_url,
            "-an", "-vf", f"fps={fps},scale=1920:1080", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
        ]
        self._frame_process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        # Frame extraction stays lightweight. Motion-selected keyframes are put
        # into the separate OCR queue by main.py, so this pipe never waits for
        # EasyOCR and never replaces a whole batch of potentially useful frames.
        frame_size = 1920 * 1080 * 3
        index = 0
        assert self._frame_process.stdout
        try:
            while True:
                raw = await self._frame_process.stdout.readexactly(frame_size)
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((1080, 1920, 3)).copy()
                timestamp = self._frame_timestamp(index)
                index += 1
                yield frame, timestamp
        except asyncio.IncompleteReadError:
            return
        finally:
            await self.stop_frames()

    async def stop_frames(self) -> None:
        if self._frame_process and self._frame_process.returncode is None:
            self._frame_process.terminate()
            with contextlib.suppress(ProcessLookupError):
                await self._frame_process.wait()
        self._frame_process = None

    async def close(self) -> None:
        await self.stop_frames()
        if self._cache_prune_task:
            self._cache_prune_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cache_prune_task
        self._cache_prune_task = None
        if self._cache_process and self._cache_process.returncode is None:
            self._cache_process.terminate()
            with contextlib.suppress(ProcessLookupError):
                await self._cache_process.wait()
        self._cache_process = None
