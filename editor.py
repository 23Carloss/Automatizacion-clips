"""FFmpeg-only 9:16 editor with Apex HUD overlays."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from config import Settings
from ffmpeg_resources import FfmpegCpuLimiter, low_priority_process_kwargs
from video_encoders import encoder_arguments, encoder_candidates, encoder_is_available

LOGGER = logging.getLogger("apex_clipper.editor")
CANCELLATION_POLL_SECONDS = 1.0

CancelCheck = Callable[[], Awaitable[bool]]
ProgressCallback = Callable[[float], Awaitable[None]]


class RenderCancelled(RuntimeError):
    """Raised after an active FFmpeg render has been stopped by the user."""


class ApexVerticalEditor:
    def __init__(
        self, settings: Settings, cpu_limiter: FfmpegCpuLimiter | None = None,
    ) -> None:
        self.settings = settings
        self._encoder: str | None = None
        self._encoder_chain = encoder_candidates(
            settings.ffmpeg_vertical_encoder,
            ("h264_amf", "h264_qsv", "h264_nvenc", "libx264"),
        )
        self._render_lock = asyncio.Lock()
        self._cpu_limiter = cpu_limiter or FfmpegCpuLimiter(
            settings.ffmpeg_max_concurrent_encodes
        )

    async def render(
        self,
        input_path: Path,
        *,
        cancel_requested: CancelCheck | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """Render one clip at a time and fall back when a hardware encoder fails."""
        async with self._render_lock:
            return await self._render_locked(
                input_path, cancel_requested, progress_callback
            )

    async def _render_locked(
        self,
        input_path: Path,
        cancel_requested: CancelCheck | None,
        progress_callback: ProgressCallback | None,
    ) -> Path:
        output_path = input_path.with_name(input_path.stem.replace("_source", "") + "_vertical.mp4")
        duration: float | None = None
        if progress_callback is not None:
            try:
                duration = await self._probe_duration(input_path)
            except Exception:
                LOGGER.exception("Could not determine clip duration for render progress.")
            await self._report_progress(progress_callback, 0.0)
        encoder = await self._select_encoder()
        gameplay, top = self.settings.gameplay_crop, self.settings.top_hud_roi
        health, ammo = self.settings.health_hud_roi, self.settings.ammo_hud_roi
        complex_filter = (
            "[0:v]scale=1920:1080:flags=lanczos,split=5[v0][v1][v2][v3][v4];"
            "[v4]scale=270:480:force_original_aspect_ratio=increase:flags=lanczos,"
            "crop=270:480,boxblur=10:1,scale=1080:1920:flags=lanczos[back];"
            f"[v0]crop={gameplay.width}:{gameplay.height}:{gameplay.x}:{gameplay.y},"
            "scale=1080:-2:flags=lanczos[game];"
            "[back][game]overlay=(W-w)/2:(H-h)/2[base];"
            f"[v1]crop={top.width}:{top.height}:{top.x}:{top.y}[top];"
            f"[v2]crop={health.width}:{health.height}:{health.x}:{health.y}[health];"
            f"[v3]crop={ammo.width}:{ammo.height}:{ammo.x}:{ammo.y}[ammo];"
            "[base][top]overlay=290:50[layer1];"
            "[layer1][health]overlay=30:1750[layer2];"
            "[layer2][ammo]overlay=600:1750,format=yuv420p[out]"
        )
        try:
            while True:
                stderr = await self._run_render(
                    input_path,
                    output_path,
                    complex_filter,
                    encoder,
                    cancel_requested=cancel_requested,
                    duration=duration,
                    progress_callback=progress_callback,
                )
                if stderr is None:
                    await self._report_progress(progress_callback, 100.0)
                    return output_path
                fallback = await self._select_fallback_encoder(encoder)
                if fallback is None:
                    raise RuntimeError(f"FFmpeg vertical render failed: {stderr}")
                LOGGER.warning(
                    "%s failed during vertical render; retrying with %s: %s",
                    encoder,
                    fallback,
                    stderr,
                )
                encoder = fallback
        except (RenderCancelled, asyncio.CancelledError):
            await self._remove_partial_output(output_path)
            raise

    async def _run_render(
        self, input_path: Path, output_path: Path, complex_filter: str, encoder: str,
        cancel_requested: CancelCheck | None = None,
        duration: float | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> str | None:
        command = self._build_render_command(
            input_path, output_path, complex_filter, encoder
        )
        async with self._cpu_limiter.encoding_slot():
            if await self._cancellation_requested(cancel_requested):
                raise RenderCancelled("Video processing was cancelled before FFmpeg started.")
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **low_priority_process_kwargs(self.settings.ffmpeg_low_priority),
            )
            if progress_callback is not None and duration:
                assert process.stdout is not None
                assert process.stderr is not None
                progress_task = asyncio.create_task(
                    self._consume_ffmpeg_progress(
                        process.stdout, duration, progress_callback
                    )
                )
                stderr_task = asyncio.create_task(process.stderr.read())
                wait_task = asyncio.create_task(process.wait())
                try:
                    while not wait_task.done():
                        await asyncio.wait(
                            {wait_task}, timeout=CANCELLATION_POLL_SECONDS
                        )
                        if wait_task.done() or cancel_requested is None:
                            continue
                        if await self._cancellation_requested(cancel_requested):
                            LOGGER.info(
                                "Stopping FFmpeg after a dashboard cancellation request."
                            )
                            await self._terminate_process(process, wait_task)
                            await asyncio.gather(
                                progress_task, stderr_task, return_exceptions=True
                            )
                            raise RenderCancelled(
                                "Video processing was cancelled from the dashboard."
                            )
                    await wait_task
                    await progress_task
                    stderr = await stderr_task
                    return (
                        stderr.decode(errors="replace").strip()
                        if process.returncode != 0 else None
                    )
                except asyncio.CancelledError:
                    await self._terminate_process(process, wait_task)
                    await asyncio.gather(
                        progress_task, stderr_task, return_exceptions=True
                    )
                    raise

            communicate_task = asyncio.create_task(process.communicate())
            try:
                while not communicate_task.done():
                    await asyncio.wait(
                        {communicate_task}, timeout=CANCELLATION_POLL_SECONDS
                    )
                    if communicate_task.done() or cancel_requested is None:
                        continue
                    if await self._cancellation_requested(cancel_requested):
                        LOGGER.info("Stopping FFmpeg after a dashboard cancellation request.")
                        await self._terminate_process(process, communicate_task)
                        raise RenderCancelled(
                            "Video processing was cancelled from the dashboard."
                        )
                _, stderr = await communicate_task
                return stderr.decode(errors="replace").strip() if process.returncode != 0 else None
            except asyncio.CancelledError:
                await self._terminate_process(process, communicate_task)
                raise

    @staticmethod
    async def _cancellation_requested(cancel_requested: CancelCheck | None) -> bool:
        if cancel_requested is None:
            return False
        try:
            return await cancel_requested()
        except Exception:
            LOGGER.exception("Could not check whether the render was cancelled.")
            return False

    async def _probe_duration(self, input_path: Path) -> float:
        process = await asyncio.create_subprocess_exec(
            self.settings.ffprobe_binary,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"FFprobe could not inspect input duration: "
                f"{stderr.decode(errors='replace').strip()}"
            )
        duration = float(stdout.decode().strip())
        if duration <= 0:
            raise ValueError("FFprobe returned a non-positive input duration.")
        return duration

    async def _consume_ffmpeg_progress(
        self,
        stream: asyncio.StreamReader,
        duration: float,
        progress_callback: ProgressCallback,
    ) -> None:
        last_percentage = -1.0
        while line := await stream.readline():
            key, separator, raw_value = line.decode(errors="replace").strip().partition("=")
            if not separator:
                continue
            if key in {"out_time_us", "out_time_ms"}:
                try:
                    elapsed = int(raw_value) / 1_000_000
                except ValueError:
                    continue
                percentage = min(99.0, max(0.0, elapsed / duration * 100))
                if percentage - last_percentage >= 0.5:
                    last_percentage = percentage
                    await self._report_progress(progress_callback, percentage)
            elif key == "progress" and raw_value == "end":
                await self._report_progress(progress_callback, 100.0)

    @staticmethod
    async def _report_progress(
        progress_callback: ProgressCallback | None, percentage: float,
    ) -> None:
        if progress_callback is None:
            return
        try:
            await progress_callback(percentage)
        except Exception:
            LOGGER.exception("Could not persist vertical render progress.")

    @staticmethod
    async def _await_process_task(task: asyncio.Task, timeout: float) -> bool:
        """Wait for FFmpeg even while asyncio.run is propagating cancellation."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        current = asyncio.current_task()
        while not task.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
            except asyncio.CancelledError:
                # Ctrl+C may cancel the main task more than once. Consume those
                # requests only during process cleanup; the caller re-raises the
                # original cancellation after FFmpeg has released its files.
                if current is not None:
                    current.uncancel()
                continue
            except asyncio.TimeoutError:
                return False
        return True

    @classmethod
    async def _terminate_process(cls, process, communicate_task: asyncio.Task) -> None:
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        if await cls._await_process_task(communicate_task, 5):
            return
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        if not await cls._await_process_task(communicate_task, 5):
            LOGGER.warning("FFmpeg did not exit within 10 seconds after cancellation.")

    @staticmethod
    async def _remove_partial_output(output_path: Path) -> None:
        """Allow Windows a short grace period to release a terminated MP4."""
        for attempt in range(20):
            try:
                output_path.unlink(missing_ok=True)
                return
            except PermissionError:
                if attempt == 19:
                    LOGGER.warning("Could not remove locked partial output %s.", output_path)
                    return
                await asyncio.sleep(0.1)

    def _build_render_command(
        self, input_path: Path, output_path: Path, complex_filter: str, encoder: str
    ) -> list[str]:
        """Build a high-quality master command without inventing extra frames."""
        codec_args = encoder_arguments(
            encoder,
            quality=self.settings.ffmpeg_vertical_quality,
            threads=self.settings.ffmpeg_encoding_threads,
            fast=False,
            target_kbps=self.settings.vertical_video_target_kbps,
            max_kbps=self.settings.vertical_video_max_kbps,
        )
        return [
            self.settings.ffmpeg_binary, "-hide_banner", "-loglevel", "error",
            "-progress", "pipe:1", "-nostats", "-i", str(input_path),
            "-filter_complex_threads", str(self.settings.ffmpeg_encoding_threads),
            "-filter_complex", complex_filter, "-map", "[out]", "-map", "0:a?",
            "-c:a", "aac", "-b:a", "128k", *codec_args,
            "-fps_mode", "passthrough", "-pix_fmt", "yuv420p",
            "-color_range", "tv", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709",
            "-movflags", "+faststart", "-y", str(output_path),
        ]

    async def _select_encoder(self) -> str:
        """Select the first encoder that can initialize on this computer."""
        if self._encoder is not None:
            return self._encoder
        for encoder in self._encoder_chain:
            if encoder == "libx264" or await encoder_is_available(
                self.settings.ffmpeg_binary, encoder
            ):
                self._encoder = encoder
                LOGGER.info("Vertical clips will use %s.", encoder)
                return encoder
        raise RuntimeError("No usable H.264 encoder was found.")

    async def _select_fallback_encoder(self, failed: str) -> str | None:
        """Move to the next configured encoder after a runtime driver failure."""
        try:
            start = self._encoder_chain.index(failed) + 1
        except ValueError:
            start = len(self._encoder_chain) - 1
        for encoder in self._encoder_chain[start:]:
            if encoder == "libx264" or await encoder_is_available(
                self.settings.ffmpeg_binary, encoder
            ):
                self._encoder = encoder
                return encoder
        return None
