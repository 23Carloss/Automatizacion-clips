"""Create a horizontal source clip around a detected visual event."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path

from config import Settings
from ffmpeg_resources import FfmpegCpuLimiter, low_priority_process_kwargs
from video_encoders import encoder_arguments, encoder_candidates, encoder_is_available

LOGGER = logging.getLogger("apex_clipper.clipper")


class Clipper:
    def __init__(
        self, settings: Settings, cpu_limiter: FfmpegCpuLimiter | None = None,
    ) -> None:
        self.settings = settings
        self._snapshot_lock = asyncio.Lock()
        self._source_encoder: str | None = None
        self._source_encoder_chain = encoder_candidates(
            settings.ffmpeg_source_encoder,
            ("h264_qsv", "h264_amf", "h264_nvenc", "libx264"),
        )
        self._cpu_limiter = cpu_limiter or FfmpegCpuLimiter(
            settings.ffmpeg_max_concurrent_encodes
        )

    async def create_clip(self, event_timestamp: float, source_url: str, event_kind: str = "") -> Path:
        """Produce a fixed-window clip for compatibility with existing callers."""
        pre_seconds, post_seconds = self.settings.clip_window(event_kind)
        return await self.create_clip_window(
            max(0.0, event_timestamp - pre_seconds),
            event_timestamp + post_seconds,
            source_url,
            stamp_timestamp=event_timestamp,
        )

    async def create_clip_window(
        self,
        start_timestamp: float,
        end_timestamp: float,
        source_url: str,
        *,
        stamp_timestamp: float | None = None,
        reservation_id: str | None = None,
        name_prefix: str = "apex",
        normalize_to_1080: bool = False,
    ) -> Path:
        """Produce one 16:9 MP4 for an already-grouped event window.

        A VOD is seekable by timeline. A live source is assembled from one-second
        rolling transport-stream chunks, so the pre-event content is retained
        without retaining the full broadcast.
        """
        if end_timestamp <= start_timestamp:
            raise ValueError("Clip end timestamp must be greater than its start timestamp.")
        if not re.fullmatch(r"[A-Za-z0-9_]+", name_prefix):
            raise ValueError("Invalid clip filename prefix.")
        stamp = int(stamp_timestamp if stamp_timestamp is not None else start_timestamp)
        destination = self.settings.clips_dir / f"{name_prefix}_{stamp}_source.mp4"
        duration = end_timestamp - start_timestamp
        try:
            if self.settings.source_kind == "vod":
                await self._run_source_encode([
                    "-ss", f"{max(0.0, start_timestamp):.3f}", "-i", source_url, "-t",
                    str(duration),
                    "-map", "0:v:0", "-map", "0:a?",
                    *(
                        ["-vf", "scale=1920:1080:flags=lanczos"]
                        if normalize_to_1080 else []
                    ),
                ], ["-c:a", "aac", "-movflags", "+faststart", "-y", str(destination)])
            else:
                # Give FFmpeg enough time to close the segment containing the final
                # post-event frames before taking a stable snapshot of the cache.
                ready_at = end_timestamp + 2.5
                await asyncio.sleep(max(0.0, ready_at - time.time()))
                await self._clip_live_window(
                    start_timestamp, end_timestamp, destination, reservation_id=reservation_id
                )
            return destination
        finally:
            if reservation_id:
                await self.release_live_reservation(reservation_id)

    async def clear_stale_reservations(self) -> None:
        """Remove reservation folders left behind by a previous process crash."""
        reservations = self.settings.cache_dir / "reservations"
        if reservations.exists():
            await asyncio.to_thread(shutil.rmtree, reservations, True)

    async def reserve_live_segments(
        self,
        start_timestamp: float,
        end_timestamp: float,
        reservation_id: str | None = None,
    ) -> str | None:
        """Pin currently available segments for a group before normal pruning."""
        if self.settings.source_kind != "live":
            return None
        if reservation_id is None:
            reservation_id = f"event_{int(start_timestamp * 1000)}_{uuid.uuid4().hex[:8]}"
        directory = self._reservation_dir(reservation_id)
        async with self._snapshot_lock:
            reserved = await asyncio.to_thread(
                self._reserve_live_segments_sync,
                start_timestamp,
                end_timestamp,
                directory,
            )
        LOGGER.info(
            "Reserved %d cached segment(s) for %s (%.3f..%.3f).",
            reserved,
            reservation_id,
            start_timestamp,
            end_timestamp,
        )
        return reservation_id

    def _reserve_live_segments_sync(
        self, start_timestamp: float, end_timestamp: float, directory: Path,
    ) -> int:
        directory.mkdir(parents=True, exist_ok=True)
        reserved = 0
        for segment in self.settings.cache_dir.glob("live_*.ts"):
            try:
                modified = segment.stat().st_mtime
            except FileNotFoundError:
                continue
            if not start_timestamp - 4 <= modified <= end_timestamp + 4:
                continue
            target = directory / segment.name
            if target.exists():
                continue
            try:
                os.link(segment, target)
            except FileNotFoundError:
                continue
            except OSError:
                try:
                    shutil.copy2(segment, target)
                except FileNotFoundError:
                    continue
            reserved += 1
        return reserved

    async def release_live_reservation(self, reservation_id: str) -> None:
        directory = self._reservation_dir(reservation_id)
        await asyncio.to_thread(shutil.rmtree, directory, True)

    def _reservation_dir(self, reservation_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", reservation_id):
            raise ValueError("Invalid live-cache reservation identifier.")
        return self.settings.cache_dir / "reservations" / reservation_id

    async def _clip_from_live_cache(
        self, event_timestamp: float, destination: Path, pre_seconds: float, post_seconds: float,
    ) -> None:
        """Compatibility wrapper around arbitrary live-cache windows."""
        await self._clip_live_window(
            event_timestamp - pre_seconds,
            event_timestamp + post_seconds,
            destination,
        )

    async def _clip_live_window(
        self,
        start_timestamp: float,
        end_timestamp: float,
        destination: Path,
        *,
        reservation_id: str | None = None,
    ) -> None:
        # Stream-copy segmentation can only cut cleanly at source keyframes, so
        # use a four-second selection margin instead of assuming one-second files.
        start = start_timestamp - 4
        end = end_timestamp + 4
        staging_dir = self.settings.cache_dir / (
            f"clip_{int(start_timestamp * 1000)}_{uuid.uuid4().hex[:8]}"
        )
        stable_segments: list[Path] = []
        try:
            async with self._snapshot_lock:
                candidates: dict[str, Path] = {}
                directories = [self.settings.cache_dir]
                if reservation_id:
                    directories.append(self._reservation_dir(reservation_id))
                for directory in directories:
                    for segment in directory.glob("live_*.ts"):
                        try:
                            modified = segment.stat().st_mtime
                        except FileNotFoundError:
                            continue
                        if start <= modified <= end:
                            # Reserved files are inspected last and therefore win
                            # when the rolling-cache pathname has already vanished.
                            candidates[segment.name] = segment
                segments = sorted(candidates.values(), key=lambda path: path.stat().st_mtime)
                if not segments:
                    raise RuntimeError(
                        "Live cache has no segments for this event window; the cache may still be warming up."
                    )
                staging_dir.mkdir(parents=True, exist_ok=False)
                for index, segment in enumerate(segments):
                    stable = staging_dir / f"{index:04d}.ts"
                    try:
                        os.link(segment, stable)
                    except FileNotFoundError:
                        continue
                    except OSError:
                        try:
                            shutil.copy2(segment, stable)
                        except FileNotFoundError:
                            continue
                    stable_segments.append(stable)
                if not stable_segments:
                    raise RuntimeError("Cached segments disappeared before they could be reserved.")

                list_path = staging_dir / "concat.txt"
                lines = [
                    "file '" + path.resolve().as_posix().replace("'", "'\\''") + "'"
                    for path in stable_segments
                ]
                list_path.write_text("\n".join(lines), encoding="utf-8")

            # mtime marks when FFmpeg closed the first segment. Probe its real
            # duration because stream-copy segments follow source keyframes and
            # are not guaranteed to be exactly one second long.
            first_segment_end = stable_segments[0].stat().st_mtime
            first_duration = await self._probe_duration(stable_segments[0])
            first_segment_start = first_segment_end - first_duration
            available_end = stable_segments[-1].stat().st_mtime
            coverage_tolerance = 2.5
            if first_segment_start > start_timestamp + coverage_tolerance:
                missing = first_segment_start - start_timestamp
                raise RuntimeError(
                    f"Live cache is missing {missing:.1f}s from the start of this clip; "
                    "wait for the rolling cache to warm up before testing detections."
                )
            if available_end < end_timestamp - coverage_tolerance:
                missing = end_timestamp - available_end
                raise RuntimeError(
                    f"Live cache is missing {missing:.1f}s from the end of this clip window."
                )
            trim_start = max(0.0, start_timestamp - first_segment_start)
            await self._run_source_encode([
                "-f", "concat", "-safe", "0", "-i", str(list_path), "-ss", f"{trim_start:.3f}",
                "-t", str(end_timestamp - start_timestamp), "-map", "0:v:0", "-map", "0:a?",
            ], ["-c:a", "aac", "-movflags", "+faststart", "-y", str(destination)])
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
            self._prune_cache()

    def _prune_cache(self) -> None:
        """Remove compressed chunks older than the configured circular window."""
        cutoff = time.time() - self.settings.buffer_seconds
        for segment in self.settings.cache_dir.glob("live_*.ts"):
            try:
                if segment.stat().st_mtime < cutoff:
                    segment.unlink(missing_ok=True)
            except (FileNotFoundError, PermissionError):
                # The rolling FFmpeg process, antivirus, or another cleanup pass
                # can briefly own the file on Windows. It is safe to retry later.
                continue

    async def _probe_duration(self, segment: Path) -> float:
        process = await asyncio.create_subprocess_exec(
            self.settings.ffprobe_binary, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(segment),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"FFprobe could not inspect cached segment: {stderr.decode(errors='replace').strip()}"
            )
        try:
            return float(stdout.decode().strip())
        except ValueError as exc:
            raise RuntimeError("FFprobe returned an invalid cached-segment duration.") from exc

    async def _run(self, arguments: list[str]) -> None:
        async with self._cpu_limiter.encoding_slot():
            process = await asyncio.create_subprocess_exec(
                self.settings.ffmpeg_binary, "-hide_banner", "-loglevel", "error", *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **low_priority_process_kwargs(self.settings.ffmpeg_low_priority),
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg clip operation failed: {stderr.decode(errors='replace').strip()}"
                )

    async def _select_source_encoder(self) -> str:
        if self._source_encoder is not None:
            return self._source_encoder
        for encoder in self._source_encoder_chain:
            if encoder == "libx264" or await encoder_is_available(
                self.settings.ffmpeg_binary, encoder
            ):
                self._source_encoder = encoder
                LOGGER.info("Source clips will use %s.", encoder)
                return encoder
        raise RuntimeError("No usable H.264 encoder was found for source clips.")

    async def _run_source_encode(
        self, arguments_before_video: list[str], arguments_after_video: list[str]
    ) -> None:
        """Encode a source clip and fall back safely if a device becomes unavailable."""
        encoder = await self._select_source_encoder()
        start = self._source_encoder_chain.index(encoder)
        last_error: RuntimeError | None = None
        for candidate in self._source_encoder_chain[start:]:
            if candidate != encoder and candidate != "libx264" and not await encoder_is_available(
                self.settings.ffmpeg_binary, candidate
            ):
                continue
            arguments = [
                *arguments_before_video,
                *encoder_arguments(
                    candidate,
                    # Keep the intermediate visually transparent so the final
                    # vertical render does not amplify first-pass artifacts.
                    quality=self.settings.ffmpeg_source_quality,
                    threads=self.settings.ffmpeg_encoding_threads,
                    fast=True,
                ),
                *arguments_after_video,
            ]
            try:
                await self._run(arguments)
                self._source_encoder = candidate
                return
            except RuntimeError as exc:
                last_error = exc
                LOGGER.warning("%s failed for a source clip; trying the next encoder: %s", candidate, exc)
        assert last_error is not None
        raise last_error
