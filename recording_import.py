"""Analyze a local PS5 recording incrementally and create clips only for visual events."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Awaitable, Callable

import cv2
import numpy as np

from clipper import Clipper
from config import Settings
from database import ClipRepository
from editor import ApexVerticalEditor, RenderCancelled
from event_detector import ApexEventDetector
from event_pipeline import EventGroup, EventGroupPlanner
from ffmpeg_resources import FfmpegCpuLimiter


LOGGER = logging.getLogger("apex_clipper.recording_import")
ProgressCallback = Callable[["RecordingProgress"], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RecordingProgress:
    stage: str
    percent: float
    events: int
    clips_done: int
    clips_total: int


@dataclass(frozen=True, slots=True)
class RecordingResult:
    duration_seconds: float
    events: int
    groups: int
    clips_created: int
    clips_failed: int


def probe_recording(path: Path, ffprobe: str = "ffprobe") -> float:
    if not path.is_file():
        raise FileNotFoundError(f"No existe la grabación: {path}")
    if path.suffix.lower() not in {".mp4", ".mov", ".webm", ".mkv"}:
        raise ValueError("La grabación debe ser MP4, MOV, WEBM o MKV.")
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,width,height", "-of", "json", str(path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"No se pudo inspeccionar la grabación: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    video = next(
        (stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video is None:
        raise ValueError("La grabación no contiene video.")
    width, height = int(video["width"]), int(video["height"])
    if abs(width / height - 16 / 9) > 0.05:
        raise ValueError("La grabación debe ser horizontal 16:9 para los recortes de Apex.")
    duration = float(data["format"]["duration"])
    if duration <= 0:
        raise ValueError("La grabación no tiene una duración válida.")
    return duration


async def scan_recording(
    path: Path,
    settings: Settings,
    detector: ApexEventDetector,
    duration: float,
    on_progress: ProgressCallback | None = None,
    import_id: str = "",
) -> tuple[list[EventGroup], int]:
    """Decode two frames per second through a pipe; never load the full file."""
    frame_size = 1920 * 1080 * 3
    process = await asyncio.create_subprocess_exec(
        settings.ffmpeg_binary, "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-an", "-vf",
        f"fps={settings.sample_fps},scale=1920:1080:flags=lanczos",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None and process.stderr is not None

    async def drain_stderr() -> str:
        recent = b""
        while chunk := await process.stderr.read(4096):
            recent = (recent + chunk)[-65536:]
        return recent.decode(errors="replace").strip()

    stderr_task = asyncio.create_task(drain_stderr())
    planner = EventGroupPlanner(settings)
    frame_index = 0
    event_count = 0
    last_reported_at = float("-inf")
    try:
        while True:
            try:
                raw = await process.stdout.readexactly(frame_size)
            except asyncio.IncompleteReadError as exc:
                if exc.partial:
                    raise RuntimeError("La grabación terminó en medio de un fotograma.") from exc
                break
            timestamp = frame_index / settings.sample_fps
            frame_index += 1
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((1080, 1920, 3)).copy()
            event = None
            try:
                prepared = detector.prepare_frame(frame, timestamp)
                if prepared is not None:
                    event = await asyncio.to_thread(detector.process_prepared, prepared)
            except Exception:
                LOGGER.exception("No se pudo inspeccionar el segundo %.2f", timestamp)
            if event is not None:
                event_count += 1
                planner.add_event(event)
                try:
                    _save_evidence(settings, import_id, event, frame)
                except Exception:
                    LOGGER.exception("No se pudo guardar la evidencia del segundo %.2f", timestamp)
            planner.advance(timestamp, observed_event=event is not None)
            if on_progress is not None and timestamp - last_reported_at >= 10:
                await on_progress(RecordingProgress(
                    "Analizando", min(100.0, 100.0 * timestamp / duration),
                    event_count, 0, 0,
                ))
                last_reported_at = timestamp
        code = await process.wait()
        stderr = await stderr_task
        if code:
            raise RuntimeError(f"FFmpeg no pudo leer la grabación: {stderr}")
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        if not stderr_task.done():
            stderr_task.cancel()
    groups = planner.finish()
    for group in groups:
        group.clip_end = min(duration, group.clip_end)
    if on_progress is not None:
        await on_progress(RecordingProgress("Analizando", 100.0, event_count, 0, len(groups)))
    return groups, event_count


def _save_evidence(settings: Settings, import_id: str, event, frame: np.ndarray) -> None:
    settings.evidence_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_]", "", import_id)
    stem = f"recording_{safe_id}_{int(event.timestamp * 1000)}_{event.kind.lower()}"
    image_path = settings.evidence_dir / f"{stem}.jpg"
    cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (settings.evidence_dir / f"{stem}.json").write_text(
        json.dumps({
            "timestamp": event.timestamp,
            "kind": event.kind,
            "confidence": event.confidence,
            "region": event.region,
            "evidence": event.evidence,
            "raw_ocr_text": event.raw_ocr_text,
            "player_gamertag": settings.player_gamertag,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def _send_approval(
    settings: Settings, repository: ClipRepository, clip_id: int,
    vertical: Path, event_kind: str, confidence: float,
) -> None:
    from telegram_bot import send_approval_once

    await send_approval_once(
        settings, repository, clip_id, vertical, event_kind, confidence
    )


async def import_recording(
    path: Path, settings: Settings, repository: ClipRepository,
    on_progress: ProgressCallback | None = None,
) -> RecordingResult:
    """Inspect a local file, then render and queue only detected event windows."""
    path = path.expanduser().resolve()
    duration = await asyncio.to_thread(probe_recording, path, settings.ffprobe_binary)
    settings.prepare_directories()
    vod_settings = replace(settings, source_kind="vod")
    detector = ApexEventDetector(vod_settings)
    await detector.initialize()
    detector.reset_session()
    import_id = uuid.uuid4().hex[:10]
    groups, events = await scan_recording(
        path, vod_settings, detector, duration, on_progress, import_id
    )
    if not groups:
        return RecordingResult(duration, events, 0, 0, 0)

    limiter = FfmpegCpuLimiter(settings.ffmpeg_max_concurrent_encodes)
    clipper = Clipper(vod_settings, limiter)
    editor = ApexVerticalEditor(vod_settings, limiter)
    created = failed = 0
    for index, group in enumerate(groups, start=1):
        if on_progress is not None:
            await on_progress(RecordingProgress(
                "Creando clips", 100.0 * (index - 1) / len(groups),
                events, index - 1, len(groups),
            ))
        clip_id = None
        source_clip = None
        try:
            source_clip = await clipper.create_clip_window(
                group.clip_start, group.clip_end, str(path),
                stamp_timestamp=group.first_event_at,
                name_prefix=f"ps5_{import_id}",
                normalize_to_1080=True,
            )
            clip_id = await repository.create_clip(
                source_clip.name, "PROCESSING", "PS5_RECORDING"
            )
            vertical = await editor.render(
                source_clip,
                cancel_requested=lambda: repository.processing_should_stop(clip_id),
                progress_callback=lambda progress: repository.set_processing_progress(
                    clip_id, progress
                ),
            )
            moved = await repository.transition_clip_status(
                clip_id, from_statuses=("PROCESSING",),
                to_status="PENDING_APPROVAL", filename=vertical.name,
            )
            if not moved:
                raise RuntimeError(f"El clip #{clip_id} cambió de estado durante el render.")
            created += 1
            try:
                await _send_approval(
                    settings, repository, clip_id, vertical, group.label, group.confidence
                )
            except Exception:
                LOGGER.exception(
                    "El clip #%s quedó pendiente; no se pudo enviar a Telegram.", clip_id
                )
        except RenderCancelled:
            failed += 1
            if clip_id is not None:
                await repository.transition_clip_status(
                    clip_id, from_statuses=("CANCEL_REQUESTED",), to_status="DISCARDED"
                )
            if source_clip is not None:
                source_clip.unlink(missing_ok=True)
        except Exception:
            failed += 1
            LOGGER.exception("No se pudo crear el grupo %s de %s.", index, len(groups))
            if clip_id is not None:
                await repository.transition_clip_status(
                    clip_id, from_statuses=("PROCESSING",),
                    to_status="UPLOADED", filename=source_clip.name if source_clip else None,
                )
            elif source_clip is not None:
                source_clip.unlink(missing_ok=True)
        if on_progress is not None:
            await on_progress(RecordingProgress(
                "Creando clips", 100.0 * index / len(groups),
                events, index, len(groups),
            ))
    return RecordingResult(duration, events, len(groups), created, failed)


async def _cli(path: Path) -> None:
    settings = Settings.from_env()
    repository = ClipRepository(settings)
    await repository.initialize()

    async def report(progress: RecordingProgress) -> None:
        print(
            f"{progress.stage}: {progress.percent:.0f}% · "
            f"{progress.events} eventos · {progress.clips_done}/{progress.clips_total} clips",
            flush=True,
        )

    result = await import_recording(path, settings, repository, report)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="Absolute local path to a PS5 recording")
    args = parser.parse_args()
    asyncio.run(_cli(args.recording))
