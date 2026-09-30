"""Async entry point for the visual-only Twitch-to-clips workflow."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from pathlib import Path

import cv2
from streamlink.exceptions import StreamlinkError

from clipper import Clipper
from config import ROOT_DIR, Settings
from database import ClipRepository
from editor import ApexVerticalEditor, RenderCancelled
from event_detector import ApexEventDetector, DetectedEvent, PreparedOcrFrame
from event_pipeline import EventGroup, EventGroupPlanner, TimelineWatermark
from ffmpeg_resources import FfmpegCpuLimiter
from instance_lock import AlreadyRunningError, SingleInstanceLock
from ocr_queue import coalesce_ocr_backlog, offer_ocr_keyframe
from stream_monitor import StreamMonitor
from telegram_bot import TelegramApprovalBot, delete_clip_files
from uploader import AutoUploader

class SecretRedactionFilter(logging.Filter):
    """Prevent Telegram bot tokens from being copied into diagnostic logs."""

    _telegram_token = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        record.msg = self._telegram_token.sub("<TELEGRAM_TOKEN_REDACTED>", message)
        record.args = ()
        return True


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for log_handler in logging.getLogger().handlers:
    log_handler.addFilter(SecretRedactionFilter())
# httpx logs Telegram bot tokens as part of request URLs at INFO level.
logging.getLogger("httpx").setLevel(logging.WARNING)
LOGGER = logging.getLogger("apex_clipper")


async def run() -> None:
    settings = Settings.from_env()
    if not settings.twitch_url:
        raise ValueError("TWITCH_URL is required. Copy .env.example to .env and configure it.")
    settings.prepare_directories()
    repository = ClipRepository(settings)
    await repository.initialize()
    cpu_limiter = FfmpegCpuLimiter(settings.ffmpeg_max_concurrent_encodes)
    monitor = StreamMonitor(settings)
    detector = ApexEventDetector(settings)
    clipper = Clipper(settings, cpu_limiter)
    editor = ApexVerticalEditor(settings, cpu_limiter)
    uploader = AutoUploader(settings, repository)
    bot = TelegramApprovalBot(settings, repository, uploader, cpu_limiter)

    tasks: set[asyncio.Task[None]] = set()
    uploader_started = False
    bot_started = False
    try:
        await clipper.clear_stale_reservations()
        await detector.initialize()
        LOGGER.info(
            "Live cache keeps %.0fs; FFmpeg encodes use at most %d thread(s) and %d concurrent job(s).",
            settings.buffer_seconds,
            settings.ffmpeg_encoding_threads,
            settings.ffmpeg_max_concurrent_encodes,
        )
        uploader.set_live_active(settings.source_kind == "live")
        await uploader.start()
        uploader_started = True
        await bot.start()
        bot_started = True
        while True:
            try:
                await monitor.start()
            except (RuntimeError, StreamlinkError) as exc:
                if settings.source_kind != "live":
                    raise
                # Telegram approvals and queued publications remain available
                # while an offline Twitch channel is checked periodically.
                uploader.set_live_active(False)
                LOGGER.warning("Twitch is unavailable; approvals remain active. Retrying in 60 seconds: %s", exc)
                await asyncio.sleep(60)
                continue

            uploader.set_live_active(settings.source_kind == "live")
            LOGGER.info("Monitoring %s at %.1f FPS; only visual signals are inspected.", settings.source_kind, settings.sample_fps)
            detector.reset_session()
            ocr_queue: asyncio.Queue[PreparedOcrFrame | None] = asyncio.Queue(
                maxsize=settings.ocr_queue_size
            )
            event_queue: asyncio.Queue[DetectedEvent | TimelineWatermark | None] = asyncio.Queue()
            ocr_idle = asyncio.Event()
            ocr_idle.set()
            assert monitor.source_url is not None
            source_url = monitor.source_url
            ocr_worker = asyncio.create_task(
                process_ocr_queue(ocr_queue, event_queue, detector, settings, ocr_idle),
                name="ocr-worker",
            )
            event_worker = asyncio.create_task(
                process_event_queue(
                    event_queue, ocr_queue, ocr_idle, settings, source_url,
                    clipper, editor, bot, repository, tasks,
                ),
                name="event-group-worker",
            )
            try:
                async for frame, timestamp in monitor.frames():
                    try:
                        prepared = detector.prepare_frame(frame, timestamp)
                    except Exception:
                        LOGGER.exception("Could not inspect OCR regions at %.3f", timestamp)
                        continue
                    if prepared is not None:
                        offer_ocr_keyframe(ocr_queue, prepared)
            finally:
                await ocr_queue.put(None)
                await ocr_worker
                await event_worker

            await monitor.close()
            uploader.set_live_active(False)
            if settings.source_kind == "vod":
                LOGGER.info("VOD ended; waiting for Telegram approvals. Press Ctrl+C to stop.")
                await asyncio.Event().wait()
            LOGGER.info("Live source ended; approvals remain active. Retrying Twitch in 30 seconds.")
            await asyncio.sleep(30)
    finally:
        await monitor.close()
        uploader.set_live_active(False)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if bot_started:
            await bot.close()
        if uploader_started:
            await uploader.close()


async def process_ocr_queue(
    queue: asyncio.Queue[PreparedOcrFrame | None],
    event_queue: asyncio.Queue[DetectedEvent | TimelineWatermark | None],
    detector: ApexEventDetector,
    settings: Settings,
    ocr_idle: asyncio.Event,
) -> None:
    """Consume the freshest keyframes and publish ordered, timestamped events."""
    skipped_since_warning = 0
    last_stale_warning_at = float("-inf")
    while True:
        prepared = await queue.get()
        stop_after = False
        try:
            if prepared is None:
                await event_queue.put(None)
                return
            ocr_idle.clear()
            if settings.source_kind == "live":
                prepared, stop_after, discarded = coalesce_ocr_backlog(queue, prepared)
                if discarded:
                    LOGGER.debug(
                        "Discarded %d pending OCR keyframe(s) in favor of %.3f.",
                        discarded,
                        prepared.timestamp,
                    )
            lag = 0.0
            if settings.source_kind == "live":
                lag = max(0.0, time.time() - prepared.timestamp)
            event: DetectedEvent | None = None
            if lag > settings.ocr_max_live_lag_seconds:
                skipped_since_warning += 1
                now = time.monotonic()
                if now - last_stale_warning_at >= 10:
                    LOGGER.warning(
                        "Skipped %d stale OCR keyframe(s); newest was %.1fs behind live video.",
                        skipped_since_warning,
                        lag,
                    )
                    skipped_since_warning = 0
                    last_stale_warning_at = now
            else:
                try:
                    event = await asyncio.to_thread(detector.process_prepared, prepared)
                except Exception:
                    LOGGER.exception("OCR failed for keyframe at %.3f", prepared.timestamp)
            if event is not None:
                try:
                    evidence_path = await asyncio.to_thread(
                        save_event_evidence, settings, event, prepared.frame
                    )
                except Exception:
                    LOGGER.exception("Could not persist event evidence; clip processing will continue.")
                    evidence_path = None
                LOGGER.info(
                    "Detected %s (%.0f%%): %s; queued for temporal grouping.",
                    event.kind,
                    event.confidence * 100,
                    event.evidence,
                )
                if evidence_path:
                    LOGGER.info("Saved detection evidence to %s", evidence_path)
                await event_queue.put(event)
            await event_queue.put(TimelineWatermark(prepared.timestamp, event is not None))
            if stop_after:
                await event_queue.put(None)
                return
        finally:
            ocr_idle.set()
            queue.task_done()


async def process_event_queue(
    queue: asyncio.Queue[DetectedEvent | TimelineWatermark | None],
    ocr_queue: asyncio.Queue[PreparedOcrFrame | None],
    ocr_idle: asyncio.Event,
    settings: Settings,
    source_url: str,
    clipper: Clipper,
    editor: ApexVerticalEditor,
    bot: TelegramApprovalBot,
    repository: ClipRepository,
    tasks: set[asyncio.Task[None]],
) -> None:
    """Group continuous detections while capture and OCR remain independent."""
    planner = EventGroupPlanner(settings)
    scheduled_groups = 0

    async def reserve_group(group: EventGroup) -> None:
        if settings.source_kind != "live":
            return
        try:
            group.reservation_id = await clipper.reserve_live_segments(
                group.clip_start,
                group.clip_end,
                group.reservation_id,
            )
        except Exception:
            # Continue with the rolling cache if a hardlink/copy races with the
            # FFmpeg segment writer; final coverage validation remains strict.
            LOGGER.exception(
                "Could not reserve live-cache segments for group %.3f..%.3f.",
                group.first_event_at,
                group.last_event_at,
            )

    async def schedule_ready_groups() -> None:
        nonlocal scheduled_groups
        while scheduled_groups < len(planner.completed):
            group = planner.completed[scheduled_groups]
            scheduled_groups += 1
            # At finalization, post-event segments now exist. Add them to the
            # reservation before the group waits for the shared encoder slot.
            await reserve_group(group)
            task = asyncio.create_task(
                process_event_group(group, source_url, clipper, editor, bot, repository),
                name=f"clip-group-{int(group.first_event_at)}",
            )
            tasks.add(task)
            task.add_done_callback(tasks.discard)

    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=settings.event_merge_gap_seconds)
        except asyncio.TimeoutError:
            # A disappearing HUD message may leave no changed ROI to OCR. Close
            # the group only when there is no keyframe queued or being processed.
            if planner.active is not None and ocr_idle.is_set() and ocr_queue.empty():
                planner.finalize_idle()
                await schedule_ready_groups()
            continue
        try:
            if item is None:
                planner.finish()
                await schedule_ready_groups()
                return
            if isinstance(item, DetectedEvent):
                previous_group = planner.active
                previous_end = previous_group.clip_end if previous_group is not None else float("-inf")
                accepted = planner.add_event(item)
                if planner.active is not None and (
                    accepted or planner.active is not previous_group or planner.active.clip_end > previous_end
                ):
                    # Pin the pre-fight history as soon as the first event is
                    # known, independently of clipping and rendering work.
                    await reserve_group(planner.active)
            else:
                planner.advance(item.timestamp, observed_event=item.observed_event)
            await schedule_ready_groups()
        finally:
            queue.task_done()


def save_event_evidence(settings: Settings, event: DetectedEvent, frame) -> Path:
    """Persist the exact trigger frame and OCR metadata for later auditing."""
    stem = f"event_{int(event.timestamp * 1000)}_{event.kind.lower()}"
    image_path = settings.evidence_dir / f"{stem}.jpg"
    metadata_path = settings.evidence_dir / f"{stem}.json"
    if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not save event evidence image: {image_path}")
    metadata_path.write_text(json.dumps({
        "timestamp": event.timestamp,
        "kind": event.kind,
        "confidence": event.confidence,
        "region": event.region,
        "evidence": event.evidence,
        "raw_ocr_text": event.raw_ocr_text,
        "player_gamertag": settings.player_gamertag,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path


async def process_event_group(
    group: EventGroup,
    source_url: str,
    clipper: Clipper,
    editor: ApexVerticalEditor,
    bot: TelegramApprovalBot,
    repository: ClipRepository,
) -> None:
    clip_id: int | None = None
    source_clip: Path | None = None
    try:
        source_clip = await clipper.create_clip_window(
            group.clip_start,
            group.clip_end,
            source_url,
            stamp_timestamp=group.first_event_at,
            reservation_id=group.reservation_id,
        )
        clip_id = await repository.create_clip(source_clip.name, "PROCESSING", "TWITCH")
        vertical_clip = await editor.render(
            source_clip,
            cancel_requested=lambda: repository.processing_should_stop(clip_id),
            progress_callback=lambda progress: repository.set_processing_progress(
                clip_id, progress
            ),
        )
        moved = await repository.transition_clip_status(
            clip_id,
            from_statuses=("PROCESSING",),
            to_status="PENDING_APPROVAL",
            filename=vertical_clip.name,
        )
        if not moved:
            if await repository.processing_should_stop(clip_id):
                raise RenderCancelled("Processing was cancelled after FFmpeg completed.")
            raise RuntimeError("Clip status changed before rendering completed.")
        try:
            await bot.request_approval(
                clip_id, vertical_clip, group.label, group.confidence
            )
        except Exception:
            # Rendering is already durable and the dashboard provides a resend
            # action. A Telegram outage must not delete or reprocess the master.
            LOGGER.exception(
                "Rendered %s, but Telegram delivery failed; clip #%s remains pending approval.",
                vertical_clip.name,
                clip_id,
            )
            return
        LOGGER.info(
            "Sent %s for Telegram approval (%s).",
            vertical_clip.name,
            EventGroupPlanner.describe(group),
        )
    except RenderCancelled:
        if clip_id is not None:
            await repository.transition_clip_status(
                clip_id,
                from_statuses=("CANCEL_REQUESTED",),
                to_status="DISCARDED",
            )
        if source_clip is not None:
            delete_clip_files(editor.settings.clips_dir, source_clip.name)
        LOGGER.info("Cancelled processing for clip #%s and removed partial files.", clip_id)
    except asyncio.CancelledError:
        # Ctrl+C cancels background group tasks. Restore an interrupted render
        # to a retryable state before propagating shutdown, so it cannot remain
        # indefinitely displayed as PROCESSING.
        if clip_id is not None and source_clip is not None:
            current = asyncio.current_task()
            if current is not None:
                while current.cancelling():
                    current.uncancel()
            try:
                await repository.set_processing_progress(clip_id, 0.0)
                restored = await repository.transition_clip_status(
                    clip_id,
                    from_statuses=("PROCESSING",),
                    to_status="UPLOADED",
                    filename=source_clip.name,
                )
                if restored:
                    LOGGER.info(
                        "Returned interrupted clip #%s to UPLOADED for retry.", clip_id
                    )
            except Exception:
                LOGGER.exception(
                    "Could not restore interrupted clip #%s during shutdown.", clip_id
                )
        raise
    except Exception:
        clip = await repository.get_clip(clip_id) if clip_id is not None else None
        if clip is not None and clip.status in {"CANCEL_REQUESTED", "DISCARDED"}:
            await repository.transition_clip_status(
                clip_id,
                from_statuses=("CANCEL_REQUESTED",),
                to_status="DISCARDED",
            )
            if source_clip is not None:
                delete_clip_files(editor.settings.clips_dir, source_clip.name)
            LOGGER.info(
                "Cancelled processing for clip #%s after FFmpeg stopped with an error.",
                clip_id,
            )
            return
        if clip is not None and clip.status == "PROCESSING" and source_clip is not None:
            await repository.set_processing_progress(clip_id, 0.0)
            await repository.transition_clip_status(
                clip_id,
                from_statuses=("PROCESSING",),
                to_status="UPLOADED",
                filename=source_clip.name,
            )
            LOGGER.exception(
                "Could not render event group %.3f..%.3f (%s); clip #%s returned to UPLOADED.",
                group.first_event_at,
                group.last_event_at,
                group.label,
                clip_id,
            )
            return
        LOGGER.exception(
            "Could not process event group %.3f..%.3f (%s).",
            group.first_event_at,
            group.last_event_at,
            group.label,
        )


if __name__ == "__main__":
    try:
        with SingleInstanceLock(ROOT_DIR / ".apex_clipper.lock"):
            with contextlib.suppress(KeyboardInterrupt):
                asyncio.run(run())
    except AlreadyRunningError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc
