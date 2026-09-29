"""Replay a video through an isolated OCR -> event queue -> merge planner.

This module never invokes the production clipper, editor, database, or Telegram
bot. It is intentionally safe to run against a local test video before changing
the main application.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Final

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings
from event_detector import ApexEventDetector, DetectedEvent, PreparedOcrFrame
from event_pipeline import EventGroup, EventGroupPlanner, TimelineWatermark

LOGGER = logging.getLogger("apex_clipper.diagnostics.event_queue")
OCR_END: Final = object()
EVENT_END: Final = object()

def capture_worker(
    video_path: Path,
    detector: ApexEventDetector,
    ocr_queue: queue.Queue[PreparedOcrFrame | object],
    errors: queue.Queue[BaseException],
    *,
    start_seconds: float,
    end_seconds: float | None,
    pace_realtime: bool,
) -> None:
    """Read sampled frames continuously without ever waiting for OCR."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / source_fps
        stop_at = min(duration, end_seconds) if end_seconds is not None else duration
        interval = 1.0 / detector.settings.sample_fps
        wall_started = time.monotonic()
        timestamp = start_seconds
        while timestamp < stop_at:
            if pace_realtime:
                due = wall_started + timestamp - start_seconds
                time.sleep(max(0.0, due - time.monotonic()))
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                break
            prepared = detector.prepare_frame(frame, timestamp)
            if prepared is not None:
                # The harness does not save evidence images, so do not retain a
                # full 1080p frame while slow CPU OCR drains the diagnostic queue.
                compact = replace(prepared, frame=np.empty((0, 0, 3), dtype=np.uint8))
                ocr_queue.put_nowait(compact)
                LOGGER.info(
                    "[CAPTURE -> OCR_QUEUE] t=%.3f regions=%s pending=%d",
                    timestamp,
                    ",".join(prepared.changed_regions),
                    ocr_queue.qsize(),
                )
            timestamp += interval
    except BaseException as exc:
        errors.put(exc)
    finally:
        capture.release()
        ocr_queue.put(OCR_END)


def ocr_worker(
    detector: ApexEventDetector,
    ocr_queue: queue.Queue[PreparedOcrFrame | object],
    event_queue: queue.Queue[DetectedEvent | TimelineWatermark | object],
    errors: queue.Queue[BaseException],
) -> None:
    """Own the stateful OCR engine and publish timestamped events in order."""
    try:
        while True:
            item = ocr_queue.get()
            try:
                if item is OCR_END:
                    event_queue.put(EVENT_END)
                    return
                assert isinstance(item, PreparedOcrFrame)
                started = time.perf_counter()
                event = detector.process_prepared(item)
                elapsed = time.perf_counter() - started
                if event is not None:
                    event_queue.put(event)
                    LOGGER.info(
                        "[OCR -> EVENT_QUEUE] t=%.3f kind=%s confidence=%.0f%% OCR=%.2fs",
                        event.timestamp,
                        event.kind,
                        event.confidence * 100,
                        elapsed,
                    )
                else:
                    LOGGER.info("[OCR NO EVENT] t=%.3f OCR=%.2fs", item.timestamp, elapsed)
                event_queue.put(TimelineWatermark(item.timestamp, event is not None))
            finally:
                ocr_queue.task_done()
    except BaseException as exc:
        errors.put(exc)
        event_queue.put(EVENT_END)


def grouping_worker(
    planner: EventGroupPlanner,
    event_queue: queue.Queue[DetectedEvent | TimelineWatermark | object],
    errors: queue.Queue[BaseException],
) -> None:
    try:
        while True:
            item = event_queue.get()
            try:
                if item is EVENT_END:
                    planner.finish()
                    return
                if isinstance(item, DetectedEvent):
                    LOGGER.info(
                        "[EVENT DEQUEUED] t=%.3f kind=%s evidence=%s",
                        item.timestamp,
                        item.kind,
                        item.evidence,
                    )
                    planner.add_event(item)
                else:
                    assert isinstance(item, TimelineWatermark)
                    planner.advance(item.timestamp, observed_event=item.observed_event)
            finally:
                event_queue.task_done()
    except BaseException as exc:
        errors.put(exc)


def run_harness(args: argparse.Namespace) -> list[EventGroup]:
    video_path = args.video.resolve()
    settings = Settings.from_env()
    settings.source_kind = "vod"
    settings.twitch_url = str(video_path)
    settings.event_cooldown_seconds = 0.0
    settings.vod_start_offset_seconds = 0.0
    detector = ApexEventDetector(settings)
    asyncio.run(detector.initialize())
    detector.reset_session()

    ocr_items: queue.Queue[PreparedOcrFrame | object] = queue.Queue()
    event_items: queue.Queue[DetectedEvent | TimelineWatermark | object] = queue.Queue()
    errors: queue.Queue[BaseException] = queue.Queue()
    planner = EventGroupPlanner(
        settings,
        merge_gap_seconds=args.merge_gap,
        duplicate_window_seconds=args.duplicate_window,
    )
    threads = [
        threading.Thread(
            target=capture_worker,
            name="diagnostic-capture",
            args=(video_path, detector, ocr_items, errors),
            kwargs={
                "start_seconds": args.start,
                "end_seconds": args.end,
                "pace_realtime": args.realtime,
            },
        ),
        threading.Thread(
            target=ocr_worker,
            name="diagnostic-ocr",
            args=(detector, ocr_items, event_items, errors),
        ),
        threading.Thread(
            target=grouping_worker,
            name="diagnostic-grouper",
            args=(planner, event_items, errors),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if not errors.empty():
        raise errors.get()

    LOGGER.info("[SUMMARY] raw_groups=%d", len(planner.completed))
    for index, group in enumerate(planner.completed, start=1):
        LOGGER.info("[SUMMARY CLIP %d] %s", index, planner.describe(group))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([group.to_dict() for group in planner.completed], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Wrote diagnostic plan to %s", args.output)
    return planner.completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Local MP4 used only for diagnosis")
    parser.add_argument("--start", type=float, default=0.0, help="First video second to inspect")
    parser.add_argument("--end", type=float, default=None, help="Stop before this video second")
    parser.add_argument("--merge-gap", type=float, default=8.0)
    parser.add_argument("--duplicate-window", type=float, default=8.0)
    parser.add_argument("--realtime", action="store_true", help="Pace capture using the video clock")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON clip plan")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
    )
    run_harness(parse_args())
