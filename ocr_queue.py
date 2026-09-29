"""Bounded, freshness-oriented queue helpers for CPU OCR keyframes."""
from __future__ import annotations

import asyncio
import logging

from event_detector import PreparedOcrFrame

LOGGER = logging.getLogger("apex_clipper.ocr_queue")


def offer_ocr_keyframe(
    queue: asyncio.Queue[PreparedOcrFrame | None], prepared: PreparedOcrFrame,
) -> None:
    """Keep capture live by coalescing the oldest pending OCR keyframe."""
    if queue.full():
        stale = queue.get_nowait()
        queue.task_done()
        if stale is not None:
            LOGGER.debug(
                "Coalesced OCR keyframe %.3f in favor of newer keyframe %.3f.",
                stale.timestamp,
                prepared.timestamp,
            )
    queue.put_nowait(prepared)


def coalesce_ocr_backlog(
    queue: asyncio.Queue[PreparedOcrFrame | None],
    prepared: PreparedOcrFrame,
) -> tuple[PreparedOcrFrame, bool, int]:
    """Keep only the newest pending live keyframe before starting expensive OCR.

    Returns the selected frame, whether the shutdown sentinel was consumed, and
    how many older frames were discarded. Every drained queue item is balanced
    with ``task_done``; the caller remains responsible for its original item.
    """
    latest = prepared
    stop_after = False
    discarded = 0
    while True:
        try:
            pending = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if pending is None:
            queue.task_done()
            stop_after = True
            break
        queue.task_done()
        discarded += 1
        latest = pending
    return latest, stop_after, discarded
