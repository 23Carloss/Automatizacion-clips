"""Temporal event grouping for continuous OCR detection.

The detector reports every confirmed visual event.  This module owns the
separate concern of deduplicating lingering OCR text and combining nearby
events into one clip window without pausing frame capture.
"""
from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field

from config import Settings
from event_detector import DetectedEvent

LOGGER = logging.getLogger("apex_clipper.event_pipeline")


@dataclass(frozen=True, slots=True)
class TimelineWatermark:
    """Latest video timestamp for which OCR processing has completed."""

    timestamp: float
    observed_event: bool = False


@dataclass(slots=True)
class EventGroup:
    """A burst of related events rendered as one continuous clip."""

    first_event_at: float
    last_event_at: float
    clip_start: float
    clip_end: float
    label: str
    events: list[DetectedEvent] = field(default_factory=list)
    reservation_id: str | None = None

    @property
    def confidence(self) -> float:
        return max((event.confidence for event in self.events), default=0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "first_event_at": round(self.first_event_at, 3),
            "last_event_at": round(self.last_event_at, 3),
            "clip_start": round(self.clip_start, 3),
            "clip_end": round(self.clip_end, 3),
            "duration": round(self.clip_end - self.clip_start, 3),
            "events": [asdict(event) for event in self.events],
        }


class EventGroupPlanner:
    """Deduplicate detections and turn temporal bursts into clip windows."""

    LABEL_PRIORITY = {
        "KNOCKED": 1,
        "BLEEDOUT": 2,
        "ELIMINATED": 2,
        "SQUAD_ELIMINATED": 3,
    }

    def __init__(
        self,
        settings: Settings,
        *,
        merge_gap_seconds: float | None = None,
        duplicate_window_seconds: float | None = None,
    ) -> None:
        self.settings = settings
        self.merge_gap_seconds = (
            settings.event_merge_gap_seconds
            if merge_gap_seconds is None
            else merge_gap_seconds
        )
        self.duplicate_window_seconds = (
            settings.event_duplicate_window_seconds
            if duplicate_window_seconds is None
            else duplicate_window_seconds
        )
        self.active: EventGroup | None = None
        self.completed: list[EventGroup] = []
        self._blank_observations = 0

    def add_event(self, event: DetectedEvent) -> bool:
        """Add a new event and return whether it changed the active group."""
        if self._is_duplicate(event):
            LOGGER.info(
                "Deduplicated %s at %.3f inside the %.1fs persistence window.",
                event.kind,
                event.timestamp,
                self.duplicate_window_seconds,
            )
            self._blank_observations = 0
            return False
        self._blank_observations = 0
        if self.active is None:
            self.active = self._new_group(event)
            LOGGER.info("Started event group: %s", self.describe(self.active))
            return True
        gap = event.timestamp - self.active.last_event_at
        if gap < self.merge_gap_seconds:
            self._merge(event)
            LOGGER.info("Extended event group after %.3fs: %s", gap, self.describe(self.active))
            return True
        self._finalize("next event is outside merge gap")
        self.active = self._new_group(event)
        LOGGER.info("Started event group: %s", self.describe(self.active))
        return True

    def advance(self, timestamp: float, *, observed_event: bool = False) -> None:
        """Finalize only after OCR has processed beyond the grouping window."""
        self._blank_observations = 0 if observed_event else self._blank_observations + 1
        if self.active and timestamp - self.active.last_event_at >= self.merge_gap_seconds:
            self._finalize("OCR watermark passed merge gap")

    def finalize_idle(self) -> None:
        """Close an active group after the OCR pipeline itself has gone idle."""
        if self.active:
            self._finalize("OCR pipeline was idle for the merge gap")

    def finish(self) -> list[EventGroup]:
        if self.active:
            self._finalize("source ended")
        return self.completed

    def _new_group(self, event: DetectedEvent) -> EventGroup:
        pre, post = self.settings.clip_window(event.kind)
        return EventGroup(
            first_event_at=event.timestamp,
            last_event_at=event.timestamp,
            clip_start=max(0.0, event.timestamp - pre),
            clip_end=event.timestamp + post,
            label=event.kind,
            events=[event],
        )

    def _merge(self, event: DetectedEvent) -> None:
        assert self.active is not None
        pre, post = self.settings.clip_window(event.kind)
        self.active.last_event_at = max(self.active.last_event_at, event.timestamp)
        self.active.clip_start = max(0.0, min(self.active.clip_start, event.timestamp - pre))
        self.active.clip_end = max(self.active.clip_end, event.timestamp + post)
        if self.LABEL_PRIORITY.get(event.kind, 0) > self.LABEL_PRIORITY.get(self.active.label, 0):
            self.active.label = event.kind
        self.active.events.append(event)

    def _is_duplicate(self, event: DetectedEvent) -> bool:
        if self.active is None or self._blank_observations >= 2:
            return False
        incoming = self._event_fingerprint(event)
        for previous in reversed(self.active.events):
            if event.timestamp - previous.timestamp > self.duplicate_window_seconds:
                break
            if previous.kind != event.kind:
                continue
            incoming_tail = self._rightmost_identity(incoming)
            previous_tail = self._rightmost_identity(self._event_fingerprint(previous))
            if difflib.SequenceMatcher(None, incoming_tail, previous_tail).ratio() >= 0.85:
                return True
        return False

    @staticmethod
    def _rightmost_identity(fingerprint: str) -> str:
        tokens = fingerprint.split()
        if not tokens:
            return ""
        last = tokens[-1]
        if (
            last.isdigit()
            and len(last) <= 4
            and len(tokens) >= 2
            and tokens[-2].isalpha()
            and len(tokens[-2]) >= 5
        ):
            return tokens[-2] + last
        return last

    @staticmethod
    def _event_fingerprint(event: DetectedEvent) -> str:
        text = event.raw_ocr_text or event.evidence
        decomposed = unicodedata.normalize("NFKD", text.upper())
        plain = "".join(char for char in decomposed if not unicodedata.combining(char))
        return re.sub(r"[^A-Z0-9]+", " ", plain).strip()

    def _finalize(self, reason: str) -> None:
        assert self.active is not None
        LOGGER.info("Event group ready (%s): %s", reason, self.describe(self.active))
        self.completed.append(self.active)
        self.active = None

    @staticmethod
    def describe(group: EventGroup) -> str:
        kinds = " -> ".join(event.kind for event in group.events)
        return (
            f"label={group.label} events=[{kinds}] "
            f"window={group.clip_start:.3f}..{group.clip_end:.3f} "
            f"duration={group.clip_end - group.clip_start:.3f}s"
        )
