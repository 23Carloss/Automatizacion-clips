"""Temporal event grouping for continuous OCR detection.

The detector reports every confirmed visual event.  This module owns the
separate concern of deduplicating lingering OCR text and combining nearby
events into one clip window without pausing frame capture.
"""
from __future__ import annotations

import difflib
import logging
from dataclasses import asdict, dataclass, field

from config import Settings
from event_detector import ApexEventDetector, DetectedEvent
from event_identity import center_victim, killfeed_victim, normalize_text, readable_name

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
    last_activity_at: float | None = None

    @property
    def confidence(self) -> float:
        return max((event.confidence for event in self.events), default=0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "first_event_at": round(self.first_event_at, 3),
            "last_event_at": round(self.last_event_at, 3),
            "last_activity_at": round(self.last_activity_at if self.last_activity_at is not None else self.last_event_at, 3),
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
        self._identity_detector = ApexEventDetector(settings)
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
        self._last_seen_by_kind: dict[str, float] = {}
        self._victim_variants: dict[int, set[str]] = {}
        self._victim_cache: dict[DetectedEvent, str | None] = {}

    def add_event(self, event: DetectedEvent) -> bool:
        """Observe an OCR event; return True only for a new, counted event."""
        if self.active is not None:
            last_activity = self.active.last_activity_at or self.active.last_event_at
            if event.timestamp - last_activity >= self.merge_gap_seconds:
                self._finalize("next event is outside merge gap")
        if self._is_duplicate(event):
            assert self.active is not None
            _, post = self.settings.clip_window(event.kind)
            self.active.last_activity_at = max(
                self.active.last_activity_at or self.active.last_event_at, event.timestamp
            )
            self.active.clip_end = max(self.active.clip_end, event.timestamp + post)
            self._last_seen_by_kind[event.kind] = event.timestamp
            self._blank_observations = 0
            LOGGER.info("Repeated %s at %.3f; extended clip to %.3f.",
                        event.kind, event.timestamp, self.active.clip_end)
            return False
        self._blank_observations = 0
        if self.active is None:
            self.active = self._new_group(event)
            self._last_seen_by_kind = {event.kind: event.timestamp}
            self._victim_variants = {}
            self._victim_cache = {}
            LOGGER.info("Started event group: %s", self.describe(self.active))
            return True
        gap = event.timestamp - (self.active.last_activity_at or self.active.last_event_at)
        self._merge(event)
        self._last_seen_by_kind[event.kind] = event.timestamp
        LOGGER.info("Extended event group after %.3fs: %s", gap, self.describe(self.active))
        return True

    def advance(self, timestamp: float, *, observed_event: bool = False) -> None:
        """Finalize only after OCR has processed beyond the grouping window."""
        self._blank_observations = 0 if observed_event else self._blank_observations + 1
        if self.active and timestamp - (self.active.last_activity_at or self.active.last_event_at) >= self.merge_gap_seconds:
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
            last_activity_at=event.timestamp,
        )

    def _merge(self, event: DetectedEvent) -> None:
        assert self.active is not None
        pre, post = self.settings.clip_window(event.kind)
        self.active.last_event_at = max(self.active.last_event_at, event.timestamp)
        self.active.last_activity_at = max(self.active.last_activity_at or self.active.last_event_at, event.timestamp)
        self.active.clip_start = max(0.0, min(self.active.clip_start, event.timestamp - pre))
        self.active.clip_end = max(self.active.clip_end, event.timestamp + post)
        if self.LABEL_PRIORITY.get(event.kind, 0) > self.LABEL_PRIORITY.get(self.active.label, 0):
            self.active.label = event.kind
        self.active.events.append(event)

    def _is_duplicate(self, event: DetectedEvent) -> bool:
        if self.active is None:
            return False
        incoming = self._victim(event)
        for previous in reversed(self.active.events):
            if previous.kind != event.kind:
                continue
            prior = self._victim(previous)
            if incoming is not None and prior is not None:
                variants = self._victim_variants.setdefault(id(previous), {prior})
                if any(difflib.SequenceMatcher(None, incoming, variant).ratio() >= 0.92
                       for variant in variants):
                    variants.add(incoming)
                    return True
                continue
            # An unreadable name is not an identity. Only suppress a nearby
            # ambiguous repeat while the same visual notification persists.
            if incoming is None and self._blank_observations < 2:
                last_seen = self._last_seen_by_kind.get(event.kind, previous.timestamp)
                return event.timestamp - last_seen <= self.duplicate_window_seconds
            if incoming is not None and prior is None:
                continue
            return False
        return False

    def _victim(self, event: DetectedEvent) -> str | None:
        if event not in self._victim_cache:
            self._victim_cache[event] = self._extract_victim(event)
        return self._victim_cache[event]

    def _extract_victim(self, event: DetectedEvent) -> str | None:
        if event.victim:
            return readable_name(normalize_text(event.victim).split())
        text = event.raw_ocr_text
        if not text:
            return None
        normalized = normalize_text(text)
        if event.region == "notification" or (event.region not in ("owned_killfeed",) and
            normalized.split()[:1] and normalized.split()[0] in {"DERRIBADO", "ELIMINADO", "ASISTENCIA", "ESCUADRON", "SQUAD", "KNOCKED"}):
            match = ApexEventDetector._match_alias(
                text, ApexEventDetector.CENTER_EVENT_KINDS,
                self.settings.ocr_fuzzy_match_threshold, allow_fuzzy=True,
            )
            return center_victim(text, match[1], match[3]) if match else None
        # Compatibility for events created before structured OCR identities.
        actor = self._identity_detector._find_gamertag(normalized)
        if actor is None:
            return None
        match = ApexEventDetector._match_alias(
            text, ApexEventDetector.KILLFEED_EVENT_KINDS,
            self.settings.ocr_fuzzy_match_threshold, allow_fuzzy=False,
        )
        return killfeed_victim(text, actor[1], match[1], match[3]) if match else killfeed_victim(text, actor[1])

    def _finalize(self, reason: str) -> None:
        assert self.active is not None
        LOGGER.info("Event group ready (%s): %s", reason, self.describe(self.active))
        self.completed.append(self.active)
        self.active = None
        self._last_seen_by_kind.clear()
        self._victim_variants.clear()
        self._victim_cache.clear()

    @staticmethod
    def describe(group: EventGroup) -> str:
        kinds = " -> ".join(event.kind for event in group.events)
        return (
            f"label={group.label} events=[{kinds}] "
            f"window={group.clip_start:.3f}..{group.clip_end:.3f} "
            f"duration={group.clip_end - group.clip_start:.3f}s"
        )
