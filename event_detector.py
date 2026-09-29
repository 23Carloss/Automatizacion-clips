"""OCR-only detector for player-owned Apex visual events."""
from __future__ import annotations

import asyncio
import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from config import Roi, Settings

LOGGER = logging.getLogger("apex_clipper.detector")

try:
    import pytesseract
except ImportError:  # OCR is optional at runtime.
    pytesseract = None

try:
    import easyocr
except ImportError:  # Tesseract remains available as a fallback.
    easyocr = None


@dataclass(frozen=True, slots=True)
class OcrLine:
    text: str
    confidence: float
    y_center: float


@dataclass(frozen=True, slots=True)
class EventMatch:
    kind: str
    alias: str
    region: str
    text: str
    confidence: float
    keyword_similarity: float

    @property
    def confirmation_key(self) -> tuple[str, str, str]:
        return self.region, self.kind, self.alias


@dataclass(frozen=True, slots=True)
class DetectedEvent:
    timestamp: float
    kind: str
    confidence: float
    evidence: str
    raw_ocr_text: str = ""
    region: str = ""


@dataclass(frozen=True, slots=True)
class PreparedOcrFrame:
    """A motion-selected, preprocessed keyframe ready for the slow OCR worker."""

    timestamp: float
    frame: np.ndarray
    image: np.ndarray
    notification_height: int
    changed_regions: tuple[str, ...]


class ApexEventDetector:
    """Confirm the same high-confidence, player-owned evidence across frames."""

    EVENT_ALIASES = (
        ("SQUAD_ELIMINATED", (
            "ESCUADRON ELIMINADO", "ESCUADRON ELIMINADA", "SQUAD ELIMINATED", "SQUAD WIPE",
        )),
        ("KNOCKED", (
            "ASISTENCIA DERRIBADO", "DERRIBADO", "DERRIBASTE", "DOWNED", "KNOCKED",
        )),
        ("BLEEDOUT", ("DESANGRADO", "DESANGRO", "BLEED OUT", "BLEEDOUT")),
        ("ELIMINATED", (
            "ELIMINADO", "ELIMINACION", "ELIMINASTE", "ELIMINATED", "ELIMINATION",
        )),
    )
    CENTER_EVENT_KINDS = frozenset({"SQUAD_ELIMINATED", "KNOCKED", "ELIMINATED"})
    KILLFEED_EVENT_KINDS = frozenset({"KNOCKED", "BLEEDOUT", "ELIMINATED"})
    REQUIRED_CONSECUTIVE_FRAMES = 3
    OWNED_KILLFEED_REQUIRED_FRAMES = 1
    COMBINED_REGION_GAP = 20
    VICTIM_PATTERNS = (
        "ELIMINADO POR", "DERRIBADO POR", "ELIMINATED BY", "KNOCKED BY",
    )
    NON_COMBAT_KILLFEED_PATTERNS = (
        "FABRICANDO BANNER",
        "BANNER DE JUGADOR",
        "REANIMANDO A",
        "GRACIAS",
        "EL CIRCULO",
        "THE RING",
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._last_event_at = float("-inf")
        self._candidate_key: tuple[str, str, str] | None = None
        self._candidate_count = 0
        self._candidate_timestamp = 0.0
        self._candidate_last_timestamp = 0.0
        self._candidate_confidences: list[float] = []
        self._candidate_text = ""
        self._easyocr_reader: Any | None = None
        self._easyocr_unavailable = easyocr is None
        self._previous_motion_images: dict[str, np.ndarray] = {}
        self._last_keyframe_at = float("-inf")

    async def initialize(self) -> None:
        """Prepare an OCR engine before opening the live-video frame pipe."""
        if not self._easyocr_unavailable:
            await asyncio.to_thread(self._initialize_easyocr)
        if self._easyocr_reader is not None:
            LOGGER.info(
                "Visual detector ready with EasyOCR; minimum confidence %.0f%%; player %s.",
                self.settings.ocr_min_confidence * 100, self.settings.player_gamertag,
            )
            return
        if self.settings.tesseract_enabled and pytesseract is not None:
            try:
                await asyncio.to_thread(pytesseract.get_tesseract_version)
                LOGGER.info(
                    "Visual detector ready with Tesseract; minimum confidence %.0f%%; player %s.",
                    self.settings.ocr_min_confidence * 100, self.settings.player_gamertag,
                )
                return
            except Exception:
                pass
        raise RuntimeError(
            "No usable OCR engine was found. Install EasyOCR dependencies or the Tesseract executable."
        )

    @staticmethod
    def _scaled_roi(frame: np.ndarray, roi: Roi) -> np.ndarray:
        height, width = frame.shape[:2]
        sx, sy = width / 1920, height / 1080
        x, y = int(roi.x * sx), int(roi.y * sy)
        w, h = int(roi.width * sx), int(roi.height * sy)
        return frame[y : min(y + h, height), x : min(x + w, width)]

    @staticmethod
    def _text_image(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)[1]

    def reset_session(self) -> None:
        """Discard motion/candidate state when a source is opened again."""
        self._previous_motion_images.clear()
        self._last_keyframe_at = float("-inf")
        self._reset_candidate()

    def _motion_signature(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        scale = self.settings.ocr_motion_size / max(height, width)
        signature = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.GaussianBlur(signature, (5, 5), 0)

    def _has_significant_motion(self, region: str, image: np.ndarray) -> bool:
        """Update the cheap ROI baseline and report a meaningful visual change."""
        signature = self._motion_signature(image)
        previous = self._previous_motion_images.get(region)
        self._previous_motion_images[region] = signature
        if previous is None or previous.shape != signature.shape:
            return True
        difference = cv2.absdiff(previous, signature)
        changed_ratio = float(np.count_nonzero(
            difference >= self.settings.ocr_motion_pixel_threshold
        )) / difference.size
        return changed_ratio >= self.settings.ocr_motion_threshold

    def _preprocess_roi(self, image: np.ndarray) -> np.ndarray:
        """Create a compact, high-contrast grayscale input for OCR."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary_mask = cv2.threshold(
            gray, self.settings.ocr_binary_threshold, 255, cv2.THRESH_BINARY
        )[1]
        # Preserve grayscale antialiasing inside the thresholded foreground.
        # Pure black/white input was slower and less accurate on the supplied capture.
        return cv2.bitwise_and(gray, gray, mask=binary_mask)

    def _reset_candidate(self) -> None:
        self._candidate_key = None
        self._candidate_count = 0
        self._candidate_timestamp = 0.0
        self._candidate_last_timestamp = 0.0
        self._candidate_confidences.clear()
        self._candidate_text = ""

    @classmethod
    def _combine_regions(cls, notification: np.ndarray, killfeed: np.ndarray) -> np.ndarray:
        """Place both ROIs on one canvas while keeping their vertical identity."""
        height = notification.shape[0] + cls.COMBINED_REGION_GAP + killfeed.shape[0]
        width = max(notification.shape[1], killfeed.shape[1])
        shape = (height, width) + notification.shape[2:]
        combined = np.zeros(shape, dtype=notification.dtype)
        combined[: notification.shape[0], : notification.shape[1], ...] = notification
        offset = notification.shape[0] + cls.COMBINED_REGION_GAP
        combined[offset : offset + killfeed.shape[0], : killfeed.shape[1], ...] = killfeed
        return combined

    def prepare_frame(self, frame: np.ndarray, timestamp: float) -> PreparedOcrFrame | None:
        """Perform ROI cropping, motion gating and preprocessing without invoking OCR."""
        if timestamp - self._last_event_at < self.settings.event_cooldown_seconds:
            return None
        notification = self._scaled_roi(frame, self.settings.notification_roi)
        killfeed = self._scaled_roi(frame, self.settings.killfeed_roi)
        if notification.size == 0 or killfeed.size == 0:
            LOGGER.warning("OCR ROI is empty for a frame with shape %s", frame.shape)
            return None

        changed = {
            "notification": self._has_significant_motion("notification", notification),
            "killfeed": self._has_significant_motion("killfeed", killfeed),
        }
        if not any(changed.values()):
            return None
        if timestamp - self._last_keyframe_at < self.settings.ocr_keyframe_interval_seconds:
            return None
        self._last_keyframe_at = timestamp

        notification_image = (
            self._preprocess_roi(notification)
            if changed["notification"] else np.zeros(notification.shape[:2], dtype=np.uint8)
        )
        killfeed_image = (
            self._preprocess_roi(killfeed)
            if changed["killfeed"] else np.zeros(killfeed.shape[:2], dtype=np.uint8)
        )
        combined = self._combine_regions(notification_image, killfeed_image)
        scale = min(1.0, self.settings.ocr_max_image_width / combined.shape[1])
        notification_height = notification.shape[0]
        if scale < 1.0:
            combined = cv2.resize(
                combined,
                (round(combined.shape[1] * scale), round(combined.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
            notification_height = round(notification_height * scale)
        return PreparedOcrFrame(
            timestamp,
            frame,
            combined,
            notification_height,
            tuple(region for region, did_change in changed.items() if did_change),
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text.upper())
        without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
        return re.sub(r"[^A-Z0-9]+", " ", without_accents).strip()

    @classmethod
    def _aliases_for(cls, allowed_kinds: frozenset[str]) -> Iterable[tuple[str, str]]:
        for event_kind, aliases in cls.EVENT_ALIASES:
            if event_kind in allowed_kinds:
                for alias in aliases:
                    yield event_kind, cls._normalize_text(alias)

    @classmethod
    def _match_alias(
        cls, text: str, allowed_kinds: frozenset[str], fuzzy_threshold: float, *, allow_fuzzy: bool,
    ) -> tuple[str, str, float, int] | None:
        normalized = cls._normalize_text(text)
        if not normalized:
            return None
        tokens = normalized.split()
        best: tuple[str, str, float, int] | None = None
        for event_kind, alias in cls._aliases_for(allowed_kinds):
            exact = re.search(rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])", normalized)
            if exact:
                return event_kind, alias, 1.0, exact.start()
            if not allow_fuzzy:
                continue
            alias_tokens = alias.split()
            width = len(alias_tokens)
            for index in range(max(0, len(tokens) - width + 1)):
                candidate = " ".join(tokens[index : index + width])
                score = difflib.SequenceMatcher(None, candidate, alias).ratio()
                if score < fuzzy_threshold:
                    continue
                char_index = len(" ".join(tokens[:index])) + (1 if index else 0)
                proposed = event_kind, alias, score, char_index
                if best is None or proposed[2] > best[2]:
                    best = proposed
        return best

    @classmethod
    def _match_event(cls, ocr_text: str) -> tuple[str, str] | None:
        """Compatibility helper for center-notification matching."""
        match = cls._match_alias(ocr_text, cls.CENTER_EVENT_KINDS, 0.88, allow_fuzzy=True)
        return (match[0], match[1]) if match else None

    def _initialize_easyocr(self) -> None:
        if self._easyocr_unavailable or self._easyocr_reader is not None:
            return
        try:
            self._easyocr_reader = easyocr.Reader(["es", "en"], gpu=False)
        except Exception as exc:
            LOGGER.warning("EasyOCR could not initialize; trying Tesseract instead: %s", exc)
            self._easyocr_unavailable = True
            self._easyocr_reader = None

    @staticmethod
    def _group_ocr_results(
        results: list[tuple[Any, str, float]], *, coordinate_scale: float = 1.0,
    ) -> list[OcrLine]:
        entries: list[tuple[float, float, str, float]] = []
        for box, text, confidence in results:
            if not str(text).strip():
                continue
            xs = [float(point[0]) / coordinate_scale for point in box]
            ys = [float(point[1]) / coordinate_scale for point in box]
            entries.append((sum(ys) / len(ys), min(xs), str(text), float(confidence)))
        entries.sort(key=lambda value: (value[0], value[1]))
        groups: list[list[tuple[float, float, str, float]]] = []
        for entry in entries:
            target = next(
                (group for group in groups if abs(entry[0] - sum(item[0] for item in group) / len(group)) <= 18),
                None,
            )
            if target is None:
                groups.append([entry])
            else:
                target.append(entry)
        lines = []
        for group in groups:
            ordered = sorted(group, key=lambda value: value[1])
            lines.append(OcrLine(
                text=" ".join(item[2] for item in ordered),
                # A weakly recognized victim name must not invalidate a strong
                # gamertag or event keyword elsewhere on the same visual row.
                confidence=max(item[3] for item in ordered),
                y_center=sum(item[0] for item in ordered) / len(ordered),
            ))
        return lines

    def _ocr_with_easyocr(self, image: np.ndarray) -> list[OcrLine]:
        if self._easyocr_unavailable:
            return []
        try:
            if self._easyocr_reader is None:
                self._initialize_easyocr()
            if self._easyocr_reader is None:
                return []
            results = self._easyocr_reader.readtext(
                image,
                detail=1,
                paragraph=False,
                decoder="greedy",
                beamWidth=1,
                batch_size=1,
                workers=0,
                canvas_size=self.settings.ocr_max_image_width,
                mag_ratio=1.0,
            )
            return self._group_ocr_results(results)
        except Exception:
            self._easyocr_unavailable = True
            self._easyocr_reader = None
            return []

    def _ocr_with_tesseract(self, image: np.ndarray) -> list[OcrLine]:
        if not self.settings.tesseract_enabled or pytesseract is None:
            return []
        try:
            data = pytesseract.image_to_data(
                self._text_image(image), config="--psm 11", output_type=pytesseract.Output.DICT
            )
        except Exception:
            return []
        grouped: dict[tuple[int, int, int], list[int]] = {}
        for index, text in enumerate(data["text"]):
            try:
                confidence = float(data["conf"][index]) / 100
            except (TypeError, ValueError):
                continue
            if not str(text).strip() or confidence < 0:
                continue
            key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            grouped.setdefault(key, []).append(index)
        synthetic_results = []
        for indexes in grouped.values():
            left = min(data["left"][index] for index in indexes)
            top = min(data["top"][index] for index in indexes)
            right = max(data["left"][index] + data["width"][index] for index in indexes)
            bottom = max(data["top"][index] + data["height"][index] for index in indexes)
            text = " ".join(str(data["text"][index]) for index in indexes)
            confidence = min(float(data["conf"][index]) / 100 for index in indexes)
            synthetic_results.append((
                ((left, top), (right, top), (right, bottom), (left, bottom)), text, confidence,
            ))
        return self._group_ocr_results(synthetic_results)

    def _ocr_lines(self, image: np.ndarray) -> list[OcrLine]:
        lines = self._ocr_with_easyocr(image)
        if not lines:
            lines = self._ocr_with_tesseract(image)
        return lines

    def _match_center_line(self, line: OcrLine) -> EventMatch | None:
        if line.confidence < self.settings.ocr_min_confidence:
            return None
        result = self._match_alias(
            line.text, self.CENTER_EVENT_KINDS, self.settings.ocr_fuzzy_match_threshold,
            allow_fuzzy=True,
        )
        if result is None:
            return None
        kind, alias, similarity, _ = result
        return EventMatch(
            kind, alias, "notification", line.text,
            min(line.confidence, similarity), similarity,
        )

    def _find_gamertag(self, normalized_text: str) -> tuple[int, int, float] | None:
        """Locate the configured player even when OCR makes a small character error."""
        expected = self._normalize_text(self.settings.player_gamertag)
        tokens = normalized_text.split()
        expected_width = len(expected.split())
        best: tuple[int, int, float] | None = None
        for index in range(max(0, len(tokens) - expected_width + 1)):
            candidate = " ".join(tokens[index : index + expected_width])
            score = difflib.SequenceMatcher(None, candidate, expected).ratio()
            if score < self.settings.gamertag_match_threshold:
                continue
            start = len(" ".join(tokens[:index])) + (1 if index else 0)
            proposed = start, start + len(candidate), score
            if best is None or proposed[2] > best[2]:
                best = proposed
        return best

    def _match_owned_killfeed_line(self, line: OcrLine) -> EventMatch | None:
        if line.confidence < self.settings.ocr_min_confidence:
            return None
        normalized = self._normalize_text(line.text)
        if any(pattern in normalized for pattern in self.VICTIM_PATTERNS):
            LOGGER.info("Ignored killfeed victim pattern: %r", line.text)
            return None
        if any(pattern in normalized for pattern in self.NON_COMBAT_KILLFEED_PATTERNS):
            LOGGER.info("Ignored non-combat killfeed-like row: %r", line.text)
            return None

        gamertag = self._find_gamertag(normalized)
        if gamertag is None:
            return None
        gamertag_start, gamertag_end, gamertag_score = gamertag
        # In Apex's standard "actor [weapon icon] victim" row, the actor is in
        # the left half. Also require visible text after the actor so a lone HUD
        # username cannot be mistaken for a killfeed event.
        if (gamertag_start + gamertag_end) / 2 > len(normalized) / 2:
            LOGGER.info("Ignored killfeed row where player is on the victim side: %r", line.text)
            return None
        suffix = normalized[gamertag_end:].strip()
        if not suffix:
            return None

        # Preserve explicit BLEEDOUT/KNOCKED/ELIMINATED classification when it
        # exists, but normal weapon-icon rows have no event keyword and are an
        # elimination by the configured player.
        result = self._match_alias(
            line.text, self.KILLFEED_EVENT_KINDS, self.settings.ocr_fuzzy_match_threshold,
            allow_fuzzy=False,
        )
        if result is None:
            kind, alias, similarity = "ELIMINATED", "PLAYER KILLFEED", gamertag_score
        else:
            kind, alias, similarity, alias_index = result
            if gamertag_start >= alias_index:
                LOGGER.info("Ignored killfeed row where player follows the event marker: %r", line.text)
                return None
        confidence = min(line.confidence, similarity, gamertag_score)
        return EventMatch(kind, alias, "owned_killfeed", line.text, confidence, similarity)

    def _find_match(self, lines: list[OcrLine], notification_height: int) -> EventMatch | None:
        killfeed_start = notification_height + self.COMBINED_REGION_GAP / 2
        for line in lines:
            if line.y_center < killfeed_start:
                match = self._match_center_line(line)
                if match:
                    return match
        for line in lines:
            if line.y_center >= killfeed_start:
                match = self._match_owned_killfeed_line(line)
                if match:
                    return match
        return None

    def process_prepared(self, prepared: PreparedOcrFrame) -> DetectedEvent | None:
        """Run OCR and event interpretation for a previously selected keyframe."""
        if prepared.timestamp - self._last_event_at < self.settings.event_cooldown_seconds:
            self._reset_candidate()
            return None
        lines = self._ocr_lines(prepared.image)
        match = self._find_match(lines, prepared.notification_height)
        if match is None:
            self._reset_candidate()
            return None

        key = match.confirmation_key
        required_frames = (
            self.OWNED_KILLFEED_REQUIRED_FRAMES
            if match.region == "owned_killfeed"
            else (
                1
                if match.keyword_similarity >= self.settings.ocr_single_frame_similarity
                else self.settings.ocr_confirmation_frames
            )
        )
        if (
            self._candidate_key is not None
            and prepared.timestamp - self._candidate_last_timestamp
            > self.settings.ocr_candidate_max_gap_seconds
        ):
            self._reset_candidate()
        if key != self._candidate_key:
            self._candidate_key = key
            self._candidate_count = 1
            self._candidate_timestamp = prepared.timestamp
            self._candidate_last_timestamp = prepared.timestamp
            self._candidate_confidences = [match.confidence]
            self._candidate_text = match.text
            LOGGER.info(
                "OCR candidate %s/%s from %s: 1/%d confidence=%.0f%% text=%r",
                match.kind, match.alias, match.region, required_frames,
                match.confidence * 100, match.text,
            )
            if required_frames > 1:
                return None
            return self._confirmed_event(match, prepared.timestamp)

        self._candidate_count += 1
        self._candidate_last_timestamp = prepared.timestamp
        self._candidate_confidences.append(match.confidence)
        self._candidate_text = match.text
        LOGGER.info(
            "OCR candidate %s/%s from %s: %d/%d confidence=%.0f%% text=%r",
            match.kind, match.alias, match.region, self._candidate_count,
            required_frames, match.confidence * 100, match.text,
        )
        if self._candidate_count < required_frames:
            return None

        return self._confirmed_event(match, prepared.timestamp)

    def detect(self, frame: np.ndarray, timestamp: float) -> DetectedEvent | None:
        """Synchronous compatibility API used by tests and offline callers."""
        prepared = self.prepare_frame(frame, timestamp)
        if prepared is None:
            return None
        return self.process_prepared(prepared)

    def _confirmed_event(self, match: EventMatch, confirmation_timestamp: float) -> DetectedEvent:
        event_timestamp = self._candidate_timestamp
        confidence = sum(self._candidate_confidences) / len(self._candidate_confidences)
        raw_text = self._candidate_text
        owner = f" for {self.settings.player_gamertag}" if match.region == "owned_killfeed" else ""
        confirmations = len(self._candidate_confidences)
        evidence = (
            f"{match.alias} in {match.region}{owner}; confirmed in "
            f"{confirmations} processed frame{'s' if confirmations != 1 else ''}"
        )
        self._last_event_at = confirmation_timestamp
        self._reset_candidate()
        return DetectedEvent(event_timestamp, match.kind, confidence, evidence, raw_text, match.region)
