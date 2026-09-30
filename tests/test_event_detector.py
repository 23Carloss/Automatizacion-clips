from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import Mock

import numpy as np

try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = Mock()

try:
    import dotenv  # noqa: F401
except ImportError:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda: None  # type: ignore[attr-defined]
    sys.modules["dotenv"] = dotenv_stub

from config import Settings
from event_detector import ApexEventDetector, OcrLine, PreparedOcrFrame


class ApexEventDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            twitch_url="https://example.invalid/channel", player_gamertag="xNopperabe"
        )
        self.detector = ApexEventDetector(self.settings)
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    @staticmethod
    def center(text: str, confidence: float = 0.95) -> list[OcrLine]:
        return [OcrLine(text, confidence, 70)]

    @staticmethod
    def killfeed(text: str, confidence: float = 0.95) -> list[OcrLine]:
        return [OcrLine(text, confidence, 220)]

    def prepared(self, timestamp: float) -> PreparedOcrFrame:
        return PreparedOcrFrame(timestamp, self.frame, np.zeros((430, 1000), dtype=np.uint8), 180, ("notification", "killfeed"))

    def test_combines_central_and_killfeed_rois_once(self) -> None:
        captured_shapes: list[tuple[int, ...]] = []

        def capture(image: np.ndarray) -> list[OcrLine]:
            captured_shapes.append(image.shape)
            return []

        self.detector._ocr_lines = capture  # type: ignore[method-assign]
        self.assertIsNone(self.detector.detect(self.frame, 100.0))
        self.assertEqual(captured_shapes, [(310, 720)])

    def test_requires_same_alias_in_two_selected_keyframes(self) -> None:
        self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
            side_effect=(
                self.center("DERRIBAD0"),
                self.center("KNOCKEDD"),
                self.center("KNOCKEDD"),
            )
        )

        events = [self.detector.process_prepared(self.prepared(value)) for value in (1.0, 1.5, 2.0)]

        self.assertEqual(events[:2], [None, None])
        self.assertIsNotNone(events[2])
        assert events[2] is not None
        self.assertEqual(events[2].kind, "KNOCKED")
        self.assertEqual(events[2].timestamp, 1.5)

    def test_one_frame_flash_resets_confirmation(self) -> None:
        self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
            side_effect=(
                self.center("DERRIBAD0"), [], self.center("DERRIBAD0")
            )
        )
        for timestamp in (1.0, 1.5, 2.0):
            self.assertIsNone(self.detector.process_prepared(self.prepared(timestamp)))

    def test_low_confidence_text_is_rejected(self) -> None:
        self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
            return_value=self.center("ELIMINADO", confidence=0.39)
        )
        for timestamp in (1.0, 1.333, 1.667):
            self.assertIsNone(self.detector.process_prepared(self.prepared(timestamp)))

    def test_third_party_bleedout_is_rejected(self) -> None:
        self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
            return_value=self.killfeed("ThickerMilk69 [Desangrado] asteria_athxna")
        )
        for timestamp in (1.0, 1.333, 1.667):
            self.assertIsNone(self.detector.process_prepared(self.prepared(timestamp)))

    def test_killfeed_keeps_actor_and_victim_separate(self) -> None:
        self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
            side_effect=(
                self.killfeed("xNopperabe R301 VictimOne"),
                self.killfeed("xNopperabe R301 VictimTwo"),
            )
        )
        first = self.detector.process_prepared(self.prepared(1.0))
        second = self.detector.process_prepared(self.prepared(2.0))
        assert first is not None and second is not None
        self.assertEqual((first.victim, second.victim), ("VICTIMONE", "VICTIMTWO"))
        self.assertEqual((first.kind, second.kind), ("ELIMINATED", "ELIMINATED"))

    def test_center_victim_requires_a_readable_name(self) -> None:
        self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
            side_effect=(
                self.center("DERRIBADO 811071 150"),
                self.center("DERRIBADO VictimOne 100"),
            )
        )
        first = self.detector.process_prepared(self.prepared(1.0))
        second = self.detector.process_prepared(self.prepared(2.0))
        assert first is not None and second is not None
        self.assertIsNone(first.victim)
        self.assertEqual(second.victim, "VICTIMONE")

    def test_player_bleedout_as_actor_is_confirmed(self) -> None:
        self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
            return_value=self.killfeed("[TEAM] xNopperabe [Desangrado] asteria_athxna")
        )

        event = self.detector.process_prepared(self.prepared(1.0))

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.kind, "BLEEDOUT")
        self.assertEqual(event.region, "owned_killfeed")
        self.assertIn("xNopperabe", event.evidence)
        self.assertIn("1 processed frame", event.evidence)
        self.assertNotEqual(event.confidence, 0.99)

    def test_single_owned_elimination_is_enough_after_marginal_squad_candidate(self) -> None:
        self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
            side_effect=(
                self.center("ESCUADR0N ELIMINAD0 +100", confidence=0.75),
                self.killfeed("xNopperabe [Desangrado] Zonic2322", confidence=0.91),
            )
        )

        self.assertIsNone(self.detector.process_prepared(self.prepared(1.0)))
        event = self.detector.process_prepared(self.prepared(1.333))

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.kind, "BLEEDOUT")
        self.assertEqual(event.timestamp, 1.333)
        self.assertAlmostEqual(event.confidence, 0.91)

    def test_player_as_victim_does_not_count_as_owned_kill(self) -> None:
        self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
            return_value=self.killfeed("ThickerMilk69 [Desangrado] xNopperabe")
        )
        for timestamp in (1.0, 1.333, 1.667):
            self.assertIsNone(self.detector.process_prepared(self.prepared(timestamp)))

    def test_personal_center_notification_does_not_require_gamertag(self) -> None:
        self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
            return_value=self.center("ESCUADRÓN ELIMINADO +100")
        )
        event = self.detector.process_prepared(self.prepared(1.0))
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.kind, "SQUAD_ELIMINATED")
        self.assertEqual(event.region, "notification")

    def test_high_similarity_partial_assist_notification_triggers_once(self) -> None:
        self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
            return_value=self.center("SISTENCIA DERRIBADO ANASSASSINA386 100", confidence=0.75)
        )

        event = self.detector.process_prepared(self.prepared(1.0))

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.kind, "KNOCKED")
        self.assertIn("1 processed frame", event.evidence)
        self.assertAlmostEqual(event.confidence, 0.75)

    def test_tolerates_one_character_center_notification_typo(self) -> None:
        match = ApexEventDetector._match_event("DERRIBAD0 jugador")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match[0], "KNOCKED")

    def test_spanish_elimination_is_an_exact_single_frame_alias(self) -> None:
        result = ApexEventDetector._match_alias(
            "ASISTENCIA ELIMINACIÓN [VALK] JpB_144441",
            ApexEventDetector.CENTER_EVENT_KINDS,
            self.settings.ocr_fuzzy_match_threshold,
            allow_fuzzy=True,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[:3], ("ELIMINATED", "ELIMINACION", 1.0))

    def test_standard_killfeed_without_keyword_is_player_elimination(self) -> None:
        self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
            return_value=self.killfeed("xNopperabe R301 VictimPlayer")
        )

        event = self.detector.process_prepared(self.prepared(1.0))

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.kind, "ELIMINATED")
        self.assertEqual(event.region, "owned_killfeed")

    def test_explicit_eliminated_by_pattern_is_rejected(self) -> None:
        for text in ("ELIMINADO POR xNopperabe", "DERRIBADO POR xNopperabe"):
            with self.subTest(text=text):
                self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
                    return_value=self.killfeed(text)
                )
                self.assertIsNone(self.detector.process_prepared(self.prepared(1.0)))

    def test_non_combat_status_rows_are_not_player_eliminations(self) -> None:
        rows = (
            "xNopperabe está fabricando Banner de jugador",
            "BigFawn57098836 Reanimando a xNopperabe",
            "Gracias xNopperabe",
            "xNopperabe [El circulo] Durbed peas",
        )
        for index, text in enumerate(rows):
            with self.subTest(text=text):
                self.detector._ocr_lines = Mock(  # type: ignore[method-assign]
                    return_value=self.killfeed(text)
                )
                self.assertIsNone(
                    self.detector.process_prepared(self.prepared(float(index + 1)))
                )

    def test_unchanged_rois_skip_second_ocr_call(self) -> None:
        self.detector._ocr_lines = Mock(return_value=[])  # type: ignore[method-assign]

        self.detector.detect(self.frame, 1.0)
        self.detector.detect(self.frame.copy(), 1.5)

        self.assertEqual(self.detector._ocr_lines.call_count, 1)

    def test_changed_frames_are_throttled_to_one_keyframe_per_second(self) -> None:
        self.detector._ocr_lines = Mock(return_value=[])  # type: ignore[method-assign]
        changed = np.full_like(self.frame, 255)

        self.detector.detect(self.frame, 1.0)
        self.detector.detect(changed, 1.5)
        self.detector.detect(self.frame, 2.0)

        self.assertEqual(self.detector._ocr_lines.call_count, 2)


if __name__ == "__main__":
    unittest.main()
