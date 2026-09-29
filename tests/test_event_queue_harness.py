from __future__ import annotations

import unittest

from config import Settings
from event_detector import DetectedEvent
from event_pipeline import EventGroupPlanner


def event(timestamp: float, kind: str, text: str | None = None) -> DetectedEvent:
    raw_text = text or kind
    return DetectedEvent(timestamp, kind, 0.95, kind, raw_text, "test")


class EventGroupPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(twitch_url="test", player_gamertag="xNopperabe")

    def test_multikill_is_merged_and_later_fight_is_independent(self) -> None:
        planner = EventGroupPlanner(self.settings, merge_gap_seconds=8.0)

        planner.add_event(event(30.0, "KNOCKED"))
        planner.add_event(event(34.0, "BLEEDOUT"))
        planner.add_event(event(37.0, "SQUAD_ELIMINATED"))
        planner.add_event(event(47.0, "SQUAD_ELIMINATED"))
        groups = planner.finish()

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].label, "SQUAD_ELIMINATED")
        self.assertEqual([item.kind for item in groups[0].events], [
            "KNOCKED", "BLEEDOUT", "SQUAD_ELIMINATED",
        ])
        self.assertEqual((groups[0].clip_start, groups[0].clip_end), (9.0, 42.0))
        self.assertEqual(groups[1].label, "SQUAD_ELIMINATED")
        self.assertEqual((groups[1].clip_start, groups[1].clip_end), (27.0, 52.0))

    def test_repeated_ocr_reading_is_deduplicated(self) -> None:
        planner = EventGroupPlanner(self.settings, merge_gap_seconds=8.0)

        planner.add_event(event(30.0, "KNOCKED"))
        planner.add_event(event(31.0, "KNOCKED"))
        groups = planner.finish()

        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].events), 1)

    def test_different_victim_is_not_deduplicated(self) -> None:
        planner = EventGroupPlanner(self.settings, merge_gap_seconds=8.0)

        planner.add_event(event(30.0, "ELIMINATED", "xNopperabe R301 VictimOne"))
        planner.add_event(event(33.0, "ELIMINATED", "xNopperabe R301 VictimTwo"))
        groups = planner.finish()

        self.assertEqual(len(groups[0].events), 2)

    def test_single_blank_ocr_does_not_rearm_lingering_killfeed(self) -> None:
        planner = EventGroupPlanner(self.settings, merge_gap_seconds=8.0)
        first = event(20.0, "ELIMINATED", "xNopperabe donMiguel01989")
        repeated = event(24.0, "ELIMINATED", "Nopperabe donMiguelo1989")

        planner.add_event(first)
        planner.advance(23.0, observed_event=False)
        planner.add_event(repeated)
        groups = planner.finish()

        self.assertEqual(len(groups[0].events), 1)

    def test_real_ocr_sequence_separates_second_fight_at_eight_seconds(self) -> None:
        planner = EventGroupPlanner(self.settings, merge_gap_seconds=8.0)

        planner.add_event(event(18.0, "KNOCKED", "DERRIBADO 811071 150"))
        planner.add_event(event(19.0, "KNOCKED", "DERRIBADO ~ulll L 150 ^"))
        planner.add_event(event(20.0, "ELIMINATED", "xNopperabe donMiguelo]989"))
        planner.add_event(event(21.0, "ELIMINATED", "InS Bumblebe xNopperabe donMiguelol989"))
        planner.add_event(event(22.0, "ELIMINATED", "Nupperabe donMiguelol989"))
        planner.advance(23.0)
        planner.add_event(event(24.0, "ELIMINATED", "Nopperabe donMiguelo1989"))
        planner.add_event(event(25.0, "ELIMINATED", "Nopperabe donMigueld1989"))
        planner.advance(26.0)
        planner.advance(27.0)
        planner.add_event(event(28.0, "KNOCKED", "DERRIBADO"))
        planner.add_event(event(33.0, "SQUAD_ELIMINATED", "ESCUADRON ELIMINADO 100"))
        groups = planner.finish()

        self.assertEqual(len(groups), 2)
        self.assertEqual([item.kind for item in groups[0].events], ["KNOCKED", "ELIMINATED"])
        self.assertEqual(groups[1].label, "SQUAD_ELIMINATED")

    def test_watermark_finalizes_group_without_waiting_for_another_event(self) -> None:
        planner = EventGroupPlanner(self.settings, merge_gap_seconds=8.0)
        planner.add_event(event(30.0, "KNOCKED"))

        planner.advance(39.0)

        self.assertIsNone(planner.active)
        self.assertEqual(len(planner.completed), 1)


if __name__ == "__main__":
    unittest.main()
