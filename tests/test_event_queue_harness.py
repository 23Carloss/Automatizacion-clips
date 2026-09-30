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

    def test_real_ocr_sequence_keeps_later_activity_in_one_fight(self) -> None:
        planner = EventGroupPlanner(self.settings)

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

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].label, "SQUAD_ELIMINATED")
        self.assertEqual([item.kind for item in groups[0].events], ["KNOCKED", "ELIMINATED", "KNOCKED", "SQUAD_ELIMINATED"])

    def test_repeated_victim_extends_clip_without_counting_another_kill(self) -> None:
        planner = EventGroupPlanner(self.settings)
        self.assertEqual(self.settings.event_merge_gap_seconds, 50.0)
        self.assertTrue(planner.add_event(event(100.0, "ELIMINATED", "xNopperabe R301 VictimOne")))
        self.assertFalse(planner.add_event(event(107.0, "ELIMINATED", "xNopperabe R301 VictimOne")))
        planner.advance(108.0)
        planner.advance(109.0)
        self.assertFalse(planner.add_event(event(115.0, "ELIMINATED", "xNopperabe R301 VictimOne")))
        assert planner.active is not None
        self.assertEqual(len(planner.active.events), 1)
        self.assertEqual(planner.active.last_event_at, 100.0)
        self.assertEqual(planner.active.last_activity_at, 115.0)
        self.assertEqual(planner.active.clip_end, 120.0)

        # The #106–#107 gap was 43.29 seconds after the last OCR reading.
        planner.advance(158.28)
        self.assertIsNotNone(planner.active)
        self.assertTrue(planner.add_event(event(158.29, "ELIMINATED", "xNopperabe R301 VictimTwo")))
        planner.advance(208.28)
        self.assertIsNotNone(planner.active)
        planner.advance(208.29)
        self.assertIsNone(planner.active)
        self.assertEqual(len(planner.completed), 1)
        group = planner.completed[0]
        self.assertEqual(len(group.events), 2)
        self.assertAlmostEqual(group.clip_end, 163.29)

    def test_unreadable_center_text_is_not_a_victim_identity(self) -> None:
        planner = EventGroupPlanner(self.settings)
        self.assertTrue(planner.add_event(event(10.0, "KNOCKED", "DERRIBADO 811071 150")))
        self.assertFalse(planner.add_event(event(11.0, "KNOCKED", "DERRIBADO ~ulll L 150 ^")))
        self.assertFalse(planner.add_event(event(12.0, "KNOCKED", "DERRIBADO")))
        assert planner.active is not None
        self.assertEqual(len(planner.active.events), 1)
        self.assertEqual(planner.active.last_activity_at, 12.0)
        self.assertEqual(planner.active.clip_end, 17.0)
        self.assertTrue(planner.add_event(event(21.0, "KNOCKED", "DERRIBADO")))
        self.assertEqual(len(planner.active.events), 2)

    def test_known_victim_matches_across_an_ambiguous_notification(self) -> None:
        planner = EventGroupPlanner(self.settings)
        self.assertTrue(planner.add_event(event(10.0, "KNOCKED", "DERRIBADO VictimOne 100")))
        self.assertTrue(planner.add_event(event(20.0, "KNOCKED", "DERRIBADO")))
        self.assertFalse(planner.add_event(event(21.0, "KNOCKED", "DERRIBADO VictimOne 100")))
        self.assertEqual(len(planner.finish()[0].events), 2)

    def test_similar_but_distinct_victims_are_counted(self) -> None:
        planner = EventGroupPlanner(self.settings)
        self.assertTrue(planner.add_event(event(10.0, "ELIMINATED", "xNopperabe R301 Enemy1234")))
        self.assertTrue(planner.add_event(event(11.0, "ELIMINATED", "xNopperabe R301 Enemy1235")))
        self.assertEqual(len(planner.finish()[0].events), 2)

    def test_two_readable_center_victims_count_as_two_knocks(self) -> None:
        planner = EventGroupPlanner(self.settings)
        self.assertTrue(planner.add_event(event(10.0, "KNOCKED", "DERRIBADO VictimOne 100")))
        self.assertTrue(planner.add_event(event(11.0, "KNOCKED", "DERRIBADO VictimTwo 100")))
        self.assertFalse(planner.add_event(event(12.0, "KNOCKED", "DERRIBADO VictimOne 100")))
        self.assertEqual(len(planner.finish()[0].events), 2)

    def test_watermark_finalizes_group_without_waiting_for_another_event(self) -> None:
        planner = EventGroupPlanner(self.settings, merge_gap_seconds=8.0)
        planner.add_event(event(30.0, "KNOCKED"))

        planner.advance(39.0)

        self.assertIsNone(planner.active)
        self.assertEqual(len(planner.completed), 1)


if __name__ == "__main__":
    unittest.main()
