from __future__ import annotations

import asyncio
import importlib
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from config import Settings
from event_detector import DetectedEvent, PreparedOcrFrame
from event_pipeline import EventGroup, TimelineWatermark


def event(timestamp: float, kind: str, text: str) -> DetectedEvent:
    return DetectedEvent(timestamp, kind, 0.95, kind, text, "test")


class MainEventPipelineTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def group() -> EventGroup:
        detected = event(20.0, "ELIMINATED", "xNopperabe Victim")
        return EventGroup(20.0, 20.0, 0.0, 25.0, "ELIMINATED", [detected])

    async def test_render_failure_returns_clip_to_retryable_uploaded_state(self) -> None:
        main = importlib.import_module("main")
        source = Path("apex_20_source.mp4")
        clipper = AsyncMock()
        clipper.create_clip_window.return_value = source
        editor = AsyncMock()
        editor.render.side_effect = RuntimeError("driver failed")
        repository = AsyncMock()
        repository.create_clip.return_value = 88
        repository.get_clip.return_value = SimpleNamespace(status="PROCESSING")
        repository.transition_clip_status.return_value = True

        await main.process_event_group(
            self.group(), "source", clipper, editor, AsyncMock(), repository
        )

        repository.set_processing_progress.assert_awaited_with(88, 0.0)
        repository.transition_clip_status.assert_awaited_with(
            88,
            from_statuses=("PROCESSING",),
            to_status="UPLOADED",
            filename=source.name,
        )

    async def test_telegram_failure_leaves_rendered_clip_pending_approval(self) -> None:
        main = importlib.import_module("main")
        source = Path("apex_20_source.mp4")
        vertical = Path("apex_20_vertical.mp4")
        clipper = AsyncMock()
        clipper.create_clip_window.return_value = source
        editor = AsyncMock()
        editor.render.return_value = vertical
        bot = AsyncMock()
        bot.request_approval.side_effect = RuntimeError("network unavailable")
        repository = AsyncMock()
        repository.create_clip.return_value = 89
        repository.transition_clip_status.return_value = True

        with patch.object(main, "delete_clip_files") as delete_files:
            await main.process_event_group(
                self.group(), "source", clipper, editor, bot, repository
            )

        repository.transition_clip_status.assert_awaited_once_with(
            89,
            from_statuses=("PROCESSING",),
            to_status="PENDING_APPROVAL",
            filename=vertical.name,
        )
        delete_files.assert_not_called()

    async def test_live_ocr_worker_skips_stale_frame_and_shuts_down_cleanly(self) -> None:
        main = importlib.import_module("main")
        settings = Settings(
            twitch_url="test",
            source_kind="live",
            player_gamertag="xNopperabe",
            ocr_max_live_lag_seconds=1.0,
        )
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        image = np.zeros((1, 1), dtype=np.uint8)
        prepared = PreparedOcrFrame(
            time.time() - 10,
            frame,
            image,
            1,
            ("notification",),
        )
        ocr_queue: asyncio.Queue[PreparedOcrFrame | None] = asyncio.Queue()
        event_queue: asyncio.Queue[DetectedEvent | TimelineWatermark | None] = asyncio.Queue()
        await ocr_queue.put(prepared)
        await ocr_queue.put(None)
        detector = Mock()
        ocr_idle = asyncio.Event()
        ocr_idle.set()

        await main.process_ocr_queue(
            ocr_queue, event_queue, detector, settings, ocr_idle
        )

        detector.process_prepared.assert_not_called()
        watermark = event_queue.get_nowait()
        self.assertIsInstance(watermark, TimelineWatermark)
        self.assertIsNone(event_queue.get_nowait())
        self.assertTrue(ocr_idle.is_set())

    async def test_event_worker_merges_multikill_and_keeps_later_fight(self) -> None:
        main = importlib.import_module("main")
        settings = Settings(
            twitch_url="test",
            player_gamertag="xNopperabe",
            event_merge_gap_seconds=8.0,
        )
        event_queue: asyncio.Queue[DetectedEvent | TimelineWatermark | None] = asyncio.Queue()
        ocr_queue: asyncio.Queue[PreparedOcrFrame | None] = asyncio.Queue()
        ocr_idle = asyncio.Event()
        ocr_idle.set()
        tasks: set[asyncio.Task[None]] = set()
        clipper = AsyncMock()
        clipper.reserve_live_segments.return_value = "reservation-test"

        await event_queue.put(event(30.0, "KNOCKED", "DERRIBADO VictimOne"))
        await event_queue.put(event(34.0, "BLEEDOUT", "xNopperabe VictimOne"))
        await event_queue.put(event(37.0, "SQUAD_ELIMINATED", "ESCUADRON ELIMINADO"))
        await event_queue.put(event(47.0, "SQUAD_ELIMINATED", "ESCUADRON ELIMINADO DOS"))
        await event_queue.put(None)

        process_group = AsyncMock()
        with patch.object(main, "process_event_group", process_group):
            await main.process_event_queue(
                event_queue,
                ocr_queue,
                ocr_idle,
                settings,
                "https://example.invalid/live.m3u8",
                clipper,
                AsyncMock(),
                AsyncMock(),
                AsyncMock(),
                tasks,
            )
            if tasks:
                await asyncio.gather(*tasks)
            await asyncio.sleep(0)

        self.assertEqual(process_group.await_count, 2)
        self.assertGreaterEqual(clipper.reserve_live_segments.await_count, 4)
        first_group = process_group.await_args_list[0].args[0]
        second_group = process_group.await_args_list[1].args[0]
        self.assertEqual(first_group.label, "SQUAD_ELIMINATED")
        self.assertEqual(first_group.reservation_id, "reservation-test")
        self.assertEqual(
            [item.kind for item in first_group.events],
            ["KNOCKED", "BLEEDOUT", "SQUAD_ELIMINATED"],
        )
        self.assertEqual(second_group.first_event_at, 47.0)


    async def test_event_worker_reserves_extended_window_for_repeated_victim(self) -> None:
        main = importlib.import_module("main")
        settings = Settings(twitch_url="test", player_gamertag="xNopperabe")
        event_queue: asyncio.Queue[DetectedEvent | TimelineWatermark | None] = asyncio.Queue()
        ocr_queue: asyncio.Queue[PreparedOcrFrame | None] = asyncio.Queue()
        ocr_idle = asyncio.Event()
        ocr_idle.set()
        tasks: set[asyncio.Task[None]] = set()
        clipper = AsyncMock()
        clipper.reserve_live_segments.return_value = "reservation-test"

        await event_queue.put(event(100.0, "ELIMINATED", "xNopperabe R301 VictimOne"))
        await event_queue.put(event(115.0, "ELIMINATED", "xNopperabe R301 VictimOne"))
        await event_queue.put(event(158.29, "ELIMINATED", "xNopperabe R301 VictimTwo"))
        await event_queue.put(None)

        process_group = AsyncMock()
        with patch.object(main, "process_event_group", process_group):
            await main.process_event_queue(
                event_queue, ocr_queue, ocr_idle, settings,
                "https://example.invalid/live.m3u8",
                clipper, AsyncMock(), AsyncMock(), AsyncMock(), tasks,
            )
            if tasks:
                await asyncio.gather(*tasks)

        process_group.assert_awaited_once()
        group = process_group.await_args.args[0]
        self.assertEqual(len(group.events), 2)
        self.assertAlmostEqual(group.clip_end, 163.29)
        reserved_ends = [call.args[1] for call in clipper.reserve_live_segments.await_args_list]
        self.assertIn(120.0, reserved_ends)
        self.assertIn(163.29, reserved_ends)

if __name__ == "__main__":
    unittest.main()
