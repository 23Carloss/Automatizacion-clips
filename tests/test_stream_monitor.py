from __future__ import annotations

import unittest
from unittest.mock import patch

from config import Settings
from event_detector import PreparedOcrFrame
from ocr_queue import coalesce_ocr_backlog, offer_ocr_keyframe
from stream_monitor import StreamMonitor


class StreamMonitorTests(unittest.TestCase):
    def test_monitor_does_not_keep_a_stale_frame_batch(self) -> None:
        monitor = StreamMonitor(Settings(twitch_url="https://example.invalid/channel"))
        self.assertFalse(hasattr(monitor, "_latest_batch"))
        self.assertEqual(monitor.settings.sample_fps, 2.0)

    def test_live_timestamp_uses_frame_arrival_time_not_process_start(self) -> None:
        monitor = StreamMonitor(Settings(twitch_url="https://example.invalid/channel"))
        with patch("stream_monitor.time.time", return_value=1234.5):
            self.assertEqual(monitor._frame_timestamp(99), 1234.5)

    def test_vod_timestamp_preserves_seekable_timeline(self) -> None:
        monitor = StreamMonitor(Settings(
            twitch_url="https://example.invalid/vod",
            source_kind="vod",
            sample_fps=2.0,
            vod_start_offset_seconds=30.0,
        ))
        self.assertEqual(monitor._frame_timestamp(5), 32.5)

    def test_full_ocr_queue_coalesces_oldest_pending_keyframe(self) -> None:
        import asyncio
        import numpy as np

        queue: asyncio.Queue[PreparedOcrFrame | None] = asyncio.Queue(maxsize=3)

        def item(timestamp: float) -> PreparedOcrFrame:
            image = np.zeros((1, 1), dtype=np.uint8)
            frame = np.zeros((1, 1, 3), dtype=np.uint8)
            return PreparedOcrFrame(timestamp, frame, image, 1, ("notification",))

        for timestamp in (1.0, 2.0, 3.0):
            queue.put_nowait(item(timestamp))
        offer_ocr_keyframe(queue, item(4.0))

        self.assertEqual(
            [queue.get_nowait().timestamp for _ in range(3)],  # type: ignore[union-attr]
            [2.0, 3.0, 4.0],
        )

    def test_worker_coalesces_all_pending_frames_to_newest(self) -> None:
        import asyncio
        import numpy as np

        queue: asyncio.Queue[PreparedOcrFrame | None] = asyncio.Queue(maxsize=3)

        def item(timestamp: float) -> PreparedOcrFrame:
            image = np.zeros((1, 1), dtype=np.uint8)
            frame = np.zeros((1, 1, 3), dtype=np.uint8)
            return PreparedOcrFrame(timestamp, frame, image, 1, ("notification",))

        first = item(1.0)
        queue.put_nowait(item(2.0))
        queue.put_nowait(item(3.0))

        selected, stop_after, discarded = coalesce_ocr_backlog(queue, first)

        self.assertEqual(selected.timestamp, 3.0)
        self.assertEqual(discarded, 2)
        self.assertFalse(stop_after)
        self.assertTrue(queue.empty())


if __name__ == "__main__":
    unittest.main()
