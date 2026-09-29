from __future__ import annotations

import tempfile
import unittest
import sys
import types
from pathlib import Path

try:
    import dotenv  # noqa: F401
except ImportError:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda: None  # type: ignore[attr-defined]
    sys.modules["dotenv"] = dotenv_stub

from config import Settings
from instance_lock import AlreadyRunningError, SingleInstanceLock


class SettingsAndLockTests(unittest.TestCase):
    def test_default_clip_window_is_exactly_25_seconds(self) -> None:
        settings = Settings(twitch_url="https://example.invalid/channel")
        self.assertEqual(settings.pre_event_seconds, 20.0)
        self.assertEqual(settings.post_event_seconds, 5.0)
        self.assertEqual(settings.pre_event_seconds + settings.post_event_seconds, 25.0)

    def test_bleedout_keeps_twenty_five_seconds_before_event(self) -> None:
        settings = Settings(twitch_url="https://example.invalid/channel")
        self.assertEqual(settings.clip_window("BLEEDOUT"), (25.0, 5.0))
        self.assertEqual(settings.clip_window("KNOCKED"), (20.0, 5.0))
        self.assertEqual(settings.buffer_seconds, 120.0)
        self.assertEqual(settings.ffmpeg_encoding_threads, 2)
        self.assertEqual(settings.ffmpeg_max_concurrent_encodes, 1)

    def test_rois_match_1080p_event_regions(self) -> None:
        settings = Settings(twitch_url="https://example.invalid/channel")
        self.assertEqual(
            (settings.notification_roi.x, settings.notification_roi.y,
             settings.notification_roi.width, settings.notification_roi.height),
            (500, 700, 1000, 180),
        )
        self.assertEqual(
            (settings.bleedout_roi.x, settings.bleedout_roi.y,
             settings.bleedout_roi.width, settings.bleedout_roi.height),
            (1150, 110, 730, 230),
        )
        self.assertEqual(settings.sample_fps, 2.0)

    def test_second_instance_cannot_acquire_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "app.lock"
            with SingleInstanceLock(lock_path):
                with self.assertRaises(AlreadyRunningError):
                    with SingleInstanceLock(lock_path):
                        pass


if __name__ == "__main__":
    unittest.main()
