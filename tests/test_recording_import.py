from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from config import Settings
from event_detector import DetectedEvent
from event_pipeline import EventGroup
from recording_import import import_recording, probe_recording, scan_recording


class FakeDetector:
    def prepare_frame(self, frame, timestamp: float):
        return SimpleNamespace(timestamp=timestamp)

    def process_prepared(self, prepared):
        if prepared.timestamp == 0.5:
            return DetectedEvent(
                0.5, "ELIMINATED", 0.95, "test event", "ELIMINADO", "notification"
            )
        return None


class RecordingImportTests(unittest.IsolatedAsyncioTestCase):
    async def test_scan_streams_frames_and_groups_only_detected_events(self) -> None:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("FFmpeg is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "hour_source.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=black:s=320x180:r=2:d=2",
                    "-c:v", "libx264", "-preset", "ultrafast", "-y", str(source),
                ],
                check=True, capture_output=True,
            )
            settings = Settings(
                twitch_url="https://example.invalid", source_kind="vod",
                evidence_dir=Path(directory) / "evidence",
            )
            duration = probe_recording(source)
            groups, events = await scan_recording(
                source, settings, FakeDetector(), duration, import_id="test"
            )
            self.assertEqual(events, 1)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].clip_start, 0.0)
            self.assertEqual(groups[0].clip_end, duration)
            self.assertTrue(list((Path(directory) / "evidence").glob("*.json")))

    async def test_detected_group_becomes_one_clip_without_copying_recording(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recording.mp4"
            source.write_bytes(b"original")
            source_clip = root / "clips" / "ps5_example_100_source.mp4"
            vertical_clip = root / "clips" / "ps5_example_100_vertical.mp4"
            settings = Settings(
                twitch_url="https://example.invalid",
                clips_dir=root / "clips",
                cache_dir=root / "cache",
                evidence_dir=root / "evidence",
            )
            event = DetectedEvent(
                100.0, "ELIMINATED", 0.95, "test", "ELIMINADO", "notification"
            )
            group = EventGroup(100.0, 100.0, 80.0, 105.0, "ELIMINATED", [event])
            repository = SimpleNamespace(
                create_clip=AsyncMock(return_value=7),
                transition_clip_status=AsyncMock(return_value=True),
                processing_should_stop=AsyncMock(return_value=False),
                set_processing_progress=AsyncMock(),
            )
            with patch("recording_import.probe_recording", return_value=3600.0), patch(
                "recording_import.scan_recording",
                AsyncMock(return_value=([group], 1)),
            ), patch("recording_import.ApexEventDetector") as detector_class, patch(
                "recording_import.Clipper"
            ) as clipper_class, patch(
                "recording_import.ApexVerticalEditor"
            ) as editor_class, patch(
                "recording_import._send_approval", AsyncMock()
            ) as approval:
                detector_class.return_value.initialize = AsyncMock()
                clipper_class.return_value.create_clip_window = AsyncMock(
                    return_value=source_clip
                )
                editor_class.return_value.render = AsyncMock(
                    return_value=vertical_clip
                )
                result = await import_recording(source, settings, repository)
            self.assertEqual((result.events, result.groups, result.clips_created), (1, 1, 1))
            self.assertEqual(result.clips_failed, 0)
            self.assertEqual(source.read_bytes(), b"original")
            kwargs = clipper_class.return_value.create_clip_window.await_args.kwargs
            self.assertTrue(kwargs["name_prefix"].startswith("ps5_"))
            self.assertTrue(kwargs["normalize_to_1080"])
            repository.create_clip.assert_awaited_once_with(
                source_clip.name, "PROCESSING", "PS5_RECORDING"
            )
            approval.assert_awaited_once()

    async def test_no_events_creates_no_clip_or_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "recording.mp4"
            source.write_bytes(b"original")
            settings = Settings(
                twitch_url="https://example.invalid",
                clips_dir=Path(directory) / "clips",
                cache_dir=Path(directory) / "cache",
                evidence_dir=Path(directory) / "evidence",
            )
            repository = SimpleNamespace()
            with patch("recording_import.probe_recording", return_value=3600.0), patch(
                "recording_import.scan_recording", AsyncMock(return_value=([], 0))
            ), patch(
                "recording_import.ApexEventDetector"
            ) as detector_class, patch(
                "recording_import.Clipper"
            ) as clipper_class:
                detector_class.return_value.initialize = AsyncMock()
                result = await import_recording(source, settings, repository)
            self.assertEqual(result.groups, 0)
            self.assertEqual(result.clips_created, 0)
            self.assertEqual(source.read_bytes(), b"original")
            self.assertFalse(list(settings.clips_dir.glob("*.mp4")))
            clipper_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
