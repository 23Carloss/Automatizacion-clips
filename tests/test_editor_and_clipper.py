from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

try:
    import dotenv  # noqa: F401
except ImportError:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda: None  # type: ignore[attr-defined]
    sys.modules["dotenv"] = dotenv_stub

from clipper import Clipper
from config import Settings
from editor import ApexVerticalEditor, RenderCancelled
from video_encoders import encoder_arguments, encoder_candidates, preview_encoder_arguments


class EditorAndClipperTests(unittest.IsolatedAsyncioTestCase):
    def test_hybrid_encoder_chains_keep_cpu_as_last_resort(self) -> None:
        self.assertEqual(encoder_candidates("h264_amf", ("h264_qsv",)), ["h264_amf", "libx264"])
        qsv = encoder_arguments("h264_qsv", quality=18, threads=2, fast=True)
        amf = encoder_arguments("h264_amf", quality=19, threads=2, fast=False)
        self.assertIn("h264_qsv", qsv)
        self.assertIn("h264_amf", amf)
        preview = preview_encoder_arguments("h264_qsv", threads=2)
        self.assertIn("h264_qsv", preview)
        self.assertEqual(preview[preview.index("-maxrate") + 1], "2500k")

    async def test_nvenc_render_failure_retries_with_cpu(self) -> None:
        settings = Settings(twitch_url="https://example.invalid/channel")
        editor = ApexVerticalEditor(settings)
        editor._select_encoder = AsyncMock(return_value="h264_nvenc")  # type: ignore[method-assign]
        editor._run_render = AsyncMock(side_effect=("Cannot load nvcuda.dll", None))  # type: ignore[method-assign]

        output = await editor.render(Path("sample_source.mp4"))

        self.assertEqual(output, Path("sample_vertical.mp4"))
        self.assertEqual(editor._encoder, "libx264")
        self.assertEqual(editor._run_render.await_count, 2)
        self.assertEqual(editor._run_render.await_args_list[0].args[-1], "h264_nvenc")
        self.assertEqual(editor._run_render.await_args_list[1].args[-1], "libx264")
        video_filter = editor._run_render.await_args_list[0].args[2]
        self.assertIn("crop=960:1080:480:0", video_filter)
        self.assertIn("scale=1080:-2:flags=lanczos", video_filter)
        self.assertIn("boxblur=10:1", video_filter)
        self.assertIn("overlay=(W-w)/2:(H-h)/2", video_filter)
        self.assertNotIn("scale=1080:1920:flags=lanczos[base]", video_filter)

    async def test_dashboard_cancellation_terminates_ffmpeg_and_removes_partial_output(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.returncode = None
                self.finished = asyncio.Event()
                self.terminate = MagicMock(side_effect=self._finish)
                self.kill = MagicMock(side_effect=self._finish)

            def _finish(self) -> None:
                self.returncode = 1
                self.finished.set()

            async def communicate(self):
                await self.finished.wait()
                return b"", b"cancelled"

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample_source.mp4"
            output = Path(directory) / "sample_vertical.mp4"
            source.write_bytes(b"source")
            output.write_bytes(b"partial")
            process = FakeProcess()
            settings = Settings(twitch_url="https://example.invalid/channel")
            editor = ApexVerticalEditor(settings)
            editor._select_encoder = AsyncMock(return_value="libx264")  # type: ignore[method-assign]

            with patch(
                "editor.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ), patch("editor.CANCELLATION_POLL_SECONDS", 0.01):
                with self.assertRaises(RenderCancelled):
                    await editor.render(
                        source,
                        cancel_requested=AsyncMock(side_effect=(False, True)),
                    )

            process.terminate.assert_called_once_with()
            process.kill.assert_not_called()
            self.assertFalse(output.exists())

    async def test_task_cancellation_reaps_ffmpeg_before_removing_partial_output(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.returncode = None
                self.finished = asyncio.Event()
                self.terminate = MagicMock(side_effect=self._finish)
                self.kill = MagicMock(side_effect=self._finish)

            def _finish(self) -> None:
                self.returncode = 1
                self.finished.set()

            async def communicate(self):
                await self.finished.wait()
                return b"", b"cancelled"

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample_source.mp4"
            output = Path(directory) / "sample_vertical.mp4"
            source.write_bytes(b"source")
            output.write_bytes(b"partial")
            process = FakeProcess()
            editor = ApexVerticalEditor(Settings(twitch_url="https://example.invalid"))
            editor._select_encoder = AsyncMock(return_value="libx264")  # type: ignore[method-assign]

            with patch(
                "editor.asyncio.create_subprocess_exec", AsyncMock(return_value=process)
            ):
                render = asyncio.create_task(editor.render(source))
                await asyncio.sleep(0)
                render.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await render

            process.terminate.assert_called_once_with()
            self.assertFalse(output.exists())

    async def test_ffmpeg_progress_is_converted_to_percentage(self) -> None:
        stream = asyncio.StreamReader()
        stream.feed_data(
            b"out_time_us=2500000\nprogress=continue\n"
            b"out_time_us=7500000\nprogress=end\n"
        )
        stream.feed_eof()
        callback = AsyncMock()
        editor = ApexVerticalEditor(Settings(twitch_url="https://example.invalid/channel"))

        await editor._consume_ffmpeg_progress(stream, 10.0, callback)

        percentages = [call.args[0] for call in callback.await_args_list]
        self.assertEqual(percentages, [25.0, 75.0, 100.0])

    def test_master_command_uses_quality_profile_and_preserves_source_fps(self) -> None:
        settings = Settings(twitch_url="https://example.invalid/channel")
        editor = ApexVerticalEditor(settings)

        command = editor._build_render_command(
            Path("sample_source.mp4"),
            Path("sample_vertical.mp4"),
            "[0:v]format=yuv420p[out]",
            "libx264",
        )

        self.assertEqual(command[command.index("-crf") + 1], "16")
        self.assertEqual(command[command.index("-preset") + 1], "slow")
        self.assertIn("colorprim=bt709", command[command.index("-x264-params") + 1])
        self.assertEqual(command[command.index("-b:a") + 1], "128k")
        self.assertEqual(command[command.index("-maxrate") + 1], "22000k")
        self.assertEqual(command[command.index("-bufsize") + 1], "44000k")
        self.assertEqual(command[command.index("-fps_mode") + 1], "passthrough")
        self.assertEqual(command[command.index("-colorspace") + 1], "bt709")
        self.assertNotIn("-r", command)

    def test_vertical_encoders_apply_platform_bitrate_ceiling(self) -> None:
        for encoder in ("h264_amf", "h264_qsv", "h264_nvenc", "libx264"):
            with self.subTest(encoder=encoder):
                arguments = encoder_arguments(
                    encoder, quality=16, threads=2, fast=False,
                    target_kbps=18000, max_kbps=22000,
                )
                self.assertEqual(arguments[arguments.index("-maxrate") + 1], "22000k")
                self.assertEqual(arguments[arguments.index("-bufsize") + 1], "44000k")
                if encoder != "libx264":
                    self.assertEqual(arguments[arguments.index("-b:v") + 1], "18000k")
        source = encoder_arguments("h264_qsv", quality=14, threads=2, fast=True)
        self.assertNotIn("-maxrate", source)
        self.assertEqual(source[source.index("-global_quality") + 1], "14")

    async def test_vod_clipper_requests_twenty_before_and_five_after(self) -> None:
        settings = Settings(twitch_url="https://example.invalid/vod", source_kind="vod")
        clipper = Clipper(settings)
        clipper._run = AsyncMock()  # type: ignore[method-assign]

        await clipper.create_clip(100.0, "https://example.invalid/video.m3u8")

        arguments = clipper._run.await_args.args[0]
        self.assertEqual(arguments[arguments.index("-ss") + 1], "80.000")
        self.assertEqual(arguments[arguments.index("-t") + 1], "25.0")
        self.assertEqual(arguments[arguments.index("-threads") + 1], "2")
        self.assertEqual(arguments[arguments.index("-crf") + 1], "14")

    async def test_local_recording_clip_uses_unique_name_and_1080_source(self) -> None:
        settings = Settings(
            twitch_url="https://example.invalid", source_kind="vod"
        )
        clipper = Clipper(settings)
        clipper._run = AsyncMock()  # type: ignore[method-assign]

        output = await clipper.create_clip_window(
            80.0, 105.0, "D:/recording.mp4", stamp_timestamp=100.0,
            name_prefix="ps5_session", normalize_to_1080=True,
        )

        arguments = clipper._run.await_args.args[0]
        self.assertEqual(output.name, "ps5_session_100_source.mp4")
        self.assertEqual(
            arguments[arguments.index("-vf") + 1],
            "scale=1920:1080:flags=lanczos",
        )

    async def test_bleedout_vod_requests_twenty_five_before_and_five_after(self) -> None:
        settings = Settings(twitch_url="https://example.invalid/vod", source_kind="vod")
        clipper = Clipper(settings)
        clipper._run = AsyncMock()  # type: ignore[method-assign]

        await clipper.create_clip(100.0, "https://example.invalid/video.m3u8", "BLEEDOUT")

        arguments = clipper._run.await_args.args[0]
        self.assertEqual(arguments[arguments.index("-ss") + 1], "75.000")
        self.assertEqual(arguments[arguments.index("-t") + 1], "30.0")

    async def test_vod_clipper_accepts_an_extended_group_window(self) -> None:
        settings = Settings(twitch_url="https://example.invalid/vod", source_kind="vod")
        clipper = Clipper(settings)
        clipper._run = AsyncMock()  # type: ignore[method-assign]

        output = await clipper.create_clip_window(
            70.0,
            109.0,
            "https://example.invalid/video.m3u8",
            stamp_timestamp=90.0,
        )

        arguments = clipper._run.await_args.args[0]
        self.assertEqual(arguments[arguments.index("-ss") + 1], "70.000")
        self.assertEqual(arguments[arguments.index("-t") + 1], "39.0")
        self.assertEqual(output.name, "apex_90_source.mp4")

    async def test_live_clipper_rejects_a_partial_prebuffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            segment = cache_dir / "live_00000001.ts"
            segment.write_bytes(b"segment")
            os.utime(segment, (101.0, 101.0))
            settings = Settings(
                twitch_url="https://example.invalid/live",
                source_kind="live",
                cache_dir=cache_dir,
                clips_dir=cache_dir,
            )
            clipper = Clipper(settings)
            clipper._probe_duration = AsyncMock(return_value=1.0)  # type: ignore[method-assign]
            clipper._run = AsyncMock()  # type: ignore[method-assign]

            with self.assertRaisesRegex(RuntimeError, "missing 20.0s from the start"):
                await clipper._clip_live_window(80.0, 105.0, cache_dir / "output.mp4")

            clipper._run.assert_not_awaited()

    async def test_reserved_segments_survive_rolling_cache_pruning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            for index, modified in enumerate((80.0, 90.0, 105.0), start=1):
                segment = cache_dir / f"live_{index:08d}.ts"
                segment.write_bytes(f"segment-{index}".encode())
                os.utime(segment, (modified, modified))
            settings = Settings(
                twitch_url="https://example.invalid/live",
                source_kind="live",
                cache_dir=cache_dir,
                clips_dir=cache_dir,
            )
            clipper = Clipper(settings)
            reservation_id = await clipper.reserve_live_segments(80.0, 105.0)
            self.assertIsNotNone(reservation_id)
            for segment in cache_dir.glob("live_*.ts"):
                segment.unlink()

            clipper._probe_duration = AsyncMock(return_value=1.0)  # type: ignore[method-assign]
            clipper._run = AsyncMock()  # type: ignore[method-assign]
            output = await clipper.create_clip_window(
                80.0,
                105.0,
                "https://example.invalid/live.m3u8",
                stamp_timestamp=100.0,
                reservation_id=reservation_id,
            )

            clipper._run.assert_awaited_once()
            self.assertEqual(output.name, "apex_100_source.mp4")
            self.assertFalse((cache_dir / "reservations" / str(reservation_id)).exists())


if __name__ == "__main__":
    unittest.main()
