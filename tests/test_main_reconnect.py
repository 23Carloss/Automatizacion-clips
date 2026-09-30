from __future__ import annotations

import asyncio
import importlib
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, Mock, patch

from streamlink.exceptions import PluginError

from config import Settings


class MainReconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_streamlink_error_retries_without_stopping_telegram(self) -> None:
        main = importlib.import_module("main")
        settings = Settings(twitch_url="https://www.twitch.tv/example", source_kind="live")
        monitor = Mock(
            start=AsyncMock(side_effect=[PluginError("DNS lookup failed"), asyncio.CancelledError()]),
            close=AsyncMock(),
        )
        repository = Mock(initialize=AsyncMock())
        detector = Mock(initialize=AsyncMock())
        clipper = Mock(clear_stale_reservations=AsyncMock())
        uploader = Mock(start=AsyncMock(), close=AsyncMock())
        bot = Mock(start=AsyncMock(), close=AsyncMock())

        with ExitStack() as stack:
            stack.enter_context(patch.object(main.Settings, "from_env", return_value=settings))
            stack.enter_context(patch.object(main.Settings, "prepare_directories"))
            for name, instance in {
                "ClipRepository": repository,
                "StreamMonitor": monitor,
                "ApexEventDetector": detector,
                "Clipper": clipper,
                "ApexVerticalEditor": Mock(),
                "AutoUploader": uploader,
                "TelegramApprovalBot": bot,
            }.items():
                stack.enter_context(patch.object(main, name, return_value=instance))
            sleep = stack.enter_context(patch.object(main.asyncio, "sleep", new_callable=AsyncMock))

            with self.assertRaises(asyncio.CancelledError):
                await main.run()

        self.assertEqual(monitor.start.await_count, 2)
        sleep.assert_awaited_once_with(60)
        bot.start.assert_awaited_once()
        bot.close.assert_awaited_once()
        uploader.set_live_active.assert_any_call(False)