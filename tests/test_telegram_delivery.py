from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

try:
    import dotenv  # noqa: F401
except ImportError:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda: None  # type: ignore[attr-defined]
    sys.modules["dotenv"] = dotenv_stub

# The bundled test runtime does not include the network/database packages. These
# small stubs let the local preview and delivery orchestration be tested without
# contacting Telegram or MySQL.
telegram_stub = types.ModuleType("telegram")
telegram_stub.Bot = object
telegram_stub.InlineKeyboardButton = lambda *args, **kwargs: (args, kwargs)
telegram_stub.InlineKeyboardMarkup = lambda value: value
telegram_stub.Update = object
telegram_ext_stub = types.ModuleType("telegram.ext")
telegram_ext_stub.Application = object
telegram_ext_stub.CallbackQueryHandler = object
telegram_ext_stub.ContextTypes = object
telegram_error_stub = types.ModuleType("telegram.error")
telegram_error_stub.NetworkError = type("NetworkError", (Exception,), {})
telegram_error_stub.TimedOut = type("TimedOut", (telegram_error_stub.NetworkError,), {})
sys.modules.setdefault("telegram", telegram_stub)
sys.modules.setdefault("telegram.ext", telegram_ext_stub)
sys.modules.setdefault("telegram.error", telegram_error_stub)

database_stub = types.ModuleType("database")
database_stub.ClipRepository = object
uploader_stub = types.ModuleType("uploader")
uploader_stub.AutoUploader = object
sys.modules.setdefault("database", database_stub)
sys.modules.setdefault("uploader", uploader_stub)

import telegram_bot
from config import Settings


class TelegramDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_callback_timeout_is_swallowed(self) -> None:
        query = types.SimpleNamespace(
            answer=AsyncMock(side_effect=telegram_error_stub.TimedOut("timeout"))
        )

        acknowledged = await telegram_bot.TelegramApprovalBot._answer_callback(query)

        self.assertFalse(acknowledged)

    async def test_large_master_uses_temporary_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            master = Path(directory) / "clip_vertical.mp4"
            master.write_bytes(b"master")

            async def write_preview(settings, source, output):
                output.write_bytes(b"p")

            with patch.object(telegram_bot, "MAX_TELEGRAM_VIDEO_BYTES", 1), patch.object(
                telegram_bot, "_render_telegram_preview", side_effect=write_preview
            ):
                path, temporary = await telegram_bot._telegram_upload_path(
                    Settings(twitch_url="https://example.invalid"), master
                )

            self.assertTrue(temporary)
            self.assertEqual(path.name, "clip_vertical_telegram_preview.mp4")
            self.assertEqual(path.read_bytes(), b"p")

    async def test_send_records_message_and_removes_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            master = Path(directory) / "clip_vertical.mp4"
            preview = Path(directory) / "preview.mp4"
            master.write_bytes(b"master")
            preview.write_bytes(b"preview")
            bot = types.SimpleNamespace(
                send_video=AsyncMock(return_value=types.SimpleNamespace(message_id=321))
            )
            repository = types.SimpleNamespace(update_clip=AsyncMock())
            settings = Settings(
                twitch_url="https://example.invalid", telegram_token="token", telegram_chat_id=123,
            )

            with patch.object(
                telegram_bot, "_telegram_upload_path", AsyncMock(return_value=(preview, True))
            ):
                message_id = await telegram_bot.send_approval_message(
                    bot, settings, repository, 7, master, "KNOCKED", 0.99
                )

            self.assertEqual(message_id, "321")
            self.assertFalse(preview.exists())
            repository.update_clip.assert_awaited_once_with(7, telegram_message_id="321")

    async def test_delete_clip_files_removes_source_master_and_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clips_dir = Path(directory)
            source = clips_dir / "apex_1_source.mp4"
            master = clips_dir / "apex_1_vertical.mp4"
            preview = clips_dir / "apex_1_vertical_telegram_preview.mp4"
            for path in (source, master, preview):
                path.write_bytes(b"video")

            failed = telegram_bot.delete_clip_files(clips_dir, master.name)

            self.assertEqual(failed, [])
            self.assertFalse(source.exists())
            self.assertFalse(master.exists())
            self.assertFalse(preview.exists())


if __name__ == "__main__":
    unittest.main()
