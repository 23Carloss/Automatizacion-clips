from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from config import Settings
from database import ClipRepository


class DatabasePaginationTests(unittest.TestCase):
    def test_status_pages_share_connection_and_clamp_stale_offset(self) -> None:
        cursor = MagicMock()
        cursor.fetchall.side_effect = [
            [{"status": "UPLOADED", "total": 12}],
            [{
                "id": 2,
                "filename": "clip_source.mp4",
                "status": "UPLOADED",
                "source": "PS_APP",
                "timestamp": datetime(2026, 9, 22, 10, 0),
                "telegram_message_id": None,
                "processing_progress": 42.5,
            }],
            [],
        ]
        connection = MagicMock()
        connection.cursor.return_value = cursor
        repository = ClipRepository(Settings(twitch_url="https://example.invalid"))

        with patch.object(repository, "_connect", return_value=connection):
            pages, totals = repository._list_clip_pages_by_status(
                {"UPLOADED": 999, "PUBLISHED": 10}, limit=5
            )

        self.assertEqual(totals, {"UPLOADED": 12, "PUBLISHED": 0})
        self.assertEqual([clip.id for clip in pages["UPLOADED"]], [2])
        self.assertEqual(pages["UPLOADED"][0].processing_progress, 42.5)
        self.assertEqual(pages["PUBLISHED"], [])
        self.assertEqual(cursor.execute.call_args_list[1].args[1], ("UPLOADED", 5, 10))
        self.assertEqual(cursor.execute.call_args_list[2].args[1], ("PUBLISHED", 5, 0))
        connection.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
