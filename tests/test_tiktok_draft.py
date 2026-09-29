from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config import Settings
from uploader import AutoUploader


class FakeResponse:
    def __init__(self, payload=None, status_code: int = 200, reason: str = "OK") -> None:
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.reason = reason
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._payload


def tiktok_settings(**overrides) -> Settings:
    values = {
        "twitch_url": "https://example.invalid/channel",
        "tiktok_access_token": "secret-token",
        "tiktok_status_poll_interval_seconds": 0.001,
        "tiktok_status_timeout_seconds": 1,
        "upload_retry_base_seconds": 0,
    }
    values.update(overrides)
    return Settings(**values)


class TikTokDraftUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.uploader = AutoUploader(tiktok_settings(), SimpleNamespace())

    def test_small_video_uses_one_chunk(self) -> None:
        self.assertEqual(self.uploader._tiktok_chunk_plan(4_000_000), (4_000_000, 1))
        self.assertEqual(self.uploader._tiktok_chunk_plan(64_000_000), (64_000_000, 1))

    def test_large_video_uses_tiktok_floor_chunk_count(self) -> None:
        self.assertEqual(self.uploader._tiktok_chunk_plan(71_000_000), (10_000_000, 7))

    def test_uploads_to_inbox_and_waits_for_draft_delivery(self) -> None:
        session = Mock()
        session.post.side_effect = [
            FakeResponse({
                "data": {
                    "publish_id": "v_pub_123",
                    "upload_url": "https://upload.example/signed",
                },
                "error": {"code": "ok", "message": "", "log_id": "log-1"},
            }),
            FakeResponse({
                "data": {"status": "SEND_TO_USER_INBOX", "uploaded_bytes": 5},
                "error": {"code": "ok", "message": "", "log_id": "log-2"},
            }),
        ]
        session.put.return_value = FakeResponse(status_code=201)

        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "clip.mp4"
            clip.write_bytes(b"video")
            with patch("uploader.requests.Session", return_value=session):
                status, detail = self.uploader._tiktok_upload(clip)

        self.assertEqual(status, "draft_ready")
        self.assertIn("v_pub_123", detail)
        init_call = session.post.call_args_list[0]
        self.assertTrue(init_call.args[0].endswith("/inbox/video/init/"))
        self.assertNotIn("post_info", init_call.kwargs["json"])
        self.assertEqual(
            init_call.kwargs["json"]["source_info"],
            {
                "source": "FILE_UPLOAD",
                "video_size": 5,
                "chunk_size": 5,
                "total_chunk_count": 1,
            },
        )
        status_call = session.post.call_args_list[1]
        self.assertTrue(status_call.args[0].endswith("/status/fetch/"))
        self.assertEqual(status_call.kwargs["json"], {"publish_id": "v_pub_123"})
        put_headers = session.put.call_args.kwargs["headers"]
        self.assertEqual(put_headers["Content-Length"], "5")
        self.assertEqual(put_headers["Content-Range"], "bytes 0-4/5")


if __name__ == "__main__":
    unittest.main()
