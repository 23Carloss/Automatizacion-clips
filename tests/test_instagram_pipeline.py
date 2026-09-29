from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import requests

from config import Settings
from r2_storage import R2VideoStorage, RemoteVideo
from uploader import AutoUploader, InstagramPublishingError, UploadResult


def instagram_settings(**overrides) -> Settings:
    values = {
        "twitch_url": "https://example.invalid/channel",
        "instagram_access_token": "secret-token",
        "instagram_user_id": "17841400000000000",
        "r2_account_id": "account",
        "r2_access_key_id": "access",
        "r2_secret_access_key": "secret",
        "r2_bucket": "clips",
        "upload_retry_base_seconds": 0,
    }
    values.update(overrides)
    return Settings(**values)


class FakeResponse:
    def __init__(self, payload, status_code: int = 200, reason: str = "OK") -> None:
        self._payload = payload
        self.status_code = status_code
        self.reason = reason
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._payload


class R2VideoStorageTests(unittest.TestCase):
    def test_upload_returns_presigned_get_url_and_delete_uses_same_key(self) -> None:
        client = Mock()
        client.generate_presigned_url.return_value = "https://signed.example/video"
        storage = R2VideoStorage(instagram_settings(), client=client)
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "clip con espacios.mp4"
            clip.write_bytes(b"video")
            remote = storage.upload(clip)

        self.assertEqual(remote.key, "clips/clip_con_espacios.mp4")
        self.assertEqual(remote.url, "https://signed.example/video")
        client.upload_file.assert_called_once_with(
            str(clip), "clips", remote.key,
            ExtraArgs={"ContentType": "video/mp4", "CacheControl": "no-store"},
        )
        client.generate_presigned_url.assert_called_once_with(
            "get_object", Params={"Bucket": "clips", "Key": remote.key}, ExpiresIn=21600,
        )
        storage.delete(remote.key)
        client.delete_object.assert_called_once_with(Bucket="clips", Key=remote.key)


class InstagramPublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = SimpleNamespace()
        self.uploader = AutoUploader(instagram_settings(), self.repository)

    def test_uploads_waits_publishes_then_deletes_r2_object(self) -> None:
        storage = Mock()
        storage.upload.return_value = RemoteVideo("clips/test.mp4", "https://signed.example/test")
        session = Mock()
        session.post.side_effect = [FakeResponse({"id": "container-1"}), FakeResponse({"id": "media-1"})]
        session.get.return_value = FakeResponse({"status_code": "FINISHED"})
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "test.mp4"
            clip.write_bytes(b"video")
            with patch("uploader.R2VideoStorage", return_value=storage), patch(
                "uploader.requests.Session", return_value=session
            ):
                result = self.uploader._instagram_upload(clip, "caption")

        self.assertEqual(result, "Instagram media ID media-1")
        create_data = session.post.call_args_list[0].kwargs["data"]
        self.assertEqual(create_data["video_url"], "https://signed.example/test")
        storage.delete.assert_called_once_with("clips/test.mp4")

    def test_unknown_publish_outcome_is_not_retryable_or_deleted(self) -> None:
        storage = Mock()
        storage.upload.return_value = RemoteVideo("clips/test.mp4", "https://signed.example/test")
        session = Mock()
        session.post.side_effect = [FakeResponse({"id": "container-1"}), requests.Timeout("timeout")]
        session.get.return_value = FakeResponse({"status_code": "FINISHED"})
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "test.mp4"
            clip.write_bytes(b"video")
            with patch("uploader.R2VideoStorage", return_value=storage), patch(
                "uploader.requests.Session", return_value=session
            ), self.assertRaises(InstagramPublishingError) as caught:
                self.uploader._instagram_upload(clip, "caption")

        self.assertFalse(caught.exception.retryable)
        self.assertIn("avoid a duplicate Reel", str(caught.exception))
        storage.delete.assert_not_called()


class PublicationRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_attempts_are_persisted(self) -> None:
        repository = SimpleNamespace(
            start_publication_attempt=AsyncMock(side_effect=[(101, 1), (102, 2)]),
            finish_publication_attempt=AsyncMock(),
        )
        uploader = AutoUploader(instagram_settings(upload_max_attempts=3), repository)
        action = AsyncMock(side_effect=[
            UploadResult("Instagram Reels", "failed", "temporary", retryable=True),
            UploadResult("Instagram Reels", "published", "media-1"),
        ])

        result = await uploader._run_with_retries(7, "Instagram Reels", action)

        self.assertEqual(result.status, "published")
        self.assertEqual(action.await_count, 2)
        self.assertEqual(repository.finish_publication_attempt.await_count, 2)
        repository.finish_publication_attempt.assert_any_await(101, "failed", "temporary")
        repository.finish_publication_attempt.assert_any_await(102, "published", "media-1")


if __name__ == "__main__":
    unittest.main()
