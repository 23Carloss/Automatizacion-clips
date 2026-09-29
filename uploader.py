"""Official-API upload adapters for approved clips.

Each platform adapter intentionally returns an explanatory skipped result until
the credentials and platform permissions required by that official API exist.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from config import Settings
from database import ClipRepository
from r2_storage import R2VideoStorage, RemoteVideo

YOUTUBE_SCOPE = ["https://www.googleapis.com/auth/youtube.upload"]
LOGGER = logging.getLogger("apex_clipper.uploader")
PLATFORMS = ("YouTube Shorts", "Instagram Reels", "TikTok")
TIKTOK_API_BASE = "https://open.tiktokapis.com/v2/post/publish"
TIKTOK_MIN_CHUNK_BYTES = 5_000_000
TIKTOK_MAX_CHUNK_BYTES = 64_000_000
TIKTOK_MAX_CHUNKS = 1_000


@dataclass(frozen=True, slots=True)
class UploadResult:
    platform: str
    status: str
    detail: str
    retryable: bool = False


class InstagramPublishingError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class TikTokUploadError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def make_metadata(kind: str = "Apex highlight") -> tuple[str, str]:
    title = f"{kind} en Apex Legends 🔥 #ApexLegends #ApexClips #Gamer #Shorts"
    caption = f"{kind} desde PS5/Twitch. #ApexLegends #ApexClips #Gaming #BattleRoyale"
    return title[:100], caption


class AutoUploader:
    """Single-worker publication queue that can yield bandwidth to a live stream."""

    def __init__(self, settings: Settings, repository: ClipRepository) -> None:
        self.settings = settings
        self.repository = repository
        self._queue: asyncio.Queue[tuple[int, Path, str]] = asyncio.Queue()
        self._queued_clip_ids: set[int] = set()
        self._live_active = False
        self._worker: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    def set_live_active(self, active: bool) -> None:
        self._live_active = active

    async def start(self) -> None:
        if self._worker is None:
            self._stopping.clear()
            await self.repository.recover_interrupted_publications()
            await self._restore_queued_clips()
            self._worker = asyncio.create_task(self._worker_loop(), name="upload-queue")

    async def close(self) -> None:
        self._stopping.set()
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def enqueue(self, clip_id: int, clip_path: Path, event_kind: str) -> None:
        if clip_id in self._queued_clip_ids:
            return
        self._queued_clip_ids.add(clip_id)
        await self._queue.put((clip_id, clip_path, event_kind))

    async def _restore_queued_clips(self) -> None:
        for clip in await self.repository.list_clips_by_status("APPROVED_QUEUED"):
            path = self.settings.clips_dir / clip.filename
            if path.is_file():
                await self.enqueue(clip.id, path, "Apex highlight")
            else:
                LOGGER.error("Queued clip #%s is missing locally: %s", clip.id, path)

    async def _worker_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                clip_id, clip_path, event_kind = await asyncio.wait_for(self._queue.get(), timeout=10)
            except asyncio.TimeoutError:
                await self._restore_queued_clips()
                continue
            try:
                while self.settings.pause_uploads_while_live and self._live_active and not self._stopping.is_set():
                    await asyncio.sleep(5)
                if self._stopping.is_set():
                    return
                started = await self.repository.transition_clip_status(
                    clip_id,
                    from_statuses=("APPROVED_QUEUED",),
                    to_status="PUBLISHING",
                )
                if not started:
                    LOGGER.info(
                        "Skipping queued clip #%s because it is no longer approved.", clip_id
                    )
                    continue
                results = await self.upload_all(clip_path, event_kind, clip_id=clip_id)
                failures = [result for result in results if result.status == "failed"]
                completed = [result for result in results if result.status in {"published", "draft_ready"}]
                if failures or not completed:
                    await self.repository.update_clip(clip_id, status="PUBLISH_FAILED")
                    for result in failures:
                        LOGGER.error("Clip #%s failed on %s: %s", clip_id, result.platform, result.detail)
                else:
                    await self.repository.update_clip(clip_id, status="PUBLISHED")
                await asyncio.sleep(max(0, self.settings.upload_delay_seconds))
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Unexpected publication failure for clip #%s", clip_id)
                await self.repository.update_clip(clip_id, status="PUBLISH_FAILED")
            finally:
                self._queued_clip_ids.discard(clip_id)
                self._queue.task_done()

    async def upload_all(
        self, clip_path: Path, event_kind: str, *, clip_id: int | None = None,
    ) -> list[UploadResult]:
        title, caption = make_metadata(event_kind)
        completed = await self.repository.get_published_platforms(clip_id) if clip_id is not None else set()
        actions: dict[str, Callable[[int | None], Awaitable[UploadResult]]] = {
            "YouTube Shorts": lambda _attempt_id: self.upload_youtube_short(clip_path, title, caption),
            "Instagram Reels": lambda attempt_id: self.upload_instagram_reel(
                clip_path, caption, attempt_id=attempt_id,
            ),
            "TikTok": lambda attempt_id: self.upload_tiktok(
                clip_path, caption, attempt_id=attempt_id,
            ),
        }

        async def run(platform: str) -> UploadResult:
            if platform in completed:
                return UploadResult(platform, "published", "Already completed in an earlier attempt.")
            return await self._run_with_retries(clip_id, platform, actions[platform])

        return list(await asyncio.gather(*(run(platform) for platform in PLATFORMS)))

    async def _run_with_retries(
        self, clip_id: int | None, platform: str,
        action: Callable[[int | None], Awaitable[UploadResult]],
    ) -> UploadResult:
        result = UploadResult(platform, "failed", "Publication did not start.", retryable=True)
        for local_attempt in range(1, self.settings.upload_max_attempts + 1):
            attempt_id: int | None = None
            attempt_number = local_attempt
            if clip_id is not None:
                attempt_id, attempt_number = await self.repository.start_publication_attempt(clip_id, platform)
            LOGGER.info("Publishing clip #%s to %s (attempt %s)", clip_id, platform, attempt_number)
            try:
                result = await action(attempt_id)
            except Exception as exc:
                LOGGER.exception("Unhandled %s adapter error", platform)
                result = UploadResult(platform, "failed", str(exc), retryable=True)
            if attempt_id is not None:
                await self.repository.finish_publication_attempt(attempt_id, result.status, result.detail)
            if result.status != "failed" or not result.retryable:
                return result
            if local_attempt < self.settings.upload_max_attempts:
                delay = max(0, self.settings.upload_retry_base_seconds) * (2 ** (local_attempt - 1))
                LOGGER.warning("Retrying %s for clip #%s in %.1f seconds", platform, clip_id, delay)
                await asyncio.sleep(delay)
        return result

    async def upload_youtube_short(self, clip_path: Path, title: str, description: str) -> UploadResult:
        if not self.settings.youtube_client_secrets:
            return UploadResult("YouTube Shorts", "skipped", "YOUTUBE_CLIENT_SECRETS is not configured.")
        try:
            video_id = await asyncio.to_thread(self._youtube_upload, clip_path, title, description)
            return UploadResult("YouTube Shorts", "published", f"https://youtube.com/shorts/{video_id}")
        except Exception as exc:
            return UploadResult("YouTube Shorts", "failed", str(exc), retryable=True)

    def _youtube_upload(self, clip_path: Path, title: str, description: str) -> str:
        token_path = Path(self.settings.youtube_token_file)
        credentials: Credentials | None = Credentials.from_authorized_user_file(token_path, YOUTUBE_SCOPE) if token_path.exists() else None
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.settings.youtube_client_secrets, YOUTUBE_SCOPE)
                credentials = flow.run_local_server(port=0)
            token_path.write_text(credentials.to_json(), encoding="utf-8")
        youtube = build("youtube", "v3", credentials=credentials)
        request = youtube.videos().insert(
            part="snippet,status",
            body={"snippet": {"title": title, "description": description, "categoryId": "20"},
                  "status": {"privacyStatus": "private", "selfDeclaredMadeForKids": False}},
            media_body=MediaFileUpload(str(clip_path), mimetype="video/mp4", resumable=True),
        )
        response = None
        while response is None:
            _, response = request.next_chunk()
        return response["id"]

    async def upload_instagram_reel(
        self, clip_path: Path, caption: str, *, attempt_id: int | None = None,
    ) -> UploadResult:
        s = self.settings
        if not (s.instagram_access_token and s.instagram_user_id):
            return UploadResult("Instagram Reels", "skipped", "INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_USER_ID are not configured.")
        if s.r2_partially_configured:
            return UploadResult("Instagram Reels", "failed", "R2 configuration is incomplete.")
        if not s.r2_configured and not self._valid_legacy_template():
            return UploadResult(
                "Instagram Reels", "failed",
                "Configure the four required R2 variables; INSTAGRAM_VIDEO_URL_TEMPLATE is only a legacy fallback.",
            )
        try:
            detail = await asyncio.to_thread(self._instagram_upload, clip_path, caption, attempt_id)
            return UploadResult("Instagram Reels", "published", detail)
        except InstagramPublishingError as exc:
            return UploadResult("Instagram Reels", "failed", str(exc), retryable=exc.retryable)
        except Exception as exc:
            return UploadResult("Instagram Reels", "failed", str(exc), retryable=True)

    def _valid_legacy_template(self) -> bool:
        template = self.settings.instagram_video_url_template
        return template.startswith("https://") and "{filename}" in template and "your-cdn.example" not in template

    def _progress(self, attempt_id: int | None, status: str, detail: str) -> None:
        if attempt_id is None:
            return
        try:
            self.repository.set_publication_attempt_progress(attempt_id, status, detail)
        except Exception:
            LOGGER.exception("Could not persist publication progress for attempt #%s", attempt_id)

    def _instagram_upload(
        self, clip_path: Path, caption: str, attempt_id: int | None = None,
    ) -> str:
        s = self.settings
        storage: R2VideoStorage | None = None
        remote: RemoteVideo | None = None
        if s.r2_configured:
            self._progress(attempt_id, "R2_UPLOADING", "Subiendo el master local a Cloudflare R2.")
            storage = R2VideoStorage(s)
            try:
                remote = storage.upload(clip_path)
            except Exception as exc:
                raise InstagramPublishingError(
                    f"R2 upload failed ({type(exc).__name__}).", retryable=True,
                ) from exc
            video_url = remote.url
            LOGGER.info("Uploaded Instagram source to R2 key %s", remote.key)
            self._progress(attempt_id, "R2_READY", "Vídeo disponible mediante URL HTTPS temporal.")
        else:
            video_url = s.instagram_video_url_template.format(filename=quote(clip_path.name))
            self._progress(attempt_id, "SOURCE_READY", "Vídeo disponible en el CDN configurado.")

        base = f"{s.instagram_graph_base_url}/{s.meta_graph_api_version}"
        session = requests.Session()
        self._progress(attempt_id, "CONTAINER_CREATING", "Creando el contenedor del Reel en Instagram.")
        create = self._graph_post(
            session,
            f"{base}/{s.instagram_user_id}/media",
            {
                "media_type": "REELS", "video_url": video_url, "caption": caption,
                "share_to_feed": "true", "access_token": s.instagram_access_token,
            },
            "Instagram container creation",
        )
        container_id = create.get("id")
        if not container_id:
            raise InstagramPublishingError("Instagram did not return a container ID.", retryable=False)
        LOGGER.info("Instagram container %s created", container_id)
        self._progress(attempt_id, "PROCESSING", f"Instagram está procesando el contenedor {container_id}.")
        self._wait_for_instagram_container(session, base, str(container_id), attempt_id)
        self._progress(attempt_id, "CONTAINER_READY", "El contenedor terminó de procesarse.")
        self._progress(attempt_id, "PUBLISHING", "Publicando el Reel en la cuenta de Instagram.")
        published = self._graph_post(
            session,
            f"{base}/{s.instagram_user_id}/media_publish",
            {"creation_id": container_id, "access_token": s.instagram_access_token},
            "Instagram media publication",
            outcome_unknown_on_network_error=True,
        )
        media_id = published.get("id")
        if not media_id:
            raise InstagramPublishingError("Instagram did not return a published media ID.", retryable=False)
        if storage is not None and remote is not None and s.r2_delete_after_publish:
            self._progress(attempt_id, "R2_CLEANUP", "Reel publicado; eliminando el objeto temporal de R2.")
            try:
                storage.delete(remote.key)
                LOGGER.info("Deleted published Instagram source from R2 key %s", remote.key)
            except Exception:
                # Do not create a duplicate Reel because temporary cleanup failed.
                LOGGER.exception("Instagram Reel published, but R2 cleanup failed for %s", remote.key)
        return f"Instagram media ID {media_id}"

    def _wait_for_instagram_container(
        self, session: requests.Session, base: str, container_id: str,
        attempt_id: int | None = None,
    ) -> None:
        deadline = time.monotonic() + self.settings.instagram_processing_timeout_seconds
        last_state = "UNKNOWN"
        while time.monotonic() < deadline:
            response = self._graph_get(
                session,
                f"{base}/{container_id}",
                {"fields": "status_code", "access_token": self.settings.instagram_access_token},
                "Instagram container status",
            )
            last_state = str(response.get("status_code", "UNKNOWN")).upper()
            self._progress(
                attempt_id, "PROCESSING",
                f"Estado del contenedor {container_id}: {last_state}.",
            )
            if last_state == "FINISHED":
                return
            if last_state in {"ERROR", "EXPIRED"}:
                raise InstagramPublishingError(
                    f"Instagram container ended in {last_state}.", retryable=last_state == "ERROR",
                )
            time.sleep(max(0.1, self.settings.instagram_poll_interval_seconds))
        raise InstagramPublishingError(
            f"Instagram processing timed out in state {last_state}.", retryable=True,
        )

    def _graph_error(self, response: requests.Response, stage: str) -> InstagramPublishingError:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        raw_error = payload.get("error", {}) if isinstance(payload, dict) else {}
        error = raw_error if isinstance(raw_error, dict) else {"message": raw_error}
        message = str(error.get("message") or response.reason or "unknown Graph API error")
        for secret in (
            self.settings.instagram_access_token,
            self.settings.r2_access_key_id,
            self.settings.r2_secret_access_key,
        ):
            if secret:
                message = message.replace(secret, "<REDACTED>")
        code = error.get("code")
        retryable = bool(error.get("is_transient")) or response.status_code == 429 or response.status_code >= 500
        suffix = f" (code {code}, HTTP {response.status_code})" if code is not None else f" (HTTP {response.status_code})"
        return InstagramPublishingError(f"{stage} failed: {message}{suffix}", retryable=retryable)

    def _graph_post(
        self, session: requests.Session, url: str, data: dict[str, Any], stage: str,
        *, outcome_unknown_on_network_error: bool = False,
    ) -> dict[str, Any]:
        try:
            response = session.post(url, data=data, timeout=(10, 60))
        except requests.RequestException as exc:
            message = f"{stage} network error ({type(exc).__name__})."
            if outcome_unknown_on_network_error:
                message += " Publication outcome is unknown; automatic retry was stopped to avoid a duplicate Reel."
            raise InstagramPublishingError(message, retryable=not outcome_unknown_on_network_error) from exc
        if not response.ok:
            error = self._graph_error(response, stage)
            if outcome_unknown_on_network_error and response.status_code >= 500:
                raise InstagramPublishingError(
                    f"{error} Publication outcome is unknown; automatic retry was stopped to avoid a duplicate Reel.",
                    retryable=False,
                )
            raise error
        try:
            payload = response.json()
        except ValueError as exc:
            raise InstagramPublishingError(f"{stage} returned invalid JSON.", retryable=True) from exc
        if not isinstance(payload, dict):
            raise InstagramPublishingError(f"{stage} returned an unexpected response.", retryable=True)
        if "error" in payload:
            raise self._graph_error(response, stage)
        return payload

    def _graph_get(
        self, session: requests.Session, url: str, params: dict[str, Any], stage: str,
    ) -> dict[str, Any]:
        try:
            response = session.get(url, params=params, timeout=(10, 30))
        except requests.RequestException as exc:
            raise InstagramPublishingError(
                f"{stage} network error ({type(exc).__name__}).", retryable=True,
            ) from exc
        if not response.ok:
            raise self._graph_error(response, stage)
        try:
            payload = response.json()
        except ValueError as exc:
            raise InstagramPublishingError(f"{stage} returned invalid JSON.", retryable=True) from exc
        if not isinstance(payload, dict):
            raise InstagramPublishingError(f"{stage} returned an unexpected response.", retryable=True)
        if "error" in payload:
            raise self._graph_error(response, stage)
        return payload

    async def upload_tiktok(
        self, clip_path: Path, caption: str, *, attempt_id: int | None = None,
    ) -> UploadResult:
        if not self.settings.tiktok_access_token:
            return UploadResult("TikTok", "skipped", "TIKTOK_ACCESS_TOKEN is not configured.")
        try:
            status, detail = await asyncio.to_thread(self._tiktok_upload, clip_path, attempt_id)
            return UploadResult("TikTok", status, detail)
        except TikTokUploadError as exc:
            return UploadResult("TikTok", "failed", str(exc), retryable=exc.retryable)
        except Exception as exc:
            return UploadResult("TikTok", "failed", str(exc), retryable=True)

    def _tiktok_chunk_plan(self, video_size: int) -> tuple[int, int]:
        if video_size <= 0:
            raise TikTokUploadError("TikTok cannot upload an empty video.", retryable=False)
        if video_size <= TIKTOK_MAX_CHUNK_BYTES:
            # TikTok explicitly permits a sub-5 MB final/only chunk.
            return video_size, 1
        chunk_size = max(
            TIKTOK_MIN_CHUNK_BYTES,
            min(self.settings.tiktok_chunk_size_bytes, TIKTOK_MAX_CHUNK_BYTES),
        )
        total_chunks = video_size // chunk_size
        if total_chunks > TIKTOK_MAX_CHUNKS:
            chunk_size = min(
                TIKTOK_MAX_CHUNK_BYTES,
                max(TIKTOK_MIN_CHUNK_BYTES, (video_size + TIKTOK_MAX_CHUNKS - 1) // TIKTOK_MAX_CHUNKS),
            )
            total_chunks = video_size // chunk_size
        if total_chunks > TIKTOK_MAX_CHUNKS:
            raise TikTokUploadError("Video is too large for TikTok's 1000-chunk limit.", retryable=False)
        return chunk_size, total_chunks

    def _tiktok_upload(
        self, clip_path: Path, attempt_id: int | None = None,
    ) -> tuple[str, str]:
        video_size = clip_path.stat().st_size
        chunk_size, total_chunks = self._tiktok_chunk_plan(video_size)
        session = requests.Session()
        headers = {
            "Authorization": f"Bearer {self.settings.tiktok_access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        }
        self._progress(attempt_id, "TIKTOK_INITIALIZING", "Solicitando una carga de borrador a TikTok.")
        response = self._tiktok_post_json(
            session,
            f"{TIKTOK_API_BASE}/inbox/video/init/",
            {
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": video_size,
                    "chunk_size": chunk_size,
                    "total_chunk_count": total_chunks,
                }
            },
            "TikTok draft initialization",
        )
        data = response.get("data", {})
        upload_url = data.get("upload_url") if isinstance(data, dict) else None
        publish_id = data.get("publish_id") if isinstance(data, dict) else None
        if not upload_url or not publish_id:
            raise TikTokUploadError(
                "TikTok did not return both an upload URL and publish ID.", retryable=True,
            )
        self._upload_tiktok_chunks(
            session, clip_path, str(upload_url), video_size, chunk_size, total_chunks, attempt_id,
        )
        return self._wait_for_tiktok_draft(session, str(publish_id), attempt_id)

    def _upload_tiktok_chunks(
        self, session: requests.Session, clip_path: Path, upload_url: str,
        video_size: int, chunk_size: int, total_chunks: int,
        attempt_id: int | None,
    ) -> None:
        offset = 0
        with clip_path.open("rb") as video:
            for index in range(total_chunks):
                current_size = chunk_size if index < total_chunks - 1 else video_size - offset
                chunk = video.read(current_size)
                if len(chunk) != current_size:
                    raise TikTokUploadError("The local video changed while it was being uploaded.", retryable=False)
                end = offset + current_size - 1
                self._progress(
                    attempt_id,
                    "TIKTOK_UPLOADING",
                    f"Subiendo parte {index + 1} de {total_chunks} ({end + 1}/{video_size} bytes).",
                )
                expected_status = 201 if index == total_chunks - 1 else 206
                self._put_tiktok_chunk(
                    session, upload_url, chunk, offset, end, video_size, expected_status,
                )
                offset = end + 1

    def _put_tiktok_chunk(
        self, session: requests.Session, upload_url: str, chunk: bytes,
        start: int, end: int, video_size: int, expected_status: int,
    ) -> None:
        headers = {
            "Content-Type": "video/mp4",
            "Content-Length": str(len(chunk)),
            "Content-Range": f"bytes {start}-{end}/{video_size}",
        }
        max_attempts = max(1, self.settings.upload_max_attempts)
        for attempt in range(1, max_attempts + 1):
            try:
                response = session.put(upload_url, data=chunk, headers=headers, timeout=(10, 300))
            except requests.RequestException as exc:
                if attempt < max_attempts:
                    time.sleep(max(0, self.settings.upload_retry_base_seconds) * (2 ** (attempt - 1)))
                    continue
                raise TikTokUploadError(
                    f"TikTok video transfer failed ({type(exc).__name__}); automatic reinitialization was stopped to avoid duplicate drafts.",
                    retryable=False,
                ) from exc
            if response.status_code == expected_status:
                return
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts:
                time.sleep(max(0, self.settings.upload_retry_base_seconds) * (2 ** (attempt - 1)))
                continue
            raise TikTokUploadError(
                f"TikTok video transfer failed (HTTP {response.status_code}); automatic reinitialization was stopped to avoid duplicate drafts.",
                retryable=False,
            )

    def _wait_for_tiktok_draft(
        self, session: requests.Session, publish_id: str, attempt_id: int | None,
    ) -> tuple[str, str]:
        deadline = time.monotonic() + self.settings.tiktok_status_timeout_seconds
        last_state = "UNKNOWN"
        while time.monotonic() < deadline:
            try:
                response = self._tiktok_post_json(
                    session,
                    f"{TIKTOK_API_BASE}/status/fetch/",
                    {"publish_id": publish_id},
                    "TikTok draft status",
                )
            except TikTokUploadError as exc:
                if not exc.retryable:
                    raise
                self._progress(attempt_id, "TIKTOK_PROCESSING", "TikTok no respondió; se volverá a consultar el mismo borrador.")
                time.sleep(max(0.1, self.settings.tiktok_status_poll_interval_seconds))
                continue
            data = response.get("data", {})
            last_state = str(data.get("status", "UNKNOWN")).upper() if isinstance(data, dict) else "UNKNOWN"
            uploaded_bytes = data.get("uploaded_bytes") if isinstance(data, dict) else None
            detail = f"Estado TikTok: {last_state}."
            if uploaded_bytes is not None:
                detail += f" {uploaded_bytes} bytes recibidos."
            self._progress(attempt_id, "TIKTOK_PROCESSING", detail)
            if last_state == "SEND_TO_USER_INBOX":
                self._progress(attempt_id, "DRAFT_READY", "Borrador entregado a la bandeja de TikTok.")
                return (
                    "draft_ready",
                    f"Borrador TikTok listo (publish ID {publish_id}). Abre la bandeja de entrada de TikTok para editarlo y publicarlo manualmente.",
                )
            if last_state == "PUBLISH_COMPLETE":
                return "published", f"TikTok publish ID {publish_id} was completed in the TikTok app."
            if last_state == "FAILED":
                reason = data.get("fail_reason", "unknown reason") if isinstance(data, dict) else "unknown reason"
                raise TikTokUploadError(f"TikTok rejected the draft: {reason}.", retryable=False)
            time.sleep(max(0.1, self.settings.tiktok_status_poll_interval_seconds))
        raise TikTokUploadError(
            f"TikTok draft status timed out in state {last_state}; check the TikTok inbox before retrying.",
            retryable=False,
        )

    def _tiktok_post_json(
        self, session: requests.Session, url: str, payload: dict[str, Any], stage: str,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.settings.tiktok_access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        }
        try:
            response = session.post(url, json=payload, headers=headers, timeout=(10, 60))
        except requests.RequestException as exc:
            raise TikTokUploadError(
                f"{stage} network error ({type(exc).__name__}).", retryable=True,
            ) from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise TikTokUploadError(f"{stage} returned invalid JSON.", retryable=response.status_code >= 500) from exc
        if not isinstance(body, dict):
            raise TikTokUploadError(f"{stage} returned an unexpected response.", retryable=True)
        error = body.get("error", {})
        error = error if isinstance(error, dict) else {}
        error_code = str(error.get("code", "ok"))
        if not response.ok or error_code != "ok":
            message = str(error.get("message") or response.reason or "unknown TikTok API error")
            token = self.settings.tiktok_access_token
            if token:
                message = message.replace(token, "<REDACTED>")
            retryable = response.status_code == 429 or response.status_code >= 500
            raise TikTokUploadError(
                f"{stage} failed: {message} (code {error_code}, HTTP {response.status_code}).",
                retryable=retryable,
            )
        return body
