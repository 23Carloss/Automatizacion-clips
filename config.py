"""Central configuration for the visual-only Apex clip pipeline.

Copy ``.env.example`` to ``.env`` (or export the variables) before running.
No audio input is used anywhere in this project.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent

DEFAULT_NOTIFICATION_ROI = (500, 700, 1000, 180)
DEFAULT_KILLFEED_ROI = (1150, 110, 730, 230)


@dataclass(frozen=True, slots=True)
class Roi:
    """An ROI expressed against a 1920x1080 source frame."""

    x: int
    y: int
    width: int
    height: int


def _roi_from_env(name: str, default: tuple[int, int, int, int]) -> Roi:
    """Read an ``x,y,width,height`` ROI without scattering coordinates in code."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return Roi(*default)
    try:
        values = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError(f"{name} must contain four integers: x,y,width,height.") from exc
    if len(values) != 4 or any(value < 0 for value in values[:2]) or any(value <= 0 for value in values[2:]):
        raise ValueError(f"{name} must be x,y,width,height with a positive width and height.")
    return Roi(*values)


@dataclass(slots=True)
class Settings:
    twitch_url: str
    source_kind: Literal["live", "vod"] = "live"
    twitch_quality: str = "best"
    vod_start_offset_seconds: float = 0.0
    sample_fps: float = 2.0
    # Kept as a compatibility safety valve. Continuous event grouping replaces
    # the old global cooldown, so the normal value is zero.
    event_cooldown_seconds: float = 0.0
    event_merge_gap_seconds: float = 50.0
    event_duplicate_window_seconds: float = 8.0
    pre_event_seconds: float = 20.0
    post_event_seconds: float = 5.0
    bleedout_pre_event_seconds: float = 25.0
    bleedout_post_event_seconds: float = 5.0
    buffer_seconds: float = 120.0
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    ffmpeg_encoding_threads: int = 2
    ffmpeg_max_concurrent_encodes: int = 1
    ffmpeg_low_priority: bool = True
    ffmpeg_source_encoder: str = "libx264"
    ffmpeg_source_quality: int = 14
    ffmpeg_vertical_encoder: str = "libx264"
    ffmpeg_vertical_quality: int = 16
    vertical_video_target_kbps: int = 18000
    vertical_video_max_kbps: int = 22000
    ffmpeg_preview_encoder: str = "auto"
    tesseract_enabled: bool = True
    player_gamertag: str = ""
    ocr_min_confidence: float = 0.40
    ocr_fuzzy_match_threshold: float = 0.88
    ocr_single_frame_similarity: float = 0.94
    gamertag_match_threshold: float = 0.82
    ocr_motion_threshold: float = 0.008
    ocr_motion_pixel_threshold: int = 18
    ocr_motion_size: int = 160
    ocr_max_image_width: int = 720
    ocr_binary_threshold: int = 120
    ocr_confirmation_frames: int = 2
    ocr_candidate_max_gap_seconds: float = 2.0
    ocr_keyframe_interval_seconds: float = 1.0
    ocr_queue_size: int = 1
    ocr_max_live_lag_seconds: float = 3.0
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "apex_clipper"
    mysql_password: str = ""
    mysql_database: str = "apex_clipper"
    telegram_token: str = ""
    telegram_chat_id: int | None = None
    clips_dir: Path = field(default_factory=lambda: ROOT_DIR / "clips")
    cache_dir: Path = field(default_factory=lambda: ROOT_DIR / "cache")
    evidence_dir: Path = field(default_factory=lambda: ROOT_DIR / "event_evidence")
    # Event regions expressed against a 1920x1080 source frame.
    notification_roi: Roi = field(default_factory=lambda: Roi(*DEFAULT_NOTIFICATION_ROI))
    killfeed_roi: Roi = field(default_factory=lambda: Roi(*DEFAULT_KILLFEED_ROI))
    # Include the match HUD and every killfeed row, including victim rank icons.
    top_hud_roi: Roi = field(default_factory=lambda: Roi(1150, 30, 730, 310))
    health_hud_roi: Roi = field(default_factory=lambda: Roi(30, 910, 450, 140))
    ammo_hud_roi: Roi = field(default_factory=lambda: Roi(1440, 910, 450, 140))
    gameplay_crop: Roi = field(default_factory=lambda: Roi(480, 0, 960, 1080))
    youtube_client_secrets: str = ""
    youtube_token_file: str = "youtube_token.json"
    instagram_access_token: str = ""
    instagram_user_id: str = ""
    # Legacy escape hatch for a pre-existing CDN. R2 is preferred because it
    # uploads the local master before asking Instagram to fetch it.
    instagram_video_url_template: str = ""
    instagram_graph_base_url: str = "https://graph.instagram.com"
    meta_graph_api_version: str = "v23.0"
    instagram_processing_timeout_seconds: float = 600.0
    instagram_poll_interval_seconds: float = 5.0
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""
    r2_public_base_url: str = ""
    r2_key_prefix: str = "clips"
    r2_presigned_url_ttl_seconds: int = 21600
    r2_delete_after_publish: bool = True
    tiktok_access_token: str = ""
    tiktok_status_timeout_seconds: float = 300.0
    tiktok_status_poll_interval_seconds: float = 3.0
    tiktok_chunk_size_bytes: int = 10_000_000
    pause_uploads_while_live: bool = True
    upload_delay_seconds: float = 5.0
    upload_max_attempts: int = 3
    upload_retry_base_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "Settings":
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        setting = cls(
            twitch_url=os.getenv("TWITCH_URL", "").strip(),
            source_kind=os.getenv("SOURCE_KIND", "live").lower(),  # type: ignore[arg-type]
            twitch_quality=os.getenv("TWITCH_QUALITY", "best"),
            vod_start_offset_seconds=float(os.getenv("VOD_START_OFFSET_SECONDS", "0")),
            sample_fps=float(os.getenv("SAMPLE_FPS", "2")),
            event_cooldown_seconds=float(os.getenv("EVENT_COOLDOWN_SECONDS", "0")),
            event_merge_gap_seconds=float(os.getenv("EVENT_MERGE_GAP_SECONDS", "50")),
            event_duplicate_window_seconds=float(os.getenv("EVENT_DUPLICATE_WINDOW_SECONDS", "8")),
            pre_event_seconds=float(os.getenv("PRE_EVENT_SECONDS", "20")),
            post_event_seconds=float(os.getenv("POST_EVENT_SECONDS", "5")),
            bleedout_pre_event_seconds=float(os.getenv("BLEEDOUT_PRE_EVENT_SECONDS", "25")),
            bleedout_post_event_seconds=float(os.getenv("BLEEDOUT_POST_EVENT_SECONDS", "5")),
            buffer_seconds=float(os.getenv("BUFFER_SECONDS", "120")),
            ffmpeg_binary=os.getenv("FFMPEG_BINARY", "ffmpeg"),
            ffprobe_binary=os.getenv("FFPROBE_BINARY", "ffprobe"),
            ffmpeg_encoding_threads=int(os.getenv("FFMPEG_ENCODING_THREADS", "2")),
            ffmpeg_max_concurrent_encodes=int(os.getenv("FFMPEG_MAX_CONCURRENT_ENCODES", "1")),
            ffmpeg_low_priority=os.getenv("FFMPEG_LOW_PRIORITY", "true").lower() == "true",
            ffmpeg_source_encoder=os.getenv("FFMPEG_SOURCE_ENCODER", "libx264").strip().lower(),
            ffmpeg_source_quality=int(os.getenv("FFMPEG_SOURCE_QUALITY", "14")),
            ffmpeg_vertical_encoder=os.getenv("FFMPEG_VERTICAL_ENCODER", "libx264").strip().lower(),
            ffmpeg_vertical_quality=int(os.getenv("FFMPEG_VERTICAL_QUALITY", "16")),
            vertical_video_target_kbps=int(os.getenv("VERTICAL_VIDEO_TARGET_KBPS", "18000")),
            vertical_video_max_kbps=int(os.getenv("VERTICAL_VIDEO_MAX_KBPS", "22000")),
            ffmpeg_preview_encoder=os.getenv("FFMPEG_PREVIEW_ENCODER", "auto").strip().lower(),
            tesseract_enabled=os.getenv("TESSERACT_ENABLED", "true").lower() == "true",
            player_gamertag=os.getenv("PLAYER_GAMERTAG", "").strip(),
            ocr_min_confidence=float(os.getenv("OCR_MIN_CONFIDENCE", "0.40")),
            ocr_fuzzy_match_threshold=float(os.getenv("OCR_FUZZY_MATCH_THRESHOLD", "0.88")),
            ocr_single_frame_similarity=float(os.getenv("OCR_SINGLE_FRAME_SIMILARITY", "0.94")),
            gamertag_match_threshold=float(os.getenv("GAMERTAG_MATCH_THRESHOLD", "0.82")),
            ocr_motion_threshold=float(os.getenv("OCR_MOTION_THRESHOLD", "0.008")),
            ocr_motion_pixel_threshold=int(os.getenv("OCR_MOTION_PIXEL_THRESHOLD", "18")),
            ocr_motion_size=int(os.getenv("OCR_MOTION_SIZE", "160")),
            ocr_max_image_width=int(os.getenv("OCR_MAX_IMAGE_WIDTH", "720")),
            ocr_binary_threshold=int(os.getenv("OCR_BINARY_THRESHOLD", "120")),
            ocr_confirmation_frames=int(os.getenv("OCR_CONFIRMATION_FRAMES", "2")),
            ocr_candidate_max_gap_seconds=float(os.getenv("OCR_CANDIDATE_MAX_GAP_SECONDS", "2")),
            ocr_keyframe_interval_seconds=float(os.getenv("OCR_KEYFRAME_INTERVAL_SECONDS", "1")),
            ocr_queue_size=int(os.getenv("OCR_QUEUE_SIZE", "1")),
            ocr_max_live_lag_seconds=float(os.getenv("OCR_MAX_LIVE_LAG_SECONDS", "3")),
            notification_roi=_roi_from_env("NOTIFICATION_ROI", DEFAULT_NOTIFICATION_ROI),
            killfeed_roi=_roi_from_env("KILLFEED_ROI", DEFAULT_KILLFEED_ROI),
            gameplay_crop=_roi_from_env("GAMEPLAY_CROP", (480, 0, 960, 1080)),
            mysql_host=os.getenv("MYSQL_HOST", "127.0.0.1").strip(),
            mysql_port=int(os.getenv("MYSQL_PORT", "3306")),
            mysql_user=os.getenv("MYSQL_USER", "apex_clipper").strip(),
            mysql_password=os.getenv("MYSQL_PASSWORD", ""),
            mysql_database=os.getenv("MYSQL_DATABASE", "apex_clipper").strip(),
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=int(chat_id) if chat_id else None,
            youtube_client_secrets=os.getenv("YOUTUBE_CLIENT_SECRETS", "").strip(),
            youtube_token_file=os.getenv("YOUTUBE_TOKEN_FILE", "youtube_token.json").strip(),
            instagram_access_token=os.getenv("INSTAGRAM_ACCESS_TOKEN", "").strip(),
            instagram_user_id=os.getenv("INSTAGRAM_USER_ID", "").strip(),
            instagram_video_url_template=os.getenv("INSTAGRAM_VIDEO_URL_TEMPLATE", "").strip(),
            instagram_graph_base_url=os.getenv(
                "INSTAGRAM_GRAPH_BASE_URL", "https://graph.instagram.com"
            ).strip().rstrip("/"),
            meta_graph_api_version=os.getenv("META_GRAPH_API_VERSION", "v23.0").strip(),
            instagram_processing_timeout_seconds=float(os.getenv("INSTAGRAM_PROCESSING_TIMEOUT_SECONDS", "600")),
            instagram_poll_interval_seconds=float(os.getenv("INSTAGRAM_POLL_INTERVAL_SECONDS", "5")),
            r2_account_id=os.getenv("R2_ACCOUNT_ID", "").strip(),
            r2_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            r2_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            r2_bucket=os.getenv("R2_BUCKET", "").strip(),
            r2_public_base_url=os.getenv("R2_PUBLIC_BASE_URL", "").strip(),
            r2_key_prefix=os.getenv("R2_KEY_PREFIX", "clips").strip().strip("/"),
            r2_presigned_url_ttl_seconds=int(os.getenv("R2_PRESIGNED_URL_TTL_SECONDS", "21600")),
            r2_delete_after_publish=os.getenv("R2_DELETE_AFTER_PUBLISH", "true").lower() == "true",
            tiktok_access_token=os.getenv("TIKTOK_ACCESS_TOKEN", "").strip(),
            tiktok_status_timeout_seconds=float(os.getenv("TIKTOK_STATUS_TIMEOUT_SECONDS", "300")),
            tiktok_status_poll_interval_seconds=float(os.getenv("TIKTOK_STATUS_POLL_INTERVAL_SECONDS", "3")),
            tiktok_chunk_size_bytes=int(os.getenv("TIKTOK_CHUNK_SIZE_BYTES", "10000000")),
            pause_uploads_while_live=os.getenv("PAUSE_UPLOADS_WHILE_LIVE", "true").lower() == "true",
            upload_delay_seconds=float(os.getenv("UPLOAD_DELAY_SECONDS", "5")),
            upload_max_attempts=int(os.getenv("UPLOAD_MAX_ATTEMPTS", "3")),
            upload_retry_base_seconds=float(os.getenv("UPLOAD_RETRY_BASE_SECONDS", "10")),
        )
        if setting.source_kind not in ("live", "vod"):
            raise ValueError("SOURCE_KIND must be 'live' or 'vod'.")
        if not setting.player_gamertag:
            raise ValueError("PLAYER_GAMERTAG is required to reject third-party killfeed events.")
        if not 0 <= setting.ocr_min_confidence <= 1:
            raise ValueError("OCR_MIN_CONFIDENCE must be between 0 and 1.")
        if not 0 <= setting.ocr_fuzzy_match_threshold <= 1:
            raise ValueError("OCR_FUZZY_MATCH_THRESHOLD must be between 0 and 1.")
        if not setting.ocr_fuzzy_match_threshold <= setting.ocr_single_frame_similarity <= 1:
            raise ValueError(
                "OCR_SINGLE_FRAME_SIMILARITY must be between OCR_FUZZY_MATCH_THRESHOLD and 1."
            )
        if not 0 <= setting.gamertag_match_threshold <= 1:
            raise ValueError("GAMERTAG_MATCH_THRESHOLD must be between 0 and 1.")
        if setting.sample_fps <= 0 or setting.sample_fps > 2:
            raise ValueError("SAMPLE_FPS must be greater than 0 and no more than 2.")
        if not 0 <= setting.ocr_motion_threshold <= 1:
            raise ValueError("OCR_MOTION_THRESHOLD must be between 0 and 1.")
        if not 0 <= setting.ocr_motion_pixel_threshold <= 255:
            raise ValueError("OCR_MOTION_PIXEL_THRESHOLD must be between 0 and 255.")
        if setting.ocr_motion_size < 32 or setting.ocr_max_image_width < 320:
            raise ValueError("OCR motion/image sizes are too small for reliable text detection.")
        if not 0 <= setting.ocr_binary_threshold <= 255:
            raise ValueError("OCR_BINARY_THRESHOLD must be between 0 and 255.")
        if setting.ocr_confirmation_frames < 1 or setting.ocr_queue_size < 1:
            raise ValueError("OCR confirmation frames and queue size must be at least 1.")
        if setting.ocr_candidate_max_gap_seconds <= 0 or setting.ocr_keyframe_interval_seconds <= 0:
            raise ValueError("OCR candidate gap and keyframe interval must be positive.")
        if setting.ocr_max_live_lag_seconds <= 0:
            raise ValueError("OCR_MAX_LIVE_LAG_SECONDS must be positive.")
        if setting.event_cooldown_seconds < 0:
            raise ValueError("EVENT_COOLDOWN_SECONDS cannot be negative.")
        if setting.event_merge_gap_seconds <= 0 or setting.event_duplicate_window_seconds <= 0:
            raise ValueError("Event merge and duplicate windows must be positive.")
        if not 1 <= setting.ffmpeg_encoding_threads <= 16:
            raise ValueError("FFMPEG_ENCODING_THREADS must be between 1 and 16.")
        if not 1 <= setting.ffmpeg_max_concurrent_encodes <= 4:
            raise ValueError("FFMPEG_MAX_CONCURRENT_ENCODES must be between 1 and 4.")
        valid_encoders = {"auto", "h264_amf", "h264_qsv", "h264_nvenc", "libx264"}
        if setting.ffmpeg_source_encoder not in valid_encoders:
            raise ValueError("FFMPEG_SOURCE_ENCODER is not a supported H.264 encoder.")
        if setting.ffmpeg_vertical_encoder not in valid_encoders:
            raise ValueError("FFMPEG_VERTICAL_ENCODER is not a supported H.264 encoder.")
        if setting.ffmpeg_preview_encoder not in valid_encoders:
            raise ValueError("FFMPEG_PREVIEW_ENCODER is not a supported H.264 encoder.")
        if not 0 <= setting.ffmpeg_source_quality <= 51:
            raise ValueError("FFMPEG_SOURCE_QUALITY must be between 0 and 51.")
        if not 0 <= setting.ffmpeg_vertical_quality <= 51:
            raise ValueError("FFMPEG_VERTICAL_QUALITY must be between 0 and 51.")
        if not 1000 <= setting.vertical_video_target_kbps <= setting.vertical_video_max_kbps <= 24000:
            raise ValueError("Vertical video bitrate must satisfy 1000 <= target <= max <= 24000 kbps.")
        crop = setting.gameplay_crop
        if crop.x + crop.width > 1920 or crop.y + crop.height > 1080:
            raise ValueError("GAMEPLAY_CROP must fit inside the 1920x1080 source frame.")
        if crop.width * 1080 < 720 * crop.height:
            raise ValueError("GAMEPLAY_CROP is too narrow and would require excessive enlargement.")
        if not setting.meta_graph_api_version.startswith("v"):
            raise ValueError("META_GRAPH_API_VERSION must look like 'v23.0'.")
        if not setting.instagram_graph_base_url.startswith("https://"):
            raise ValueError("INSTAGRAM_GRAPH_BASE_URL must use HTTPS.")
        if not 1 <= setting.r2_presigned_url_ttl_seconds <= 604800:
            raise ValueError("R2_PRESIGNED_URL_TTL_SECONDS must be between 1 and 604800.")
        if setting.r2_public_base_url and not setting.r2_public_base_url.startswith("https://"):
            raise ValueError("R2_PUBLIC_BASE_URL must use HTTPS.")
        if setting.instagram_processing_timeout_seconds <= 0 or setting.instagram_poll_interval_seconds <= 0:
            raise ValueError("Instagram processing timeout and poll interval must be positive.")
        if setting.tiktok_status_timeout_seconds <= 0 or setting.tiktok_status_poll_interval_seconds <= 0:
            raise ValueError("TikTok status timeout and poll interval must be positive.")
        if not 5_000_000 <= setting.tiktok_chunk_size_bytes <= 64_000_000:
            raise ValueError("TIKTOK_CHUNK_SIZE_BYTES must be between 5000000 and 64000000.")
        if setting.upload_max_attempts < 1:
            raise ValueError("UPLOAD_MAX_ATTEMPTS must be at least 1.")
        windows = (
            (setting.pre_event_seconds, setting.post_event_seconds),
            (setting.bleedout_pre_event_seconds, setting.bleedout_post_event_seconds),
        )
        if any(pre < 0 or post < 0 or pre + post <= 0 for pre, post in windows):
            raise ValueError("Clip windows must be non-negative and have a positive total duration.")
        longest_window = max(pre + post for pre, post in windows)
        if setting.buffer_seconds < longest_window + 5:
            raise ValueError("BUFFER_SECONDS must exceed the longest clip window by at least 5 seconds.")
        return setting

    def clip_window(self, event_kind: str) -> tuple[float, float]:
        """Return the capture window around a confirmed visual event."""
        if event_kind.upper() == "BLEEDOUT":
            return self.bleedout_pre_event_seconds, self.bleedout_post_event_seconds
        return self.pre_event_seconds, self.post_event_seconds

    @property
    def bleedout_roi(self) -> Roi:
        """Compatibility alias for older callers; this region is the full killfeed."""
        return self.killfeed_roi

    @property
    def r2_configured(self) -> bool:
        return all((self.r2_account_id, self.r2_access_key_id, self.r2_secret_access_key, self.r2_bucket))

    @property
    def r2_partially_configured(self) -> bool:
        values = (self.r2_account_id, self.r2_access_key_id, self.r2_secret_access_key, self.r2_bucket)
        return any(values) and not all(values)

    def prepare_directories(self) -> None:
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
