"""Durable human approval controls for rendered vertical clips."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import Application, CallbackQueryHandler, ContextTypes

from config import Settings
from database import ClipRepository
from ffmpeg_resources import FfmpegCpuLimiter, low_priority_process_kwargs
from uploader import AutoUploader
from video_encoders import (
    encoder_candidates,
    encoder_is_available,
    preview_encoder_arguments,
)

LOGGER = logging.getLogger("apex_clipper.telegram")
MAX_TELEGRAM_VIDEO_BYTES = 48_000_000


def delete_clip_files(clips_dir: Path, filename: str) -> list[Path]:
    """Delete a clip's master, source, and Telegram preview when present.

    Returns files that still exist because Windows had them locked.
    """
    name = Path(filename).name
    names = {name}
    if name.endswith("_vertical.mp4"):
        names.add(name.removesuffix("_vertical.mp4") + "_source.mp4")
        names.add(name.removesuffix(".mp4") + "_telegram_preview.mp4")
    elif name.endswith("_source.mp4"):
        vertical_name = name.removesuffix("_source.mp4") + "_vertical.mp4"
        names.add(vertical_name)
        names.add(vertical_name.removesuffix(".mp4") + "_telegram_preview.mp4")

    failed: list[Path] = []
    for candidate_name in names:
        path = clips_dir / candidate_name
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            failed.append(path)
    return failed


def _approval_keyboard(clip_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🟢 Aprobar y Publicar", callback_data=f"approve:{clip_id}")],
        [InlineKeyboardButton("🔴 Descartar", callback_data=f"discard:{clip_id}")],
    ])


async def _render_telegram_preview(
    settings: Settings,
    source: Path,
    output: Path,
    cpu_limiter: FfmpegCpuLimiter | None = None,
) -> None:
    """Create a compact approval copy without modifying the publication master."""
    limiter = cpu_limiter or FfmpegCpuLimiter(settings.ffmpeg_max_concurrent_encodes)
    candidates = encoder_candidates(
        settings.ffmpeg_preview_encoder,
        ("h264_qsv", "h264_amf", "h264_nvenc", "libx264"),
    )
    last_error = "No encoder was attempted."
    for encoder in candidates:
        if encoder != "libx264" and not await encoder_is_available(
            settings.ffmpeg_binary, encoder
        ):
            continue
        command = [
            settings.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a?",
            "-vf", "scale=720:1280:flags=lanczos,format=yuv420p", "-r", "30",
            *preview_encoder_arguments(
                encoder, threads=settings.ffmpeg_encoding_threads
            ),
            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
            "-y", str(output),
        ]
        async with limiter.encoding_slot():
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **low_priority_process_kwargs(settings.ffmpeg_low_priority),
            )
            _, stderr = await process.communicate()
        if process.returncode == 0:
            LOGGER.info("Telegram previews will use %s.", encoder)
            return
        last_error = stderr.decode(errors="replace").strip()
        LOGGER.warning(
            "%s failed for a Telegram preview; trying the next encoder: %s",
            encoder,
            last_error,
        )
    raise RuntimeError(f"FFmpeg Telegram preview failed: {last_error}")


async def _telegram_upload_path(
    settings: Settings,
    vertical_path: Path,
    cpu_limiter: FfmpegCpuLimiter | None = None,
) -> tuple[Path, bool]:
    if vertical_path.stat().st_size <= MAX_TELEGRAM_VIDEO_BYTES:
        return vertical_path, False
    preview = vertical_path.with_name(f"{vertical_path.stem}_telegram_preview.mp4")
    try:
        if cpu_limiter is None:
            await _render_telegram_preview(settings, vertical_path, preview)
        else:
            await _render_telegram_preview(settings, vertical_path, preview, cpu_limiter)
        if preview.stat().st_size > MAX_TELEGRAM_VIDEO_BYTES:
            raise RuntimeError(
                f"Telegram preview is still too large ({preview.stat().st_size / 1_000_000:.1f} MB)."
            )
        LOGGER.info(
            "Created %.1f MB Telegram preview for %.1f MB master.",
            preview.stat().st_size / 1_000_000,
            vertical_path.stat().st_size / 1_000_000,
        )
        return preview, True
    except Exception:
        preview.unlink(missing_ok=True)
        raise


async def send_approval_message(
    bot: Bot, settings: Settings, repository: ClipRepository, clip_id: int,
    vertical_path: Path, event_kind: str, confidence: float,
    cpu_limiter: FfmpegCpuLimiter | None = None,
) -> str:
    """Send an approval video using an initialized Bot or Application bot."""
    if not vertical_path.is_file():
        raise FileNotFoundError(f"Approval video does not exist: {vertical_path}")
    upload_path, temporary = await _telegram_upload_path(settings, vertical_path, cpu_limiter)
    caption = f"Evento visual: {event_kind}\nConfianza: {confidence:.0%}\nMaster: 1080x1920 / 60 FPS"
    try:
        with upload_path.open("rb") as video:
            message = await bot.send_video(
                chat_id=settings.telegram_chat_id,
                video=video,
                caption=caption,
                supports_streaming=True,
                reply_markup=_approval_keyboard(clip_id),
                connect_timeout=30,
                read_timeout=120,
                write_timeout=120,
            )
    finally:
        if temporary:
            upload_path.unlink(missing_ok=True)
    await repository.update_clip(clip_id, telegram_message_id=str(message.message_id))
    return str(message.message_id)


async def send_approval_once(
    settings: Settings, repository: ClipRepository, clip_id: int,
    vertical_path: Path, event_kind: str = "CARGA_MANUAL", confidence: float = 1.0,
) -> str:
    """Send from the dashboard without starting a second getUpdates poller."""
    if not settings.telegram_token or settings.telegram_chat_id is None:
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required.")
    bot = Bot(token=settings.telegram_token)
    await bot.initialize()
    try:
        return await send_approval_message(
            bot, settings, repository, clip_id, vertical_path, event_kind, confidence
        )
    finally:
        await bot.shutdown()


class TelegramApprovalBot:
    def __init__(
        self,
        settings: Settings,
        repository: ClipRepository,
        uploader: AutoUploader,
        cpu_limiter: FfmpegCpuLimiter | None = None,
    ) -> None:
        if not settings.telegram_token or settings.telegram_chat_id is None:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required.")
        self.settings = settings
        self.repository = repository
        self.uploader = uploader
        self.cpu_limiter = cpu_limiter
        self.application = Application.builder().token(settings.telegram_token).build()
        self.application.add_handler(CallbackQueryHandler(self._on_callback, pattern=r"^(approve|discard):"))
        self.application.add_error_handler(self._on_error)

    async def start(self) -> None:
        await self.application.initialize()
        await self.application.start()
        if self.application.updater is None:
            raise RuntimeError("Telegram updater is unavailable.")
        await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES)

    async def close(self) -> None:
        if self.application.updater:
            await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()

    async def request_approval(self, clip_id: int, vertical_path: Path, event_kind: str, confidence: float) -> None:
        await send_approval_message(
            self.application.bot, self.settings, self.repository,
            clip_id, vertical_path, event_kind, confidence, self.cpu_limiter,
        )

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or query.data is None:
            return
        if query.message is None or query.message.chat_id != self.settings.telegram_chat_id:
            await self._answer_callback(query, "No autorizado.", show_alert=True)
            return
        action, raw_clip_id = query.data.split(":", 1)
        try:
            clip_id = int(raw_clip_id)
        except ValueError:
            await self._answer_callback(query, "Identificador de clip inválido.", show_alert=True)
            return
        clip = await self.repository.get_clip(clip_id)
        if clip is None or clip.status not in {"PENDING_APPROVAL", "APPROVED_QUEUED"}:
            await self._answer_callback(query, "Este clip ya fue procesado o no está disponible.", show_alert=True)
            return
        vertical_path = self.settings.clips_dir / clip.filename
        if action == "discard":
            changed = await self.repository.transition_clip_status(
                clip_id,
                from_statuses=("PENDING_APPROVAL", "APPROVED_QUEUED"),
                to_status="DISCARDED",
            )
            if not changed:
                await self._answer_callback(
                    query, "El clip cambió de estado antes de descartarse.", show_alert=True
                )
                return
            await self._answer_callback(query)
            delete_clip_files(self.settings.clips_dir, clip.filename)
            await query.edit_message_caption(caption="🔴 Clip descartado y archivos temporales eliminados.")
            return
        changed = await self.repository.transition_clip_status(
            clip_id,
            from_statuses=("PENDING_APPROVAL",),
            to_status="APPROVED_QUEUED",
        )
        if not changed:
            await self._answer_callback(
                query, "El clip cambió de estado antes de aprobarse.", show_alert=True
            )
            return
        await self._answer_callback(query)
        await self.uploader.enqueue(clip_id, vertical_path, "Apex highlight")
        await query.edit_message_caption(caption="🟢 Aprobado. Se publicará cuando la transmisión no esté activa.")

    @staticmethod
    async def _answer_callback(query, text: str | None = None, **kwargs) -> bool:
        """A callback acknowledgement failure must not cancel the chosen action."""
        try:
            await query.answer(text, **kwargs)
            return True
        except (TimedOut, NetworkError) as exc:
            LOGGER.warning("Could not acknowledge Telegram callback: %s", exc)
            return False

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        error = context.error
        if isinstance(error, (TimedOut, NetworkError)):
            LOGGER.warning("Transient Telegram network error: %s", error)
            return
        if isinstance(error, BaseException):
            LOGGER.error(
                "Unhandled Telegram error", exc_info=(type(error), error, error.__traceback__)
            )
        else:
            LOGGER.error("Unhandled Telegram error: %r", error)

    @staticmethod
    def _delete_files(vertical_path: Path) -> None:
        delete_clip_files(vertical_path.parent, vertical_path.name)
