"""H.264 encoder discovery and quality presets shared by clip stages."""
from __future__ import annotations

import asyncio
from collections.abc import Iterable


SUPPORTED_H264_ENCODERS = ("auto", "h264_amf", "h264_qsv", "h264_nvenc", "libx264")


def encoder_candidates(preferred: str, automatic_order: Iterable[str]) -> list[str]:
    """Return a unique hardware-first chain ending in the CPU fallback."""
    if preferred == "auto":
        requested = list(automatic_order)
    else:
        requested = [preferred]
    return list(dict.fromkeys([*requested, "libx264"]))


async def encoder_is_available(ffmpeg_binary: str, encoder: str) -> bool:
    """Verify that FFmpeg can initialize the encoder, not merely list it."""
    process = await asyncio.create_subprocess_exec(
        ffmpeg_binary,
        "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=128x128:r=30:d=0.1",
        "-frames:v", "2", "-an", "-c:v", encoder, "-f", "null", "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await process.communicate()
    return process.returncode == 0


def encoder_arguments(encoder: str, *, quality: int, threads: int, fast: bool) -> list[str]:
    """Build comparable high-quality settings for each supported H.264 encoder."""
    if encoder == "h264_amf":
        return [
            "-c:v", encoder,
            "-usage", "high_quality",
            "-quality", "quality",
            "-rc", "cqp",
            "-qp_i", str(quality),
            "-qp_p", str(quality),
            "-qp_b", str(min(51, quality + 2)),
            "-profile:v", "high",
        ]
    if encoder == "h264_qsv":
        return [
            "-c:v", encoder,
            "-preset", "fast" if fast else "slow",
            "-global_quality", str(quality),
            "-profile:v", "high",
        ]
    if encoder == "h264_nvenc":
        return [
            "-c:v", encoder,
            "-preset", "p4" if fast else "p6",
            "-rc", "vbr",
            "-cq", str(quality),
            "-b:v", "0",
            "-profile:v", "high",
        ]
    if encoder == "libx264":
        return [
            "-c:v", encoder,
            "-crf", str(quality),
            "-preset", "fast" if fast else "slow",
            "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv",
            "-threads", str(threads),
        ]
    raise ValueError(f"Unsupported H.264 encoder: {encoder}")


def preview_encoder_arguments(encoder: str, *, threads: int) -> list[str]:
    """Build a fast, size-bounded encoder profile for Telegram previews."""
    common_rate = ["-b:v", "2200k", "-maxrate", "2500k", "-bufsize", "5000k"]
    if encoder == "h264_amf":
        return [
            "-c:v", encoder, "-usage", "transcoding", "-quality", "speed",
            "-rc", "vbr_peak", *common_rate, "-profile:v", "high",
        ]
    if encoder == "h264_qsv":
        return ["-c:v", encoder, "-preset", "veryfast", *common_rate, "-profile:v", "high"]
    if encoder == "h264_nvenc":
        return ["-c:v", encoder, "-preset", "p3", "-rc", "vbr", *common_rate, "-profile:v", "high"]
    if encoder == "libx264":
        return [
            "-c:v", encoder, "-preset", "fast", "-crf", "27", *common_rate,
            "-threads", str(threads),
        ]
    raise ValueError(f"Unsupported H.264 encoder: {encoder}")
