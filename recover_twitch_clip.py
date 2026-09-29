"""Recover a deleted source clip from a known Twitch VOD window."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime

from clipper import Clipper
from config import Settings
from database import ClipRepository
from stream_monitor import StreamMonitor


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("VOD start must include a UTC offset or end in Z.")
    return parsed


async def recover(
    clip_id: int,
    vod_id: str,
    vod_started_at: datetime,
    event_timestamp: float,
    event_kind: str,
) -> None:
    settings = Settings.from_env()
    settings.prepare_directories()
    repository = ClipRepository(settings)
    await repository.initialize()
    clip = await repository.get_clip(clip_id)
    if clip is None:
        raise LookupError(f"Clip #{clip_id} does not exist.")
    if clip.status not in {"CANCEL_REQUESTED", "DISCARDED"}:
        raise RuntimeError(
            f"Clip #{clip_id} is {clip.status}; recovery is limited to cancelled clips."
        )
    expected_name = f"apex_{int(event_timestamp)}_source.mp4"
    if clip.filename != expected_name:
        raise RuntimeError(
            f"Clip #{clip_id} expects {clip.filename}, not {expected_name}."
        )

    settings.source_kind = "vod"
    settings.twitch_url = f"https://www.twitch.tv/videos/{vod_id}"
    monitor = StreamMonitor(settings)
    source_url = await asyncio.to_thread(monitor.resolve_stream)
    event_offset = event_timestamp - vod_started_at.timestamp()
    pre_seconds, post_seconds = settings.clip_window(event_kind)
    start = max(0.0, event_offset - pre_seconds)
    end = event_offset + post_seconds
    print(
        f"Recovering clip #{clip_id} from VOD {vod_id}, "
        f"window {start:.3f}..{end:.3f}s",
        flush=True,
    )
    output = await Clipper(settings).create_clip_window(
        start,
        end,
        source_url,
        stamp_timestamp=event_timestamp,
    )
    if output.name != expected_name or not output.is_file():
        raise RuntimeError("Recovered source did not match the expected clip file.")
    moved = await repository.transition_clip_status(
        clip_id,
        from_statuses=("CANCEL_REQUESTED", "DISCARDED"),
        to_status="UPLOADED",
    )
    if not moved:
        raise RuntimeError(f"Clip #{clip_id} changed state during VOD recovery.")
    print(f"Recovered source: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_id", type=int)
    parser.add_argument("--vod-id", required=True)
    parser.add_argument("--vod-started-at", required=True, type=parse_utc)
    parser.add_argument("--event-timestamp", required=True, type=float)
    parser.add_argument("--event-kind", required=True)
    arguments = parser.parse_args()
    asyncio.run(recover(
        arguments.clip_id,
        arguments.vod_id,
        arguments.vod_started_at,
        arguments.event_timestamp,
        arguments.event_kind,
    ))
