"""Retry interrupted vertical renders by clip id.

Use ``--force-stale-processing`` only after verifying that no FFmpeg process
still owns a row left in PROCESSING by an earlier application run.
"""
from __future__ import annotations

import argparse
import asyncio

from config import Settings
from database import ClipRecord, ClipRepository
from editor import ApexVerticalEditor, RenderCancelled
from telegram_bot import send_approval_once


async def retry_clip(
    clip: ClipRecord,
    settings: Settings,
    repository: ClipRepository,
    editor: ApexVerticalEditor,
    *,
    force_stale_processing: bool,
    recover_cancelled: bool,
) -> None:
    if clip.status == "PROCESSING":
        if not force_stale_processing:
            raise RuntimeError(
                f"Clip #{clip.id} is PROCESSING; verify it is stale and pass "
                "--force-stale-processing."
            )
        released = await repository.transition_clip_status(
            clip.id,
            from_statuses=("PROCESSING",),
            to_status="UPLOADED",
        )
        if not released:
            raise RuntimeError(f"Clip #{clip.id} changed state before it could be retried.")
    elif clip.status in {"CANCEL_REQUESTED", "DISCARDED"}:
        if not recover_cancelled:
            raise RuntimeError(
                f"Clip #{clip.id} is {clip.status}; pass --recover-cancelled only "
                "when its source file still exists."
            )
        recovered = await repository.transition_clip_status(
            clip.id,
            from_statuses=("CANCEL_REQUESTED", "DISCARDED"),
            to_status="UPLOADED",
        )
        if not recovered:
            raise RuntimeError(f"Clip #{clip.id} changed state before recovery.")
    elif clip.status != "UPLOADED":
        raise RuntimeError(
            f"Clip #{clip.id} is {clip.status}; only UPLOADED or verified stale "
            "PROCESSING clips can be retried."
        )

    claimed = await repository.transition_clip_status(
        clip.id,
        from_statuses=("UPLOADED",),
        to_status="PROCESSING",
    )
    if not claimed:
        raise RuntimeError(f"Clip #{clip.id} could not be claimed for processing.")
    await repository.set_processing_progress(clip.id, 0.0)

    source_path = settings.clips_dir / clip.filename
    if not source_path.is_file():
        await repository.transition_clip_status(
            clip.id,
            from_statuses=("PROCESSING",),
            to_status="UPLOADED",
        )
        raise FileNotFoundError(f"Source file does not exist: {source_path}")

    last_printed = -1

    async def report_progress(progress: float) -> None:
        nonlocal last_printed
        await repository.set_processing_progress(clip.id, progress)
        whole = int(progress)
        if whole >= last_printed + 5 or whole == 100:
            last_printed = whole
            print(f"Clip #{clip.id}: {progress:.1f}%", flush=True)

    try:
        output_path = await editor.render(
            source_path,
            cancel_requested=lambda: repository.processing_should_stop(clip.id),
            progress_callback=report_progress,
        )
        moved = await repository.transition_clip_status(
            clip.id,
            from_statuses=("PROCESSING",),
            to_status="PENDING_APPROVAL",
            filename=output_path.name,
        )
        if not moved:
            raise RenderCancelled(f"Clip #{clip.id} was cancelled before completion.")
    except RenderCancelled:
        print(f"Clip #{clip.id}: cancelled from the dashboard.", flush=True)
        return
    except Exception:
        await repository.transition_clip_status(
            clip.id,
            from_statuses=("PROCESSING",),
            to_status="UPLOADED",
        )
        raise

    try:
        await send_approval_once(settings, repository, clip.id, output_path)
        print(f"Clip #{clip.id}: rendered and sent to Telegram for approval.", flush=True)
    except Exception as exc:
        print(
            f"Clip #{clip.id}: rendered successfully, but Telegram delivery failed: {exc}",
            flush=True,
        )


async def main(
    clip_ids: list[int], force_stale_processing: bool, recover_cancelled: bool,
) -> None:
    settings = Settings.from_env()
    settings.prepare_directories()
    repository = ClipRepository(settings)
    await repository.initialize()
    editor = ApexVerticalEditor(settings)
    for clip_id in clip_ids:
        clip = await repository.get_clip(clip_id)
        if clip is None:
            raise LookupError(f"Clip #{clip_id} does not exist.")
        print(f"Retrying clip #{clip.id}: {clip.filename}", flush=True)
        await retry_clip(
            clip,
            settings,
            repository,
            editor,
            force_stale_processing=force_stale_processing,
            recover_cancelled=recover_cancelled,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_ids", nargs="+", type=int)
    parser.add_argument("--force-stale-processing", action="store_true")
    parser.add_argument("--recover-cancelled", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(main(
        arguments.clip_ids,
        arguments.force_stale_processing,
        arguments.recover_cancelled,
    ))
