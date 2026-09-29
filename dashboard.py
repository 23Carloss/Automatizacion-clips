"""Streamlit operations console for the Apex visual-clip pipeline.

Run with: ``streamlit run dashboard.py``.  It shares the same MySQL database
and clips directory as ``main.py``; do not expose it to the public internet.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from math import ceil
from pathlib import Path

import streamlit as st

from config import Roi, Settings
from database import ClipRecord, ClipRepository
from editor import ApexVerticalEditor, RenderCancelled
from telegram_bot import delete_clip_files, send_approval_once

PIPELINE_STATES = (
    "UPLOADED", "PROCESSING", "CANCEL_REQUESTED", "PENDING_APPROVAL", "APPROVED_QUEUED",
    "PUBLISHING", "PUBLISH_FAILED", "PUBLISHED", "DISCARDED",
)
PIPELINE_STAGE_LABELS = {
    "UPLOADED": "📥 Cargados",
    "PROCESSING": "⚙️ Procesando",
    "CANCEL_REQUESTED": "⏹️ Cancelando",
    "PENDING_APPROVAL": "👀 Por aprobar",
    "APPROVED_QUEUED": "🕓 En cola",
    "PUBLISHING": "🚀 Publicando",
    "PUBLISH_FAILED": "⚠️ Con error",
    "PUBLISHED": "✅ Publicados",
    "DISCARDED": "🗑️ Descartados",
}
PIPELINE_STAGE_DESCRIPTIONS = {
    "UPLOADED": "Archivos recibidos que todavía no completan el render vertical.",
    "PROCESSING": "Render vertical activo. Aquí puedes revisar el porcentaje o cancelar FFmpeg.",
    "CANCEL_REQUESTED": "Cancelaciones solicitadas que están deteniendo FFmpeg y limpiando archivos.",
    "PENDING_APPROVAL": "Clips listos para enviar o reenviar a Telegram y decidir su publicación.",
    "APPROVED_QUEUED": "Clips aprobados que esperan su turno de publicación.",
    "PUBLISHING": "Clips que se están enviando a las plataformas configuradas.",
    "PUBLISH_FAILED": "Publicaciones fallidas disponibles para reintentar o descartar.",
    "PUBLISHED": "Historial de clips publicados correctamente.",
    "DISCARDED": "Historial de clips cancelados o descartados manualmente.",
}
DISCARDABLE_STATES = (
    "UPLOADED", "PENDING_APPROVAL", "APPROVED_QUEUED", "PUBLISH_FAILED",
)
PIPELINE_PAGE_SIZE = 5
PIPELINE_COLUMN_HEIGHT = 720

STAGE_LABELS = {
    "STARTED": "Iniciando",
    "R2_UPLOADING": "Subiendo a R2",
    "R2_READY": "URL HTTPS lista",
    "SOURCE_READY": "Origen HTTPS listo",
    "CONTAINER_CREATING": "Creando contenedor",
    "PROCESSING": "Procesando en Instagram",
    "CONTAINER_READY": "Contenedor listo",
    "PUBLISHING": "Publicando",
    "R2_CLEANUP": "Limpiando R2",
    "TIKTOK_INITIALIZING": "Iniciando borrador",
    "TIKTOK_UPLOADING": "Subiendo a TikTok",
    "TIKTOK_PROCESSING": "Procesando en TikTok",
    "DRAFT_READY": "Borrador listo",
    "PUBLISHED": "Publicado",
    "FAILED": "Falló",
    "SKIPPED": "Omitido",
}


def run_async(coro):
    """Run a short background operation from Streamlit's synchronous script."""
    return asyncio.run(coro)


@st.cache_resource
def services() -> tuple[Settings, ClipRepository]:
    settings = Settings.from_env()
    settings.prepare_directories()
    repository = ClipRepository(settings)
    run_async(repository.initialize())
    return settings, repository


def process_ps_app_upload(uploaded_file, settings: Settings, repository: ClipRepository) -> int:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix != ".mp4":
        raise ValueError("Solo se permiten archivos .mp4.")
    name = f"psapp_{datetime.now():%Y%m%d_%H%M%S_%f}_source.mp4"
    source_path = settings.clips_dir / name
    source_path.write_bytes(uploaded_file.getbuffer())
    clip_id = run_async(repository.create_clip(name, "UPLOADED", "PS_APP"))
    try:
        run_async(repository.update_clip(clip_id, status="PROCESSING"))
        output_path = run_async(ApexVerticalEditor(settings).render(
            source_path,
            cancel_requested=lambda: repository.processing_should_stop(clip_id),
            progress_callback=lambda progress: repository.set_processing_progress(
                clip_id, progress
            ),
        ))
    except RenderCancelled:
        run_async(repository.transition_clip_status(
            clip_id,
            from_statuses=("CANCEL_REQUESTED",),
            to_status="DISCARDED",
        ))
        delete_clip_files(settings.clips_dir, name)
        raise
    except Exception as exc:
        if run_async(repository.processing_should_stop(clip_id)):
            run_async(repository.transition_clip_status(
                clip_id,
                from_statuses=("CANCEL_REQUESTED",),
                to_status="DISCARDED",
            ))
            delete_clip_files(settings.clips_dir, name)
            raise RenderCancelled(
                "El procesamiento fue cancelado desde el dashboard."
            ) from exc
        # The source stays available for a retry; its row makes the failed stage visible.
        run_async(repository.transition_clip_status(
            clip_id,
            from_statuses=("PROCESSING",),
            to_status="UPLOADED",
        ))
        raise
    moved = run_async(repository.transition_clip_status(
        clip_id,
        from_statuses=("PROCESSING",),
        to_status="PENDING_APPROVAL",
        filename=output_path.name,
    ))
    if not moved:
        if run_async(repository.processing_should_stop(clip_id)):
            run_async(repository.transition_clip_status(
                clip_id,
                from_statuses=("CANCEL_REQUESTED",),
                to_status="DISCARDED",
            ))
            delete_clip_files(settings.clips_dir, output_path.name)
            raise RenderCancelled("El procesamiento fue cancelado desde el dashboard.")
        raise RuntimeError("El clip cambió de estado antes de terminar el procesamiento.")
    # Sending messages does not call getUpdates, so this does not compete with
    # the main process that owns Telegram callback polling.
    run_async(send_approval_once(settings, repository, clip_id, output_path))
    return clip_id


def clip_card(clip: ClipRecord, settings: Settings, repository: ClipRepository) -> None:
    path = settings.clips_dir / clip.filename
    st.caption(f"#{clip.id} · {clip.source} · {clip.timestamp:%Y-%m-%d %H:%M}")
    st.write(f"`{clip.filename}`")
    if clip.status in {"PROCESSING", "CANCEL_REQUESTED"}:
        progress = max(0.0, min(float(clip.processing_progress), 100.0))
        label = (
            f"Deteniendo en {progress:.1f}%…"
            if clip.status == "CANCEL_REQUESTED"
            else f"{progress:.1f}% procesado"
        )
        st.progress(progress / 100, text=label)
    if path.exists() and path.suffix.lower() == ".mp4":
        st.video(str(path))
    else:
        st.info("Archivo local no disponible todavía.")
    if clip.status == "PROCESSING":
        with st.popover("⏹️ Cancelar procesamiento", use_container_width=True):
            st.warning("Se detendrá FFmpeg y se eliminarán los archivos parciales.")
            if st.button("Confirmar cancelación", key=f"cancel-processing-{clip.id}", type="primary"):
                try:
                    changed = run_async(repository.transition_clip_status(
                        clip.id,
                        from_statuses=("PROCESSING",),
                        to_status="DISCARDED",
                    ))
                    if changed:
                        delete_clip_files(settings.clips_dir, clip.filename)
                        st.success("Cancelación solicitada. FFmpeg se detendrá en unos segundos.")
                        st.rerun()
                    else:
                        st.warning("El clip ya cambió de etapa; el tablero se actualizará.")
                except Exception as exc:
                    st.error(f"No se pudo solicitar la cancelación: {exc}")
    elif clip.status == "CANCEL_REQUESTED":
        st.info("Deteniendo FFmpeg…")
        if st.button(
            "Finalizar descarte",
            key=f"finish-cancellation-{clip.id}",
            use_container_width=True,
        ):
            changed = run_async(repository.transition_clip_status(
                clip.id,
                from_statuses=("CANCEL_REQUESTED",),
                to_status="DISCARDED",
            ))
            if changed:
                delete_clip_files(settings.clips_dir, clip.filename)
                st.rerun()
    if clip.status == "PENDING_APPROVAL" and path.is_file():
        action = "Reenviar a Telegram" if clip.telegram_message_id else "Enviar a Telegram"
        if st.button(action, key=f"telegram-{clip.id}"):
            try:
                with st.spinner("Preparando y enviando vista previa…"):
                    run_async(send_approval_once(settings, repository, clip.id, path))
                st.success("Mensaje de aprobación enviado a Telegram.")
                st.rerun()
            except Exception as exc:
                st.error(f"No se pudo enviar a Telegram: {exc}")
    if clip.status == "PUBLISH_FAILED" and path.is_file():
        if st.button("Reintentar publicación", key=f"retry-publish-{clip.id}"):
            try:
                run_async(repository.update_clip(clip.id, status="APPROVED_QUEUED"))
                st.success("Reencolado. El proceso principal lo retomará en unos segundos.")
                st.rerun()
            except Exception as exc:
                st.error(f"No se pudo reencolar: {exc}")
    if clip.status in DISCARDABLE_STATES:
        with st.popover("🗑️ Descartar", use_container_width=True):
            st.warning("El clip y sus archivos temporales se eliminarán.")
            if st.button("Confirmar descarte", key=f"confirm-discard-{clip.id}", type="primary"):
                try:
                    changed = run_async(repository.transition_clip_status(
                        clip.id,
                        from_statuses=DISCARDABLE_STATES,
                        to_status="DISCARDED",
                    ))
                    if not changed:
                        st.warning("El clip cambió de etapa. Actualiza el tablero e inténtalo de nuevo.")
                    else:
                        locked_files = delete_clip_files(settings.clips_dir, clip.filename)
                        if locked_files:
                            st.warning(
                                "El clip fue descartado, pero algún archivo está en uso y no pudo eliminarse."
                            )
                        else:
                            st.success("Clip descartado y archivos temporales eliminados.")
                        st.rerun()
                except Exception as exc:
                    st.error(f"No se pudo descartar el clip: {exc}")
    if clip.status in {"PUBLISHING", "PUBLISH_FAILED", "PUBLISHED"}:
        attempts = run_async(repository.list_publication_attempts(clip.id, limit=12))
        if attempts:
            with st.expander(
                "Progreso de publicación",
                expanded=clip.status in {"PUBLISHING", "PUBLISH_FAILED"},
            ):
                for attempt in attempts:
                    label = STAGE_LABELS.get(attempt.status, attempt.status)
                    icon = "✅" if attempt.status in {"PUBLISHED", "DRAFT_READY"} else "❌" if attempt.status == "FAILED" else "⏳"
                    st.markdown(
                        f"{icon} **{attempt.platform}** · intento {attempt.attempt_number} · {label}"
                    )
                    if attempt.detail:
                        st.caption(attempt.detail)


@st.fragment(run_every="3s")
def pipeline_board(settings: Settings, repository: ClipRepository) -> None:
    st.caption(
        f"Actualización automática cada 3 segundos · {PIPELINE_PAGE_SIZE} clips por página."
    )
    toolbar = st.columns([5, 1])
    selected_status = toolbar[0].segmented_control(
        "Etapa del pipeline",
        PIPELINE_STATES,
        default="PROCESSING",
        format_func=lambda status: PIPELINE_STAGE_LABELS[status],
        key="pipeline-stage-menu",
        label_visibility="collapsed",
    ) or "PROCESSING"
    if toolbar[1].button("Actualizar", use_container_width=True):
        st.rerun(scope="fragment")

    page_key = f"pipeline-page-{selected_status}"
    requested_page = int(st.session_state.get(page_key, 1))
    matching, total = run_async(repository.list_clips_by_status_page(
        selected_status,
        limit=PIPELINE_PAGE_SIZE,
        offset=(requested_page - 1) * PIPELINE_PAGE_SIZE,
    ))
    total_pages = max(1, ceil(total / PIPELINE_PAGE_SIZE))
    current_page = min(max(1, requested_page), total_pages)
    if st.session_state.get(page_key) != current_page:
        st.session_state[page_key] = current_page

    heading, pagination = st.columns([4, 1])
    heading.subheader(f"{PIPELINE_STAGE_LABELS[selected_status]} ({total})")
    heading.caption(PIPELINE_STAGE_DESCRIPTIONS[selected_status])
    if total_pages > 1:
        pagination.selectbox(
            "Página",
            options=range(1, total_pages + 1),
            key=page_key,
            format_func=lambda page: f"Página {page}/{total_pages}",
            label_visibility="collapsed",
        )

    with st.container(height=PIPELINE_COLUMN_HEIGHT, border=True):
        if not matching:
            st.info("No hay clips en esta etapa.")
        card_columns = st.columns(2)
        for index, clip in enumerate(matching):
            with card_columns[index % len(card_columns)]:
                with st.container(border=True):
                    clip_card(clip, settings, repository)


def roi_inputs(label: str, roi: Roi) -> Roi:
    st.markdown(f"**{label}**")
    cols = st.columns(4)
    values = []
    for column, field, value in zip(cols, ("x", "y", "ancho", "alto"), (roi.x, roi.y, roi.width, roi.height)):
        values.append(column.number_input(field, min_value=0, value=value, step=1, key=f"{label}_{field}"))
    return Roi(*map(int, values))


def main() -> None:
    st.set_page_config(page_title="Apex Clips", page_icon="🎮", layout="wide")
    st.title("🎮 Apex Clips — monitor visual")
    try:
        settings, repository = services()
    except Exception as exc:
        st.error(f"No se pudo iniciar MySQL: {exc}")
        st.stop()
    upload_tab, pipeline_tab, settings_tab = st.tabs(["Carga manual PS App", "Pipeline de clips", "Ajustes"])

    with upload_tab:
        st.write("Carga un clip horizontal descargado desde PS App. Se renderiza a 9:16 y queda pendiente de aprobación.")
        uploaded_file = st.file_uploader("Clip .mp4", type=["mp4"])
        if uploaded_file and st.button("Procesar clip", type="primary"):
            try:
                with st.spinner("Renderizando el clip vertical…"):
                    process_ps_app_upload(uploaded_file, settings, repository)
                st.success("Listo: el clip está pendiente de aprobación y fue enviado a Telegram.")
            except RenderCancelled:
                st.info("Procesamiento cancelado; se eliminaron los archivos temporales.")
            except Exception as exc:
                st.error(f"El clip quedó disponible para reintento: {exc}")

    with pipeline_tab:
        pipeline_board(settings, repository)

    with settings_tab:
        st.write("El detector usa exclusivamente la ROI de notificación central. Los cambios se aplican a esta sesión del dashboard; copia los valores verificados a `config.py` para hacerlos permanentes.")
        try:
            version = run_async(repository.test_connection())
            st.success(f"MySQL conectado: {version}")
        except Exception as exc:
            st.error(str(exc))
        with st.form("roi-form"):
            notification = roi_inputs("Notificación central", settings.notification_roi)
            bleedout = roi_inputs("Desangrado / killfeed", settings.bleedout_roi)
            if st.form_submit_button("Aplicar en esta sesión"):
                settings.notification_roi = notification
                settings.bleedout_roi = bleedout
                st.success("ROI aplicada en esta sesión.")


if __name__ == "__main__":
    main()
