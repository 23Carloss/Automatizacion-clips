# Apex Twitch Visual Clipper

Pipeline modular, sin análisis de audio, para detectar señales visuales de Apex Legends en Twitch, producir clips verticales y pedir aprobación en Telegram.

## Inicio rápido

1. Instala **FFmpeg** y, opcionalmente, el ejecutable **Tesseract OCR** y asegúrate de que ambos estén en `PATH`.
2. Crea y activa un entorno virtual; instala `pip install -r requirements.txt`.
3. Crea una cuenta MySQL con permiso para crear la base y tablas indicadas; copia `.env.example` como `.env` y completa Twitch, Telegram y MySQL.
4. Ejecuta `python main.py`. En otra terminal, ejecuta `streamlit run dashboard.py` para abrir el panel local.

El flujo usa Streamlink para resolver Twitch. Para directos, FFmpeg conserva continuamente un búfer circular comprimido de 120 segundos. Los eventos normales extraen 20 s anteriores y 5 s posteriores; `BLEEDOUT`, que suele aparecer al final de una pelea, extrae 25 s anteriores y 5 s posteriores. Para VOD se aplican las mismas ventanas usando el timestamp de su línea temporal. Al iniciar una prueba en vivo conviene dejar calentar el búfer al menos 25 segundos; si todavía no existe el historial solicitado, el programa rechaza el recorte parcial y lo explica en el log.

## Detección y edición

`event_detector.py` analiza dos ROI configurables para 1080p: la notificación central (`NOTIFICATION_ROI=500,700,1000,180`) y el killfeed superior derecho (`KILLFEED_ROI=1150,110,730,230`). Antes de usar OCR compara una firma gris reducida de cada ROI con la muestra anterior; solo los cambios que superan `OCR_MOTION_THRESHOLD` pasan por CLAHE, suavizado y una máscara binaria configurable. La imagen combinada se limita con `OCR_MAX_IMAGE_WIDTH` para reducir el coste de EasyOCR; Tesseract continúa disponible como alternativa.

La notificación central usa coincidencia difusa para eliminaciones, derribos, asistencias y escuadrones eliminados. Una lectura exacta o de similitud alta puede confirmar en un fotograma; una lectura marginal usa `OCR_CONFIRMATION_FRAMES` (dos de forma predeterminada). En el killfeed, `PLAYER_GAMERTAG` debe estar en la mitad izquierda de la línea. Esto reconoce el formato normal `jugador [icono] víctima` aunque no haya palabras clave; `ELIMINADO POR` y `DERRIBADO POR` se rechazan explícitamente.

FFmpeg muestrea dos imágenes por segundo de forma predeterminada y `OCR_KEYFRAME_INTERVAL_SECONDS=1` limita el trabajo pesado a un fotograma clave por segundo. El chequeo de movimiento es inmediato y un único trabajador ejecuta OCR fuera del bucle principal. La cola conserva solo el fotograma significativo más reciente (`OCR_QUEUE_SIZE=1`); antes de cada inferencia descarta cualquier pendiente más antiguo. En directos, cada fotograma recibe la hora real de llegada —no una estimación basada en el inicio de FFmpeg— para que la latencia normal de Twitch/HLS no se confunda con atraso de OCR. El trabajo que aun así supere `OCR_MAX_LIVE_LAG_SECONDS` se omite, avanzando la línea temporal sin permitir que EasyOCR se aleje indefinidamente del vídeo. Cada evento confirmado guarda el fotograma y un JSON con texto OCR, confianza, región y gamertag en `event_evidence/`. El flujo no inspecciona ni requiere audio; FFmpeg conserva la pista que ya esté presente en el vídeo fuente para el archivo final.

Después del OCR existe una segunda cola dedicada a eventos. `EVENT_COOLDOWN_SECONDS=0` mantiene la escucha continua mientras se recorta, renderiza o envía otro clip. Las detecciones separadas por menos de `EVENT_MERGE_GAP_SECONDS=50` desde la última actividad OCR válida se agrupan y extienden una única ventana; su etiqueta sube automáticamente a `SQUAD_ELIMINATED` cuando aparece esa señal. Una lectura repetida de la misma víctima no suma otra baja, pero mantiene abierta la pelea y extiende el final del clip mientras siga apareciendo. Las víctimas distintas sí suman eventos al mismo clip. Cuando no se puede leer un nombre, `EVENT_DUPLICATE_WINDOW_SECONDS=8` suprime solo las repeticiones ambiguas cercanas; el grupo se cierra tras 50 segundos sin actividad detectable.

Al aceptar el primer evento de un grupo, el programa crea hardlinks de los segmentos disponibles en `cache/reservations/`. Esa reserva fija inmediatamente el prebúfer y se amplía al cerrar el grupo para incluir el postbúfer. La limpieza circular no toca esos enlaces; se eliminan automáticamente cuando el clip fuente termina de construirse y las reservas de una ejecución interrumpida se limpian al siguiente arranque.

Las codificaciones pesadas comparten un límite configurable (`FFMPEG_MAX_CONCURRENT_ENCODES`) y los caminos por CPU usan como máximo `FFMPEG_ENCODING_THREADS` hilos con prioridad baja en Windows. `FFMPEG_SOURCE_ENCODER` controla la creación del clip fuente y `FFMPEG_VERTICAL_ENCODER` el render final. Admiten `h264_amf` (AMD), `h264_qsv` (Intel), `h264_nvenc` (NVIDIA), `libx264` (CPU) o `auto`; cada acelerador se prueba con una codificación real y siempre se conserva una caída segura a CPU.

`editor.py` no analiza audio. Ahora toma una zona central de 960×1080 y la muestra sobre un fondo desenfocado de 1080×1920; el gameplay se amplía aproximadamente 1.125 veces en vez de 1.78. `GAMEPLAY_CROP` permite ajustar esa zona con coordenadas x,y,width,height del video fuente de 1920×1080. Las tres capas HUD se conservan. El perfil master conserva los FPS de la fuente, etiqueta el color BT.709 y codifica el audio AAC a 128 kbps. `FFMPEG_SOURCE_QUALITY` mantiene el clip intermedio visualmente transparente. El master usa `VERTICAL_VIDEO_TARGET_KBPS=18000` y `VERTICAL_VIDEO_MAX_KBPS=22000` para mantenerse dentro del límite de 25 Mbps que Meta documenta para [Reels publicados por API](https://www.postman.com/meta/instagram/folder/830j7my/reels-publishing). `FFMPEG_VERTICAL_QUALITY` controla el objetivo de calidad cuando el codificador lo admite. Las vistas previas grandes de Telegram usan su propio `FFMPEG_PREVIEW_ENCODER`.

Solo se permite una instancia de `main.py`. Un segundo intento termina inmediatamente con un mensaje claro, evitando conflictos de Telegram y escrituras simultáneas en la caché. La limpieza de segmentos tolera bloqueos temporales de Windows y vuelve a intentarlo en una ejecución posterior.

## Calidad y grabaciones de PS5

En la pestaña Calidad del dashboard puedes elegir un clip y comparar imágenes del mismo segundo del archivo fuente, el render vertical y una copia local del Reel publicado. La comparación guarda el reporte en diagnostics/quality_reports/. También está disponible por consola:

```powershell
python -m diagnostics.clip_quality clips/apex_1790651769_source.mp4 clips/apex_1790651769_vertical.mp4 --at 8 --at 16
```

Una carga corta desde PS App sigue siendo un solo clip horizontal que se renderiza completo. Para buscar eventos en una grabación larga, usa Cargas y grabaciones > Analizar una grabación completa y escribe la ruta local del archivo, por ejemplo desde una unidad USB. El detector lee aproximadamente dos imágenes por segundo; no copia la hora de video al navegador ni crea un master de una hora. Solo recorta, renderiza y envía a aprobación los grupos donde detecta eventos. El archivo original permanece intacto y debe seguir disponible hasta terminar el análisis. Si no hay eventos, no crea clips. También se puede iniciar con `python recording_import.py RUTA_DEL_VIDEO`.

La PS5 permite [guardar hasta una hora de juego](https://www.playstation.com/es-es/support/games/capture-ps5-gameplay-screenshots/). PS App solo transfiere automáticamente videos de menos de tres minutos que no sean 4K ([ayuda oficial](https://www.playstation.com/es-es/support/games/ps5-game-captures-ps-app/)), por lo que una grabación larga se copia desde la consola a una unidad USB y luego al equipo. El panel lee ese archivo desde su ruta local para evitar una segunda copia completa.

## Publicación

Telegram es el punto de control humano. Cada cambio queda en MySQL (`clips`), incluyendo el id del mensaje de Telegram. Descartar borra los dos MP4 temporales. Aprobar cambia a `APPROVED_QUEUED`; el único worker de publicación espera a que el directo termine cuando `PAUSE_UPLOADS_WHILE_LIVE=true`.

Las cargas manuales del dashboard se envían automáticamente a Telegram. Si el master excede el límite del Bot API, se genera una vista previa temporal 720×1280/30 FPS de menos de 48 MB; el master 1080×1920/60 FPS no se modifica. Los clips pendientes muestran un botón para enviar o reenviar el mensaje. `main.py` debe estar activo para recibir los botones de aprobación y descarte.

El dashboard permite cargar MP4 cortos de PS App, analizar grabaciones largas de PS5 desde una ruta local, comparar la calidad de las etapas, navegar el pipeline y probar MySQL. Solo se consultan y renderizan las tarjetas de la etapa seleccionada; cada menú conserva su propia paginación y un área con scroll. En **Procesando**, FFmpeg reporta el porcentaje real según el tiempo codificado frente a la duración del clip. El render puede cancelarse desde su tarjeta: el proceso se detiene en unos segundos, libera CPU/GPU y elimina los archivos parciales. También puedes descartar manualmente clips cargados, pendientes, en cola o con una publicación fallida; la confirmación elimina sus MP4 temporales. El tablero muestra el intento y la etapa actual de cada plataforma. Sus controles ROI son para probar una sesión: pasa los valores que funcionen a `config.py` para conservarlos. No expongas el dashboard a Internet.

Un fallo normal de render o un cierre controlado devuelve automáticamente el registro a `UPLOADED`, conservando su fuente para reintento. Si un apagado forzado deja un registro antiguo en `PROCESSING` sin un proceso FFmpeg activo, puede reclamarse y renderizarse nuevamente con `python retry_clips.py --force-stale-processing ID`. Verifica primero que el registro sea realmente huérfano. Cuando un clip cancelado todavía conserva su MP4 fuente, puede recuperarse con `python retry_clips.py --recover-cancelled ID`. El comando procesa los IDs en serie, actualiza su porcentaje y los deja en `PENDING_APPROVAL`.

Si el fuente de un clip cancelado ya fue eliminado pero el VOD de Twitch continúa disponible, `recover_twitch_clip.py` permite reconstruir solo su ventana original indicando el ID del VOD, su fecha de inicio UTC, el timestamp del evento y su tipo. La utilidad valida el nombre esperado antes de devolver el registro a `UPLOADED`; después se procesa normalmente con `retry_clips.py ID`.

Los conectores oficiales configurados son:

- YouTube sube inicialmente como privado por seguridad.
- Instagram sube primero el master local a Cloudflare R2, entrega a Graph API una URL HTTPS temporal, espera el procesamiento del contenedor y publica el Reel.
- TikTok usa Content Posting Upload API (`video.upload`), transfiere el MP4 a la bandeja de entrada de la app y espera hasta confirmar que el borrador quedó listo. La edición final y la publicación se hacen manualmente desde TikTok.

Las claves no se guardan en código ni se versionan.

### Configurar Instagram + Cloudflare R2

1. En Cloudflare crea un bucket R2 y un token S3 limitado a ese bucket con permiso **Object Read & Write**.
2. Copia a `.env` el Account ID, Access Key ID, Secret Access Key y nombre del bucket. No es necesario hacer público el bucket: si `R2_PUBLIC_BASE_URL` queda vacío, el programa genera una URL GET firmada con vigencia de seis horas.
3. Conserva en `.env` tu `INSTAGRAM_ACCESS_TOKEN` e `INSTAGRAM_USER_ID`. El flujo de Instagram Login usa `INSTAGRAM_GRAPH_BASE_URL=https://graph.instagram.com`; `META_GRAPH_API_VERSION` permite actualizar la versión sin cambiar código.
4. Instala las dependencias actualizadas con `pip install -r requirements.txt` y reinicia `main.py` para que MySQL amplíe el esquema automáticamente.

Variables mínimas que debes completar:

```env
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET=
```

El flujo usa `R2_KEY_PREFIX=clips`, URLs firmadas de seis horas y elimina el objeto remoto únicamente después de recibir el ID publicado de Instagram. Cambia `R2_DELETE_AFTER_PUBLISH=false` si quieres conservarlo. Si ya dispones de un CDN que recibe los archivos por otro mecanismo, puedes dejar R2 vacío y usar el `INSTAGRAM_VIDEO_URL_TEMPLATE` heredado; R2 tiene prioridad cuando está completo.

### Configurar borradores de TikTok

1. En TikTok for Developers añade **Content Posting API** y solicita el scope `video.upload`; para este flujo no se necesita `video.publish`.
2. Autoriza tu cuenta TikTok mediante OAuth y copia el access token resultante en `TIKTOK_ACCESS_TOKEN` dentro de `.env`.
3. Reinicia `main.py`. Cuando apruebes un clip, el dashboard mostrará **Iniciando borrador**, **Subiendo a TikTok**, **Procesando en TikTok** y finalmente **Borrador listo**.
4. TikTok enviará una notificación a la bandeja de entrada de la app móvil. Ábrela para añadir o ajustar el texto, editar el vídeo y publicarlo manualmente.

El programa carga automáticamente en partes los archivos de más de 64 MB y vuelve a intentar partes fallidas sobre la misma sesión. Si el estado final queda incierto, detiene los reintentos para no crear borradores duplicados. TikTok limita la cantidad de cargas pendientes; completa o elimina los borradores anteriores antes de repetir una prueba.

### Estados y reintentos de publicación

Al aprobar, el clip pasa por `APPROVED_QUEUED` y `PUBLISHING`. Cada intento por plataforma se registra en `publication_attempts`, incluyendo resultado y error. Los fallos transitorios se repiten hasta `UPLOAD_MAX_ATTEMPTS` con espera exponencial basada en `UPLOAD_RETRY_BASE_SECONDS`. El resultado final fallido queda como `PUBLISH_FAILED` y el dashboard muestra **Reintentar publicación**. El worker revisa la base periódicamente, así que no es necesario reiniciar el proceso.

Los éxitos previos y los borradores TikTok ya entregados se consultan antes de un reintento para no duplicar publicaciones. Si la conexión se corta exactamente al enviar `media_publish`, el sistema no reintenta Instagram automáticamente porque el resultado podría haber sido publicado; revisa la cuenta antes de usar el botón manual. Un cierre inesperado también deja el clip en `PUBLISH_FAILED` para revisión segura.

Para consultar el historial directamente:

```sql
SELECT clip_id, platform, attempt_number, status, detail, started_at, finished_at
FROM publication_attempts
ORDER BY started_at DESC;
```
