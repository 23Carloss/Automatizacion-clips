"""Create a local, time-aligned visual report for clip quality diagnosis."""
from __future__ import annotations

import argparse
import html
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    video_mbps: float | None
    file_mbps: float | None


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed for {command[-1]}: {result.stderr.strip()}")
    return result.stdout


def _fps(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if separator:
        return float(numerator) / float(denominator)
    return float(value)


def probe(path: Path, ffprobe: str) -> VideoInfo:
    if not path.is_file():
        raise FileNotFoundError(path)
    details = json.loads(_run([
        ffprobe, "-v", "error", "-show_entries",
        "format=duration,bit_rate:stream=codec_type,width,height,avg_frame_rate,bit_rate",
        "-of", "json", str(path),
    ]))
    video = next(
        (stream for stream in details.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video is None:
        raise ValueError(f"No video stream in {path}")
    file_data = details.get("format", {})
    def mbps(value: str | None) -> float | None:
        return round(int(value) / 1_000_000, 2) if value else None
    return VideoInfo(
        path=str(path.resolve()),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=round(_fps(video["avg_frame_rate"]), 2),
        duration_seconds=round(float(file_data["duration"]), 2),
        video_mbps=mbps(video.get("bit_rate")),
        file_mbps=mbps(file_data.get("bit_rate")),
    )


def create_report(
    source: Path, vertical: Path, published: Path | None = None, *,
    seconds: list[float] | None = None, published_offset: float = 0.0,
    output_dir: Path | None = None, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
) -> Path:
    videos = {"Fuente Twitch": source, "Vertical local": vertical}
    if published is not None:
        videos["Publicado en Instagram"] = published
    info = {label: probe(path, ffprobe) for label, path in videos.items()}
    common_duration = min(item.duration_seconds for item in info.values())
    if seconds is None:
        seconds = [round(common_duration * portion, 2) for portion in (0.2, 0.5, 0.8)]
    if not seconds or any(at < 0 or at >= common_duration for at in seconds):
        raise ValueError(f"Choose times from 0 up to {common_duration:.2f} seconds.")
    if published is not None and any(
        at + published_offset < 0
        or at + published_offset >= info["Publicado en Instagram"].duration_seconds
        for at in seconds
    ):
        raise ValueError("The Instagram offset puts a frame outside the published video.")

    output_dir = output_dir or Path(__file__).resolve().parent / "quality_reports" / source.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[list[tuple[str, str]]] = []
    for index, at in enumerate(seconds, start=1):
        row: list[tuple[str, str]] = []
        for label, path in videos.items():
            capture_at = at + published_offset if label == "Publicado en Instagram" else at
            image_name = f"{index:02d}_{list(videos).index(label) + 1}.png"
            _run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{capture_at:.3f}",
                "-i", str(path), "-frames:v", "1", "-an", "-y", str(output_dir / image_name),
            ])
            row.append((label, image_name))
        frame_rows.append(row)

    metadata = {"times_seconds": seconds, "published_offset_seconds": published_offset,
                "videos": {label: asdict(item) for label, item in info.items()}}
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cells = "".join(
        "<tr><th>{label}</th><td>{size}</td><td>{fps}</td><td>{video_rate}</td>"
        "<td>{duration}</td></tr>".format(
            label=html.escape(label),
            size=f"{item.width} × {item.height}",
            fps=item.fps,
            video_rate=f"{item.video_mbps:.2f}" if item.video_mbps is not None else "—",
            duration=f"{item.duration_seconds:.2f} s",
        )
        for label, item in info.items()
    )
    galleries = "".join(
        '<section><h2>Segundo {at:.2f}</h2><div class="frames">{images}</div></section>'.format(
            at=at,
            images="".join(
                '<figure><img src="{name}" alt="{label} en segundo {at:.2f}"><figcaption>{label}</figcaption></figure>'.format(
                    name=html.escape(name), label=html.escape(label), at=at
                )
                for label, name in row
            ),
        )
        for at, row in zip(seconds, frame_rows)
    )
    document = f"""<!doctype html>
<html lang="es"><meta charset="utf-8"><title>Comparación de calidad</title>
<style>
body{{font:16px system-ui,sans-serif;background:#111827;color:#f9fafb;max-width:1440px;margin:auto;padding:24px}}
p{{line-height:1.5;color:#d1d5db}}table{{border-collapse:collapse;width:100%;margin:20px 0 32px}}
th,td{{padding:10px;border-bottom:1px solid #374151;text-align:left}}
.frames{{display:flex;gap:16px;overflow-x:auto}}figure{{flex:1;min-width:260px;margin:0}}
img{{width:100%;height:440px;object-fit:contain;background:#030712}}
figcaption{{padding:8px 0;font-weight:600}}section{{margin:28px 0 44px}}
</style><body>
<h1>Comparación de calidad del clip</h1>
<p>Las imágenes muestran los mismos segundos de cada etapa, sin la compresión del reproductor.
Si la fuente ya tiene bloques, el problema aparece antes de la edición. Si la fuente se ve bien
y el vertical no, revisa el encuadre y la exportación. Si solo empeora el archivo publicado,
revisa el procesamiento de Instagram.</p>
<table><thead><tr><th>Etapa</th><th>Resolución</th><th>FPS</th><th>Video Mbps</th><th>Duración</th></tr></thead>
<tbody>{cells}</tbody></table>{galleries}</body></html>"""
    report = output_dir / "report.html"
    report.write_text(document, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local _source.mp4")
    parser.add_argument("vertical", type=Path, help="Local _vertical.mp4")
    parser.add_argument("--published", type=Path, help="Downloaded Instagram MP4, if available")
    parser.add_argument("--at", type=float, action="append", help="Second to compare; repeat as needed")
    parser.add_argument("--published-offset", type=float, default=0.0,
                        help="Adjust Instagram timeline if it starts later or earlier")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    report = create_report(
        args.source, args.vertical, args.published, seconds=args.at,
        published_offset=args.published_offset, output_dir=args.output_dir,
        ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
    )
    print(report.resolve())


if __name__ == "__main__":
    main()
