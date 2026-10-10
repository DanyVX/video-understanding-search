"""Local diagnostic CLI; ingest and search arrive in later milestones."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from vsearch import __version__
from vsearch.environment import collect_environment
from vsearch.index.manifest import Manifest
from vsearch.ingest.pipeline import IngestOptions, delete_video, ingest
from vsearch.settings import load_settings

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Timestamped video-moment retrieval. M0 includes environment diagnostics.",
)


@app.command()
def version() -> None:
    """Print this package's current prerelease version."""
    typer.echo(__version__)


@app.command()
def doctor(
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable report")] = False,
    check_network: Annotated[
        bool, typer.Option("--check-network", help="Opt in to a PyPI reachability probe")
    ] = False,
) -> None:
    """Inspect tools, hardware, dependency versions, and git revision."""
    try:
        load_settings()
    except ValidationError:
        typer.echo("Invalid VSEARCH_ configuration. Review .env.example.", err=True)
        raise typer.Exit(code=2) from None

    report = collect_environment(check_network=check_network)
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key, value in report.items():
            if key == "library_versions":
                typer.echo("library_versions:")
                for package, version_value in value.items():
                    typer.echo(f"  {package}: {version_value}")
            else:
                typer.echo(f"{key}: {value}")


@app.command("ingest")
def ingest_command(
    path: Path,
    fps: Annotated[float, typer.Option(help="Uniform samples per second")] = 0.5,
    keep_source: Annotated[bool, typer.Option(help="Allow local source path persistence")] = False,
) -> None:
    """Ingest short media locally with resumable, bounded ffmpeg extraction."""
    from vsearch.ingest.probe import MediaError

    try:
        settings = load_settings()
        result = ingest(
            path, Manifest(settings.data_dir), IngestOptions(fps=fps, keep_source=keep_source)
        )
        typer.echo(json.dumps(result))
    except (ValidationError, MediaError, OSError, ValueError) as exc:
        typer.echo(f"Ingest failed: {type(exc).__name__}", err=True)
        raise typer.Exit(code=2) from None


@app.command("videos")
def videos_command() -> None:
    """List ingested video metadata."""
    settings = load_settings()
    typer.echo(json.dumps(Manifest(settings.data_dir).list_videos()))


@app.command("verify")
def verify_command() -> None:
    """Check thumbnail references against persisted manifest."""
    errors = Manifest(load_settings().data_dir).verify()
    typer.echo(json.dumps({"consistent": not errors, "errors": errors}))
    if errors:
        raise typer.Exit(code=1)


@app.command("repair")
def repair_command() -> None:
    """Mark missing-thumbnail videos failed; safe reingest repairs them."""
    errors = Manifest(load_settings().data_dir).repair()
    typer.echo(json.dumps({"marked_failed": errors}))


@app.command("delete")
def delete_command(video_id: str) -> None:
    """Erase derived thumbnails and manifest rows for one content hash."""
    existed = delete_video(Manifest(load_settings().data_dir), video_id)
    typer.echo(json.dumps({"deleted": existed}))
    if not existed:
        raise typer.Exit(code=1)


@app.command("index-visual")
def index_visual_command(video_id: str, batch_size: int = 8) -> None:
    """Opt in to loading OpenCLIP weights to index one video's thumbnails."""
    from vsearch.embed.visual import OpenClipEncoder, index_frames
    from vsearch.index.vectors import SQLiteVectorStore

    config = load_settings()
    db = Manifest(config.data_dir)
    video = db.get_video(video_id)
    if video is None or video["status"] != "done":
        typer.echo("Video must exist and finish ingest first", err=True)
        raise typer.Exit(code=2)
    try:
        model = OpenClipEncoder()
        indexed = index_frames(
            db, SQLiteVectorStore(config.data_dir), model, video_id, batch_size=batch_size
        )
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        typer.echo(f"Visual indexing failed: {type(exc).__name__}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps({"indexed_frames": indexed, "model_key": model.model_key}))


@app.command("search")
def search_command(
    query: str,
    limit: Annotated[int, typer.Option(min=1, max=50)] = 10,
    visual: bool = False,
) -> None:
    """Search indexed speech, optionally adding true CLIP image/text retrieval."""
    from dataclasses import asdict

    from vsearch.retrieve.service import search_moments

    try:
        result = search_moments(load_settings().data_dir, query, limit=limit, include_visual=visual)
    except (ValueError, ImportError, RuntimeError) as exc:
        typer.echo(f"Search failed: {type(exc).__name__}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps([asdict(m) for m in result]))


@app.command("transcribe")
def transcribe_command(video_id: str, model_name: str = "tiny") -> None:
    """Run optional local Whisper model on an explicitly retained source video."""
    from vsearch.asr.transcript import WhisperTranscriber, windows
    from vsearch.index.transcripts import TranscriptIndex

    config = load_settings()
    record = Manifest(config.data_dir).get_video(video_id)
    if not record or not record["source_path"]:
        typer.echo("Source not retained. Reingest with --keep-source.", err=True)
        raise typer.Exit(code=2)
    try:
        segments = windows(
            WhisperTranscriber(model_name).transcribe(Path(str(record["source_path"])))
        )
        TranscriptIndex(config.data_dir).replace(video_id, segments)
    except (RuntimeError, ValueError, OSError) as exc:
        typer.echo(f"Transcription failed: {type(exc).__name__}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps({"segments": len(segments)}))


@app.command("serve")
def serve_command(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the local service. Public hosting requires authentication first."""
    import uvicorn

    if host not in ("127.0.0.1", "::1", "localhost"):
        typer.echo("Refusing non-loopback binding without auth", err=True)
        raise typer.Exit(code=2)
    uvicorn.run("vsearch.api.app:app", host=host, port=port)
