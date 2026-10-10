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
        result = ingest(path, Manifest(settings.data_dir), IngestOptions(fps=fps, keep_source=keep_source))
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
