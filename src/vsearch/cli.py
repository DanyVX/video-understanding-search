"""Local diagnostic CLI; ingest and search arrive in later milestones."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from pydantic import ValidationError

from vsearch import __version__
from vsearch.environment import collect_environment
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
