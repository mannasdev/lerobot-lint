"""The `lelint` command. Wires engine.check_dataset() to the console report
and the 0/1/2 exit-code contract (matches trajlens's convention, so tooling
written against one behaves the same against the other)."""

import json
import platform

import typer

from lerobot_lint.config import load_profile
from lerobot_lint.engine import check_dataset
from lerobot_lint.loader import DatasetLoadError
from lerobot_lint.report.console import render_console_report
from lerobot_lint.report.json_report import render_json_report
from lerobot_lint.types import Finding

app = typer.Typer()


def _debug_context(repo_id_or_path: str) -> str:
    from importlib.metadata import version as pkg_version

    return (
        f"lerobot-lint {pkg_version('lerobot-lint')} | "
        f"python {platform.python_version()} | dataset: {repo_id_or_path}"
    )


def determine_exit_code(findings: list[Finding]) -> int:
    """0 = clean, 1 = warnings/info only, 2 = any error (including a load
    failure, which is reported as an EPISODE_LOAD_ERROR finding with
    severity=error -- there is no separate exit code for it)."""
    if any(f.severity == "error" for f in findings):
        return 2
    if findings:
        return 1
    return 0


def _parse_episode_range(episodes: str | None) -> list[int] | None:
    if episodes is None:
        return None
    start_str, _, end_str = episodes.partition(":")
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else start + 1
    return list(range(start, end))


@app.command()
def check(
    repo_id_or_path: str,
    profile: str = typer.Option("default", help="Robot profile: default, so101, koch"),
    episodes: str | None = typer.Option(None, "--episodes", help="Episode range, e.g. 0:50"),
    no_video: bool = typer.Option(False, "--no-video", help="Skip camera/video checks entirely"),
    json_out: str | None = typer.Option(None, "--json", help="Also write a JSON report to this path"),
    verbose: bool = typer.Option(False, "--verbose", help="Print debug context on error/crash"),
) -> None:
    """Check a LeRobotDataset for behavioral/kinematic data bugs."""
    loaded_profile = load_profile(profile)
    episode_indices = _parse_episode_range(episodes)

    try:
        findings = check_dataset(
            repo_id_or_path,
            loaded_profile,
            episode_indices=episode_indices,
            download_videos=not no_video,
        )
    except DatasetLoadError as e:
        typer.echo(f"Could not load dataset: {e}", err=True)
        if verbose:
            typer.echo(_debug_context(repo_id_or_path), err=True)
            typer.echo(f"Full error: {e!r}", err=True)
        raise typer.Exit(code=2) from e
    except Exception as e:
        # never let an unexpected crash surface as a raw traceback to a stranger
        typer.echo(f"lelint crashed unexpectedly: {e}", err=True)
        if verbose:
            typer.echo(_debug_context(repo_id_or_path), err=True)
            import traceback

            typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(code=2) from e

    report = render_console_report(findings, repo_id_or_path)
    typer.echo(report)

    if json_out is not None:
        json_report = render_json_report(findings, repo_id_or_path, profile)
        with open(json_out, "w") as f:
            json.dump(json_report, f, indent=2)

    raise typer.Exit(code=determine_exit_code(findings))


@app.command()
def profiles() -> None:
    """List built-in robot profiles."""
    for name in ["default", "so101", "koch"]:
        typer.echo(name)


@app.command()
def version() -> None:
    """Print the lerobot-lint version."""
    from importlib.metadata import version as pkg_version

    typer.echo(pkg_version("lerobot-lint"))


if __name__ == "__main__":
    app()
