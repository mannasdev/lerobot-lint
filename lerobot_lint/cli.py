"""The `lelint` command. Wires engine.check_dataset() to the console report
and the 0/1/2 exit-code contract (matches trajlens's convention, so tooling
written against one behaves the same against the other)."""

import typer

from lerobot_lint.config import load_profile
from lerobot_lint.engine import check_dataset
from lerobot_lint.loader import DatasetLoadError
from lerobot_lint.report.console import render_console_report
from lerobot_lint.types import Finding

app = typer.Typer()


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
        raise typer.Exit(code=2) from e

    report = render_console_report(findings, repo_id_or_path)
    typer.echo(report)
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
