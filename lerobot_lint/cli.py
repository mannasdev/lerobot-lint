"""The `lelint` command. Wires engine.check_dataset() to the console report
and the 0/1/2 exit-code contract (matches trajlens's convention, so tooling
written against one behaves the same against the other)."""

import enum
import json
import platform

import typer

from lerobot_lint.config import detect_profile_name, list_profile_names, load_profile
from lerobot_lint.engine import check_dataset
from lerobot_lint.loader import DatasetLoadError, get_joint_names
from lerobot_lint.report.console import render_console_report
from lerobot_lint.report.json_report import render_json_report
from lerobot_lint.types import Finding

app = typer.Typer()


class FailOnLevel(str, enum.Enum):
    info = "info"
    warning = "warning"
    error = "error"


class UnitsChoice(str, enum.Enum):
    auto = "auto"
    radians = "radians"
    degrees = "degrees"
    normalized = "normalized"


def _debug_context(repo_id_or_path: str) -> str:
    from importlib.metadata import version as pkg_version

    return (
        f"lerobot-lint {pkg_version('lerobot-lint')} | "
        f"python {platform.python_version()} | dataset: {repo_id_or_path}"
    )


_SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2}


def determine_exit_code(findings: list[Finding], fail_on: str = "error") -> int:
    """0 = clean, 1 = findings present but below the fail_on threshold,
    2 = a finding at or above the fail_on threshold (default: error --
    including a load failure, reported as an EPISODE_LOAD_ERROR finding
    with severity=error, which has no separate exit code of its own)."""
    threshold = _SEVERITY_RANK[fail_on]
    if any(_SEVERITY_RANK[f.severity] >= threshold for f in findings):
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


def _resolve_profile(repo_id_or_path: str, profile: str | None) -> tuple[str, str]:
    """Returns (profile_name, disclosure_message). --profile always wins; with
    no --profile, tries to auto-detect from the dataset's joint names and
    falls back to 'default' if nothing matches. The disclosure must always be
    shown -- a confidently-wrong auto-match silently applying the wrong joint
    tolerances would produce false positives/negatives on exactly the bugs
    this tool exists to catch."""
    if profile is not None:
        return profile, f"Using profile: {profile} (explicit, via --profile)"

    detected = detect_profile_name(get_joint_names(repo_id_or_path))
    if detected is not None:
        return detected, f"Using profile: {detected} (auto-detected from joint names, override with --profile)"
    return "default", "Using profile: default (no match, override with --profile)"


@app.command()
def check(
    repo_id_or_path: str,
    profile: str | None = typer.Option(
        None, "--profile", help="Robot profile: default, so101, koch (default: auto-detect from joint names)"
    ),
    episodes: str | None = typer.Option(None, "--episodes", help="Episode range, e.g. 0:50"),
    no_video: bool = typer.Option(False, "--no-video", help="Skip camera/video checks entirely"),
    units: UnitsChoice = typer.Option(
        UnitsChoice.auto,
        "--units",
        help="Joint-state units: radians, degrees, or normalized (default: infer from the data)",
    ),
    json_out: str | None = typer.Option(None, "--json", help="Also write a JSON report to this path"),
    verbose: bool = typer.Option(False, "--verbose", help="Print debug context on error/crash"),
    fail_on: FailOnLevel = typer.Option(
        FailOnLevel.error,
        "--fail-on",
        help="Minimum severity that causes a nonzero exit: info, warning, or error",
    ),
) -> None:
    """Check a LeRobotDataset for behavioral/kinematic data bugs."""
    profile_name, disclosure = _resolve_profile(repo_id_or_path, profile)
    typer.echo(disclosure)
    episode_indices = _parse_episode_range(episodes)

    try:
        loaded_profile = load_profile(profile_name)
        findings = check_dataset(
            repo_id_or_path,
            loaded_profile,
            episode_indices=episode_indices,
            download_videos=not no_video,
            units=units.value,
        )
    except ValueError as e:
        typer.echo(f"{e}", err=True)
        if verbose:
            typer.echo(_debug_context(repo_id_or_path), err=True)
        raise typer.Exit(code=2) from e
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

    report = render_console_report(findings, repo_id_or_path, profile_disclosure=disclosure)
    typer.echo(report)

    if json_out is not None:
        json_report = render_json_report(findings, repo_id_or_path, profile_name)
        with open(json_out, "w") as f:
            json.dump(json_report, f, indent=2)

    raise typer.Exit(code=determine_exit_code(findings, fail_on=fail_on.value))


@app.command()
def profiles() -> None:
    """List built-in robot profiles."""
    for name in list_profile_names():
        typer.echo(name)


@app.command()
def version() -> None:
    """Print the lerobot-lint version."""
    from importlib.metadata import version as pkg_version

    typer.echo(pkg_version("lerobot-lint"))


if __name__ == "__main__":
    app()
