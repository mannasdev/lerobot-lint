"""lerobot_lint.guard() -- a context-manager pre-flight gate. Calls into the
exact same engine.check_dataset() the CLI calls, so guard() and the CLI
cannot silently drift out of agreement on the same dataset (verified by a
conformance test)."""

from lerobot_lint.config import load_profile
from lerobot_lint.engine import check_dataset
from lerobot_lint.types import Finding

CRASHED_CHECK_SUFFIX = "_CRASHED"


class LelintCheckFailedError(Exception):
    """Raised when the dataset has a real severity=error finding -- your data
    has a problem lelint caught."""


class LelintCheckCrashedError(Exception):
    """Raised when a check itself crashed while running -- lelint broke on
    your data, not the other way around. Deliberately a different exception
    type than LelintCheckFailedError so calling code can tell these apart."""


def _has_crashed_finding(findings: list[Finding]) -> bool:
    return any(f.check.endswith(CRASHED_CHECK_SUFFIX) for f in findings)


def _has_error_finding(findings: list[Finding]) -> bool:
    return any(f.severity == "error" for f in findings)


class guard:
    """Usage: `with lerobot_lint.guard(repo_id_or_path): train(...)`. Raises
    before `train(...)` ever runs if the dataset has severity=error findings.
    Never raises on warnings/info alone. Never swallows train()'s own
    exceptions -- __exit__ always returns falsy."""

    def __init__(
        self,
        repo_id_or_path: str,
        profile_name: str = "default",
        episode_indices: list[int] | None = None,
        download_videos: bool = False,
        units: str = "auto",
    ) -> None:
        self.repo_id_or_path = repo_id_or_path
        self.profile_name = profile_name
        self.episode_indices = episode_indices
        self.download_videos = download_videos
        self.units = units

    def __enter__(self) -> "guard":
        profile = load_profile(self.profile_name)
        findings = check_dataset(
            self.repo_id_or_path,
            profile,
            episode_indices=self.episode_indices,
            download_videos=self.download_videos,
            units=self.units,
        )

        if _has_crashed_finding(findings):
            crashed = [f for f in findings if f.check.endswith(CRASHED_CHECK_SUFFIX)]
            raise LelintCheckCrashedError(
                f"{len(crashed)} check(s) crashed while linting {self.repo_id_or_path!r}: "
                f"{[f.check for f in crashed]}"
            )

        if _has_error_finding(findings):
            errors = [f for f in findings if f.severity == "error"]
            raise LelintCheckFailedError(
                f"{len(errors)} error(s) found in {self.repo_id_or_path!r}: "
                f"{[f.check for f in errors]}"
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False  # never swallow the wrapped block's own exception
