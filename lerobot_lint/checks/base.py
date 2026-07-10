from abc import ABC, abstractmethod

from lerobot_lint.types import EpisodeData, EpisodeSummary, Finding


class Check(ABC):
    """One check. Subclasses implement `run` for a single episode's data.
    Dataset-scoped checks (Group E) implement `run_dataset` instead — see
    CheckRegistry.run_dataset_checks."""

    id: str
    severity: str
    scope: str = "episode"

    @abstractmethod
    def run(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        raise NotImplementedError


class CheckRegistry:
    """Runs every registered check against an episode. A check that raises is
    isolated — it does not abort the run, and is itself reported as a finding
    (severity=error, check id suffixed `_CRASHED`) so one broken check never
    silently takes down the other checks."""

    def __init__(self) -> None:
        self._checks: list[Check] = []

    def register(self, check: Check) -> None:
        self._checks.append(check)

    def run_all(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        findings: list[Finding] = []
        for check in self._checks:
            try:
                findings.extend(check.run(episode, episode_index))
            except Exception as e:
                findings.append(
                    Finding(
                        check=f"{check.id}_CRASHED",
                        severity="error",
                        episode=episode_index,
                        joint=None,
                        frames=[],
                        message=f"Check {check.id} crashed: {e}",
                        data={"exception_type": type(e).__name__},
                    )
                )
        return findings


class DatasetCheck(ABC):
    """A Group E check: dataset-scoped, runs once against the accumulated
    per-episode summaries from pass 1 -- never against raw per-episode
    frame/state data, which pass 1 has already discarded."""

    id: str
    severity: str
    scope: str = "dataset"

    @abstractmethod
    def run_dataset(self, summaries: list[EpisodeSummary]) -> list[Finding]:
        raise NotImplementedError


class DatasetCheckRegistry:
    """Same per-check crash isolation as CheckRegistry, for dataset-scoped
    checks running against the pass-1 accumulator's output."""

    def __init__(self) -> None:
        self._checks: list[DatasetCheck] = []

    def register(self, check: DatasetCheck) -> None:
        self._checks.append(check)

    def run_all(self, summaries: list[EpisodeSummary]) -> list[Finding]:
        findings: list[Finding] = []
        for check in self._checks:
            try:
                findings.extend(check.run_dataset(summaries))
            except Exception as e:
                findings.append(
                    Finding(
                        check=f"{check.id}_CRASHED",
                        severity="error",
                        episode=None,
                        joint=None,
                        frames=[],
                        message=f"Dataset check {check.id} crashed: {e}",
                        data={"exception_type": type(e).__name__},
                    )
                )
        return findings
