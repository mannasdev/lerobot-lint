"""Wires the loader, both check registries, and the two-pass accumulator
together into one call. The CLI and guard() both call into this -- neither
duplicates this wiring, which is what the guard()/CLI conformance test
depends on to be meaningful."""

from lerobot_lint.checks.base import CheckRegistry, DatasetCheckRegistry
from lerobot_lint.checks.consistency import ActionRangeMismatchCheck, ActionStateDivergenceCheck
from lerobot_lint.checks.hygiene import (
    DurationOutlierCheck,
    LowDiversityCheck,
    MissingTaskCheck,
    ShortEpisodeCheck,
    TaskImbalanceCheck,
    TooFewEpisodesCheck,
)
from lerobot_lint.checks.kinematic import (
    DeadJointCheck,
    DiscontinuityCheck,
    FrozenStateCheck,
    GripperInertCheck,
    JitterCheck,
    SaturatedJointCheck,
)
from lerobot_lint.checks.timing import FpsMismatchCheck, FrameDropsCheck, TimestampNonMonotonicCheck
from lerobot_lint.checks.video import DegenerateImageCheck, FrozenCameraCheck, VideoLengthMismatchCheck
from lerobot_lint.config import Profile
from lerobot_lint.loader import iter_episodes
from lerobot_lint.types import Finding, summarize_episode

EPISODE_LOAD_ERROR_CHECK_ID = "EPISODE_LOAD_ERROR"


def build_episode_check_registry(profile: Profile) -> CheckRegistry:
    registry = CheckRegistry()
    registry.register(DeadJointCheck())
    registry.register(SaturatedJointCheck())
    registry.register(JitterCheck())
    registry.register(DiscontinuityCheck())
    registry.register(FrozenStateCheck())
    if profile.gripper_index is not None:
        registry.register(GripperInertCheck(gripper_index=profile.gripper_index))
    registry.register(ActionStateDivergenceCheck())
    registry.register(ActionRangeMismatchCheck())
    registry.register(TimestampNonMonotonicCheck())
    registry.register(FrameDropsCheck())
    registry.register(FpsMismatchCheck())
    registry.register(VideoLengthMismatchCheck())
    registry.register(FrozenCameraCheck())
    registry.register(DegenerateImageCheck())
    return registry


def build_dataset_check_registry() -> DatasetCheckRegistry:
    registry = DatasetCheckRegistry()
    registry.register(ShortEpisodeCheck())
    registry.register(DurationOutlierCheck())
    registry.register(MissingTaskCheck())
    registry.register(TaskImbalanceCheck())
    registry.register(LowDiversityCheck())
    registry.register(TooFewEpisodesCheck())
    return registry


def check_dataset(
    repo_id_or_path: str,
    profile: Profile,
    episode_indices: list[int] | None = None,
    download_videos: bool = False,
) -> list[Finding]:
    """Pass 1: stream episodes, run episode-scoped checks, accumulate summaries.
    Pass 2: run dataset-scoped checks against the accumulated summaries. A
    per-episode load failure is reported as its own finding and does not stop
    the rest of the dataset from being checked."""
    episode_registry = build_episode_check_registry(profile)
    dataset_registry = build_dataset_check_registry()

    findings: list[Finding] = []
    summaries = []

    for index, episode, error in iter_episodes(repo_id_or_path, episode_indices, download_videos):
        if error is not None:
            findings.append(
                Finding(
                    check=EPISODE_LOAD_ERROR_CHECK_ID,
                    severity="error",
                    episode=index,
                    joint=None,
                    frames=[],
                    message=error,
                    data={},
                )
            )
            continue

        findings.extend(episode_registry.run_all(episode, index))
        summaries.append(summarize_episode(episode, index))

    findings.extend(dataset_registry.run_all(summaries))
    return findings
