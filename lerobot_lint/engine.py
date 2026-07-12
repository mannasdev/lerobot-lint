"""Wires the loader, both check registries, and the two-pass accumulator
together into one call. The CLI and guard() both call into this -- neither
duplicates this wiring, which is what the guard()/CLI conformance test
depends on to be meaningful."""

import numpy as np

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
from lerobot_lint import units as units_mod
from lerobot_lint.config import Profile
from lerobot_lint.loader import iter_episodes
from lerobot_lint.types import EpisodeData, Finding, summarize_episode

EPISODE_LOAD_ERROR_CHECK_ID = "EPISODE_LOAD_ERROR"
NO_EPISODES_CHECK_ID = "NO_EPISODES"
UNITS_INFERRED_CHECK_ID = "UNITS_INFERRED"
UNITS_UNKNOWN_CHECK_ID = "UNITS_UNKNOWN"

# JITTER's threshold is an absolute rad/s value -- the only check that needs
# real angle units. Degrees are converted to radians before checks run, so
# both angle conventions can use it; normalized/unknown data cannot.
_JITTER_CAPABLE_UNITS = (units_mod.RADIANS, units_mod.DEGREES)


def build_episode_check_registry(profile: Profile, units: str = units_mod.RADIANS) -> CheckRegistry:
    registry = CheckRegistry()
    registry.register(DeadJointCheck())
    registry.register(SaturatedJointCheck())
    if units in _JITTER_CAPABLE_UNITS:
        registry.register(JitterCheck(max_joint_velocity=profile.max_joint_velocity))
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


def _resolve_units(episode: EpisodeData, requested: str) -> tuple[str, Finding | None]:
    """Explicit --units always wins, silently. Under auto, infer from the first
    episode's states and disclose the assumption as a finding IN the report --
    a pre-run echo line gets buried under download progress bars."""
    if requested != units_mod.AUTO:
        return requested, None

    decision = units_mod.infer_units(episode.states)

    if decision.units == units_mod.UNKNOWN:
        return decision.units, Finding(
            check=UNITS_UNKNOWN_CHECK_ID,
            severity="warning",
            episode=None,
            joint=None,
            frames=[],
            message=(
                f"Could not infer joint-state units ({decision.reason}); "
                f"velocity-threshold checks (JITTER) skipped -- pass --units to enable them"
            ),
            data={"reason": decision.reason},
        )

    consequence = {
        units_mod.DEGREES: "states converted to radians for kinematic checks",
        units_mod.RADIANS: "used as-is for kinematic checks",
        units_mod.NORMALIZED: (
            "velocity-threshold checks (JITTER) skipped: a rad/s limit is "
            "meaningless for normalized values"
        ),
    }[decision.units]
    return decision.units, Finding(
        check=UNITS_INFERRED_CHECK_ID,
        severity="info",
        episode=None,
        joint=None,
        frames=[],
        message=(
            f"Joint-state units inferred as {decision.units} ({decision.reason}); "
            f"{consequence}. Override with --units."
        ),
        data={"units": decision.units, "reason": decision.reason},
    )


def _in_radians(episode: EpisodeData, units: str) -> EpisodeData:
    if units != units_mod.DEGREES:
        return episode
    factor = np.float32(units_mod.DEGREES_TO_RADIANS)
    return EpisodeData(
        states=episode.states * factor,
        actions=episode.actions * factor,
        timestamps=episode.timestamps,
        fps=episode.fps,
        task=episode.task,
        camera_handles=episode.camera_handles,
    )


def check_dataset(
    repo_id_or_path: str,
    profile: Profile,
    episode_indices: list[int] | None = None,
    download_videos: bool = False,
    units: str = units_mod.AUTO,
) -> list[Finding]:
    """Pass 1: stream episodes, run episode-scoped checks, accumulate summaries.
    Pass 2: run dataset-scoped checks against the accumulated summaries. A
    per-episode load failure is reported as its own finding and does not stop
    the rest of the dataset from being checked.

    Units are resolved once, from the first successfully loaded episode -- a
    dataset's recording convention doesn't change mid-dataset, and single-pass
    streaming means we never see all episodes before the checks run."""
    dataset_registry = build_dataset_check_registry()

    findings: list[Finding] = []
    summaries = []
    episode_registry = None
    resolved_units = None

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

        if episode_registry is None:
            resolved_units, units_finding = _resolve_units(episode, units)
            if units_finding is not None:
                findings.append(units_finding)
            episode_registry = build_episode_check_registry(profile, units=resolved_units)

        episode = _in_radians(episode, resolved_units)
        findings.extend(episode_registry.run_all(episode, index))
        summaries.append(summarize_episode(episode, index))

    if not summaries and not any(f.check == EPISODE_LOAD_ERROR_CHECK_ID for f in findings):
        # zero episodes were even attempted (not "all attempts failed", which
        # already produces EPISODE_LOAD_ERROR findings) -- a dataset with
        # nothing to check is an error case, not a silent pass-through.
        findings.append(
            Finding(
                check=NO_EPISODES_CHECK_ID,
                severity="error",
                episode=None,
                joint=None,
                frames=[],
                message="Dataset has no episodes to check",
                data={},
            )
        )

    findings.extend(dataset_registry.run_all(summaries))
    return findings
