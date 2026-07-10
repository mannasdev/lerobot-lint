import numpy as np
import pytest

from lerobot_lint.checks.base import Check, CheckRegistry, DatasetCheck, DatasetCheckRegistry
from lerobot_lint.types import EpisodeData, EpisodeSummary, Finding


def _clean_episode(n_frames=20, n_joints=3):
    return EpisodeData(
        states=np.random.default_rng(0).normal(size=(n_frames, n_joints)).astype(np.float32),
        actions=np.random.default_rng(1).normal(size=(n_frames, n_joints)).astype(np.float32),
        timestamps=np.arange(n_frames, dtype=np.float64) / 30.0,
        fps=30.0,
        task="pick up the block",
        camera_handles={},
    )


class _AlwaysFiresCheck(Check):
    id = "ALWAYS_FIRES"
    severity = "warning"
    scope = "episode"

    def run(self, episode, episode_index):
        return [
            Finding(
                check=self.id,
                severity=self.severity,
                episode=episode_index,
                joint=None,
                frames=[0],
                message="fired on purpose",
                data={},
            )
        ]


class _CrashingCheck(Check):
    id = "CRASHES"
    severity = "error"
    scope = "episode"

    def run(self, episode, episode_index):
        raise KeyError("simulated malformed data")


def test_registry_runs_all_registered_checks():
    registry = CheckRegistry()
    registry.register(_AlwaysFiresCheck())

    findings = registry.run_all(_clean_episode(), episode_index=0)

    assert len(findings) == 1
    assert findings[0].check == "ALWAYS_FIRES"


def test_registry_isolates_a_crashing_check_and_reports_it_as_a_finding():
    registry = CheckRegistry()
    registry.register(_CrashingCheck())
    registry.register(_AlwaysFiresCheck())

    findings = registry.run_all(_clean_episode(), episode_index=0)

    # the crash must not prevent the other check from running
    assert any(f.check == "ALWAYS_FIRES" for f in findings)
    # the crash itself must be reported, not swallowed or raised
    crashed = [f for f in findings if f.check == "CRASHES_CRASHED"]
    assert len(crashed) == 1
    assert crashed[0].severity == "error"
    assert "KeyError" in crashed[0].data.get("exception_type", "")


def _summary(episode_index=0, frame_count=30, duration=1.0, task="pick up the block"):
    return EpisodeSummary(
        episode_index=episode_index,
        frame_count=frame_count,
        duration=duration,
        task=task,
        joint_means=np.zeros(3),
        joint_mins=np.zeros(3),
        joint_maxs=np.zeros(3),
    )


class _AlwaysFiresDatasetCheck(DatasetCheck):
    id = "ALWAYS_FIRES_DATASET"
    severity = "info"
    scope = "dataset"

    def run_dataset(self, summaries):
        return [
            Finding(
                check=self.id,
                severity=self.severity,
                episode=None,
                joint=None,
                frames=[],
                message=f"saw {len(summaries)} episodes",
                data={"n": len(summaries)},
            )
        ]


class _CrashingDatasetCheck(DatasetCheck):
    id = "CRASHES_DATASET"
    severity = "error"
    scope = "dataset"

    def run_dataset(self, summaries):
        raise ValueError("simulated aggregation bug")


def test_dataset_registry_runs_all_registered_checks():
    registry = DatasetCheckRegistry()
    registry.register(_AlwaysFiresDatasetCheck())

    findings = registry.run_all([_summary(), _summary(episode_index=1)])

    assert len(findings) == 1
    assert findings[0].data["n"] == 2


def test_dataset_registry_isolates_a_crashing_check():
    registry = DatasetCheckRegistry()
    registry.register(_CrashingDatasetCheck())
    registry.register(_AlwaysFiresDatasetCheck())

    findings = registry.run_all([_summary()])

    assert any(f.check == "ALWAYS_FIRES_DATASET" for f in findings)
    crashed = [f for f in findings if f.check == "CRASHES_DATASET_CRASHED"]
    assert len(crashed) == 1
    assert crashed[0].severity == "error"
