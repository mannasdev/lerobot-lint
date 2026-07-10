import numpy as np
import pytest

from lerobot_lint.checks.base import Check, CheckRegistry
from lerobot_lint.types import EpisodeData, Finding


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
