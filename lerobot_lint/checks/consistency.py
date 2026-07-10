"""Group B — action/state consistency checks (per episode)."""

import numpy as np

from lerobot_lint.checks.base import Check
from lerobot_lint.types import EpisodeData, Finding


class ActionStateDivergenceCheck(Check):
    """B1. For position-controlled arms, action[t] (commanded) should be tracked
    by observation.state[t] (achieved). Per-joint correlation below track_corr
    (source-spec default: 0.8) means the follower isn't doing what the leader
    commands -- leader/follower calibration mismatch, or torque limits. This is
    the "I was cheating during data collection" class of bug."""

    id = "ACTION_STATE_DIVERGENCE"
    severity = "warning"
    scope = "episode"

    TRACK_CORR_THRESHOLD = 0.8

    def run(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        n_joints = episode.states.shape[1]
        findings = []

        for joint_index in range(n_joints):
            action = episode.actions[:, joint_index]
            state = episode.states[:, joint_index]

            if np.std(action) == 0.0 or np.std(state) == 0.0:
                continue  # constant signal -- correlation undefined, not this check's concern

            correlation = float(np.corrcoef(action, state)[0, 1])
            if correlation < self.TRACK_CORR_THRESHOLD:
                findings.append(
                    Finding(
                        check=self.id,
                        severity=self.severity,
                        episode=episode_index,
                        joint=str(joint_index),
                        frames=[],
                        message=(
                            f"Joint {joint_index}'s achieved state tracks its commanded "
                            f"action with correlation {correlation:.2f} (below "
                            f"{self.TRACK_CORR_THRESHOLD}) -- possible leader/follower "
                            f"calibration mismatch"
                        ),
                        data={"joint_index": joint_index, "correlation": correlation},
                    )
                )
        return findings
