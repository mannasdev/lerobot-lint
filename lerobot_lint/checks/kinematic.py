"""Group A — kinematic signal checks (per joint, per episode). The core value."""

import numpy as np

from lerobot_lint.checks.base import Check
from lerobot_lint.types import EpisodeData, Finding


class DeadJointCheck(Check):
    """A1. A joint that never moves while others do -> disconnected/stuck motor.

    Episode-scoped heuristic (source-spec threshold, dataset-wide range/episode-%
    escalation deferred until Group E's cross-episode aggregation exists): a joint
    is a dead-joint candidate if its std is under 1% of the largest std among the
    episode's other joints. All-joints-static episodes (e.g. a settling frame) are
    exempt -- there must be at least one other joint that DID move.
    """

    id = "DEAD_JOINT"
    severity = "error"
    scope = "episode"

    RELATIVE_STD_THRESHOLD = 0.01

    def run(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        stds = np.std(episode.states, axis=0)
        max_std = float(np.max(stds))

        if max_std == 0.0:
            return []  # every joint static -- nothing else moved, not a dead joint

        findings = []
        for joint_index, std in enumerate(stds):
            if std < self.RELATIVE_STD_THRESHOLD * max_std:
                findings.append(
                    Finding(
                        check=self.id,
                        severity=self.severity,
                        episode=episode_index,
                        joint=str(joint_index),
                        frames=[],
                        message=(
                            f"Joint {joint_index} never moved (std={std:.6f}) while "
                            f"another joint had std={max_std:.6f}"
                        ),
                        data={"joint_index": joint_index, "std": float(std), "max_std": max_std},
                    )
                )
        return findings


class SaturatedJointCheck(Check):
    """A2. Joint pinned at its episode-observed min/max for >30% of frames ->
    hitting a mechanical limit or calibration offset."""

    id = "SATURATED_JOINT"
    severity = "warning"
    scope = "episode"

    FRACTION_THRESHOLD = 0.30
    PIN_EPSILON = 1e-6

    def run(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        n_frames, n_joints = episode.states.shape
        findings = []

        for joint_index in range(n_joints):
            values = episode.states[:, joint_index]
            joint_min, joint_max = float(np.min(values)), float(np.max(values))
            if joint_max - joint_min < self.PIN_EPSILON:
                continue  # joint never moved at all -- DEAD_JOINT's territory, not this check

            at_min = np.isclose(values, joint_min, atol=self.PIN_EPSILON)
            at_max = np.isclose(values, joint_max, atol=self.PIN_EPSILON)
            pinned_fraction = float(np.count_nonzero(at_min | at_max) / n_frames)

            if pinned_fraction > self.FRACTION_THRESHOLD:
                findings.append(
                    Finding(
                        check=self.id,
                        severity=self.severity,
                        episode=episode_index,
                        joint=str(joint_index),
                        frames=[],
                        message=(
                            f"Joint {joint_index} pinned at its min/max for "
                            f"{pinned_fraction:.0%} of frames"
                        ),
                        data={"joint_index": joint_index, "pinned_fraction": pinned_fraction},
                    )
                )
        return findings


class JitterCheck(Check):
    """A3. Physically impossible velocity spikes -> encoder noise.

    Per-frame velocity |delta q| * fps exceeding max_joint_velocity (source-spec
    default: 8 rad/s for SO-101-class hobby servos) on isolated frames. Escalates
    to error if the spike frames exceed 2% of the episode's frames.
    """

    id = "JITTER"
    severity = "warning"
    scope = "episode"

    MAX_JOINT_VELOCITY = 8.0  # rad/s, source-spec SO-101-class default
    ERROR_FRACTION_THRESHOLD = 0.02

    def run(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        n_frames, n_joints = episode.states.shape
        findings = []

        for joint_index in range(n_joints):
            values = episode.states[:, joint_index]
            velocity = np.abs(np.diff(values)) * episode.fps
            spike_mask = velocity > self.MAX_JOINT_VELOCITY

            if not np.any(spike_mask):
                continue

            # both endpoints of a one-frame spike show up as high velocity
            # (the jump in, and the jump back out) -- count distinct frames touched.
            spike_deltas = np.nonzero(spike_mask)[0]
            spike_frames = sorted(set(spike_deltas.tolist()) | set((spike_deltas + 1).tolist()))
            spike_fraction = len(spike_frames) / n_frames
            worst_idx = int(np.argmax(velocity))

            severity = "error" if spike_fraction > self.ERROR_FRACTION_THRESHOLD else self.severity
            findings.append(
                Finding(
                    check=self.id,
                    severity=severity,
                    episode=episode_index,
                    joint=str(joint_index),
                    frames=spike_frames,
                    message=(
                        f"Joint {joint_index} has {len(spike_frames)} frame(s) with velocity "
                        f"exceeding {self.MAX_JOINT_VELOCITY} rad/s (worst: "
                        f"{float(velocity[worst_idx]):.1f} rad/s at frame {worst_idx})"
                    ),
                    data={
                        "joint_index": joint_index,
                        "spike_frame_count": len(spike_frames),
                        "spike_fraction": spike_fraction,
                        "worst_velocity": float(velocity[worst_idx]),
                        "worst_frame": worst_idx,
                    },
                )
            )
        return findings
