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


class DiscontinuityCheck(Check):
    """A4. Single-frame teleport: a jump of more than half the joint's observed
    range in one frame -- the classic 12-bit encoder wraparound bug (wrist_roll
    wrapping at 2047/4096 counts). Named explicitly in the message so a report
    reads like someone who's felt this pain, not a generic anomaly flag."""

    id = "DISCONTINUITY"
    severity = "error"
    scope = "episode"

    RANGE_FRACTION_THRESHOLD = 0.5

    def run(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        n_joints = episode.states.shape[1]
        findings = []

        for joint_index in range(n_joints):
            values = episode.states[:, joint_index]
            joint_range = float(np.max(values) - np.min(values))
            if joint_range == 0.0:
                continue

            deltas = np.diff(values)
            jump_frames = np.nonzero(np.abs(deltas) > self.RANGE_FRACTION_THRESHOLD * joint_range)[0]
            if jump_frames.size == 0:
                continue

            worst_delta_idx = int(jump_frames[np.argmax(np.abs(deltas[jump_frames]))])
            worst_delta = float(deltas[worst_delta_idx])
            worst_frame = worst_delta_idx + 1  # the frame the bad value lands on
            findings.append(
                Finding(
                    check=self.id,
                    severity=self.severity,
                    episode=episode_index,
                    joint=str(joint_index),
                    frames=[worst_frame],
                    message=(
                        f"Single-frame jump of {abs(worst_delta):.3f} "
                        f"({abs(worst_delta) / joint_range:.0%} of observed range) on "
                        f"joint {joint_index} at frame {worst_frame} -- likely encoder wraparound"
                    ),
                    data={
                        "joint_index": joint_index,
                        "delta": worst_delta,
                        "range_fraction": abs(worst_delta) / joint_range,
                    },
                )
            )
        return findings


class FrozenStateCheck(Check):
    """A5. Consecutive identical state vectors for more than freeze_frames (source-
    spec default: 15 frames at 30fps = 0.5s) mid-episode -> recorder hiccup or
    serial dropout. Runs touching episode start/end are exempt -- settling there
    is normal, not a bug."""

    id = "FROZEN_STATE"
    severity = "error"
    scope = "episode"

    FREEZE_FRAMES_THRESHOLD = 15

    def run(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        n_frames = episode.states.shape[0]
        if n_frames < 2:
            return []

        same_as_previous = np.all(episode.states[1:] == episode.states[:-1], axis=1)

        findings = []
        run_start = None
        for i, is_same in enumerate(same_as_previous):
            frame = i + 1  # same_as_previous[i] compares frame i+1 to frame i
            if is_same:
                if run_start is None:
                    run_start = frame - 1
            elif run_start is not None:
                findings.extend(self._maybe_flag_run(run_start, frame - 1, n_frames, episode_index))
                run_start = None
        if run_start is not None:
            findings.extend(self._maybe_flag_run(run_start, n_frames - 1, n_frames, episode_index))
        return findings

    def _maybe_flag_run(self, start: int, end: int, n_frames: int, episode_index: int) -> list[Finding]:
        length = end - start + 1
        touches_start = start == 0
        touches_end = end == n_frames - 1
        if length <= self.FREEZE_FRAMES_THRESHOLD or touches_start or touches_end:
            return []
        return [
            Finding(
                check=self.id,
                severity=self.severity,
                episode=episode_index,
                joint=None,
                frames=list(range(start, end + 1)),
                message=(
                    f"State frozen (identical for {length} consecutive frames) "
                    f"from frame {start} to {end} -- likely a recorder hiccup or "
                    f"serial dropout"
                ),
                data={"freeze_length": length, "freeze_start": start, "freeze_end": end},
            )
        ]
