"""Group C — timing integrity checks (per episode)."""

import numpy as np

from lerobot_lint.checks.base import Check
from lerobot_lint.types import EpisodeData, Finding


class TimestampNonMonotonicCheck(Check):
    """C1. Timestamps go backwards or duplicate."""

    id = "TIMESTAMP_NON_MONOTONIC"
    severity = "error"
    scope = "episode"

    def run(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        deltas = np.diff(episode.timestamps)
        bad = np.nonzero(deltas <= 0)[0]
        if bad.size == 0:
            return []

        # report the frame the bad delta lands on (delta[i] compares frame i+1 to i)
        bad_frames = sorted((bad + 1).tolist())
        return [
            Finding(
                check=self.id,
                severity=self.severity,
                episode=episode_index,
                joint=None,
                frames=bad_frames,
                message=(
                    f"Timestamps go backwards or duplicate at frame(s) {bad_frames}"
                ),
                data={"bad_frame_count": len(bad_frames)},
            )
        ]


class FrameDropsCheck(Check):
    """C2. Gaps between consecutive timestamps larger than gap_factor (source-spec
    default: 1.5x) the declared frame period -> a dropped/skipped frame. Escalates
    from warning to error if gaps affect more than 5% of the episode's frames."""

    id = "FRAME_DROPS"
    severity = "warning"
    scope = "episode"

    GAP_FACTOR = 1.5
    ERROR_FRACTION_THRESHOLD = 0.05

    def run(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        frame_period = 1.0 / episode.fps
        gap_threshold = self.GAP_FACTOR * frame_period

        deltas = np.diff(episode.timestamps)
        gap_mask = deltas > gap_threshold
        gap_count = int(np.count_nonzero(gap_mask))
        if gap_count == 0:
            return []

        gap_fraction = gap_count / len(episode.timestamps)
        largest_gap = float(np.max(deltas[gap_mask]))
        severity = "error" if gap_fraction > self.ERROR_FRACTION_THRESHOLD else self.severity

        return [
            Finding(
                check=self.id,
                severity=severity,
                episode=episode_index,
                joint=None,
                frames=(np.nonzero(gap_mask)[0] + 1).tolist(),
                message=(
                    f"{gap_count} frame gap(s) exceeding {self.GAP_FACTOR}x the "
                    f"declared frame period (largest: {largest_gap:.4f}s vs expected "
                    f"{frame_period:.4f}s)"
                ),
                data={"gap_count": gap_count, "gap_fraction": gap_fraction, "largest_gap": largest_gap},
            )
        ]


class FpsMismatchCheck(Check):
    """C3. Median measured delta-t deviates from the declared fps by more than
    10% (source-spec default) -> the dataset lies about its rate. Temporal
    models (ACT-style chunking) silently break on this."""

    id = "FPS_MISMATCH"
    severity = "warning"
    scope = "episode"

    DEVIATION_THRESHOLD = 0.10

    def run(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        declared_dt = 1.0 / episode.fps
        measured_dt = float(np.median(np.diff(episode.timestamps)))
        deviation = abs(measured_dt - declared_dt) / declared_dt

        if deviation <= self.DEVIATION_THRESHOLD:
            return []

        measured_fps = 1.0 / measured_dt if measured_dt > 0 else float("inf")
        return [
            Finding(
                check=self.id,
                severity=self.severity,
                episode=episode_index,
                joint=None,
                frames=[],
                message=(
                    f"Declared fps ({episode.fps:.1f}) deviates {deviation:.0%} from "
                    f"the median measured rate (~{measured_fps:.1f} fps)"
                ),
                data={"declared_fps": episode.fps, "measured_fps": measured_fps, "deviation": deviation},
            )
        ]
