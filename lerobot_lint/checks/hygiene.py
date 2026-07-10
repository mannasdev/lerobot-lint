"""Group E — dataset-level hygiene checks. Run once against pass 1's
accumulated EpisodeSummary list, never against raw per-episode data."""

from collections import Counter

import numpy as np

from lerobot_lint.checks.base import DatasetCheck
from lerobot_lint.types import EpisodeSummary, Finding


class ShortEpisodeCheck(DatasetCheck):
    """E1. Episode duration < 1.0s or < 15 frames -> accidental recording."""

    id = "SHORT_EPISODE"
    severity = "warning"
    scope = "dataset"

    MIN_DURATION_SECONDS = 1.0
    MIN_FRAMES = 15

    def run_dataset(self, summaries: list[EpisodeSummary]) -> list[Finding]:
        findings = []
        for summary in summaries:
            if summary.duration < self.MIN_DURATION_SECONDS or summary.frame_count < self.MIN_FRAMES:
                findings.append(
                    Finding(
                        check=self.id,
                        severity=self.severity,
                        episode=summary.episode_index,
                        joint=None,
                        frames=[],
                        message=(
                            f"Episode {summary.episode_index} is only "
                            f"{summary.duration:.2f}s ({summary.frame_count} frames) -- "
                            f"likely an accidental recording"
                        ),
                        data={"duration": summary.duration, "frame_count": summary.frame_count},
                    )
                )
        return findings


class DurationOutlierCheck(DatasetCheck):
    """E2. Episode duration z-score > 3 vs the dataset's own distribution."""

    id = "DURATION_OUTLIER"
    severity = "info"
    scope = "dataset"

    Z_SCORE_THRESHOLD = 3.0

    def run_dataset(self, summaries: list[EpisodeSummary]) -> list[Finding]:
        if len(summaries) < 2:
            return []

        durations = np.array([s.duration for s in summaries])
        mean, std = float(np.mean(durations)), float(np.std(durations))
        if std == 0.0:
            return []

        findings = []
        for summary, duration in zip(summaries, durations):
            z_score = (duration - mean) / std
            if abs(z_score) > self.Z_SCORE_THRESHOLD:
                findings.append(
                    Finding(
                        check=self.id,
                        severity=self.severity,
                        episode=summary.episode_index,
                        joint=None,
                        frames=[],
                        message=(
                            f"Episode {summary.episode_index}'s duration ({duration:.2f}s) "
                            f"is a z-score of {z_score:.1f} vs the dataset mean "
                            f"({mean:.2f}s)"
                        ),
                        data={"duration": float(duration), "z_score": z_score, "dataset_mean": mean},
                    )
                )
        return findings
