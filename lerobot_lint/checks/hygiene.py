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


class MissingTaskCheck(DatasetCheck):
    """E3. Empty/placeholder task strings ("test", "asdf", "") -- the
    junk-upload signature."""

    id = "MISSING_TASK"
    severity = "warning"
    scope = "dataset"

    PLACEHOLDER_STRINGS = {"test", "asdf", "todo", "tbd", "n/a", "na", "xxx", "placeholder"}
    MIN_MEANINGFUL_LENGTH = 3

    def _is_placeholder(self, task: str) -> bool:
        stripped = task.strip()
        if len(stripped) < self.MIN_MEANINGFUL_LENGTH:
            return True
        return stripped.lower() in self.PLACEHOLDER_STRINGS

    def run_dataset(self, summaries: list[EpisodeSummary]) -> list[Finding]:
        findings = []
        for summary in summaries:
            if self._is_placeholder(summary.task):
                findings.append(
                    Finding(
                        check=self.id,
                        severity=self.severity,
                        episode=summary.episode_index,
                        joint=None,
                        frames=[],
                        message=(
                            f"Episode {summary.episode_index} has an empty or "
                            f"placeholder task string ({summary.task!r})"
                        ),
                        data={"task": summary.task},
                    )
                )
        return findings


class TaskImbalanceCheck(DatasetCheck):
    """E4. In a multi-task dataset, a task with under 5% of episodes -- likely
    underrepresented, a policy trained on this may not generalize to it."""

    id = "TASK_IMBALANCE"
    severity = "info"
    scope = "dataset"

    MIN_FRACTION_THRESHOLD = 0.05

    def run_dataset(self, summaries: list[EpisodeSummary]) -> list[Finding]:
        task_counts = Counter(s.task for s in summaries)
        if len(task_counts) < 2:
            return []  # single-task dataset -- imbalance doesn't apply

        total = len(summaries)
        findings = []
        for task, count in task_counts.items():
            fraction = count / total
            if fraction < self.MIN_FRACTION_THRESHOLD:
                findings.append(
                    Finding(
                        check=self.id,
                        severity=self.severity,
                        episode=None,
                        joint=None,
                        frames=[],
                        message=(
                            f"Task {task!r} makes up only {fraction:.1%} of episodes "
                            f"({count} of {total}) -- underrepresented"
                        ),
                        data={"task": task, "count": count, "fraction": fraction, "total": total},
                    )
                )
        return findings


class LowDiversityCheck(DatasetCheck):
    """E5. Workspace-coverage approximation (no forward kinematics in v1): if
    episodes' trajectory centroids barely spread relative to the dataset-wide
    joint range, the demos are near-identical and a policy won't generalize
    to moved objects.

    Subsample cap (resolves a cross-model outside-voice finding): vectorizing
    the pairwise-centroid-distance computation makes it fast, but doesn't
    change its O(n^2) complexity -- capped to a random subsample above
    SUBSAMPLE_CAP episodes so large Hub datasets stay fast regardless of
    scale, rather than depending on the field-study corpus happening to
    include a large dataset to reveal the problem."""

    id = "LOW_DIVERSITY"
    severity = "warning"
    scope = "dataset"

    RELATIVE_DISTANCE_THRESHOLD = 0.05
    SUBSAMPLE_CAP = 500
    SUBSAMPLE_SEED = 0

    def run_dataset(self, summaries: list[EpisodeSummary]) -> list[Finding]:
        if len(summaries) < 2:
            return []

        if len(summaries) > self.SUBSAMPLE_CAP:
            rng = np.random.default_rng(self.SUBSAMPLE_SEED)
            indices = rng.choice(len(summaries), size=self.SUBSAMPLE_CAP, replace=False)
            sampled = [summaries[i] for i in indices]
        else:
            sampled = summaries

        centroids = np.stack([s.joint_means for s in sampled])
        global_min = np.min(np.stack([s.joint_mins for s in summaries]), axis=0)
        global_max = np.max(np.stack([s.joint_maxs for s in summaries]), axis=0)
        range_norm = float(np.linalg.norm(global_max - global_min))
        if range_norm == 0.0:
            return []

        # vectorized pairwise Euclidean distance -- fast per-pair, still O(n^2)
        # pairs overall, which is why the subsample cap above exists.
        diffs = centroids[:, None, :] - centroids[None, :, :]
        distances = np.sqrt(np.sum(diffs**2, axis=-1))
        upper_triangle = distances[np.triu_indices_from(distances, k=1)]
        mean_pairwise_distance = float(np.mean(upper_triangle))

        if mean_pairwise_distance < self.RELATIVE_DISTANCE_THRESHOLD * range_norm:
            return [
                Finding(
                    check=self.id,
                    severity=self.severity,
                    episode=None,
                    joint=None,
                    frames=[],
                    message=(
                        f"Episodes traverse near-identical trajectories (mean pairwise "
                        f"centroid distance {mean_pairwise_distance:.3f}, "
                        f"{mean_pairwise_distance / range_norm:.1%} of the dataset's "
                        f"joint-space range) -- demos are near-identical, policy will "
                        f"not generalize to moved objects"
                    ),
                    data={
                        "mean_pairwise_distance": mean_pairwise_distance,
                        "range_norm": range_norm,
                        "sampled_episode_count": len(sampled),
                    },
                )
            ]
        return []
