import numpy as np

from lerobot_lint.checks.hygiene import (
    DurationOutlierCheck,
    MissingTaskCheck,
    ShortEpisodeCheck,
    TaskImbalanceCheck,
)
from lerobot_lint.types import EpisodeSummary


def _summary(episode_index=0, frame_count=100, duration=3.0, task="pick up the block"):
    return EpisodeSummary(
        episode_index=episode_index,
        frame_count=frame_count,
        duration=duration,
        task=task,
        joint_means=np.zeros(3),
        joint_mins=np.zeros(3),
        joint_maxs=np.ones(3),
    )


def test_short_episode_fires_on_duration_under_1_second():
    summaries = [_summary(duration=0.5, frame_count=100)]
    findings = ShortEpisodeCheck().run_dataset(summaries)

    assert len(findings) == 1
    assert findings[0].check == "SHORT_EPISODE"
    assert findings[0].severity == "warning"
    assert findings[0].episode == 0


def test_short_episode_fires_on_fewer_than_15_frames():
    summaries = [_summary(duration=3.0, frame_count=10)]
    findings = ShortEpisodeCheck().run_dataset(summaries)

    assert len(findings) == 1


def test_short_episode_does_not_fire_on_a_normal_episode():
    summaries = [_summary(duration=3.0, frame_count=100)]
    findings = ShortEpisodeCheck().run_dataset(summaries)

    assert findings == []


def test_short_episode_checks_every_episode_independently():
    summaries = [
        _summary(episode_index=0, duration=3.0, frame_count=100),
        _summary(episode_index=1, duration=0.2, frame_count=6),
    ]
    findings = ShortEpisodeCheck().run_dataset(summaries)

    assert len(findings) == 1
    assert findings[0].episode == 1


def test_duration_outlier_fires_on_an_episode_far_from_the_dataset_mean():
    # 19 episodes at ~3s, one wildly different at 60s -- a clear z-score outlier
    summaries = [_summary(episode_index=i, duration=3.0) for i in range(19)]
    summaries.append(_summary(episode_index=19, duration=60.0))

    findings = DurationOutlierCheck().run_dataset(summaries)

    assert len(findings) == 1
    assert findings[0].check == "DURATION_OUTLIER"
    assert findings[0].severity == "info"
    assert findings[0].episode == 19


def test_duration_outlier_does_not_fire_on_uniform_durations():
    summaries = [_summary(episode_index=i, duration=3.0) for i in range(20)]

    findings = DurationOutlierCheck().run_dataset(summaries)

    assert findings == []


def test_duration_outlier_does_not_crash_on_a_single_episode():
    summaries = [_summary(episode_index=0, duration=3.0)]

    findings = DurationOutlierCheck().run_dataset(summaries)

    assert findings == []


def test_missing_task_fires_on_an_empty_task_string():
    summaries = [_summary(episode_index=0, task="")]
    findings = MissingTaskCheck().run_dataset(summaries)

    assert len(findings) == 1
    assert findings[0].check == "MISSING_TASK"
    assert findings[0].severity == "warning"
    assert findings[0].episode == 0


def test_missing_task_fires_on_placeholder_task_strings():
    for placeholder in ["test", "asdf", "TODO", "  "]:
        summaries = [_summary(episode_index=0, task=placeholder)]
        findings = MissingTaskCheck().run_dataset(summaries)
        assert len(findings) == 1, f"expected a finding for task={placeholder!r}"


def test_missing_task_does_not_fire_on_a_real_task_description():
    summaries = [_summary(episode_index=0, task="pick up the red block and place it in the bin")]
    findings = MissingTaskCheck().run_dataset(summaries)

    assert findings == []


def test_task_imbalance_fires_when_one_task_is_under_5_percent():
    # 98 episodes of task A, 2 episodes of task B -- task B is 2%, under 5%
    summaries = [_summary(episode_index=i, task="pick up the block") for i in range(98)]
    summaries += [_summary(episode_index=98 + i, task="push the block") for i in range(2)]

    findings = TaskImbalanceCheck().run_dataset(summaries)

    assert len(findings) == 1
    assert findings[0].check == "TASK_IMBALANCE"
    assert findings[0].severity == "info"
    assert findings[0].data["task"] == "push the block"


def test_task_imbalance_does_not_fire_on_a_single_task_dataset():
    summaries = [_summary(episode_index=i, task="pick up the block") for i in range(50)]

    findings = TaskImbalanceCheck().run_dataset(summaries)

    assert findings == []


def test_task_imbalance_does_not_fire_when_tasks_are_reasonably_balanced():
    summaries = [_summary(episode_index=i, task="pick up the block") for i in range(30)]
    summaries += [_summary(episode_index=30 + i, task="push the block") for i in range(20)]

    findings = TaskImbalanceCheck().run_dataset(summaries)

    assert findings == []
