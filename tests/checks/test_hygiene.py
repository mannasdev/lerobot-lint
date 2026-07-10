import numpy as np

from lerobot_lint.checks.hygiene import DurationOutlierCheck, ShortEpisodeCheck
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
