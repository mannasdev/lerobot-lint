import numpy as np

from lerobot_lint.checks.timing import FpsMismatchCheck, FrameDropsCheck, TimestampNonMonotonicCheck
from lerobot_lint.types import EpisodeData


def _episode(timestamps, fps=30.0):
    n_frames = len(timestamps)
    n_joints = 2
    return EpisodeData(
        states=np.zeros((n_frames, n_joints), dtype=np.float32),
        actions=np.zeros((n_frames, n_joints), dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        fps=fps,
        task="pick up the block",
        camera_handles={},
    )


def test_fires_when_timestamps_go_backwards():
    timestamps = list(np.arange(10) / 30.0)
    timestamps[5] = timestamps[3]  # frame 5's timestamp goes backwards

    ep = _episode(timestamps)
    findings = TimestampNonMonotonicCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].check == "TIMESTAMP_NON_MONOTONIC"
    assert findings[0].severity == "error"
    assert 5 in findings[0].frames


def test_fires_when_timestamps_are_duplicated():
    timestamps = list(np.arange(10) / 30.0)
    timestamps[5] = timestamps[4]  # duplicate timestamp

    ep = _episode(timestamps)
    findings = TimestampNonMonotonicCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert 5 in findings[0].frames


def test_does_not_fire_on_strictly_increasing_timestamps():
    timestamps = list(np.arange(10) / 30.0)

    ep = _episode(timestamps)
    findings = TimestampNonMonotonicCheck().run(ep, episode_index=0)

    assert findings == []


def test_frame_drops_fires_as_warning_below_5_percent_threshold():
    fps = 30.0
    n_frames = 100
    timestamps = np.arange(n_frames) / fps
    timestamps[50:] += 1.0 / fps * 3  # one gap of 3 frame-periods -- 1 dropped frame, 1%

    ep = _episode(timestamps, fps=fps)
    findings = FrameDropsCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].check == "FRAME_DROPS"
    assert findings[0].severity == "warning"
    assert findings[0].data["gap_count"] == 1


def test_frame_drops_escalates_to_error_above_5_percent_threshold():
    fps = 30.0
    n_frames = 100
    timestamps = np.arange(n_frames) / fps
    # 10 separate gaps -> 10% of frames affected, above the 5% escalation threshold
    for i in range(10, 60, 5):
        timestamps[i:] += 1.0 / fps * 3

    ep = _episode(timestamps, fps=fps)
    findings = FrameDropsCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_frame_drops_does_not_fire_on_evenly_spaced_timestamps():
    fps = 30.0
    timestamps = np.arange(100) / fps

    ep = _episode(timestamps, fps=fps)
    findings = FrameDropsCheck().run(ep, episode_index=0)

    assert findings == []


def test_fps_mismatch_fires_when_measured_rate_deviates_over_10_percent():
    declared_fps = 30.0
    actual_fps = 20.0  # recorded at 20fps but declared 30fps
    timestamps = np.arange(100) / actual_fps

    ep = _episode(timestamps, fps=declared_fps)
    findings = FpsMismatchCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].check == "FPS_MISMATCH"
    assert findings[0].severity == "warning"


def test_fps_mismatch_does_not_fire_when_measured_rate_matches_declared():
    fps = 30.0
    timestamps = np.arange(100) / fps

    ep = _episode(timestamps, fps=fps)
    findings = FpsMismatchCheck().run(ep, episode_index=0)

    assert findings == []
