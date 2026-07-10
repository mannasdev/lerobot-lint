import numpy as np
import pytest

from lerobot_lint.types import CameraSample, EpisodeData


def _make_episode(n_frames=10, n_joints=6):
    return EpisodeData(
        states=np.zeros((n_frames, n_joints), dtype=np.float32),
        actions=np.zeros((n_frames, n_joints), dtype=np.float32),
        timestamps=np.arange(n_frames, dtype=np.float64) / 30.0,
        fps=30.0,
        task="pick up the block",
        camera_handles={},
    )


def test_episode_data_holds_fields():
    ep = _make_episode(n_frames=10, n_joints=6)
    assert ep.fps == 30.0
    assert ep.task == "pick up the block"
    assert ep.states.shape == (10, 6)


def test_episode_data_rejects_mismatched_timestamp_length():
    with pytest.raises(ValueError, match="timestamps"):
        EpisodeData(
            states=np.zeros((10, 6), dtype=np.float32),
            actions=np.zeros((10, 6), dtype=np.float32),
            timestamps=np.arange(9, dtype=np.float64) / 30.0,  # one short
            fps=30.0,
            task="pick up the block",
            camera_handles={},
        )


def test_episode_data_rejects_mismatched_action_state_length():
    with pytest.raises(ValueError, match="actions"):
        EpisodeData(
            states=np.zeros((10, 6), dtype=np.float32),
            actions=np.zeros((9, 6), dtype=np.float32),  # one short
            timestamps=np.arange(10, dtype=np.float64) / 30.0,
            fps=30.0,
            task="pick up the block",
            camera_handles={},
        )


def test_camera_sample_holds_total_frame_count_and_sampled_frames():
    frames = np.zeros((10, 8, 8, 3), dtype=np.uint8)
    sample = CameraSample(total_frame_count=300, frames=frames)

    assert sample.total_frame_count == 300
    assert sample.frames.shape == (10, 8, 8, 3)
