import numpy as np

from lerobot_lint.loader import iter_episodes
from lerobot_lint.types import EpisodeData

# Small, public, non-video-dependent smoke test against a real dataset.
# NOTE: network-dependent for now — the design doc calls for replacing this with a
# locally-committed fixture snapshot once one is captured (T-item already tracked).
REAL_REPO_ID = "lerobot/pusht"


def test_iter_episodes_yields_episode_data_for_real_dataset():
    episodes = list(iter_episodes(REAL_REPO_ID, episode_indices=[0], download_videos=False))

    assert len(episodes) == 1
    ep = episodes[0]
    assert isinstance(ep, EpisodeData)
    assert ep.states.ndim == 2
    assert ep.actions.shape == ep.states.shape
    assert ep.timestamps.shape[0] == ep.states.shape[0]
    assert ep.fps == 10
    assert ep.task == "Push the T-shaped block onto the T-shaped target."


def test_iter_episodes_states_are_float32_numpy():
    episodes = list(iter_episodes(REAL_REPO_ID, episode_indices=[0], download_videos=False))
    ep = episodes[0]

    assert isinstance(ep.states, np.ndarray)
    assert ep.states.dtype == np.float32


def test_iter_episodes_timestamps_start_at_zero_and_increase():
    episodes = list(iter_episodes(REAL_REPO_ID, episode_indices=[0], download_videos=False))
    ep = episodes[0]

    assert ep.timestamps[0] == 0.0
    assert (np.diff(ep.timestamps) > 0).all()
