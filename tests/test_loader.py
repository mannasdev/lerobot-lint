import numpy as np

from lerobot_lint.loader import get_joint_names, iter_episodes
from lerobot_lint.types import EpisodeData

# Small, public, non-video-dependent smoke test against a real dataset.
# NOTE: network-dependent for now — the design doc calls for replacing this with a
# locally-committed fixture snapshot once one is captured (T-item already tracked).
REAL_REPO_ID = "lerobot/pusht"


def test_iter_episodes_yields_index_and_episode_data_for_real_dataset():
    results = list(iter_episodes(REAL_REPO_ID, episode_indices=[0], download_videos=False))

    assert len(results) == 1
    index, ep, error = results[0]
    assert index == 0
    assert error is None
    assert isinstance(ep, EpisodeData)
    assert ep.states.ndim == 2
    assert ep.actions.shape == ep.states.shape
    assert ep.timestamps.shape[0] == ep.states.shape[0]
    assert ep.fps == 10
    assert ep.task == "Push the T-shaped block onto the T-shaped target."


def test_iter_episodes_preserves_real_index_for_a_non_contiguous_subset():
    # requesting episode 3 specifically must yield index=3, not 0 (enumerate
    # position) -- callers (e.g. the CLI engine) need the real dataset index.
    results = list(iter_episodes(REAL_REPO_ID, episode_indices=[3], download_videos=False))

    assert len(results) == 1
    index, _ep, _error = results[0]
    assert index == 3


def test_iter_episodes_states_are_float32_numpy():
    results = list(iter_episodes(REAL_REPO_ID, episode_indices=[0], download_videos=False))
    _index, ep, _error = results[0]

    assert isinstance(ep.states, np.ndarray)
    assert ep.states.dtype == np.float32


def test_iter_episodes_timestamps_start_at_zero_and_increase():
    results = list(iter_episodes(REAL_REPO_ID, episode_indices=[0], download_videos=False))
    _index, ep, _error = results[0]

    assert ep.timestamps[0] == 0.0
    assert (np.diff(ep.timestamps) > 0).all()


def test_iter_episodes_one_bad_episode_does_not_abort_the_rest():
    # pusht has 206 episodes (0-205) -- 99999 is guaranteed out of range and
    # will fail to load, mixed with two real, loadable episodes. A single
    # corrupt/missing episode must not prevent the others from being checked.
    results = list(iter_episodes(REAL_REPO_ID, episode_indices=[0, 99999, 1], download_videos=False))

    assert len(results) == 3
    by_index = {index: (ep, error) for index, ep, error in results}

    ep0, error0 = by_index[0]
    assert error0 is None
    assert isinstance(ep0, EpisodeData)

    ep_bad, error_bad = by_index[99999]
    assert ep_bad is None
    assert error_bad is not None
    assert "99999" in error_bad

    ep1, error1 = by_index[1]
    assert error1 is None
    assert isinstance(ep1, EpisodeData)


def test_get_joint_names_returns_the_motor_names_for_a_real_dataset():
    # pusht is a 2D-pixel-coordinate sim task, so its "motors" are generic
    # placeholders -- this only proves the metadata plumbing works, not that
    # pusht has real joint names (see PROGRESS.md's pusht finding).
    joint_names = get_joint_names(REAL_REPO_ID)

    assert joint_names == ["motor_0", "motor_1"]


def test_get_joint_names_returns_none_for_a_bad_repo_id_instead_of_raising():
    assert get_joint_names("not-a-real-repo/does-not-exist-xyz") is None
