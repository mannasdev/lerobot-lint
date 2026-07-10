import numpy as np

from lerobot_lint.checks.consistency import ActionStateDivergenceCheck
from lerobot_lint.types import EpisodeData


def _episode(states, actions, fps=30.0):
    n_frames = states.shape[0]
    return EpisodeData(
        states=states.astype(np.float32),
        actions=actions.astype(np.float32),
        timestamps=np.arange(n_frames, dtype=np.float64) / fps,
        fps=fps,
        task="pick up the block",
        camera_handles={},
    )


def test_fires_when_state_does_not_track_action():
    n_frames, n_joints = 100, 2
    rng = np.random.default_rng(5)
    actions = np.tile(np.linspace(0, 10, n_frames), (n_joints, 1)).T.copy()
    states = rng.normal(size=(n_frames, n_joints))  # follower ignores the leader entirely

    ep = _episode(states, actions)
    findings = ActionStateDivergenceCheck().run(ep, episode_index=0)

    assert len(findings) == n_joints
    assert all(f.check == "ACTION_STATE_DIVERGENCE" for f in findings)
    assert all(f.severity == "warning" for f in findings)


def test_does_not_fire_when_state_tracks_action_closely():
    n_frames, n_joints = 100, 2
    actions = np.tile(np.linspace(0, 10, n_frames), (n_joints, 1)).T.copy()
    states = actions.copy()  # follower tracks the leader perfectly

    ep = _episode(states, actions)
    findings = ActionStateDivergenceCheck().run(ep, episode_index=0)

    assert findings == []


def test_does_not_fire_with_small_realistic_tracking_lag():
    n_frames, n_joints = 100, 2
    actions = np.tile(np.linspace(0, 10, n_frames), (n_joints, 1)).T.copy()
    states = np.roll(actions, shift=1, axis=0)  # one-frame lag, still highly correlated
    states[0] = actions[0]

    ep = _episode(states, actions)
    findings = ActionStateDivergenceCheck().run(ep, episode_index=0)

    assert findings == []
