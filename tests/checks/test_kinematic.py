import numpy as np

from lerobot_lint.checks.kinematic import DeadJointCheck, SaturatedJointCheck
from lerobot_lint.types import EpisodeData


def _episode_with_states(states):
    n_frames = states.shape[0]
    return EpisodeData(
        states=states.astype(np.float32),
        actions=states.astype(np.float32),
        timestamps=np.arange(n_frames, dtype=np.float64) / 30.0,
        fps=30.0,
        task="pick up the block",
        camera_handles={},
    )


def test_dead_joint_fires_when_one_joint_never_moves_while_others_do():
    rng = np.random.default_rng(0)
    n_frames, n_joints = 100, 4
    states = rng.normal(scale=1.0, size=(n_frames, n_joints))
    states[:, 2] = 0.5  # joint index 2 is dead: constant, others vary

    ep = _episode_with_states(states)
    findings = DeadJointCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].check == "DEAD_JOINT"
    assert findings[0].data["joint_index"] == 2


def test_dead_joint_does_not_fire_on_clean_data():
    rng = np.random.default_rng(0)
    n_frames, n_joints = 100, 4
    states = rng.normal(scale=1.0, size=(n_frames, n_joints))  # all joints vary

    ep = _episode_with_states(states)
    findings = DeadJointCheck().run(ep, episode_index=0)

    assert findings == []


def test_dead_joint_does_not_fire_when_all_joints_are_static():
    # all joints constant (e.g. a settling frame) -- no OTHER joint moves, so
    # this shouldn't be flagged as a disconnected motor, just a static episode.
    n_frames, n_joints = 100, 4
    states = np.full((n_frames, n_joints), 0.5)

    ep = _episode_with_states(states)
    findings = DeadJointCheck().run(ep, episode_index=0)

    assert findings == []


def test_saturated_joint_fires_when_pinned_at_max_for_over_30_percent_of_frames():
    rng = np.random.default_rng(2)
    n_frames, n_joints = 100, 3
    states = rng.normal(scale=1.0, size=(n_frames, n_joints))
    states[:40, 1] = 5.0  # joint 1 pinned at its max for 40% of frames

    ep = _episode_with_states(states)
    findings = SaturatedJointCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].check == "SATURATED_JOINT"
    assert findings[0].severity == "warning"
    assert findings[0].data["joint_index"] == 1


def test_saturated_joint_does_not_fire_below_30_percent_threshold():
    rng = np.random.default_rng(2)
    n_frames, n_joints = 100, 3
    states = rng.normal(scale=1.0, size=(n_frames, n_joints))
    states[:20, 1] = 5.0  # only 20% of frames pinned -- below threshold

    ep = _episode_with_states(states)
    findings = SaturatedJointCheck().run(ep, episode_index=0)

    assert findings == []


def test_saturated_joint_does_not_fire_on_freely_moving_joint():
    rng = np.random.default_rng(2)
    n_frames, n_joints = 100, 3
    states = rng.normal(scale=1.0, size=(n_frames, n_joints))

    ep = _episode_with_states(states)
    findings = SaturatedJointCheck().run(ep, episode_index=0)

    assert findings == []
