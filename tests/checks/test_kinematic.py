import numpy as np

from lerobot_lint.checks.kinematic import (
    DeadJointCheck,
    DiscontinuityCheck,
    FrozenStateCheck,
    GripperInertCheck,
    JitterCheck,
    SaturatedJointCheck,
)
from lerobot_lint.types import EpisodeData


def _episode_with_states(states, fps=30.0, task="pick up the block"):
    n_frames = states.shape[0]
    return EpisodeData(
        states=states.astype(np.float32),
        actions=states.astype(np.float32),
        timestamps=np.arange(n_frames, dtype=np.float64) / fps,
        fps=fps,
        task=task,
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


def test_jitter_fires_on_a_single_impossible_velocity_spike():
    n_frames, n_joints, fps = 100, 2, 30.0
    # smooth, slow motion well under the 8 rad/s default threshold
    states = np.tile(np.linspace(0, 1, n_frames), (n_joints, 1)).T.copy()
    states[50, 0] += 5.0  # one-frame spike: |delta|*fps = 5.0*30 = 150 rad/s

    ep = _episode_with_states(states, fps=fps)
    findings = JitterCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].check == "JITTER"
    assert findings[0].data["joint_index"] == 0
    # frames 49 (jump in), 50 (the spike), 51 (jump back out) all register as
    # touched by the anomalous velocity -- diff-based detection sees two
    # transitions (49->50 and 50->51), spanning three frame indices.
    assert findings[0].data["spike_frame_count"] == 3


def test_jitter_does_not_fire_on_smooth_motion():
    n_frames, n_joints, fps = 100, 2, 30.0
    states = np.tile(np.linspace(0, 1, n_frames), (n_joints, 1)).T.copy()

    ep = _episode_with_states(states, fps=fps)
    findings = JitterCheck().run(ep, episode_index=0)

    assert findings == []


def test_jitter_honors_a_custom_max_joint_velocity():
    n_frames, n_joints, fps = 100, 2, 30.0
    states = np.tile(np.linspace(0, 1, n_frames), (n_joints, 1)).T.copy()
    states[50, 0] += 5.0  # 150 rad/s spike: above the 8.0 default, below 200

    ep = _episode_with_states(states, fps=fps)

    # a permissive threshold must silence the finding the default produces --
    # this is what wires profiles' max_joint_velocity into actual behavior
    assert JitterCheck().run(ep, episode_index=0) != []
    assert JitterCheck(max_joint_velocity=200.0).run(ep, episode_index=0) == []


def test_jitter_reports_the_configured_threshold_in_its_message():
    n_frames, n_joints, fps = 100, 2, 30.0
    states = np.tile(np.linspace(0, 1, n_frames), (n_joints, 1)).T.copy()
    states[50, 0] += 5.0

    ep = _episode_with_states(states, fps=fps)
    findings = JitterCheck(max_joint_velocity=100.0).run(ep, episode_index=0)

    assert len(findings) == 1
    assert "100.0 rad/s" in findings[0].message


def test_jitter_escalates_to_error_above_2_percent_of_frames():
    n_frames, n_joints, fps = 100, 2, 30.0
    states = np.tile(np.linspace(0, 1, n_frames), (n_joints, 1)).T.copy()
    # spike every other frame across the first 10 frames -> > 2% of 100 frames
    for i in range(0, 10, 2):
        states[i, 0] += 5.0

    ep = _episode_with_states(states, fps=fps)
    findings = JitterCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_discontinuity_fires_on_encoder_wraparound():
    # simulates a 12-bit encoder wraparound: wrist_roll oscillates smoothly in a
    # small band, then teleports by ~the full observed range in a single frame.
    n_frames, n_joints = 100, 2
    states = np.zeros((n_frames, n_joints))
    states[:, 0] = np.sin(np.linspace(0, 4 * np.pi, n_frames)) * 0.1  # small smooth wiggle
    states[50:, 0] += 6.0  # single-frame teleport of ~6.0, dwarfing the 0.2 wiggle range

    ep = _episode_with_states(states)
    findings = DiscontinuityCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].check == "DISCONTINUITY"
    assert findings[0].severity == "error"
    assert findings[0].data["joint_index"] == 0
    assert 50 in findings[0].frames
    assert "wraparound" in findings[0].message.lower()


def test_discontinuity_does_not_fire_on_smooth_motion():
    n_frames, n_joints = 100, 2
    states = np.zeros((n_frames, n_joints))
    states[:, 0] = np.sin(np.linspace(0, 4 * np.pi, n_frames))

    ep = _episode_with_states(states)
    findings = DiscontinuityCheck().run(ep, episode_index=0)

    assert findings == []


def test_frozen_state_fires_on_a_mid_episode_freeze_over_15_frames():
    n_frames, n_joints, fps = 100, 3 , 30.0
    rng = np.random.default_rng(3)
    states = rng.normal(size=(n_frames, n_joints))
    states[40:60] = states[40]  # 20 identical frames mid-episode -- a recorder hiccup

    ep = _episode_with_states(states, fps=fps)
    findings = FrozenStateCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].check == "FROZEN_STATE"
    assert findings[0].severity == "error"
    assert findings[0].data["freeze_length"] == 20


def test_frozen_state_does_not_fire_on_a_short_mid_episode_freeze():
    n_frames, n_joints, fps = 100, 3, 30.0
    rng = np.random.default_rng(3)
    states = rng.normal(size=(n_frames, n_joints))
    states[40:50] = states[40]  # only 10 identical frames -- below the 15-frame threshold

    ep = _episode_with_states(states, fps=fps)
    findings = FrozenStateCheck().run(ep, episode_index=0)

    assert findings == []


def test_frozen_state_ignores_settling_at_episode_start_and_end():
    n_frames, n_joints, fps = 100, 3, 30.0
    rng = np.random.default_rng(3)
    states = rng.normal(size=(n_frames, n_joints))
    states[:20] = states[0]  # 20 identical frames at the very start -- settling, expected
    states[-20:] = states[-1]  # 20 identical frames at the very end -- settling, expected

    ep = _episode_with_states(states, fps=fps)
    findings = FrozenStateCheck().run(ep, episode_index=0)

    assert findings == []


def test_frozen_state_does_not_fire_on_freely_moving_data():
    n_frames, n_joints, fps = 100, 3, 30.0
    rng = np.random.default_rng(3)
    states = rng.normal(size=(n_frames, n_joints))

    ep = _episode_with_states(states, fps=fps)
    findings = FrozenStateCheck().run(ep, episode_index=0)

    assert findings == []


def test_gripper_inert_fires_when_gripper_never_moves_on_a_grasp_task():
    n_frames, n_joints, gripper_index = 50, 4, 3
    rng = np.random.default_rng(4)
    states = rng.normal(size=(n_frames, n_joints))
    states[:, gripper_index] = 0.0  # gripper never moves

    ep = _episode_with_states(states, task="pick up the red block and place it in the bin")
    findings = GripperInertCheck(gripper_index=gripper_index).run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].check == "GRIPPER_INERT"
    assert findings[0].severity == "warning"


def test_gripper_inert_does_not_fire_when_gripper_moves():
    n_frames, n_joints, gripper_index = 50, 4, 3
    rng = np.random.default_rng(4)
    states = rng.normal(size=(n_frames, n_joints))  # gripper (col 3) varies too

    ep = _episode_with_states(states, task="pick up the red block and place it in the bin")
    findings = GripperInertCheck(gripper_index=gripper_index).run(ep, episode_index=0)

    assert findings == []


def test_gripper_inert_does_not_fire_when_task_does_not_imply_grasping():
    n_frames, n_joints, gripper_index = 50, 4, 3
    rng = np.random.default_rng(4)
    states = rng.normal(size=(n_frames, n_joints))
    states[:, gripper_index] = 0.0  # gripper never moves, but task doesn't need it to

    ep = _episode_with_states(states, task="push the T-shaped block to the target")
    findings = GripperInertCheck(gripper_index=gripper_index).run(ep, episode_index=0)

    assert findings == []
