"""Engine-level units behavior: inference from the first episode, degree->radian
conversion before kinematic checks, JITTER suppression when units are unknown,
and profiles' max_joint_velocity actually reaching JitterCheck."""

import numpy as np

from lerobot_lint import engine
from lerobot_lint.config import Profile
from lerobot_lint.types import EpisodeData


def _episode_with_states(states, fps=30.0):
    n_frames = states.shape[0]
    return EpisodeData(
        states=states.astype(np.float32),
        actions=states.astype(np.float32),
        timestamps=np.arange(n_frames, dtype=np.float64) / fps,
        fps=fps,
        task="pick up the block",
        camera_handles={},
    )


def _profile(max_joint_velocity=8.0):
    return Profile(
        name="test", gripper_index=None, joint_names=None, max_joint_velocity=max_joint_velocity
    )


def _patch_episodes(monkeypatch, episodes):
    def fake_iter(repo_id_or_path, episode_indices=None, download_videos=True):
        for i, ep in enumerate(episodes):
            yield i, ep, None

    monkeypatch.setattr(engine, "iter_episodes", fake_iter)


def _smooth_degree_scale_episode():
    # ordinary teleop motion recorded in degrees: 0 -> 40 deg over ~3.3s.
    # naively read as radians this is "12 rad/s" and floods JITTER --
    # the koch_pick_place_5_lego_random_pose false-positive storm.
    states = np.tile(np.linspace(0.0, 40.0, 100), (2, 1)).T.copy()
    return _episode_with_states(states)


def test_degree_scale_data_is_inferred_and_converted_so_jitter_stays_quiet(monkeypatch):
    _patch_episodes(monkeypatch, [_smooth_degree_scale_episode()])

    findings = engine.check_dataset("fake/repo", _profile(), download_videos=False)

    assert [f for f in findings if f.check == "JITTER"] == []
    inferred = [f for f in findings if f.check == "UNITS_INFERRED"]
    assert len(inferred) == 1
    assert inferred[0].severity == "info"
    assert "degrees" in inferred[0].message


def test_explicit_units_override_suppresses_inference(monkeypatch):
    _patch_episodes(monkeypatch, [_smooth_degree_scale_episode()])

    # the user insists the data is radians: no inference finding, and the
    # degree-scale ramp now genuinely exceeds 8 rad/s, so JITTER must fire
    findings = engine.check_dataset("fake/repo", _profile(), download_videos=False, units="radians")

    assert [f for f in findings if f.check == "UNITS_INFERRED"] == []
    assert [f for f in findings if f.check == "JITTER"] != []


def test_encoder_count_scale_data_skips_jitter_with_one_loud_finding(monkeypatch):
    states = np.tile(np.linspace(1000.0, 3000.0, 100), (2, 1)).T.copy()
    _patch_episodes(monkeypatch, [_episode_with_states(states)])

    findings = engine.check_dataset("fake/repo", _profile(), download_videos=False)

    assert [f for f in findings if f.check == "JITTER"] == []
    unknown = [f for f in findings if f.check == "UNITS_UNKNOWN"]
    assert len(unknown) == 1
    assert unknown[0].severity == "warning"
    assert "--units" in unknown[0].message


def test_profiles_max_joint_velocity_reaches_the_jitter_check(monkeypatch):
    # radian-scale motion with a genuine 150 rad/s spike
    states = np.tile(np.linspace(0.0, 1.0, 100), (2, 1)).T.copy()
    states[50, 0] += 5.0

    _patch_episodes(monkeypatch, [_episode_with_states(states)])
    strict = engine.check_dataset("fake/repo", _profile(max_joint_velocity=8.0), download_videos=False)

    _patch_episodes(monkeypatch, [_episode_with_states(states)])
    permissive = engine.check_dataset(
        "fake/repo", _profile(max_joint_velocity=1000.0), download_videos=False
    )

    assert [f for f in strict if f.check == "JITTER"] != []
    assert [f for f in permissive if f.check == "JITTER"] == []


def test_units_inference_happens_once_from_the_first_episode(monkeypatch):
    # both episodes are degree-scale; exactly one UNITS_INFERRED finding total
    _patch_episodes(monkeypatch, [_smooth_degree_scale_episode(), _smooth_degree_scale_episode()])

    findings = engine.check_dataset("fake/repo", _profile(), download_videos=False)

    assert len([f for f in findings if f.check == "UNITS_INFERRED"]) == 1
