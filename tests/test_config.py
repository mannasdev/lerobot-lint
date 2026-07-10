import pytest

from lerobot_lint.config import Profile, load_profile


def test_default_profile_has_conservative_generic_thresholds():
    profile = load_profile("default")

    assert profile.name == "default"
    assert profile.gripper_index is None  # generic profile can't know this
    assert profile.max_joint_velocity == 8.0


def test_so101_profile_has_a_gripper_index():
    profile = load_profile("so101")

    assert profile.name == "so101"
    assert profile.gripper_index is not None
    assert isinstance(profile.gripper_index, int)


def test_koch_profile_has_a_gripper_index():
    profile = load_profile("koch")

    assert profile.name == "koch"
    assert profile.gripper_index is not None


def test_load_profile_raises_on_unknown_profile_name():
    with pytest.raises(ValueError, match="unknown profile"):
        load_profile("not_a_real_robot")
