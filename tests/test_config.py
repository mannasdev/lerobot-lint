import pytest

from lerobot_lint.config import Profile, detect_profile_name, list_profile_names, load_profile


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


def test_list_profile_names_includes_all_three_built_in_profiles():
    assert list_profile_names() == ["default", "koch", "so101"]


def test_detect_profile_name_matches_the_shared_so101_koch_joint_convention():
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    detected = detect_profile_name(joint_names)

    # so101 and koch share an identical convention (and identical thresholds) --
    # either is a correct match, "koch" wins deterministically (sorted order).
    assert detected == "koch"


def test_detect_profile_name_returns_none_for_an_unrecognized_convention():
    assert detect_profile_name(["motor_0", "motor_1"]) is None


def test_detect_profile_name_returns_none_for_no_joint_names():
    assert detect_profile_name(None) is None
    assert detect_profile_name([]) is None
