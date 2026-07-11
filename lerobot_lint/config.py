"""Robot profiles: per-joint velocity limits, gripper index, and joint names
that generic statistics can't infer. User-facing via --profile; profiles/*.yaml
ship with the package."""

from dataclasses import dataclass
from importlib import resources

import yaml

PROFILES_PACKAGE = "lerobot_lint.profiles"


@dataclass
class Profile:
    name: str
    gripper_index: int | None
    joint_names: list[str] | None
    max_joint_velocity: float


def list_profile_names() -> list[str]:
    return sorted(
        p.name.removesuffix(".yaml")
        for p in resources.files(PROFILES_PACKAGE).iterdir()
        if p.name.endswith(".yaml")
    )


def detect_profile_name(joint_names: list[str] | None) -> str | None:
    """Match a dataset's joint names against a known hardware profile's naming
    convention. Returns None (falls back to the 'default' profile) if nothing
    matches. so101 and koch currently share an identical convention -- and
    identical thresholds, so which one is reported doesn't change any check's
    behavior -- the earlier one in sorted profile-name order wins."""
    if not joint_names:
        return None
    for name in list_profile_names():
        if name == "default":
            continue
        if load_profile(name).joint_names == joint_names:
            return name
    return None


def load_profile(name: str) -> Profile:
    try:
        raw = resources.files(PROFILES_PACKAGE).joinpath(f"{name}.yaml").read_text()
    except FileNotFoundError as e:
        raise ValueError(f"unknown profile {name!r} -- no {name}.yaml in profiles/") from e

    data = yaml.safe_load(raw)
    return Profile(
        name=data["name"],
        gripper_index=data.get("gripper_index"),
        joint_names=data.get("joint_names"),
        max_joint_velocity=data["max_joint_velocity"],
    )
