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
