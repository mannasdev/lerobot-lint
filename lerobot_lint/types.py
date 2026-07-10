from dataclasses import dataclass, field
from typing import Any

import numpy as np

VALID_SEVERITIES = {"error", "warning", "info"}


@dataclass
class EpisodeData:
    """Normalized per-episode view produced by loader.py — the only file that
    touches the lerobot API. states/actions are (T, J) float32; timestamps is (T,)."""

    states: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    fps: float
    task: str
    camera_handles: dict[str, Any]

    def __post_init__(self) -> None:
        n_frames = self.states.shape[0]
        if self.timestamps.shape[0] != n_frames:
            raise ValueError(
                f"timestamps length ({self.timestamps.shape[0]}) does not match "
                f"states length ({n_frames})"
            )
        if self.actions.shape[0] != n_frames:
            raise ValueError(
                f"actions length ({self.actions.shape[0]}) does not match "
                f"states length ({n_frames})"
            )


@dataclass
class Finding:
    """One check result. episode/joint may be None for dataset-level (Group E)
    findings that aren't scoped to a single episode or joint."""

    check: str
    severity: str
    episode: int | None
    joint: str | None
    frames: list[int]
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {VALID_SEVERITIES}, got {self.severity!r}"
            )
