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
class CameraSample:
    """A camera's sampled frames plus the video's actual total frame count --
    D1 (VIDEO_LENGTH_MISMATCH) needs the total; D2/D3 operate on the cheap
    sampled subset. Populated by loader.py's windowed sampling, never a full
    frame-by-frame decode."""

    total_frame_count: int
    frames: np.ndarray


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


@dataclass
class EpisodeSummary:
    """The lightweight, per-episode record pass 1 emits and pass 2 accumulates
    for Group E's dataset-scoped checks. Deliberately excludes raw per-frame
    arrays -- that's the entire point of a two-pass streaming design: raw
    state/action data is discarded once an episode's per-episode checks and
    this summary have been computed."""

    episode_index: int
    frame_count: int
    duration: float
    task: str
    joint_means: np.ndarray
    joint_mins: np.ndarray
    joint_maxs: np.ndarray


def summarize_episode(episode: EpisodeData, episode_index: int) -> EpisodeSummary:
    return EpisodeSummary(
        episode_index=episode_index,
        frame_count=episode.states.shape[0],
        duration=float(episode.timestamps[-1] - episode.timestamps[0]),
        task=episode.task,
        joint_means=np.mean(episode.states, axis=0),
        joint_mins=np.min(episode.states, axis=0),
        joint_maxs=np.max(episode.states, axis=0),
    )
