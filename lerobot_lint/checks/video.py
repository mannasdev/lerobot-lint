"""Group D — camera/video checks (sampled, cheap).

These checks operate on `episode.camera_handles[camera_name]`, a CameraSample
per camera carrying both the video's total frame count and a small sampled
subset of decoded frames. Decode-agnostic by design: the windowed sampling
(3 windows of 10 truly-consecutive frames per camera, per the design doc's
fix for isolated-sampling blind spots) and the actual av/opencv decode
integration live in loader.py, wired up separately from this check logic.
"""

import numpy as np

from lerobot_lint.checks.base import Check
from lerobot_lint.types import EpisodeData, Finding


class VideoLengthMismatchCheck(Check):
    """D1. Video frame count != state row count for an episode (per camera) ->
    desynced observation/action pairs, the silent policy-killer."""

    id = "VIDEO_LENGTH_MISMATCH"
    severity = "error"
    scope = "episode"

    def run(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        expected = episode.states.shape[0]
        findings = []
        for camera_name, sample in episode.camera_handles.items():
            if sample.total_frame_count == expected:
                continue
            findings.append(
                Finding(
                    check=self.id,
                    severity=self.severity,
                    episode=episode_index,
                    joint=None,
                    frames=[],
                    message=(
                        f"Camera {camera_name!r} has {sample.total_frame_count} video frames "
                        f"but the episode has {expected} state rows -- desynced "
                        f"observation/action pairs"
                    ),
                    data={
                        "camera": camera_name,
                        "video_frame_count": sample.total_frame_count,
                        "state_row_count": expected,
                    },
                )
            )
        return findings


class FrozenCameraCheck(Check):
    """D2. Sampled consecutive frames are pixel-identical (mean abs diff <
    0.5/255) -> camera hung mid-recording. The "identical webcams randomly
    reassigning device paths" bug class."""

    id = "FROZEN_CAMERA"
    severity = "error"
    scope = "episode"

    MEAN_ABS_DIFF_THRESHOLD = 0.5 / 255

    def run(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        findings = []
        for camera_name, sample in episode.camera_handles.items():
            frames = sample.frames
            if frames is None or len(frames) < 2:
                continue

            diffs = np.mean(
                np.abs(frames[1:].astype(np.float64) - frames[:-1].astype(np.float64)),
                axis=tuple(range(1, frames.ndim)),
            ) / 255.0
            frozen_pairs = np.nonzero(diffs < self.MEAN_ABS_DIFF_THRESHOLD)[0]
            if frozen_pairs.size == 0:
                continue

            findings.append(
                Finding(
                    check=self.id,
                    severity=self.severity,
                    episode=episode_index,
                    joint=None,
                    frames=(frozen_pairs + 1).tolist(),
                    message=(
                        f"Camera {camera_name!r} has {frozen_pairs.size} pixel-identical "
                        f"consecutive frame pair(s) in the sampled window -- camera likely "
                        f"hung mid-recording"
                    ),
                    data={"camera": camera_name, "frozen_pair_count": int(frozen_pairs.size)},
                )
            )
        return findings


class DegenerateImageCheck(Check):
    """D3. Sampled frames near-black, near-white, or near-zero variance ->
    lens cap, overexposure, or a dead feed."""

    id = "DEGENERATE_IMAGE"
    severity = "warning"
    scope = "episode"

    NEAR_BLACK_THRESHOLD = 10 / 255
    NEAR_WHITE_THRESHOLD = 245 / 255
    NEAR_ZERO_VARIANCE_THRESHOLD = 1e-6

    def run(self, episode: EpisodeData, episode_index: int) -> list[Finding]:
        findings = []
        for camera_name, sample in episode.camera_handles.items():
            frames = sample.frames
            if frames is None or len(frames) == 0:
                continue

            mean_luminance = float(np.mean(frames.astype(np.float64))) / 255.0
            variance = float(np.var(frames.astype(np.float64))) / (255.0**2)

            reason = None
            if mean_luminance < self.NEAR_BLACK_THRESHOLD:
                reason = f"near-black (mean luminance {mean_luminance:.1%})"
            elif mean_luminance > self.NEAR_WHITE_THRESHOLD:
                reason = f"near-white (mean luminance {mean_luminance:.1%})"
            elif variance < self.NEAR_ZERO_VARIANCE_THRESHOLD:
                reason = "near-zero variance (flat/blank feed)"

            if reason is None:
                continue

            findings.append(
                Finding(
                    check=self.id,
                    severity=self.severity,
                    episode=episode_index,
                    joint=None,
                    frames=[],
                    message=(
                        f"Camera {camera_name!r} sampled frames are {reason} -- "
                        f"possible lens cap, overexposure, or dead feed"
                    ),
                    data={"camera": camera_name, "mean_luminance": mean_luminance, "variance": variance},
                )
            )
        return findings
