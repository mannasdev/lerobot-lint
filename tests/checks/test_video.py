import numpy as np

from lerobot_lint.checks.video import DegenerateImageCheck, FrozenCameraCheck, VideoLengthMismatchCheck
from lerobot_lint.types import CameraSample, EpisodeData


def _episode_with_camera_samples(samples_by_camera, n_frames=20, n_joints=2):
    return EpisodeData(
        states=np.zeros((n_frames, n_joints), dtype=np.float32),
        actions=np.zeros((n_frames, n_joints), dtype=np.float32),
        timestamps=np.arange(n_frames, dtype=np.float64) / 30.0,
        fps=30.0,
        task="pick up the block",
        camera_handles=samples_by_camera,
    )


def _random_frames(rng, n=10, h=8, w=8, c=3):
    return rng.integers(0, 255, size=(n, h, w, c), dtype=np.uint8)


def test_video_length_mismatch_fires_when_camera_frame_count_differs_from_state_rows():
    rng = np.random.default_rng(9)
    sample = CameraSample(total_frame_count=15, frames=_random_frames(rng))  # episode has 20 states

    ep = _episode_with_camera_samples({"laptop": sample}, n_frames=20)
    findings = VideoLengthMismatchCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].check == "VIDEO_LENGTH_MISMATCH"
    assert findings[0].severity == "error"
    assert findings[0].data["video_frame_count"] == 15
    assert findings[0].data["state_row_count"] == 20


def test_video_length_mismatch_does_not_fire_when_counts_match():
    rng = np.random.default_rng(9)
    sample = CameraSample(total_frame_count=20, frames=_random_frames(rng))

    ep = _episode_with_camera_samples({"laptop": sample}, n_frames=20)
    findings = VideoLengthMismatchCheck().run(ep, episode_index=0)

    assert findings == []


def test_frozen_camera_fires_when_consecutive_frames_are_pixel_identical():
    rng = np.random.default_rng(7)
    frames = _random_frames(rng, n=10)
    frames[4] = frames[3]  # camera hung: frame 4 identical to frame 3
    sample = CameraSample(total_frame_count=300, frames=frames)

    ep = _episode_with_camera_samples({"laptop": sample})
    findings = FrozenCameraCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].check == "FROZEN_CAMERA"
    assert findings[0].severity == "error"
    assert findings[0].data["camera"] == "laptop"


def test_frozen_camera_does_not_fire_on_varying_frames():
    rng = np.random.default_rng(7)
    sample = CameraSample(total_frame_count=300, frames=_random_frames(rng, n=10))

    ep = _episode_with_camera_samples({"laptop": sample})
    findings = FrozenCameraCheck().run(ep, episode_index=0)

    assert findings == []


def test_frozen_camera_checks_every_camera_independently():
    rng = np.random.default_rng(7)
    good_sample = CameraSample(total_frame_count=300, frames=_random_frames(rng, n=10))
    frozen_frames = _random_frames(rng, n=10)
    frozen_frames[5] = frozen_frames[4]
    frozen_sample = CameraSample(total_frame_count=300, frames=frozen_frames)

    ep = _episode_with_camera_samples({"laptop": good_sample, "wrist": frozen_sample})
    findings = FrozenCameraCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].data["camera"] == "wrist"


def test_degenerate_image_fires_on_near_black_frames():
    frames = np.full((10, 8, 8, 3), fill_value=2, dtype=np.uint8)  # near-black, luminance < 10/255
    sample = CameraSample(total_frame_count=300, frames=frames)

    ep = _episode_with_camera_samples({"laptop": sample})
    findings = DegenerateImageCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert findings[0].check == "DEGENERATE_IMAGE"
    assert findings[0].severity == "warning"
    assert "black" in findings[0].message.lower()


def test_degenerate_image_fires_on_near_white_frames():
    frames = np.full((10, 8, 8, 3), fill_value=253, dtype=np.uint8)  # near-white, > 245/255
    sample = CameraSample(total_frame_count=300, frames=frames)

    ep = _episode_with_camera_samples({"laptop": sample})
    findings = DegenerateImageCheck().run(ep, episode_index=0)

    assert len(findings) == 1
    assert "white" in findings[0].message.lower()


def test_degenerate_image_does_not_fire_on_normal_frames():
    rng = np.random.default_rng(8)
    sample = CameraSample(total_frame_count=300, frames=rng.integers(50, 200, size=(10, 8, 8, 3), dtype=np.uint8))

    ep = _episode_with_camera_samples({"laptop": sample})
    findings = DegenerateImageCheck().run(ep, episode_index=0)

    assert findings == []
