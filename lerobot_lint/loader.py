"""The only file that touches the `lerobot` library's API directly. If lerobot's
on-disk format changes again (it has, three times: v2.0 -> v2.1 -> v3.0), this is
the one file that needs to change."""

from collections.abc import Iterator

import numpy as np

from lerobot_lint.types import EpisodeData

SUPPORTED_CODEBASE_VERSIONS = {"v2.0", "v2.1", "v3.0"}


class DatasetLoadError(Exception):
    """Raised when a dataset (or an unsupported/mismatched version of one) cannot
    be loaded. Callers report this as a load error, never let it surface as a raw
    traceback."""


def _task_string_for_index(meta, task_index: int) -> str:
    tasks_df = meta.tasks
    # tasks_df is indexed by task string, with a task_index column.
    match = tasks_df[tasks_df["task_index"] == task_index]
    if match.empty:
        raise DatasetLoadError(f"No task string found for task_index={task_index}")
    return match.index[0]


def _load_one_episode(repo_id_or_path: str, episode_index: int, meta, download_videos: bool) -> EpisodeData:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id_or_path, episodes=[episode_index], download_videos=download_videos)
    hf = ds.hf_dataset
    n_frames = len(hf)

    states = np.stack([hf[i]["observation.state"].numpy() for i in range(n_frames)]).astype(np.float32)
    actions = np.stack([hf[i]["action"].numpy() for i in range(n_frames)]).astype(np.float32)
    timestamps = np.array([hf[i]["timestamp"].item() for i in range(n_frames)], dtype=np.float64)
    task_index = int(hf[0]["task_index"].item())
    task = _task_string_for_index(meta, task_index)

    return EpisodeData(
        states=states,
        actions=actions,
        timestamps=timestamps,
        fps=float(meta.fps),
        task=task,
        camera_handles={},
    )


def get_joint_names(repo_id_or_path: str) -> list[str] | None:
    """Best-effort lookup of a dataset's observation.state joint/motor names,
    for profile auto-detection. Advisory only -- any failure (bad repo id,
    missing feature, unexpected metadata shape) returns None rather than
    raising, since the real load error (if any) will surface properly from
    iter_episodes instead."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    try:
        meta = LeRobotDatasetMetadata(repo_id_or_path)
        return list(meta.features["observation.state"]["names"]["motors"])
    except Exception:
        return None


def iter_episodes(
    repo_id_or_path: str,
    episode_indices: list[int] | None = None,
    download_videos: bool = True,
) -> Iterator[tuple[int, EpisodeData | None, str | None]]:
    """Stream one episode at a time from a LeRobotDataset, never loading the whole
    dataset's frame data into RAM at once. `episode_indices=None` iterates every
    episode in the dataset.

    Yields (real_episode_index, EpisodeData_or_None, error_message_or_None). The
    real dataset index is always the real one, not an enumerate() position --
    callers need to know which actual episode a finding or error belongs to,
    especially when a non-contiguous subset is requested.

    A single episode failing to load (corrupt shard, out-of-range index) does
    NOT abort the run -- it's reported via the error slot (episode=None,
    error=message) and iteration continues with the remaining episodes. Only a
    whole-dataset failure (can't even read metadata) raises DatasetLoadError,
    since there is nothing to iterate over at all in that case."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    try:
        meta = LeRobotDatasetMetadata(repo_id_or_path)
    except Exception as e:
        raise DatasetLoadError(f"Could not load dataset metadata for {repo_id_or_path!r}: {e}") from e

    indices = episode_indices if episode_indices is not None else range(meta.total_episodes)

    for i in indices:
        try:
            yield i, _load_one_episode(repo_id_or_path, i, meta, download_videos), None
        except Exception as e:
            yield i, None, f"Could not load episode {i} of {repo_id_or_path!r}: {e}"
