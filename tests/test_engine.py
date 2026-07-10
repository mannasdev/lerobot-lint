from lerobot_lint.config import load_profile
from lerobot_lint.engine import check_dataset
from lerobot_lint.types import Finding

REAL_REPO_ID = "lerobot/pusht"


def test_check_dataset_runs_end_to_end_against_a_real_dataset():
    profile = load_profile("default")

    findings = check_dataset(
        REAL_REPO_ID, profile, episode_indices=[0, 1, 2], download_videos=False
    )

    assert isinstance(findings, list)
    assert all(isinstance(f, Finding) for f in findings)
    # a 3-episode check is well under the 30-episode-per-task rule of thumb
    assert any(f.check == "TOO_FEW_EPISODES" for f in findings)


def test_check_dataset_treats_zero_episodes_as_an_error():
    profile = load_profile("default")

    # an empty episode_indices list -- no episodes requested, none checked
    findings = check_dataset(REAL_REPO_ID, profile, episode_indices=[], download_videos=False)

    no_episodes_findings = [f for f in findings if f.check == "NO_EPISODES"]
    assert len(no_episodes_findings) == 1
    assert no_episodes_findings[0].severity == "error"


def test_check_dataset_reports_a_bad_episode_without_aborting_the_rest():
    profile = load_profile("default")

    findings = check_dataset(
        REAL_REPO_ID, profile, episode_indices=[0, 99999, 1], download_videos=False
    )

    load_errors = [f for f in findings if f.check == "EPISODE_LOAD_ERROR"]
    assert len(load_errors) == 1
    assert load_errors[0].episode == 99999
    assert load_errors[0].severity == "error"

    # the two valid episodes must still have been checked -- dataset-level
    # checks ran against real summaries from episodes 0 and 1, not zero.
    assert any(f.check == "TOO_FEW_EPISODES" for f in findings)
