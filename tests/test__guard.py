import pytest

from lerobot_lint._guard import LelintCheckCrashedError, LelintCheckFailedError, guard

REAL_REPO_ID = "lerobot/pusht"


def test_guard_raises_check_failed_on_a_real_error_finding():
    # 2 episodes of pusht produces real JITTER errors (severity=error) --
    # guard() must raise since this dataset has real error-severity findings.
    with pytest.raises(LelintCheckFailedError):
        with guard(REAL_REPO_ID, profile_name="default", episode_indices=[0], download_videos=False):
            pass


def test_guard_does_not_raise_on_warnings_only(monkeypatch):
    # force a clean run by stubbing check_dataset to return only a warning
    import lerobot_lint._guard as guard_module
    from lerobot_lint.types import Finding

    def fake_check_dataset(*args, **kwargs):
        return [Finding(check="X", severity="warning", episode=0, joint=None, frames=[], message="m", data={})]

    monkeypatch.setattr(guard_module, "check_dataset", fake_check_dataset)

    with guard(REAL_REPO_ID, profile_name="default", episode_indices=[0]):
        pass  # must not raise -- warnings alone don't block


def test_guard_never_swallows_the_wrapped_blocks_own_exception(monkeypatch):
    import lerobot_lint._guard as guard_module

    monkeypatch.setattr(guard_module, "check_dataset", lambda *a, **k: [])  # clean dataset

    class MyOwnBug(Exception):
        pass

    with pytest.raises(MyOwnBug):
        with guard(REAL_REPO_ID, profile_name="default", episode_indices=[0]):
            raise MyOwnBug("something in my training script broke")


def test_guard_raises_crashed_error_distinct_from_failed_error(monkeypatch):
    import lerobot_lint._guard as guard_module
    from lerobot_lint.types import Finding

    def fake_check_dataset(*args, **kwargs):
        return [
            Finding(
                check="SOME_CHECK_CRASHED",
                severity="error",
                episode=0,
                joint=None,
                frames=[],
                message="Check SOME_CHECK crashed: KeyError",
                data={"exception_type": "KeyError"},
            )
        ]

    monkeypatch.setattr(guard_module, "check_dataset", fake_check_dataset)

    with pytest.raises(LelintCheckCrashedError):
        with guard(REAL_REPO_ID, profile_name="default", episode_indices=[0]):
            pass


def test_guard_treats_zero_episodes_as_a_failed_check_not_a_pass_through():
    with pytest.raises(LelintCheckFailedError):
        with guard(REAL_REPO_ID, profile_name="default", episode_indices=[]):
            pass
