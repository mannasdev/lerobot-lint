import pytest

import lerobot_lint


def test_lerobot_lint_guard_is_importable_and_callable_at_top_level():
    assert callable(lerobot_lint.guard)


def test_lerobot_lint_exposes_the_guard_exceptions():
    assert issubclass(lerobot_lint.LelintCheckFailedError, Exception)
    assert issubclass(lerobot_lint.LelintCheckCrashedError, Exception)


def test_lerobot_lint_guard_raises_via_top_level_import():
    with pytest.raises(lerobot_lint.LelintCheckFailedError):
        with lerobot_lint.guard("lerobot/pusht", episode_indices=[0], download_videos=False):
            pass
