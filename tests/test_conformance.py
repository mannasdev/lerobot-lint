import pytest
from typer.testing import CliRunner

import lerobot_lint
from lerobot_lint.cli import app

runner = CliRunner()

REAL_REPO_ID = "lerobot/pusht"


def test_a_dataset_that_exits_2_from_the_cli_also_raises_under_guard():
    # pusht episode 0 produces real JITTER errors -- CLI should exit 2.
    result = runner.invoke(
        app, ["check", REAL_REPO_ID, "--episodes", "0:1", "--no-video"]
    )
    assert result.exit_code == 2

    # guard() on the exact same fixture must raise for the same reason.
    with pytest.raises(lerobot_lint.LelintCheckFailedError):
        with lerobot_lint.guard(REAL_REPO_ID, episode_indices=[0], download_videos=False):
            pass


def test_a_clean_finding_set_does_not_raise_under_guard_or_exit_nonzero_for_errors(monkeypatch):
    # Same underlying engine call, both entry points, forced clean via monkeypatch
    # so this test doesn't depend on any real dataset happening to be clean.
    from lerobot_lint import cli as cli_module
    import lerobot_lint._guard as guard_module

    monkeypatch.setattr(cli_module, "check_dataset", lambda *a, **k: [])
    monkeypatch.setattr(guard_module, "check_dataset", lambda *a, **k: [])

    result = runner.invoke(app, ["check", REAL_REPO_ID, "--episodes", "0:1", "--no-video"])
    assert result.exit_code == 0

    with lerobot_lint.guard(REAL_REPO_ID, episode_indices=[0]):
        pass  # must not raise
