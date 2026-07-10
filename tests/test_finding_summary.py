from lerobot_lint.report.finding_summary import finding_summary
from lerobot_lint.types import Finding


def _finding(message="joint 2 never moved", check="DEAD_JOINT", episode=3):
    return Finding(
        check=check, severity="error", episode=episode, joint="2", frames=[], message=message, data={}
    )


def test_finding_summary_includes_check_episode_and_message():
    summary = finding_summary(_finding())

    assert "DEAD_JOINT" in summary
    assert "3" in summary
    assert "joint 2 never moved" in summary


def test_finding_summary_uses_dataset_for_none_episode():
    summary = finding_summary(_finding(episode=None))

    assert "dataset" in summary.lower()


def test_finding_summary_strips_control_characters_from_message():
    # dataset-derived strings (task descriptions etc.) are untrusted -- a
    # malicious task string must never carry raw control/escape sequences
    # into the console, a bug card, or anywhere else this is displayed.
    malicious_message = "pick up the block\x1b[31mFAKE ERROR\x1b[0m\x07"
    summary = finding_summary(_finding(message=malicious_message))

    assert "\x1b" not in summary
    assert "\x07" not in summary
    assert "pick up the block" in summary
    assert "FAKE ERROR" in summary  # visible text survives, only control chars stripped
