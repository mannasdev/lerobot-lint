from lerobot_lint.cli import determine_exit_code
from lerobot_lint.types import Finding


def _finding(severity, check="SOME_CHECK"):
    return Finding(
        check=check, severity=severity, episode=0, joint=None, frames=[], message="x", data={}
    )


def test_exit_code_0_when_clean():
    assert determine_exit_code([]) == 0


def test_exit_code_1_when_only_warnings_or_info():
    findings = [_finding("warning"), _finding("info")]
    assert determine_exit_code(findings) == 1


def test_exit_code_2_when_any_error():
    findings = [_finding("warning"), _finding("error")]
    assert determine_exit_code(findings) == 2


def test_exit_code_2_on_load_error():
    findings = [_finding("error", check="EPISODE_LOAD_ERROR")]
    assert determine_exit_code(findings) == 2
