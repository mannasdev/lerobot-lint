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


def test_fail_on_defaults_to_error_matching_prior_behavior():
    findings = [_finding("warning")]
    assert determine_exit_code(findings) == 1


def test_fail_on_warning_escalates_a_warning_only_finding_set_to_exit_2():
    findings = [_finding("warning")]
    assert determine_exit_code(findings, fail_on="warning") == 2


def test_fail_on_warning_still_exits_1_for_info_only_finding_set():
    findings = [_finding("info")]
    assert determine_exit_code(findings, fail_on="warning") == 1


def test_fail_on_info_escalates_any_finding_at_all_to_exit_2():
    findings = [_finding("info")]
    assert determine_exit_code(findings, fail_on="info") == 2


def test_fail_on_warning_still_exits_2_for_an_error():
    findings = [_finding("error")]
    assert determine_exit_code(findings, fail_on="warning") == 2
