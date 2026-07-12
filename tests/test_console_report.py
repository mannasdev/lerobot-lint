from lerobot_lint.report.console import render_console_report
from lerobot_lint.types import Finding


def _finding(check="DEAD_JOINT", severity="error", episode=3, message="joint 2 never moved"):
    return Finding(
        check=check,
        severity=severity,
        episode=episode,
        joint="2",
        frames=[],
        message=message,
        data={},
    )


def test_render_reports_clean_dataset():
    report = render_console_report([], repo_id_or_path="lerobot/pusht")

    assert "lerobot/pusht" in report
    assert "clean" in report.lower() or "0 errors" in report.lower()


def test_render_includes_every_finding_message():
    findings = [
        _finding(check="DEAD_JOINT", severity="error", message="joint 2 never moved"),
        _finding(check="JITTER", severity="warning", message="joint 0 has a velocity spike"),
    ]

    report = render_console_report(findings, repo_id_or_path="lerobot/pusht")

    assert "joint 2 never moved" in report
    assert "joint 0 has a velocity spike" in report
    assert "DEAD_JOINT" in report
    assert "JITTER" in report


def test_render_includes_a_summary_count_by_severity():
    findings = [
        _finding(severity="error"),
        _finding(severity="error"),
        _finding(severity="warning"),
        _finding(severity="info"),
    ]

    report = render_console_report(findings, repo_id_or_path="lerobot/pusht")

    assert "2" in report  # error count appears somewhere
    assert "error" in report.lower()
    assert "warning" in report.lower()
    assert "info" in report.lower()


def test_render_includes_the_profile_disclosure_in_the_report_body():
    # the pre-run echo of this line gets buried under download progress bars;
    # the report itself is where a user actually reads it
    report = render_console_report(
        [_finding()],
        repo_id_or_path="lerobot/pusht",
        profile_disclosure="Using profile: default (no match, override with --profile)",
    )

    assert "Using profile: default (no match, override with --profile)" in report


def test_render_includes_the_profile_disclosure_even_on_a_clean_report():
    report = render_console_report(
        [],
        repo_id_or_path="lerobot/pusht",
        profile_disclosure="Using profile: koch (auto-detected from joint names, override with --profile)",
    )

    assert "Using profile: koch" in report
