import json

from lerobot_lint.report.json_report import render_json_report
from lerobot_lint.types import Finding


def _finding(check="DEAD_JOINT", severity="error", episode=3, joint="2", frames=None, message="x", data=None):
    return Finding(
        check=check, severity=severity, episode=episode, joint=joint,
        frames=frames or [], message=message, data=data or {},
    )


def test_json_report_has_top_level_schema_fields():
    report = render_json_report([], repo_id_or_path="lerobot/pusht", profile_name="default")

    assert report["schema_version"] == 1
    assert "lelint_version" in report
    assert report["dataset"]["repo_id"] == "lerobot/pusht"
    assert report["dataset"]["robot_profile"] == "default"
    assert report["grade"] is None  # scorecard formula not designed/calibrated yet


def test_json_report_summary_counts_by_severity_and_episodes_affected():
    findings = [
        _finding(severity="error", episode=0),
        _finding(severity="error", episode=1),
        _finding(severity="warning", episode=1),
        _finding(severity="info", episode=None),
    ]

    report = render_json_report(findings, repo_id_or_path="x", profile_name="default")

    assert report["summary"]["errors"] == 2
    assert report["summary"]["warnings"] == 1
    assert report["summary"]["info"] == 1
    assert report["summary"]["episodes_affected"] == 2  # episodes 0 and 1, not the None one


def test_json_report_findings_are_fully_serializable():
    findings = [_finding(data={"delta": 5.9, "range_fraction": 0.98})]

    report = render_json_report(findings, repo_id_or_path="x", profile_name="default")

    # must not raise -- proves every field is JSON-safe
    dumped = json.dumps(report)
    reloaded = json.loads(dumped)
    assert reloaded["findings"][0]["check"] == "DEAD_JOINT"
    assert reloaded["findings"][0]["data"]["delta"] == 5.9


def test_json_report_sanitizes_message_control_characters():
    malicious = _finding(message="task\x1b[31mFAKE\x1b[0m")

    report = render_json_report([malicious], repo_id_or_path="x", profile_name="default")

    assert "\x1b" not in report["findings"][0]["message"]
    assert "FAKE" in report["findings"][0]["message"]


def test_json_report_schema_snapshot():
    # Schema-stability guard: the source spec calls this schema "the seed of
    # the future eval-CI product" -- a breaking change to field names/shape
    # should fail this test loudly, not drift silently.
    findings = [
        _finding(
            check="DISCONTINUITY",
            severity="error",
            episode=12,
            joint="wrist_roll",
            frames=[341],
            message="Single-frame jump of 5.9 rad (98% of range)",
            data={"delta": 5.9, "range_fraction": 0.98},
        )
    ]

    report = render_json_report(findings, repo_id_or_path="lerobot/pusht", profile_name="so101")

    assert set(report.keys()) == {"lelint_version", "schema_version", "dataset", "grade", "summary", "findings"}
    assert set(report["dataset"].keys()) == {"repo_id", "robot_profile"}
    assert set(report["summary"].keys()) == {"errors", "warnings", "info", "episodes_affected"}
    assert set(report["findings"][0].keys()) == {
        "check", "severity", "episode", "joint", "frames", "message", "data",
    }
