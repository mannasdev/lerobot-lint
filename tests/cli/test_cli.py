import json

from typer.testing import CliRunner

from lerobot_lint.cli import app

runner = CliRunner()


def test_check_writes_json_report_when_json_flag_passed(tmp_path):
    out_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        ["check", "lerobot/pusht", "--episodes", "0:2", "--no-video", "--json", str(out_path)],
    )

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["schema_version"] == 1
    assert data["dataset"]["repo_id"] == "lerobot/pusht"
    # console report is still printed too, JSON is additional not exclusive
    assert "lerobot-lint" in result.stdout


def test_check_on_a_bad_repo_id_reports_a_clean_error_not_a_traceback():
    result = runner.invoke(app, ["check", "not-a-real-repo/does-not-exist-xyz"])

    assert result.exit_code == 2
    assert "Traceback" not in result.stdout
    assert "Traceback" not in (result.stderr or "")


def test_check_verbose_flag_includes_debug_context_on_load_error():
    result = runner.invoke(app, ["check", "not-a-real-repo/does-not-exist-xyz", "--verbose"])

    assert result.exit_code == 2
    combined = (result.stdout or "") + (result.stderr or "")
    assert "lerobot-lint" in combined  # lelint version info shown
    assert "python" in combined.lower()  # python version shown


def test_check_fail_on_warning_escalates_a_warning_only_run_to_exit_2(monkeypatch):
    from lerobot_lint import cli as cli_module
    from lerobot_lint.types import Finding

    warning_finding = Finding(
        check="SOME_CHECK", severity="warning", episode=0, joint=None, frames=[], message="x", data={}
    )
    monkeypatch.setattr(cli_module, "check_dataset", lambda *a, **k: [warning_finding])

    result = runner.invoke(app, ["check", "lerobot/pusht", "--profile", "default", "--fail-on", "warning"])

    assert "No such option" not in result.output
    assert result.exit_code == 2


def test_check_rejects_an_invalid_fail_on_value_cleanly_without_touching_the_dataset(monkeypatch):
    from lerobot_lint import cli as cli_module

    def _fail_if_called(*a, **k):
        raise AssertionError("check_dataset should not be called for an invalid --fail-on value")

    monkeypatch.setattr(cli_module, "check_dataset", _fail_if_called)

    result = runner.invoke(app, ["check", "lerobot/pusht", "--fail-on", "catastrophic"])

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert "KeyError" not in result.output
    assert "Invalid value" in result.output


def test_check_discloses_an_explicit_profile(monkeypatch):
    from lerobot_lint import cli as cli_module

    monkeypatch.setattr(cli_module, "check_dataset", lambda *a, **k: [])

    result = runner.invoke(app, ["check", "lerobot/pusht", "--profile", "so101"])

    assert "Using profile: so101 (explicit, via --profile)" in result.output


def test_check_discloses_an_auto_detected_profile_when_none_is_passed(monkeypatch):
    from lerobot_lint import cli as cli_module

    monkeypatch.setattr(cli_module, "check_dataset", lambda *a, **k: [])
    monkeypatch.setattr(cli_module, "get_joint_names", lambda repo_id: cli_module.load_profile("koch").joint_names)

    result = runner.invoke(app, ["check", "lerobot/pusht"])

    assert "Using profile: koch (auto-detected from joint names, override with --profile)" in result.output


def test_check_discloses_a_fallback_to_default_when_nothing_matches(monkeypatch):
    from lerobot_lint import cli as cli_module

    monkeypatch.setattr(cli_module, "check_dataset", lambda *a, **k: [])
    monkeypatch.setattr(cli_module, "get_joint_names", lambda repo_id: ["motor_0", "motor_1"])

    result = runner.invoke(app, ["check", "lerobot/pusht"])

    assert "Using profile: default (no match, override with --profile)" in result.output


def test_profiles_command_lists_all_built_in_profiles():
    result = runner.invoke(app, ["profiles"])

    assert result.output.split() == ["default", "koch", "so101"]


def test_check_rejects_an_unknown_explicit_profile_cleanly():
    result = runner.invoke(app, ["check", "lerobot/pusht", "--profile", "not_a_real_robot"])

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert "unknown profile" in result.output
