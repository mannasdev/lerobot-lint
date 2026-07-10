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
