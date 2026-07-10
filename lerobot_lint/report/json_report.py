"""Stable JSON report schema (source spec §5). schema_version bumps on any
breaking change -- this is meant to be "the seed of the future eval-CI
product," held stable like a real contract, not allowed to drift silently."""

from collections import Counter
from importlib.metadata import version as pkg_version
from typing import Any

from lerobot_lint.report.finding_summary import sanitize_text
from lerobot_lint.types import Finding

SCHEMA_VERSION = 1


def render_json_report(
    findings: list[Finding],
    repo_id_or_path: str,
    profile_name: str,
) -> dict[str, Any]:
    counts = Counter(f.severity for f in findings)
    episodes_affected = len({f.episode for f in findings if f.episode is not None})

    return {
        "lelint_version": pkg_version("lerobot-lint"),
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "repo_id": repo_id_or_path,
            "robot_profile": profile_name,
        },
        "grade": None,  # A-F scorecard formula not yet designed/calibrated (needs the field study)
        "summary": {
            "errors": counts.get("error", 0),
            "warnings": counts.get("warning", 0),
            "info": counts.get("info", 0),
            "episodes_affected": episodes_affected,
        },
        "findings": [
            {
                "check": f.check,
                "severity": f.severity,
                "episode": f.episode,
                "joint": f.joint,
                "frames": f.frames,
                "message": sanitize_text(f.message),
                "data": f.data,
            }
            for f in findings
        ],
    }
