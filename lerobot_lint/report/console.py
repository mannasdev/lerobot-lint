"""Terminal report: findings grouped by severity, with a summary count. Rich
for styling; render_console_report returns plain text (via a recording
Console) so it's easy to test and easy to pipe/redirect."""

from collections import Counter

from rich.console import Console

from lerobot_lint.report.finding_summary import finding_summary
from lerobot_lint.types import Finding

SEVERITY_ORDER = ["error", "warning", "info"]
SEVERITY_LABELS = {"error": "Errors", "warning": "Warnings", "info": "Info"}


def render_console_report(findings: list[Finding], repo_id_or_path: str) -> str:
    console = Console(record=True, width=100)
    console.print(f"[bold]lerobot-lint[/bold] — {repo_id_or_path}")
    console.print()

    if not findings:
        console.print("[bold green]Clean![/bold green] No issues found.")
        return console.export_text()

    by_severity = {sev: [f for f in findings if f.severity == sev] for sev in SEVERITY_ORDER}

    for severity in SEVERITY_ORDER:
        group = by_severity[severity]
        if not group:
            continue
        console.print(f"[bold]{SEVERITY_LABELS[severity]} ({len(group)})[/bold]")
        for finding in group:
            console.print(f"  {finding_summary(finding)}", markup=False)
        console.print()

    counts = Counter(f.severity for f in findings)
    summary = ", ".join(f"{counts.get(sev, 0)} {SEVERITY_LABELS[sev].lower()}" for sev in SEVERITY_ORDER)
    console.print(f"[bold]Summary:[/bold] {summary}")

    return console.export_text()
