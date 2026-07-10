"""Shared Finding-to-text formatting, used by the console reporter and the
future bug-card renderer. Strips control characters at the source (not just
at whichever consumer got security-reviewed first) since dataset-derived
strings (task descriptions, messages built from them) originate from
arbitrary, potentially attacker-controlled Hub datasets. Every consumer of
this function inherits the same protection -- none should ever execute or
display untrusted control/escape sequences."""

import re

from lerobot_lint.types import Finding

_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _strip_control_characters(text: str) -> str:
    return _CONTROL_CHAR_PATTERN.sub("", text)


def finding_summary(finding: Finding) -> str:
    scope = f"episode {finding.episode}" if finding.episode is not None else "dataset"
    message = _strip_control_characters(finding.message)
    return f"[{finding.check}] {scope}: {message}"
