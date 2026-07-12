"""Joint-state unit inference. LeRobot's meta/info.json declares no units for
observation.state, and Hub datasets mix at least three conventions: radians,
degrees (classic Koch/SO-10x calibration), and normalized values. An absolute
velocity threshold is meaningless without knowing which, so units are inferred
from the observed state extent unless the user passes --units explicitly."""

import math
from dataclasses import dataclass

import numpy as np

RADIANS = "radians"
DEGREES = "degrees"
NORMALIZED = "normalized"
UNKNOWN = "unknown"

AUTO = "auto"
EXPLICIT_UNITS = (RADIANS, DEGREES, NORMALIZED)

DEGREES_TO_RADIANS = math.pi / 180.0

# Inference boundaries on max |state| across an episode. A limited-turn arm
# joint can't exceed one revolution, so anything past 2pi isn't radians and
# anything past 360 isn't degrees -- at that point it's raw encoder counts (or
# some convention we don't know), and guessing would recreate the false-positive
# storm this module exists to prevent.
_NORMALIZED_MAX = 1.05  # [-1, 1] with float slack
_RADIANS_MAX = 2 * math.pi
_DEGREES_MAX = 400.0  # one revolution with slack; also catches the [-100, 100]
# lerobot calibration convention, which this heuristic can't distinguish from
# small degree motions -- the reason string discloses the assumption.


@dataclass
class UnitsDecision:
    units: str
    reason: str


def infer_units(states: np.ndarray) -> UnitsDecision:
    """Classify joint-state units from the largest absolute value observed."""
    max_abs = float(np.max(np.abs(states))) if states.size else 0.0

    if max_abs <= _NORMALIZED_MAX:
        return UnitsDecision(NORMALIZED, f"max |state| = {max_abs:.3f} fits [-1, 1]")
    if max_abs <= _RADIANS_MAX:
        return UnitsDecision(RADIANS, f"max |state| = {max_abs:.2f} fits one revolution (2pi rad)")
    if max_abs <= _DEGREES_MAX:
        return UnitsDecision(
            DEGREES, f"max |state| = {max_abs:.1f} exceeds 2pi rad but fits one revolution in degrees"
        )
    return UnitsDecision(
        UNKNOWN, f"max |state| = {max_abs:.0f} exceeds any angle convention -- likely raw encoder counts"
    )
