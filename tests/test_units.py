import numpy as np

from lerobot_lint.units import DEGREES, NORMALIZED, RADIANS, UNKNOWN, infer_units


def test_infers_normalized_when_states_fit_the_unit_interval():
    states = np.linspace(-1.0, 1.0, 200, dtype=np.float32).reshape(100, 2)

    decision = infer_units(states)

    assert decision.units == NORMALIZED
    assert decision.reason  # human-readable justification, shown in the report


def test_infers_radians_when_states_exceed_unit_interval_but_fit_one_revolution():
    states = np.linspace(-3.0, 3.0, 200, dtype=np.float32).reshape(100, 2)

    decision = infer_units(states)

    assert decision.units == RADIANS


def test_infers_degrees_when_states_exceed_two_pi_but_fit_one_revolution_in_degrees():
    # the real koch_pick_place_5_lego_random_pose case: joint positions like
    # -36..41 are impossible in radians for a limited-turn joint
    states = np.linspace(-36.0, 41.0, 200, dtype=np.float32).reshape(100, 2)

    decision = infer_units(states)

    assert decision.units == DEGREES


def test_infers_unknown_for_raw_encoder_count_scale_values():
    states = np.linspace(0.0, 4096.0, 200, dtype=np.float32).reshape(100, 2)

    decision = infer_units(states)

    assert decision.units == UNKNOWN


def test_inference_uses_magnitude_not_sign():
    # all-negative degree-scale data must still read as degrees
    states = np.linspace(-300.0, -200.0, 200, dtype=np.float32).reshape(100, 2)

    decision = infer_units(states)

    assert decision.units == DEGREES
