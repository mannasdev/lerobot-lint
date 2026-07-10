import pytest

from lerobot_lint.types import Finding


def test_finding_holds_fields():
    f = Finding(
        check="DEAD_JOINT",
        severity="error",
        episode=3,
        joint="shoulder_pan",
        frames=[10, 11, 12],
        message="shoulder_pan never moved while other joints did",
        data={"std": 0.0001},
    )
    assert f.check == "DEAD_JOINT"
    assert f.severity == "error"
    assert f.episode == 3
    assert f.joint == "shoulder_pan"
    assert f.frames == [10, 11, 12]
    assert f.data["std"] == 0.0001


def test_finding_rejects_invalid_severity():
    with pytest.raises(ValueError, match="severity"):
        Finding(
            check="DEAD_JOINT",
            severity="catastrophic",
            episode=3,
            joint=None,
            frames=[],
            message="x",
            data={},
        )
