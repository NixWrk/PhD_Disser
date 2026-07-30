import pytest

from breathgeom.measure.elastix_registration import ElastixParams


def test_elastix_parameter_sequences_must_match() -> None:
    with pytest.raises(ValueError, match="equal length"):
        ElastixParams(stages=("rigid", "bspline"), iterations=(10,))


def test_elastix_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="stages"):
        ElastixParams(stages=("magic",), iterations=(10,))
