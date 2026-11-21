import numpy as np
import pytest
from facecap.pipeline import unit, similarity


def test_unit_normalization():
    assert np.linalg.norm(unit(np.array([3.,4.])))==pytest.approx(1)


def test_similarity_is_cosine_not_probability():
    assert similarity([1,0],[1,0])==pytest.approx(1.0)
    assert similarity([1,0],[0,1])==pytest.approx(0.0)
    assert similarity([1,0],[-1,0])==pytest.approx(-1.0)


def test_reject_zero_vector():
    with pytest.raises(ValueError,match='Zero'):
        unit([0,0])
