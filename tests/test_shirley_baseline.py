import numpy as np
import pytest

from algorithms.preprocessing import shirley_baseline_from_points


def test_shirley_baseline_returns_finite_arrays():
    x = np.linspace(0.0, 10.0, 101)
    y = 1.0 + np.exp(-((x - 5.0) ** 2))

    corrected, baseline, info = shirley_baseline_from_points(
        x,
        y,
        1.0,
        9.0,
        return_info=True,
    )

    assert corrected.shape == y.shape
    assert baseline.shape == y.shape
    assert np.isfinite(corrected).all()
    assert np.isfinite(baseline).all()
    assert 1 <= info["iterations"] <= 100


def test_shirley_baseline_rejects_nonpositive_tolerance():
    x = np.arange(5.0)
    y = np.array([1.0, 2.0, 4.0, 2.0, 1.0])

    with pytest.raises(ValueError, match="tolerance"):
        shirley_baseline_from_points(
            x,
            y,
            0.0,
            4.0,
            tolerance=0,
        )
