import numpy as np
import pytest

from algorithms.preprocessing import linear_baseline_from_points


def test_linear_baseline_from_points_removes_straight_line():
    x = np.arange(5.0)
    y = 2.0 * x + 3.0

    corrected, baseline = linear_baseline_from_points(
        x,
        y,
        0.0,
        4.0,
    )

    np.testing.assert_allclose(baseline, y)
    np.testing.assert_allclose(corrected, 0.0, atol=1e-12)


def test_linear_baseline_rejects_equal_limits():
    x = np.arange(5.0)
    y = np.arange(5.0)

    with pytest.raises(ValueError, match="must be different"):
        linear_baseline_from_points(
            x,
            y,
            2.0,
            2.0,
        )
