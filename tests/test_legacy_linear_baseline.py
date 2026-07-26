import numpy as np

from algorithms.preprocessing import linear_baseline_correction


def test_legacy_linear_baseline_removes_endpoint_line():
    x = np.arange(6.0)
    y = 3.0 * x + 2.0

    result = linear_baseline_correction(y, x)

    np.testing.assert_allclose(
        result,
        0.0,
        atol=1e-12,
    )
