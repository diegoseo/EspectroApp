import numpy as np

from algorithms.preprocessing import normalize_by_mean


def test_normalize_minus_one_one_reaches_expected_limits(numeric_df):
    result = normalize_by_mean(
        numeric_df,
        "Normalize to interval [-1,1]",
    )

    np.testing.assert_allclose(result.min(axis=0), -1.0)
    np.testing.assert_allclose(result.max(axis=0), 1.0)
