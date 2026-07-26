import numpy as np
import pandas as pd

from algorithms.preprocessing import normalize_by_mean


def test_standardization_centers_and_scales_each_row():
    df = pd.DataFrame(
        {
            "s1": [1.0, 2.0],
            "s2": [3.0, 6.0],
            "s3": [5.0, 10.0],
        }
    )

    result = normalize_by_mean(df, "Standardize u=0, v2=1")

    np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(result.std(axis=1, ddof=1), 1.0, atol=1e-12)
