import numpy as np
import pandas as pd

from algorithms.preprocessing import normalize_by_mean


def test_scaling_makes_each_row_standard_deviation_one():
    df = pd.DataFrame(
        {
            "s1": [1.0, 2.0],
            "s2": [3.0, 6.0],
            "s3": [5.0, 10.0],
        }
    )

    result = normalize_by_mean(df, "Scale to v2=1")

    np.testing.assert_allclose(result.std(axis=1, ddof=1), 1.0, atol=1e-12)
