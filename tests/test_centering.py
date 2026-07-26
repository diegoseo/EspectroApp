import numpy as np
import pandas as pd

from algorithms.preprocessing import normalize_by_mean


def test_centering_makes_each_row_mean_zero():
    df = pd.DataFrame(
        {
            "s1": [1.0, 4.0],
            "s2": [3.0, 8.0],
        }
    )

    result = normalize_by_mean(df, "Center to u=0")

    np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-12)
