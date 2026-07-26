import numpy as np
import pandas as pd

from algorithms.preprocessing import smooth_gaussian_filter


def test_gaussian_filter_preserves_constant_signal():
    df = pd.DataFrame({"s1": np.full(9, 4.0)})

    result = smooth_gaussian_filter(df, sigma=1.5)

    np.testing.assert_allclose(
        result["s1"],
        4.0,
        atol=1e-12,
    )


def test_gaussian_filter_preserves_shape_and_columns(numeric_df):
    result = smooth_gaussian_filter(
        numeric_df,
        sigma=1.0,
    )

    assert result.shape == numeric_df.shape
    assert list(result.columns) == list(numeric_df.columns)
