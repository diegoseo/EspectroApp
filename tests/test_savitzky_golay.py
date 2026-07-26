import numpy as np
import pandas as pd

from algorithms.preprocessing import smooth_savitzky_golay


def test_savitzky_golay_preserves_shape_and_columns(numeric_df):
    result = smooth_savitzky_golay(
        numeric_df,
        ventana=5,
        orden=2,
    )

    assert result.shape == numeric_df.shape
    assert list(result.columns) == list(numeric_df.columns)


def test_savitzky_golay_preserves_linear_signal():
    df = pd.DataFrame({"s1": np.arange(7.0)})

    result = smooth_savitzky_golay(
        df,
        ventana=5,
        orden=2,
    )

    np.testing.assert_allclose(
        result["s1"],
        df["s1"],
        atol=1e-12,
    )
