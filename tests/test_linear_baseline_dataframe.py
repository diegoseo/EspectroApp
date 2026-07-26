import numpy as np
import pandas as pd
import pytest

from algorithms.preprocessing import correct_linear_baseline


def test_correct_linear_baseline_processes_all_columns():
    x = np.arange(5.0)
    df = pd.DataFrame(
        {
            "s1": 2.0 * x + 1.0,
            "s2": -x + 5.0,
        }
    )

    result = correct_linear_baseline(
        df,
        x,
        x_start=0.0,
        x_end=4.0,
    )

    assert result.shape == df.shape
    np.testing.assert_allclose(
        result.to_numpy(),
        0.0,
        atol=1e-12,
    )


def test_correct_linear_baseline_requires_limits(numeric_df):
    x = np.arange(len(numeric_df))

    with pytest.raises(
        ValueError,
        match="requires x_start and x_end",
    ):
        correct_linear_baseline(numeric_df, x)
