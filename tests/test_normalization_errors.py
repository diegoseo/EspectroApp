import numpy as np
import pandas as pd
import pytest

from algorithms.preprocessing import normalize_by_mean


def test_normalization_rejects_missing_values():
    df = pd.DataFrame({"s1": [1.0, np.nan, 3.0]})

    with pytest.raises(ValueError, match="numeric data"):
        normalize_by_mean(df, "Center to u=0")


def test_normalization_rejects_unknown_method(numeric_df):
    with pytest.raises(
        ValueError,
        match="Unsupported normalization method",
    ):
        normalize_by_mean(numeric_df, "unknown")
