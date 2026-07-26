import numpy as np
import pandas as pd
import pytest

from algorithms.preprocessing import normalize_by_area


def test_area_normalization_produces_unit_area():
    shift = pd.Series([3.0, 2.0, 1.0, 0.0])
    df = pd.DataFrame({"s1": [2.0, 2.0, 2.0, 2.0]})

    result = normalize_by_area(df, shift)

    area = (
        np.trapezoid(
            result["s1"].to_numpy(),
            shift.to_numpy(),
        )
        * -1
    )

    assert area == pytest.approx(1.0)


def test_area_normalization_keeps_zero_signal_unchanged():
    shift = pd.Series([3.0, 2.0, 1.0])
    df = pd.DataFrame({"s1": [0.0, 0.0, 0.0]})

    result = normalize_by_area(df, shift)

    pd.testing.assert_series_equal(result["s1"], df["s1"])
