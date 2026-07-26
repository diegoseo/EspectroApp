import pandas as pd
import pytest

from algorithms.preprocessing import smooth_moving_average


def test_moving_average_expected_center_value():
    df = pd.DataFrame({"s1": [1.0, 2.0, 9.0, 4.0, 5.0]})

    result = smooth_moving_average(df, ventana=3)

    assert result.loc[2, "s1"] == pytest.approx((2.0 + 9.0 + 4.0) / 3.0)


def test_moving_average_does_not_create_edge_nan_values():
    df = pd.DataFrame({"s1": [1.0, 2.0, 3.0]})

    result = smooth_moving_average(df, ventana=3)

    assert not result.isna().any().any()
