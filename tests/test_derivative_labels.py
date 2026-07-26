import numpy as np
import pandas as pd

from algorithms.preprocessing import (
    calculate_first_derivative,
    calculate_second_derivative,
)


def test_derivative_functions_preserve_labels():
    x = np.arange(5.0)
    df = pd.DataFrame(
        {
            "A": x,
            "B": x**2,
        },
        index=[10, 11, 12, 13, 14],
    )

    first = calculate_first_derivative(df, x)
    second = calculate_second_derivative(df, x)

    assert first.index.equals(df.index)
    assert second.index.equals(df.index)
    assert list(first.columns) == list(df.columns)
    assert list(second.columns) == list(df.columns)
