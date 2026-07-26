import numpy as np
import pandas as pd

from algorithms.preprocessing import calculate_second_derivative


def test_second_derivative_of_quadratic_signal_is_two():
    x = np.linspace(-5.0, 5.0, 101)
    df = pd.DataFrame({"s1": x**2})

    result = calculate_second_derivative(df, x)

    np.testing.assert_allclose(
        result["s1"].iloc[2:-2],
        2.0,
        atol=1e-9,
    )
