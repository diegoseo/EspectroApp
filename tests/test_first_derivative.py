import numpy as np
import pandas as pd

from algorithms.preprocessing import calculate_first_derivative


def test_first_derivative_of_linear_signal_is_constant():
    x = np.linspace(0.0, 5.0, 21)
    df = pd.DataFrame({"s1": 4.0 * x + 7.0})

    result = calculate_first_derivative(df, x)

    np.testing.assert_allclose(
        result["s1"],
        4.0,
        atol=1e-10,
    )
