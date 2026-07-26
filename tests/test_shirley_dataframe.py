import numpy as np
import pandas as pd

from algorithms.preprocessing import correct_shirley_baseline


def test_correct_shirley_baseline_preserves_structure():
    x = np.linspace(0.0, 10.0, 51)
    peak = np.exp(-((x - 5.0) ** 2))

    df = pd.DataFrame(
        {
            "s1": 1.0 + peak,
            "s2": 2.0 + 2.0 * peak,
        }
    )

    result = correct_shirley_baseline(
        df,
        x,
        1.0,
        9.0,
    )

    assert result.shape == df.shape
    assert list(result.columns) == list(df.columns)
