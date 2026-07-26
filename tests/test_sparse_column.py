import numpy as np
import pandas as pd

from algorithms.preprocessing import get_column_with_fewest_rows


def test_get_column_with_fewest_rows_identifies_sparse_column():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1, np.nan, np.nan],
        }
    )

    column, count = get_column_with_fewest_rows(df)

    assert column == "b"
    assert count == 1
