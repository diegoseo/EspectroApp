import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from algorithms.clustering import _composition_text, calculate_hca


def hca_dataframe():
    return pd.DataFrame(
        {
            'x': [100, 200, 300, 400],
            's1': [1.0, 1.3, 0.8, 1.1],
            's2': [1.2, 0.9, 1.4, 0.7],
            's3': [5.0, 5.6, 4.7, 5.2],
            's4': [5.4, 4.8, 5.1, 4.5],
        }
    )


def test_composition_text_preserves_order_and_counts_unknown():
    assert _composition_text(['A', 'B', 'A', np.nan]) == '2 A, 1 B, 1 Unknown'


@pytest.mark.parametrize('distance', ['Euclidiana', 'Manhattan', 'Coseno', 'Chebyshev', 'Pearson', 'Spearman'])
def test_hca_supported_distances_return_figure_and_complete_table(distance):
    options = {distance: True, 'Average Linkage': True, 'Numero Clusters': 2}
    fig, table = calculate_hca(hca_dataframe(), None, options, ['A', 'A', 'B', 'B'])

    assert isinstance(fig, Figure)
    assert fig.axes[0].lines or fig.axes[0].collections
    assert list(table.columns) == ['Cluster', 'Label', 'Size', 'Composition']
    assert table['Size'].sum() == 4
    assert set(table['Composition'].str.contains('A|B')) == {True}


def test_hca_rejects_invalid_configuration_and_too_few_samples():
    with pytest.raises(ValueError, match='at least two'):
        calculate_hca(hca_dataframe()[['x', 's1']], None, {'Euclidiana': True, 'Ward': True}, ['A'])

    with pytest.raises(ValueError, match='distance'):
        calculate_hca(hca_dataframe(), None, {'Unknown': True, 'Ward': True}, ['A'] * 4)

    with pytest.raises(ValueError, match='linkage'):
        calculate_hca(hca_dataframe(), None, {'Euclidiana': True, 'Unknown': True}, ['A'] * 4)


def test_hca_rejects_non_finite_correlation_distance():
    constant = pd.DataFrame({'x': [1, 2, 3], 'a': [1, 1, 1], 'b': [1, 1, 1]})
    with pytest.raises(ValueError, match='non-finite'):
        calculate_hca(constant, None, {'Pearson': True, 'Average Linkage': True}, ['A', 'B'])
