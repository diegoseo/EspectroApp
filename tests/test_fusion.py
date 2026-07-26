import numpy as np
import pandas as pd
import pytest

from algorithms.fusion import (
    _normalize_component_counts,
    _prepare_mid_level_pca_input,
    cortar_df_rango_comun,
    obtener_min_max_eje_x,
    sort_samples,
    val_ejex,
)


def spectral_df(labels, x=(100, 200, 300), offset=0.0):
    rows = [['X Axis', *labels]]
    for i, value in enumerate(x):
        rows.append([value, *[offset + i + j for j in range(len(labels))]])
    return pd.DataFrame(rows)


def test_sort_samples_reorders_repeated_labels_without_dropping_columns():
    first = spectral_df(['A', 'A', 'B'])
    second = spectral_df(['B', 'A', 'A'], offset=10)
    original_width = second.shape[1]
    datasets = [first, second]

    ranges, intersects, common, labels = sort_samples(datasets)

    assert labels == ['A', 'A', 'B']
    assert datasets[1].shape[1] == original_width
    assert datasets[1].iloc[0, 1:].tolist() == [11, 12, 10]
    assert intersects is True
    assert common == (100.0, 300.0)
    assert ranges == [(100.0, 300.0), (100.0, 300.0)]


def test_sort_samples_preserves_position_for_different_labels():
    first = spectral_df(['A', 'B'])
    second = spectral_df(['X', 'Y'])
    datasets = [first, second]
    sort_samples(datasets)
    assert datasets[1].iloc[0, 1:].tolist() == [0, 1]


def test_val_ejex_sorts_axes_and_detects_no_intersection():
    first = spectral_df(['A', 'B'], x=(300, 100, 200))
    second = spectral_df(['A', 'B'], x=(500, 600, 700))
    datasets = [first, second]

    ranges, intersects, common = val_ejex(datasets)

    assert ranges == [(100.0, 300.0), (500.0, 700.0)]
    assert intersects is False
    assert common is None
    assert datasets[0].iloc[:, 0].tolist() == [100.0, 200.0, 300.0]


def test_mid_level_input_and_component_counts_validation():
    df = spectral_df(['A', 'B', 'C'])
    X, labels = _prepare_mid_level_pca_input(df)
    assert X.shape == (3, 3)
    assert labels == ['A', 'B', 'C']

    assert _normalize_component_counts(2, 3) == [2, 2, 2]
    assert _normalize_component_counts([2, 3], 2) == [2, 3]

    with pytest.raises(ValueError):
        _normalize_component_counts([2], 2)
    with pytest.raises(ValueError):
        _normalize_component_counts(1, 2)


def test_range_helpers():
    df = spectral_df(['A', 'B'], x=(100, 200, 300, 400))
    cut_list = cortar_df_rango_comun([df], (150, 350), True, False)
    cut = cut_list[0]
    assert cut.iloc[1:, 0].astype(float).tolist() == [200.0, 300.0]
    assert obtener_min_max_eje_x(df) == (100.0, 400.0)

    assert cortar_df_rango_comun([df], (150, 350), False, True)[0] is df
    with pytest.raises(ValueError):
        cortar_df_rango_comun([df], (150, 350), False, False)
