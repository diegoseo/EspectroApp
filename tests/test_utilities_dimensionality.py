import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import algorithms.dimensionality as dimensionality
from algorithms.dimensionality import pca, tsne, tsne_pca
from algorithms.utilities import (
    assign_type_colors,
    calculate_cumulative_variance,
    plot_heatmap_pca,
    prepare_pca_matrix,
)


def internal_dataframe():
    return pd.DataFrame(
        [
            ['X Axis', 'A', 'A', 'B', 'B'],
            [100.0, 1.0, 1.2, 4.0, 4.2],
            [200.0, 2.0, 2.2, 5.0, 5.2],
            [300.0, 3.0, 3.2, 6.0, 6.2],
        ]
    )


def test_prepare_pca_matrix_transposes_samples():
    matrix = prepare_pca_matrix(internal_dataframe())
    assert matrix.shape == (4, 3)
    np.testing.assert_allclose(matrix[0], [1.0, 2.0, 3.0])


def test_prepare_pca_matrix_validates_input():
    with pytest.raises(ValueError, match='empty'):
        prepare_pca_matrix(pd.DataFrame())

    bad = internal_dataframe()
    bad.iloc[2, 2] = None
    with pytest.raises(ValueError, match='missing'):
        prepare_pca_matrix(bad)

    one_sample = internal_dataframe().iloc[:, :2]
    with pytest.raises(ValueError, match='At least two'):
        prepare_pca_matrix(one_sample)


def test_pca_shapes_variance_and_invalid_components():
    X = prepare_pca_matrix(internal_dataframe())
    scores, variance = pca(X, 2)

    assert scores.shape == (4, 2)
    assert variance.shape == (2,)
    assert 99.0 <= variance.sum() <= 100.0001

    with pytest.raises(ValueError, match='between 2'):
        pca(X, 1)
    with pytest.raises(ValueError):
        pca(X, 10)


def test_cumulative_variance_and_type_colors_are_deterministic():
    individual, cumulative, count = calculate_cumulative_variance(internal_dataframe(), 95)
    assert len(individual) == len(cumulative)
    assert cumulative[-1] == pytest.approx(100.0)
    assert 1 <= count <= len(individual)

    first = assign_type_colors(['B', 'A', 'B'])
    second = assign_type_colors(['A', 'B'])
    assert first == second
    assert list(first) == ['A', 'B']


def test_heatmap_returns_matplotlib_figure_with_selected_components():
    scores = np.array([[1, 2, 3], [2, 3, 4], [8, 7, 6]])
    fig = plot_heatmap_pca(scores, ['B', 'A', 'B'], [1, 3, 99])
    assert isinstance(fig, Figure)
    assert fig.axes


def test_tsne_builds_reproducible_model_and_validates_parameters(monkeypatch):
    captured = {}

    class FakeTSNE:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit_transform(self, data):
            return np.zeros((len(data), captured["n_components"]))

    monkeypatch.setattr(dimensionality, "TSNE", FakeTSNE)
    X = np.arange(60, dtype=float).reshape(12, 5)

    result = tsne(X, 2, perplexity=50, learning_rate=150, max_iter=250)
    assert result.shape == (12, 2)
    assert captured["random_state"] == 42
    assert captured["perplexity"] == 11.0
    assert captured["max_iter"] == 250

    with pytest.raises(ValueError, match="2 or 3"):
        tsne(X, 4, max_iter=250)
    with pytest.raises(ValueError, match="at least 250"):
        tsne(X, 2, max_iter=100)
    with pytest.raises(ValueError, match="greater than 0"):
        tsne(X, 2, perplexity=0, max_iter=250)


def test_tsne_after_pca_passes_results_to_tsne(monkeypatch):
    calls = {}

    def fake_pca(data, components):
        calls["pca_components"] = components
        return np.ones((len(data), components)), np.array([100.0 / components] * components)

    def fake_tsne(data, **kwargs):
        calls["tsne_shape"] = data.shape
        calls["tsne_kwargs"] = kwargs
        return np.zeros((len(data), kwargs["n_componentes"]))

    monkeypatch.setattr(dimensionality, "pca", fake_pca)
    monkeypatch.setattr(dimensionality, "tsne", fake_tsne)

    X = np.zeros((15, 6))
    result = tsne_pca(X, cp_pca=3, cp_tsne=2, perplexity=4, max_iter=250)

    assert result.shape == (15, 2)
    assert calls["pca_components"] == 3
    assert calls["tsne_shape"] == (15, 3)
    assert calls["tsne_kwargs"]["n_componentes"] == 2

