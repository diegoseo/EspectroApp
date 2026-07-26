import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def spectral_dataset():
    axis = np.linspace(400.0, 1800.0, 41)
    labels = ["A", "A", "A", "B", "B", "B"]

    rows = [["Wavenumber", *labels]]
    for index, x_value in enumerate(axis):
        rows.append(
            [
                float(x_value),
                np.sin(index / 5.0) + 1.0,
                np.sin(index / 5.0) + 1.2,
                np.sin(index / 5.0) + 0.8,
                np.cos(index / 6.0) + 2.0,
                np.cos(index / 6.0) + 2.2,
                np.cos(index / 6.0) + 1.8,
            ]
        )

    return pd.DataFrame(rows), axis, {"A": "#1f77b4", "B": "#ff7f0e"}


def _close_widget(widget):
    widget.close()
    widget.deleteLater()


def test_all_main_spectral_plot_windows_can_be_constructed(qapp, spectral_dataset):
    from plotting import (
        SpectraPlotWindow,
        StackedSpectraPlotWindow,
        LimitedRangeSpectraPlotWindow,
        SpectraByTypePlotWindow,
        LimitedRangeSpectraByTypePlotWindow,
    )

    data, axis, colors = spectral_dataset

    widgets = [
        SpectraPlotWindow(data, axis, colors),
        StackedSpectraPlotWindow(
            data,
            axis,
            colors,
            offset_mode="automatic",
            show_labels=True,
            maximum_spectra=6,
        ),
        LimitedRangeSpectraPlotWindow(
            data,
            axis,
            colors,
            700.0,
            1500.0,
        ),
        SpectraByTypePlotWindow(
            data,
            axis,
            colors,
            "A",
        ),
        LimitedRangeSpectraByTypePlotWindow(
            data,
            axis,
            colors,
            "B",
            700.0,
            1500.0,
        ),
    ]

    try:
        for widget in widgets:
            assert widget.windowTitle()
            assert widget.layout() is not None

            if hasattr(widget, "plot_widget"):
                assert widget.plot_widget.listDataItems()
            else:
                assert hasattr(widget, "axes")
                assert len(widget.axes.lines) > 0
    finally:
        for widget in widgets:
            _close_widget(widget)


@pytest.mark.parametrize(
    ("offset_mode", "offset_value", "sample_type", "range_min", "range_max"),
    [
        ("automatic", 1.15, None, None, None),
        ("manual", 2.5, "A", None, None),
        ("automatic", 1.10, "B", 650.0, 1300.0),
    ],
)
def test_stacked_plot_configuration_branches(
    qapp,
    spectral_dataset,
    offset_mode,
    offset_value,
    sample_type,
    range_min,
    range_max,
):
    from plotting import StackedSpectraPlotWindow

    data, axis, colors = spectral_dataset
    widget = StackedSpectraPlotWindow(
        data,
        axis,
        colors,
        offset_mode=offset_mode,
        offset_value=offset_value,
        show_labels=False,
        maximum_spectra=4,
        sample_type=sample_type,
        range_min=range_min,
        range_max=range_max,
    )

    try:
        assert hasattr(widget, "axes")
        assert len(widget.axes.lines) > 0
    finally:
        _close_widget(widget)


def test_accuracy_valid_and_guard_branches():
    from plotting import calculate_accuracy

    rng = np.random.default_rng(42)

    class_a = rng.normal(loc=-2.0, scale=0.15, size=(10, 2))
    class_b = rng.normal(loc=2.0, scale=0.15, size=(10, 2))
    matrix = np.vstack([class_a, class_b])
    labels = ["A"] * 10 + ["B"] * 10

    score = calculate_accuracy(
        pd.DataFrame(matrix, columns=["PC1", "PC2"]),
        labels,
    )

    assert 90.0 <= score <= 100.0
    assert calculate_accuracy(pd.DataFrame({"label": ["x", "y"]}), ["A", "B"]) == 0.0
    assert calculate_accuracy(pd.DataFrame({"PC1": [1.0, 2.0, 3.0]}), ["A"] * 3) == 0.0
    assert calculate_accuracy(
        pd.DataFrame({"PC1": [1.0, 2.0, 3.0, 4.0]}),
        ["A", "A", "A", "A"],
    ) == 0.0


def test_legacy_accuracy_returns_percentage():
    from plotting import calculate_accuracyviejo

    data = pd.DataFrame(
        {
            "PC1": [-3, -2, -1, -0.5, 0.5, 1, 2, 3],
            "PC2": [-2, -1, -1, -0.2, 0.2, 1, 1, 2],
        },
        dtype=float,
    )
    labels = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    score = calculate_accuracyviejo(data, labels)

    assert 0.0 <= score <= 100.0


def test_cumulative_variance_figure_contains_expected_artists():
    from plotting import graficar_varianza_acumulada

    individual = np.array([55.0, 25.0, 12.0, 5.0, 3.0])
    cumulative = np.cumsum(individual)

    figure = graficar_varianza_acumulada(
        cumulative,
        individual,
        umbral=90,
        max_cp=5,
        anotar=True,
    )

    try:
        axis = figure.axes[0]
        assert len(axis.patches) == 5
        assert len(axis.lines) >= 3
        assert axis.get_ylim()[1] == 105
    finally:
        figure.clear()


def test_plotting_style_helpers(qapp):
    import pyqtgraph as pg
    from plotting import apply_color_alpha, apply_nature_style_pg

    color = apply_color_alpha("#336699", alpha=77)
    assert color.alpha() == 77

    widget = pg.PlotWidget()
    try:
        apply_nature_style_pg(widget, "Wavenumber", "Intensity")
        assert widget.getAxis("bottom").labelText == "Wavenumber"
        assert widget.getAxis("left").labelText == "Intensity"
    finally:
        _close_widget(widget)
