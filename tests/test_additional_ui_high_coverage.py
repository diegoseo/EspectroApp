import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_dataframe_dialogs_and_table_model(qapp):
    from ui.pages.dataframe_page import (
        FileNameDialog,
        LimitedRangeSampleTypeDialog,
        PandasTableModel,
        RamanRangeDialog,
        SampleTypeDialog,
    )

    frame = pd.DataFrame({"Axis": [100, 200], "A": [1.0, 2.0]})
    model = PandasTableModel(frame)
    assert model.rowCount() == 2
    assert model.columnCount() == 2
    assert model.headerData(0, 1) is not None

    name = FileNameDialog()
    name.input.setText("result")
    assert name.get_name() == "result"
    name.close()

    for dialog_cls in (RamanRangeDialog, SampleTypeDialog, LimitedRangeSampleTypeDialog):
        try:
            dialog = dialog_cls(frame)
        except TypeError:
            try:
                dialog = dialog_cls(["A", "B"])
            except TypeError:
                continue
        assert dialog.layout() is not None
        dialog.close()


def test_fusion_component_selection_and_result_dialogs(qapp, monkeypatch):
    from ui.pages.data_fusion_page import (
        ComponentSelectionDialog,
        FusionPreviewWindow,
        MidLevelResultWindow,
    )

    dialog = ComponentSelectionDialog(
        [np.array([70.0, 30.0]), np.array([60.0])],
        ["ftir.csv", "raman.csv"],
    )
    assert len(dialog.component_items) == 3
    assert dialog.selected_components() == [1, 2]
    dialog.z_combo.setCurrentIndex(1)
    assert dialog.selected_components() == [1, 2, 1]

    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args[-1]),
    )
    dialog._validate_and_accept()
    assert warnings
    dialog.close()

    spectral = pd.DataFrame(
        [
            ["Axis", "A", "B"],
            [100, 1.0, 2.0],
            [200, 2.0, 3.0],
        ]
    )
    preview = FusionPreviewWindow(spectral)
    assert preview.layout() is not None
    preview.close()

    called = []
    result = MidLevelResultWindow(
        pd.DataFrame([[1, 2], [3, 4]]),
        [np.array([70.0, 30.0])],
        ["ftir.csv"],
        lambda: called.append(True),
    )
    assert result.layout() is not None
    result.close()


def test_results_pages_replace_existing_tabs(qapp):
    from ui.pages.spectra_page import SpectraResultsPage
    from ui.pages.dimensionality_page import MultivariateResultsPage

    spectra = SpectraResultsPage(lambda: None)
    first = QWidget()
    second = QWidget()
    spectra.add_plot(first, "Spectrum")
    spectra.add_plot(second, "Spectrum")
    assert spectra.tabs.count() == 1
    assert spectra.plot_widgets["Spectrum"] is second
    spectra.close()

    results = MultivariateResultsPage(lambda: None)
    fig1 = Figure()
    fig2 = Figure()
    results.add_plot(fig1, "PCA")
    results.add_plot(fig2, "PCA")
    assert results.tabs.count() == 1
    assert results.tabs.tabText(0) == "PCA"
    assert not results.activate_figure(fig2)
    assert not results.activate_figure(Figure())
    results.cleanup()
    results.close()
