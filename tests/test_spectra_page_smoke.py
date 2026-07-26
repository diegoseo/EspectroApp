import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QWidget


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def internal_dataset():
    axis = np.linspace(400.0, 1800.0, 21)
    rows = [["Wavenumber", "A", "A", "B", "B"]]

    for index, value in enumerate(axis):
        rows.append(
            [
                float(value),
                1.0 + index * 0.1,
                1.2 + index * 0.1,
                2.0 + index * 0.2,
                2.2 + index * 0.2,
            ]
        )

    return pd.DataFrame(rows)


def _close(widget):
    widget.close()
    widget.deleteLater()


def test_spectra_results_page_adds_and_replaces_tabs(qapp):
    from ui.pages.spectra_page import SpectraResultsPage

    page = SpectraResultsPage(lambda: None)

    first = QWidget()
    second = QWidget()

    try:
        page.add_plot(first, "Full spectra")
        assert page.tabs.count() == 1
        assert page.tabs.tabText(0) == "Full spectra"

        page.add_plot(second, "Full spectra")
        assert page.tabs.count() == 1
        assert page.plot_widgets["Full spectra"] is second
    finally:
        _close(page)


def test_spectra_export_options_constructor_populates_dataset_controls(
    qapp,
    internal_dataset,
):
    from ui.pages.spectra_page import SpectraExportOptionsWindow

    page = SpectraExportOptionsWindow(
        [internal_dataset],
        ["/tmp/example_spectra.csv"],
    )

    try:
        assert page.combo_archivo.count() == 1
        assert page.combo_sample_type.count() == 2
        assert page.combo_export_sample_type.count() == 2
        assert float(page.input_range_min.text()) == 400.0
        assert float(page.input_range_max.text()) == 1800.0
        assert page.input_export_file_name.text() == "example_spectra_export.csv"
    finally:
        _close(page)


def test_spectra_export_visibility_branches(qapp, internal_dataset):
    from ui.pages.spectra_page import SpectraExportOptionsWindow

    page = SpectraExportOptionsWindow([internal_dataset], ["dataset.csv"])
    page.show()
    qapp.processEvents()

    try:
        page.check_limited_plot.setChecked(True)
        page.update_parameter_visibility()
        qapp.processEvents()
        assert not page.range_group.isHidden()

        page.check_type_plot.setChecked(True)
        page.update_parameter_visibility()
        qapp.processEvents()
        assert not page.type_group.isHidden()

        page.check_stacked_plot.setChecked(True)
        page.check_stacked_limited.setChecked(True)
        page.check_stacked_by_type.setChecked(True)
        page.update_parameter_visibility()
        qapp.processEvents()
        assert not page.stacked_group.isHidden()
        assert not page.range_group.isHidden()
        assert not page.type_group.isHidden()
    finally:
        _close(page)


def test_spectra_export_invalid_dataset_index_clears_fields(qapp, internal_dataset):
    from ui.pages.spectra_page import SpectraExportOptionsWindow

    page = SpectraExportOptionsWindow([internal_dataset], ["dataset.csv"])

    try:
        page.update_dataset_parameters(-1)
        assert page.input_range_min.text() == ""
        assert page.input_range_max.text() == ""
        assert page.combo_sample_type.count() == 0
    finally:
        _close(page)


def test_spectra_export_confirm_emits_configuration(qapp, internal_dataset):
    from ui.pages.spectra_page import SpectraExportOptionsWindow

    page = SpectraExportOptionsWindow([internal_dataset], ["dataset.csv"])
    emitted = []
    page.seleccion_confirmada.connect(lambda index, options: emitted.append((index, options)))

    try:
        page.check_full_plot.setChecked(True)
        page.confirm_selection()

        assert len(emitted) == 1
        index, options = emitted[0]
        assert index == 0
        assert options["plots"]["full"] is True
    finally:
        _close(page)
