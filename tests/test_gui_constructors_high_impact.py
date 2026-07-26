import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

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
    axis = np.linspace(400.0, 1800.0, 41)
    labels = ["A", "A", "A", "B", "B", "B"]

    rows = [["Wavenumber", *labels]]

    for index, value in enumerate(axis):
        rows.append(
            [
                float(value),
                1.0 + np.sin(index / 5.0),
                1.2 + np.sin(index / 5.0),
                0.8 + np.sin(index / 5.0),
                2.0 + np.cos(index / 6.0),
                2.2 + np.cos(index / 6.0),
                1.8 + np.cos(index / 6.0),
            ]
        )

    dataframe = pd.DataFrame(rows)
    dataframe.attrs["data_status"] = "ready"
    dataframe.attrs["detected_delimiter"] = ","
    return dataframe


@pytest.fixture
def raw_dataset():
    dataframe = pd.DataFrame(
        [
            ["Wavenumber", "A_1", "A_2", "B_1"],
            [400.0, 1.0, 1.2, 2.0],
            [500.0, 1.1, np.nan, 2.1],
            [600.0, 1.2, 1.4, 2.2],
            [700.0, 1.3, 1.5, 2.3],
        ]
    )
    dataframe.attrs["data_status"] = "raw"
    dataframe.attrs["detected_delimiter"] = ","
    return dataframe


def close_widget(widget):
    widget.close()
    widget.deleteLater()




class FakeMainMenu(QWidget):
    """Minimal main-window replacement required by embedded pages."""

    def __init__(self):
        super().__init__()
        self.welcome_calls = 0

    def show_welcome_page(self):
        self.welcome_calls += 1

    def record_analysis_step(self, *args, **kwargs):
        return None

    def add_dataframe(self, *args, **kwargs):
        return None


class FakePipelineManager:
    def list_names(self):
        return []

    def save(self, name, options):
        return str(name)

    def load(self, name):
        return {"name": name, "options": {}}

    def delete(self, name):
        return None


def test_preprocessing_window_constructor_executes_full_ui(
    qapp,
    internal_dataset,
    monkeypatch,
):
    import ui.pages.preprocessing_page as module

    monkeypatch.setattr(module, "PipelineManager", FakePipelineManager)

    menu = FakeMainMenu()
    page = module.PreprocessingWindow(
        [internal_dataset],
        ["/tmp/example.csv"],
        menu,
        embedded=True,
    )

    try:
        qapp.processEvents()

        assert page.df is not None
        assert page.selector_df.count() == 1
        assert page.selector_spectrum.count() == 6
        assert page.pipeline_selector.count() == 0
        assert page.grupo_normalizar.isCheckable()
        assert page.grupo_sg.isCheckable()
        assert page.grupo_fg.isCheckable()
        assert page.grupo_mm.isCheckable()
        assert page.preview_plot is not None
        assert page.layout() is not None
    finally:
        close_widget(page)
        close_widget(menu)


def test_preprocessing_window_toggles_multiple_controls(
    qapp,
    internal_dataset,
    monkeypatch,
):
    import ui.pages.preprocessing_page as module

    monkeypatch.setattr(module, "PipelineManager", FakePipelineManager)

    menu = FakeMainMenu()
    page = module.PreprocessingWindow(
        [internal_dataset],
        ["example.csv"],
        menu,
        embedded=True,
    )

    try:
        page.grupo_normalizar.setChecked(True)
        page.combo_normalizar.setCurrentText("Center to u=0")
        page.normalizar_a.setChecked(True)

        page.grupo_sg.setChecked(True)
        page.grupo_fg.setChecked(True)
        page.grupo_mm.setChecked(True)

        page.derivada_pd.setChecked(True)
        page.derivada_sd.setChecked(True)

        qapp.processEvents()

        assert page.grupo_normalizar.isChecked()
        assert page.normalizar_a.isChecked()
        assert page.grupo_sg.isChecked()
        assert page.grupo_fg.isChecked()
        assert page.grupo_mm.isChecked()
    finally:
        close_widget(page)
        close_widget(menu)


def test_dimensionality_window_constructor_executes_full_ui(
    qapp,
    internal_dataset,
):
    from ui.pages.dimensionality_page import DimensionalityReductionWindow

    menu = FakeMainMenu()
    page = DimensionalityReductionWindow(
        [internal_dataset],
        ["/tmp/example.csv"],
        menu,
        embedded=True,
    )

    try:
        qapp.processEvents()

        assert page.df is not None
        assert page.selector_df.count() == 1
        assert page.pca.isChecked() is False
        assert page.tsne.isChecked() is False
        assert page.tsne_pca.isChecked() is False
        assert page.input_reduccion_dim_componentes is not None
        assert page.input_perplexity is not None
        assert page.layout() is not None
    finally:
        close_widget(page)
        close_widget(menu)


def test_dimensionality_window_visibility_and_input_branches(
    qapp,
    internal_dataset,
):
    from ui.pages.dimensionality_page import DimensionalityReductionWindow

    menu = FakeMainMenu()
    page = DimensionalityReductionWindow(
        [internal_dataset],
        ["example.csv"],
        menu,
        embedded=True,
    )

    try:
        page.show()
        qapp.processEvents()

        page.pca.setChecked(True)
        page.tsne.setChecked(True)
        page.tsne_pca.setChecked(True)
        page.toggle_tsne_pca(True)

        page.input_reduccion_dim_componentes.setText("5")
        page.input_reduccion_dim_intervalo.setText("95")
        page.input_comp_tsne_direct.setText("2")
        page.input_perplexity.setText("5")
        page.input_iterations_tsne.setText("300")
        page.input_comp_pca.setText("5")
        page.input_comp_tsne.setText("2")

        qapp.processEvents()

        assert not page.contenedor_componentes_tsne_pca.isHidden()
        assert page.input_reduccion_dim_componentes.text() == "5"
        assert page.input_perplexity.text() == "5"

        page.toggle_tsne_pca(False)
        assert page.contenedor_componentes_tsne_pca.isHidden()
    finally:
        close_widget(page)
        close_widget(menu)


def test_hca_window_constructor_and_linkage_safeguard(
    qapp,
    internal_dataset,
):
    from ui.pages.hca_page import VentanaHca

    menu = FakeMainMenu()
    page = VentanaHca(
        [internal_dataset],
        ["/tmp/example.csv"],
        menu,
        embedded=True,
    )

    try:
        qapp.processEvents()

        assert page.df is not None
        assert page.selector_df.count() == 1
        assert page.input_clusters.text() == "12"

        page.euclidiana.setChecked(True)
        page.actualizar_estado_enlaces()
        assert page.ward.isEnabled()

        page.ward.setChecked(True)
        page.correlación_pearson.setChecked(True)
        page.actualizar_estado_enlaces()

        assert not page.ward.isEnabled()
        assert not page.ward.isChecked()
        assert page.average_linkage.isChecked()
    finally:
        close_widget(page)
        close_widget(menu)


def test_data_preparation_assistant_constructor_executes_detection_ui(
    qapp,
    raw_dataset,
):
    from ui.pages.data_preparation_page import DataPreparationAssistant

    page = DataPreparationAssistant(
        [raw_dataset],
        ["/tmp/raw_example.csv"],
    )

    try:
        qapp.processEvents()

        assert page.dataset.count() == 1
        assert page.active_df is not None
        assert page.raw.model() is not None
        assert page.prep.model() is None
        assert page.name.text().endswith("_prepared")
        assert page.save_button.isEnabled() is False
        assert page.orientation.count() == 3
        assert page.class_source.count() == 4
        assert page.missing.count() == 4

        page.invalidate_preview()

        assert page.preview_df is None
        assert page.save_button.isEnabled() is False
        assert page.status.text().strip() != ""
        assert page.prep.model() is None
    finally:
        close_widget(page)


def test_dataframe_information_and_selection_pages_build_cards(
    qapp,
    internal_dataset,
):
    from ui.pages.dataframe_page import (
        DataFrameInformationPage,
        DataFrameSelectionWindow,
        VerDf,
    )

    calls = []

    info_page = DataFrameInformationPage(
        internal_dataset,
        "/tmp/example.csv",
        lambda: calls.append("back"),
    )

    selection_page = DataFrameSelectionWindow(
        [internal_dataset],
        ["/tmp/example.csv"],
        lambda index: calls.append(("delete", index)),
        lambda index: calls.append(("view", index)),
        lambda index: calls.append(("info", index)),
        back_callback=lambda: calls.append("back-selection"),
        embedded=True,
    )

    view_page = VerDf(internal_dataset)

    try:
        qapp.processEvents()

        type_cards = info_page.findChildren(QWidget, "typeCard")
        assert len(type_cards) == 2

        assert selection_page.layout() is not None
        assert selection_page.dataframes[0] is internal_dataset

        assert view_page.modelo.rowCount() == internal_dataset.shape[0] - 1
        assert view_page.modelo.columnCount() == internal_dataset.shape[1]
    finally:
        close_widget(info_page)
        close_widget(selection_page)
        close_widget(view_page)
