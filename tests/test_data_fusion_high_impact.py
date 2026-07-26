import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def datasets():
    def make_dataset(start, stop, offset):
        axis = np.linspace(start, stop, 31)
        rows = [["Wavenumber", "A", "A", "B", "B"]]
        for index, value in enumerate(axis):
            rows.append(
                [
                    float(value),
                    1.0 + offset + np.sin(index / 5.0),
                    1.2 + offset + np.sin(index / 5.0),
                    2.0 + offset + np.cos(index / 6.0),
                    2.2 + offset + np.cos(index / 6.0),
                ]
            )
        dataframe = pd.DataFrame(rows)
        dataframe.attrs["data_status"] = "ready"
        return dataframe

    first = make_dataset(400.0, 1800.0, 0.0)
    second = make_dataset(500.0, 1700.0, 0.3)
    third = make_dataset(2000.0, 3200.0, 0.6)
    return first, second, third


class FakeMainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.history_calls = []
        self.added = []

    def show_welcome_page(self):
        return None

    def open_dimensionality_reduction_window(self):
        return None

    def record_analysis_step(self, **kwargs):
        self.history_calls.append(kwargs)

    def add_dataframe(self, dataframe, name):
        self.added.append((dataframe, name))

    def register_exported_dataframe(self, dataframe, name):
        self.added.append((dataframe, name))


class DummySignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in self.callbacks:
            callback(*args)


class DummyWorker:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False
        self.signal_datalowfusion = DummySignal()
        self.signal_datamidfusion = DummySignal()
        self.signal_datalowfusionsininterseccion = DummySignal()
        self.signal_datamidfusionsininterseccion = DummySignal()

    def start(self):
        self.started = True


def close_widget(widget):
    widget.close()
    widget.deleteLater()


def build_configuration(module, datasets, menu, intersection=True):
    first, second, _ = datasets
    common = (500.0, 1700.0) if intersection else None
    ranges = [(400.0, 1800.0), (500.0, 1700.0)]

    return module.DataFusionConfigurationWindow(
        [first, second],
        [first, second],
        ["/tmp/ftir.csv", "/tmp/raman.csv"],
        ranges,
        intersection,
        common,
        ["A", "B"],
        menu,
        embedded=True,
        show_buttons=False,
        show_summary=True,
    )


def test_fusion_selection_constructor_covers_dataset_cards(qapp, datasets):
    from ui.pages.data_fusion_page import DataFusionSelectionWindow

    first, second, third = datasets
    menu = FakeMainMenu()
    page = DataFusionSelectionWindow(
        [first, second, third],
        ["/tmp/ftir.csv", "/tmp/raman.csv", "/tmp/no_overlap.csv"],
        menu,
        embedded=True,
    )

    try:
        qapp.processEvents()
        assert len(page.checkboxes) == 3
        assert page.btn_accept is not None
        assert page.btn_plot_preview is not None
        assert page.btn_plot_mid_result.isEnabled() is False
        assert page.configuration_widget is None
        assert page.seleccionados == []
    finally:
        close_widget(page)
        close_widget(menu)


def test_fusion_configuration_constructor_covers_low_and_mid_controls(
    qapp,
    datasets,
):
    import ui.pages.data_fusion_page as module

    menu = FakeMainMenu()
    page = build_configuration(module, datasets, menu, intersection=True)

    try:
        qapp.processEvents()

        assert page.lowfusion.isChecked()
        assert page.rb_concat_v.isChecked()
        assert page.sin_interpolacion.isChecked()
        assert len(page.component_spinboxes) == 2
        assert page.intervalo_confianza.text() == "95"
        assert page.layout() is not None
    finally:
        close_widget(page)
        close_widget(menu)


def test_merge_columns_forces_interpolation(qapp, datasets):
    import ui.pages.data_fusion_page as module

    menu = FakeMainMenu()
    page = build_configuration(module, datasets, menu, intersection=True)

    try:
        page.rb_concat_h.setChecked(True)
        page.update_low_level_axis_options()
        qapp.processEvents()

        assert page.interpolarsi.isChecked()
        assert not page.sin_interpolacion.isEnabled()
        assert not page.contenedor_interpolacion_low.isHidden()

        page.rb_concat_v.setChecked(True)
        page.update_low_level_axis_options()
        assert page.sin_interpolacion.isEnabled()
    finally:
        close_widget(page)
        close_widget(menu)


def test_fusion_configuration_toggles_low_mid_and_interpolation(
    qapp,
    datasets,
):
    import ui.pages.data_fusion_page as module

    menu = FakeMainMenu()
    page = build_configuration(module, datasets, menu, intersection=True)
    page.show()

    try:
        page.toggle_lowfusion(True)
        qapp.processEvents()
        assert not page.contenedor_lowf.isHidden()

        page.midfusion.setChecked(True)
        page.toggle_midfusion(True)
        qapp.processEvents()
        assert not page.contenedor_midf.isHidden()

        page.interpolar_mid.setChecked(True)
        page.mostrar_opciones_interpolacion_mid()
        assert not page.contenedor_opciones_dinamicas_mid.isHidden()

        page.sin_interpolacion_mid.setChecked(True)
        page.mostrar_opciones_interpolacion_mid()
        qapp.processEvents()
        assert page.contenedor_opciones_dinamicas_mid.isHidden()
    finally:
        close_widget(page)
        close_widget(menu)


def test_fusion_component_and_confidence_validation(qapp, datasets):
    import ui.pages.data_fusion_page as module

    menu = FakeMainMenu()
    page = build_configuration(module, datasets, menu, intersection=True)

    try:
        assert len(page._component_counts()) == 2
        assert page._confidence_value() == 95.0

        page.intervalo_confianza.setText("abc")
        with pytest.raises(ValueError, match="numeric"):
            page._confidence_value()

        page.intervalo_confianza.setText("100")
        with pytest.raises(ValueError, match="between"):
            page._confidence_value()
    finally:
        close_widget(page)
        close_widget(menu)


def test_fusion_history_parameters_low_and_mid(qapp, datasets):
    import ui.pages.data_fusion_page as module

    menu = FakeMainMenu()
    page = build_configuration(module, datasets, menu, intersection=True)

    try:
        low = page._fusion_history_parameters("Low-level fusion")
        assert low["Concatenation"] == "Vertical"
        assert "Source datasets" in low

        mid = page._fusion_history_parameters("Mid-level fusion")
        assert "Principal components by dataset" in mid
        assert mid["Confidence interval"] == "95%"

        page._record_fusion_history("fusion_result", "Mid-level fusion")
        assert len(menu.history_calls) == 1
        assert menu.history_calls[0]["output_dataset"] == "fusion_result"
    finally:
        close_widget(page)
        close_widget(menu)


def test_low_level_dispatch_uses_mock_worker(
    qapp,
    datasets,
    monkeypatch,
):
    import ui.pages.data_fusion_page as module

    monkeypatch.setattr(module, "LowLevelDataFusionThread", DummyWorker)

    menu = FakeMainMenu()
    page = build_configuration(module, datasets, menu, intersection=True)

    try:
        page.lowfusion.setChecked(True)
        page.sin_interpolacion.setChecked(True)
        page.aplicar_fusion()

        assert isinstance(page.hilo, DummyWorker)
        assert page.hilo.started
    finally:
        close_widget(page)
        close_widget(menu)


def test_mid_level_dispatch_uses_mock_worker(
    qapp,
    datasets,
    monkeypatch,
):
    import ui.pages.data_fusion_page as module

    monkeypatch.setattr(module, "MidLevelDataFusionThread", DummyWorker)

    menu = FakeMainMenu()
    page = build_configuration(module, datasets, menu, intersection=True)

    try:
        page.midfusion.setChecked(True)
        page.sin_interpolacion_mid.setChecked(True)
        page.aplicar_fusion_mid()

        assert isinstance(page.hilo, DummyWorker)
        assert page.hilo.started
    finally:
        close_widget(page)
        close_widget(menu)


def test_nonintersection_low_and_mid_dispatch(
    qapp,
    datasets,
    monkeypatch,
):
    import ui.pages.data_fusion_page as module

    monkeypatch.setattr(
        module,
        "LowLevelDataFusionNoCommonRangeThread",
        DummyWorker,
    )
    monkeypatch.setattr(
        module,
        "MidLevelDataFusionThread",
        DummyWorker,
    )

    menu = FakeMainMenu()
    page = build_configuration(module, datasets, menu, intersection=False)

    try:
        page.lowfusion.setChecked(True)
        page.lineal.setChecked(True)
        page.input_n_puntos.setText("100")
        page.mostrar_opciones_interpolacionsinintersecctar()

        assert isinstance(page.hilo, DummyWorker)
        assert page.hilo.started

        page.midfusion.setChecked(True)
        page.sin_interpolacion_mid.setChecked(True)
        page.mostrar_opciones_interpolacionsinintersecctar_mid()

        assert isinstance(page.hilo, DummyWorker)
        assert page.hilo.started
    finally:
        close_widget(page)
        close_widget(menu)
