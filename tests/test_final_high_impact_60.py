import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


class FakeButton:
    def __init__(self):
        self.enabled = False

    def setEnabled(self, value):
        self.enabled = bool(value)


class FakeSelectionPage:
    def __init__(self):
        self.btn_plot_preview = FakeButton()
        self.btn_accept = FakeButton()
        self.btn_plot_mid_result = FakeButton()


class FakeMenu:
    def __init__(self):
        self.dataframes = []
        self.nombres_archivos = []
        self.data_fusion_page = FakeSelectionPage()


class FakePreviewWindow:
    def __init__(self, dataframe, title, parent=None):
        self.dataframe = dataframe
        self.title = title
        self.parent = parent
        self.shown = False

    def show(self):
        self.shown = True


def fused_dataframe():
    dataframe = pd.DataFrame(
        {
            "Axis": [400.0, 500.0, 600.0],
            "A": [1.0, 2.0, 3.0],
            "B": [2.0, 3.0, 4.0],
        }
    )
    dataframe.attrs["source"] = "test"
    return dataframe


def build_fusion_dummy(preview_mode=False):
    dummy = SimpleNamespace()
    dummy.preview_mode = preview_mode
    dummy.menu_principal = FakeMenu()
    dummy.df_concat_midfusion = None
    dummy.lista_varianza = None
    dummy.preview_window = None
    dummy.recorded = []
    dummy.mid_result_calls = 0

    def record(output_name, operation):
        dummy.recorded.append((output_name, operation))

    def show_result():
        dummy.mid_result_calls += 1

    dummy._record_fusion_history = record
    dummy.show_mid_level_result = show_result
    return dummy


def test_low_level_final_save_branch(tmp_path, monkeypatch):
    import ui.pages.data_fusion_page as module

    dummy = build_fusion_dummy(preview_mode=False)
    dataframe = fused_dataframe()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("low_result", True),
    )
    monkeypatch.setattr(
        module.QMessageBox,
        "information",
        lambda *args, **kwargs: None,
    )

    module.DataFusionConfigurationWindow.lowfusionfinal(dummy, dataframe)

    assert dummy.df_concat_midfusion is dataframe
    assert dummy.menu_principal.dataframes == [dataframe]
    assert dummy.menu_principal.nombres_archivos == ["low_result"]
    assert dummy.recorded == [("low_result", "Low-level fusion")]
    assert (tmp_path / "archivos_guardados" / "low_result.csv").exists()


def test_low_level_final_preview_branch(monkeypatch):
    import ui.pages.data_fusion_page as module

    dummy = build_fusion_dummy(preview_mode=True)
    dataframe = fused_dataframe()

    monkeypatch.setattr(module, "FusionPreviewWindow", FakePreviewWindow)

    module.DataFusionConfigurationWindow.lowfusionfinal(dummy, dataframe)

    assert dummy.preview_mode is False
    assert dummy.preview_window.shown is True
    assert dummy.menu_principal.data_fusion_page.btn_plot_preview.enabled
    assert dummy.menu_principal.data_fusion_page.btn_accept.enabled
    assert dummy.menu_principal.data_fusion_page.btn_plot_mid_result.enabled
    assert dummy.mid_result_calls == 1


def test_mid_level_final_save_branch(tmp_path, monkeypatch):
    import ui.pages.data_fusion_page as module

    dummy = build_fusion_dummy(preview_mode=False)
    dataframe = fused_dataframe()
    variance = [np.array([70.0, 30.0])]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("mid_result", True),
    )
    monkeypatch.setattr(
        module.QMessageBox,
        "information",
        lambda *args, **kwargs: None,
    )

    module.DataFusionConfigurationWindow.midfusionfinal(
        dummy,
        dataframe,
        variance,
    )

    assert dummy.lista_varianza is variance
    assert dummy.menu_principal.dataframes == [dataframe]
    assert dummy.menu_principal.nombres_archivos == ["mid_result"]
    assert dummy.recorded == [("mid_result", "Mid-level fusion")]
    assert dummy.mid_result_calls == 1
    assert (tmp_path / "archivos_guardados" / "mid_result.csv").exists()


def test_mid_level_final_preview_branch():
    import ui.pages.data_fusion_page as module

    dummy = build_fusion_dummy(preview_mode=True)
    dataframe = fused_dataframe()
    variance = [np.array([80.0, 20.0])]

    module.DataFusionConfigurationWindow.midfusionfinal(
        dummy,
        dataframe,
        variance,
    )

    selection = dummy.menu_principal.data_fusion_page
    assert dummy.preview_mode is False
    assert selection.btn_plot_preview.enabled
    assert selection.btn_accept.enabled
    assert selection.btn_plot_mid_result.enabled
    assert dummy.mid_result_calls == 1


@pytest.mark.parametrize(
    ("method_name", "result_name", "expected_operation"),
    [
        (
            "lowfusionfinalsininterseccion",
            "low_no_common",
            "Low-level fusion",
        ),
        (
            "midfusionfinalsininterseccion",
            "mid_no_common",
            "Mid-level fusion",
        ),
    ],
)
def test_no_intersection_final_save_branches(
    method_name,
    result_name,
    expected_operation,
    tmp_path,
    monkeypatch,
):
    import ui.pages.data_fusion_page as module

    dummy = build_fusion_dummy(preview_mode=False)
    dataframe = fused_dataframe()
    variance = [np.array([60.0, 40.0])]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module.QInputDialog,
        "getText",
        lambda *args, **kwargs: (result_name, True),
    )
    monkeypatch.setattr(
        module.QMessageBox,
        "information",
        lambda *args, **kwargs: None,
    )

    method = getattr(module.DataFusionConfigurationWindow, method_name)

    if method_name.startswith("mid"):
        method(dummy, dataframe, variance)
    else:
        method(dummy, dataframe)

    assert dummy.menu_principal.dataframes == [dataframe]
    assert dummy.menu_principal.nombres_archivos == [result_name]
    assert dummy.recorded == [(result_name, expected_operation)]
    assert (
        tmp_path / "archivos_guardados" / f"{result_name}.csv"
    ).exists()


class FakePlot:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FakeResultsPage:
    def __init__(self):
        self.plots = []

    def add_plot(self, widget, title):
        self.plots.append((widget, title))


def build_main_dummy():
    dataframe = pd.DataFrame(
        {
            "Axis": [400.0, 500.0, 600.0],
            "A": [1.0, 2.0, 3.0],
            "B": [2.0, 3.0, 4.0],
        }
    )

    dummy = SimpleNamespace()
    dummy.complete_df = dataframe
    dummy.raman_shift = dataframe["Axis"].to_numpy()
    dummy.color_mapping = {"A": "#111111", "B": "#222222"}
    dummy.x_label = "Wavenumber"
    dummy.y_label = "Intensity"
    dummy.nombres_archivos = ["/tmp/example.csv"]
    dummy.current_spectra_dataset_index = 0
    dummy.spectra_results_page = None
    dummy.history = []
    dummy.exports = []

    def prepare_results():
        dummy.spectra_results_page = FakeResultsPage()

    def record_analysis_step(**kwargs):
        dummy.history.append(kwargs)

    def process_export(action, configuration):
        dummy.exports.append((action, configuration))

    def translate(text, **kwargs):
        return text

    dummy.prepare_spectra_results_page = prepare_results
    dummy.record_analysis_step = record_analysis_step
    dummy.process_spectra_export = process_export
    dummy.tr = translate
    return dummy


def test_process_spectra_configuration_all_plot_branches(monkeypatch):
    import main as module

    for name in (
        "SpectraPlotWindow",
        "LimitedRangeSpectraPlotWindow",
        "SpectraByTypePlotWindow",
        "LimitedRangeSpectraByTypePlotWindow",
        "StackedSpectraPlotWindow",
    ):
        monkeypatch.setattr(module, name, FakePlot)

    dummy = build_main_dummy()

    configuration = {
        "plots": {
            "full": True,
            "limited": True,
            "type": True,
            "limited_type": True,
            "stacked": True,
        },
        "range_min": 450.0,
        "range_max": 550.0,
        "sample_type": "A",
        "stacked_options": {
            "offset_mode": "manual",
            "offset_value": 2.0,
            "show_labels": False,
            "maximum_spectra": 3,
            "sample_type": "A",
            "range_min": 450.0,
            "range_max": 550.0,
        },
        "export_action": "full_csv",
    }

    module.MainMenu.process_spectra_configuration(dummy, configuration)

    assert len(dummy.spectra_results_page.plots) == 5
    assert dummy.min_val == 450.0
    assert dummy.max_val == 550.0
    assert dummy.selected_type == "A"
    assert len(dummy.history) == 1
    assert dummy.history[0]["operation"] == "Stacked spectra visualization"
    assert dummy.exports == [("full_csv", configuration)]


def test_process_spectra_configuration_stacked_error_branch(monkeypatch):
    import main as module

    class BrokenStackedPlot:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("plot failed")

    messages = []
    monkeypatch.setattr(module, "StackedSpectraPlotWindow", BrokenStackedPlot)
    monkeypatch.setattr(
        module.QMessageBox,
        "critical",
        lambda *args: messages.append(args),
    )

    dummy = build_main_dummy()
    configuration = {
        "plots": {"stacked": True},
        "stacked_options": {},
    }

    module.MainMenu.process_spectra_configuration(dummy, configuration)

    assert len(messages) == 1
    assert "plot failed" in str(messages[0])
