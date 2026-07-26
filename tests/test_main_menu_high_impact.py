import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def avoid_unsaved_project_dialog(monkeypatch):
    """Prevent modal project-close dialogs from blocking offscreen tests."""
    from main import MainMenu

    monkeypatch.setattr(
        MainMenu,
        "_confirm_discard_unsaved_changes",
        lambda self: True,
    )


@pytest.fixture
def ready_dataset():
    rows = [
        ["Wavenumber", "A", "A", "B"],
        [400.0, 1.0, 1.1, 2.0],
        [500.0, 1.2, 1.3, 2.2],
        [600.0, 1.4, 1.5, 2.4],
    ]
    dataframe = pd.DataFrame(rows)
    dataframe.attrs["data_status"] = "ready"
    return dataframe


@pytest.fixture
def raw_dataset(ready_dataset):
    dataframe = ready_dataset.copy()
    dataframe.attrs["data_status"] = "raw"
    return dataframe


def close_widget(widget):
    widget.close()
    widget.deleteLater()


def test_main_menu_constructor_covers_dashboard_and_sidebar(qapp):
    from main import MainMenu

    window = MainMenu()

    try:
        qapp.processEvents()

        assert window.windowTitle().endswith("EspectroApp")
        assert len(window.menu_buttons) >= 8
        assert window.workspace_stack.count() >= 1
        assert window.welcome_page is not None
        assert isinstance(window.dataframes, list)
        assert isinstance(window.nombres_archivos, list)
        assert window.datasets_value_label.text().isdigit()
    finally:
        close_widget(window)


def test_main_menu_language_and_navigation(qapp):
    from main import MainMenu

    window = MainMenu()

    try:
        window.update_language("es")
        qapp.processEvents()
        assert window.current_language == "es"

        window.update_language("invalid")
        qapp.processEvents()
        assert window.current_language == "en"

        window.show_welcome_page()
        assert window.workspace_stack.currentWidget() is window.welcome_page
    finally:
        close_widget(window)


def test_main_menu_axis_label_helpers(qapp, ready_dataset):
    from main import MainMenu

    window = MainMenu()

    try:
        assert window.detect_labels_from_df(ready_dataset)[0].startswith("Wavenumber")

        raman = ready_dataset.copy()
        raman.iloc[0, 0] = "Raman Shift"
        assert window.detect_labels_from_df(raman)[0].startswith("Raman")

        assert window.detect_labels_from_df(pd.DataFrame())[0] == "X Axis"

        window.x_label = "Raman Shift (cm-1)"
        assert window.get_export_x_column_name() == "Raman Shift"

        window.x_label = "Wavenumber"
        assert window.get_export_x_column_name() == "Wavenumber"

        window.x_label = "Time"
        assert window.get_export_x_column_name() == "X Axis"
    finally:
        close_widget(window)


def test_main_menu_register_prepared_dataset_and_duplicate_name(
    qapp,
    ready_dataset,
    monkeypatch,
):
    from main import MainMenu

    monkeypatch.setattr(
        MainMenu,
        "_confirm_discard_unsaved_changes",
        lambda self: True,
    )

    window = MainMenu()

    try:
        window.register_prepared_dataset(ready_dataset, "prepared")
        window.register_prepared_dataset(ready_dataset, "prepared")

        assert window.nombres_archivos == ["prepared", "prepared_2"]
        assert len(window.dataframes) == 2
        assert all(
            df.attrs.get("data_status") == "ready"
            for df in window.dataframes
        )
    finally:
        close_widget(window)


def test_main_menu_register_exported_dataframe(
    qapp,
    monkeypatch,
):
    from main import MainMenu

    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        MainMenu,
        "_confirm_discard_unsaved_changes",
        lambda self: True,
    )

    window = MainMenu()
    exported = pd.DataFrame(
        {
            "Wavenumber": [400.0, 500.0],
            "A": [1.0, 2.0],
        }
    )

    try:
        window.register_exported_dataframe(exported, "result.csv")
        window.register_exported_dataframe(exported, "result.csv")

        assert window.nombres_archivos == ["result.csv", "result_2.csv"]
        assert len(window.dataframes) == 2
        assert window.dataframes[0].iloc[0, 0] == "Wavenumber"
    finally:
        close_widget(window)


def test_main_menu_process_raw_loaded_file(
    qapp,
    raw_dataset,
):
    from main import MainMenu

    window = MainMenu()

    try:
        window.process_loaded_files(raw_dataset, "/tmp/raw.csv")

        assert len(window.dataframes) == 1
        assert window.nombres_archivos == ["/tmp/raw.csv"]
        assert window.df_final.equals(raw_dataset)
        assert window.index_actual == 0
    finally:
        close_widget(window)


def test_main_menu_receive_modified_dataframe(
    qapp,
    ready_dataset,
):
    from main import MainMenu

    window = MainMenu()
    modified = ready_dataset.copy()
    modified.iloc[1, 1] = 99.0

    try:
        window.dataframes = [ready_dataset.copy()]
        window.index_actual = 0
        window.receive_modified_dataframe(modified)

        assert window.df.equals(modified)
        assert window.df_final.equals(modified)
        assert window.dataframes[0].equals(modified)
    finally:
        close_widget(window)


def test_dashboard_stats_counts_models(qapp):
    from main import MainMenu

    window = MainMenu()

    try:
        window.dataframes = [pd.DataFrame(), pd.DataFrame()]
        window.analysis_history._entries = [
            SimpleNamespace(operation="PCA analysis"),
            SimpleNamespace(operation="Dataset loaded"),
            SimpleNamespace(operation="HCA analysis"),
        ]
        window.update_dashboard_stats()

        assert window.datasets_value_label.text() == "2"
        assert window.operations_value_label.text() == "3"
        assert window.models_value_label.text() == "0"
    finally:
        window.analysis_history._entries = []
        close_widget(window)


def test_history_parameter_formatting(qapp):
    from main import MainMenu

    window = MainMenu()

    try:
        enabled = window._format_history_parameter("Interpolation", True)
        sequence = window._format_history_parameter(
            "Source datasets",
            ["FTIR", "Raman"],
        )

        assert ":" in enabled
        assert "FTIR" in sequence
        assert "Raman" in sequence
    finally:
        close_widget(window)
