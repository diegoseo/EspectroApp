import pandas as pd
import sys, os
import matplotlib
import numpy as np
import tempfile
from uuid import uuid4
from pathlib import Path
import pyqtgraph as pg

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl, QTimer, QSettings

from PySide6.QtWidgets import (
    QTableView,
    QTabWidget,
    QGridLayout,
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QInputDialog,
    QLabel,
    QDialog,
    QLineEdit,
    QCheckBox,
    QGroupBox,
    QComboBox,
    QHeaderView,
    QMainWindow,
    QScrollArea,
    QButtonGroup,
    QRadioButton,
    QSplitter,
    QStackedWidget,
    QFrame,
    QSizePolicy,
    QMenu,
)

from PySide6.QtGui import QIcon, QFont, QAction
from PySide6.QtCore import Qt, QSize, Signal, QAbstractTableModel, QModelIndex
from functools import partial
from thread import (
    FileLoaderThread,
    SpectraPlotThread,
    PreprocessingThread,
    DimensionalityReductionThread,
    HcaThread,
    DataFusionThread,
    LowLevelDataFusionThread,
    LowLevelDataFusionNoCommonRangeThread,
    MidLevelDataFusionThread,
    MidLevelDataFusionNoCommonRangeThread,
    MidLevelPlotThread,
)
from plotting import (
    SpectraPlotWindow,
    LimitedRangeSpectraPlotWindow,
    SpectraByTypePlotWindow,
    LimitedRangeSpectraByTypePlotWindow,
    StackedSpectraPlotWindow,
    graficar_varianza_acumulada,
)
from functions import (
    get_column_with_fewest_rows,
    calculate_cumulative_variance,
    assign_type_colors,
    linear_baseline_from_points,
    shirley_baseline_from_points,
    normalize_by_mean,
    normalize_by_area,
    smooth_savitzky_golay,
    smooth_gaussian_filter,
    smooth_moving_average,
    calculate_first_derivative,
    calculate_second_derivative,
)

from ui.pages.preprocessing_page import PreprocessingWindow
from ui.pages.dimensionality_page import (
    DimensionalityReductionWindow,
    VentanaGraficoPCA2D,
    VentanaGraficoPCA3D,
    VentanaGraficoTSNE2D,
    VentanaGraficoTSNE3D,
    VentanaGraficoLoading,
)

from ui.pages.hca_page import VentanaHca
from ui.pages.data_fusion_page import DataFusionSelectionWindow

from ui.pages.dataframe_page import (
    DataFrameInformationPage,
    DataFrameSelectionWindow,
    DataFrameFixWindow,
    VerDf,
    CsvGenerator,
    normalize_visual_dataframe,
)
from ui.pages.spectra_page import (
    SpectraResultsPage,
    SpectraExportOptionsWindow,
)
from ui.pages.data_preparation_page import DataPreparationAssistant
from ui.pages.fitted_models_page import FittedModelsPage

from ui.styles import MAIN_WINDOW_STYLE, MENU_BUTTON_STYLE
from core.analysis_history import AnalysisHistoryManager, AnalysisHistoryEntry
from methods import FittedModelManager, create_default_method_registry

from core.project_manager import (
    PROJECT_EXTENSION,
    load_project_file,
    save_project_file,
)
from core.translations import (
    TRANSLATIONS,
    translate,
    set_language,
    retranslate_widget_tree,
)

matplotlib.use("QtAgg")


def resource_path(relative_path):
    """Return a resource path compatible with development and PyInstaller."""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


class MainMenu(QWidget):
    """
    Provides the main graphical entry point for the spectral data analysis application.
    The window lets users load spectra, explore tables and plots, run preprocessing and multivariate analysis, and launch data fusion workflows.

    """

    def __init__(self):
        super().__init__()
        self.app_settings = QSettings("EspectroApp", "EspectroApp")
        self.current_language = str(self.app_settings.value("language", "en"))
        if self.current_language not in TRANSLATIONS:
            self.current_language = "en"
        set_language(self.current_language)
        self.menu_buttons = {}
        self.menu_section_labels = {}
        self.setWindowTitle("EspectroApp")
        self.setMinimumSize(1350, 820)
        self.resize(1600, 950)
        self.setStyleSheet(MAIN_WINDOW_STYLE)
        content_widget = QWidget()
        content_widget.setObjectName("mainPanel")
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(14, 18, 14, 16)
        layout.setSpacing(5)

        self.threads = []
        self.dataframes = []
        self.nombres_archivos = []
        self.df_final = None
        self.analysis_history = AnalysisHistoryManager()
        self.analysis_history.changed.connect(self.refresh_history_view)
        self.method_registry = create_default_method_registry()
        self.fitted_models = FittedModelManager(self)
        self.fitted_models.changed.connect(self.update_dashboard_stats)
        self.project_path = None
        self.project_name = self.tr("Untitled project")
        self.project_modified = False

        self.app_title_label = QLabel()
        self.app_title_label.setObjectName("appTitle")
        self.app_title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.app_title_label)

        self.menu_section_labels["loading"] = self.create_menu_label("")
        layout.addWidget(self.menu_section_labels["loading"])

        self.menu_buttons["load"] = self.create_button(
            "",
            resource_path("icom/sidebar/upload.svg"),
            self.open_file_dialog,
        )
        layout.addWidget(self.menu_buttons["load"])

        self.menu_buttons["prepare"] = self.create_button(
            "",
            resource_path("icom/sidebar/data-preparation.svg"),
            self.open_data_preparation_assistant,
        )
        layout.addWidget(self.menu_buttons["prepare"])

        self.menu_buttons["view"] = self.create_button(
            "",
            resource_path("icom/sidebar/table.svg"),
            self.view_dataframe,
        )
        layout.addWidget(self.menu_buttons["view"])

        self.menu_buttons["display"] = self.create_button(
            "",
            resource_path("icom/sidebar/chart-spline.svg"),
            self.open_spectra_window,
        )
        layout.addWidget(self.menu_buttons["display"])

        layout.addSpacing(10)

        self.menu_section_labels["processing"] = self.create_menu_label("")
        layout.addWidget(self.menu_section_labels["processing"])

        self.menu_buttons["preprocess"] = self.create_button(
            "",
            resource_path("icom/sidebar/sliders-horizontal.svg"),
            self.open_preprocessing_window,
        )
        layout.addWidget(self.menu_buttons["preprocess"])

        self.menu_buttons["pca"] = self.create_button(
            "",
            resource_path("icom/sidebar/network.svg"),
            self.open_dimensionality_reduction_window,
        )
        layout.addWidget(self.menu_buttons["pca"])

        self.menu_buttons["models_page"] = self.create_button(
            "",
            resource_path("icom/sidebar/database-star.svg"),
            self.open_fitted_models_page,
        )
        layout.addWidget(self.menu_buttons["models_page"])

        self.menu_buttons["hca"] = self.create_button(
            "",
            resource_path("icom/sidebar/workflow.svg"),
            self.open_hca_window,
        )
        layout.addWidget(self.menu_buttons["hca"])

        layout.addSpacing(10)

        self.menu_section_labels["fusion_section"] = self.create_menu_label("")
        layout.addWidget(self.menu_section_labels["fusion_section"])

        self.menu_buttons["fusion"] = self.create_button(
            "",
            resource_path("icom/sidebar/git-merge.svg"),
            self.open_data_fusion_window,
        )
        layout.addWidget(self.menu_buttons["fusion"])

        layout.addStretch()

        self.settings_button = QPushButton()
        self.settings_button.setObjectName("settingsButton")
        self.settings_button.setIcon(QIcon(resource_path("icom/sidebar/settings.svg")))
        self.settings_button.setIconSize(QSize(21, 21))
        self.settings_button.setCursor(Qt.PointingHandCursor)
        self.settings_button.setToolTip(self.tr("Settings"))
        self.settings_button.clicked.connect(self.show_settings_menu)
        layout.addWidget(self.settings_button, alignment=Qt.AlignLeft)

        scroll = QScrollArea()
        scroll.setObjectName("mainScroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)

        scroll.setMinimumWidth(230)
        scroll.setMaximumWidth(255)

        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.workspace = QFrame()
        self.workspace.setObjectName("workspace")

        workspace_layout = QVBoxLayout(self.workspace)
        workspace_layout.setContentsMargins(34, 26, 34, 28)
        workspace_layout.setSpacing(8)

        self.workspace_title = QLabel()
        self.workspace_title.setObjectName("workspaceTitle")
        self.workspace_title.setAlignment(Qt.AlignCenter)

        self.workspace_subtitle = QLabel()
        self.workspace_subtitle.setObjectName("workspaceSubtitle")
        self.workspace_subtitle.setAlignment(Qt.AlignCenter)
        self.workspace_subtitle.setWordWrap(True)

        workspace_layout.addWidget(self.workspace_title)
        workspace_layout.addWidget(self.workspace_subtitle)

        self.workspace_stack = QStackedWidget()
        self.workspace_stack.setObjectName("workspaceStack")

        self.welcome_page = self.create_welcome_page()
        self.workspace_stack.addWidget(self.welcome_page)

        workspace_layout.addWidget(self.workspace_stack, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self.workspace)

        splitter.setSizes([245, 1105])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.addWidget(splitter)

        self.setLayout(main_layout)
        self.update_language(self.current_language)
        self.update_project_title()

    def tr(self, key, **values):
        return translate(key, self.current_language, **values)

    def update_project_title(self):
        marker = "*" if self.project_modified else ""
        name = str(self.project_name or self.tr("Untitled project"))
        self.setWindowTitle(f"{name}{marker} — EspectroApp")

    def mark_project_modified(self):
        self.project_modified = True
        self.update_project_title()

    def _active_page_key(self):
        current = self.workspace_stack.currentWidget()
        page_map = {
            getattr(self, "welcome_page", None): "welcome",
            getattr(self, "data_preparation_page", None): "prepare",
            getattr(self, "dataframe_selection_page", None): "view",
            getattr(self, "preprocessing_page", None): "preprocess",
            getattr(self, "dimensionality_page", None): "pca",
            getattr(self, "hca_page", None): "hca",
            getattr(self, "fitted_models_page", None): "models_page",
            getattr(self, "data_fusion_page", None): "fusion",
        }
        return page_map.get(current, "welcome")

    def _restore_active_page(self, page_key):
        actions = {
            "prepare": self.open_data_preparation_assistant,
            "view": self.view_dataframe,
            "preprocess": self.open_preprocessing_window,
            "pca": self.open_dimensionality_reduction_window,
            "hca": self.open_hca_window,
            "models_page": self.open_fitted_models_page,
            "fusion": self.open_data_fusion_window,
        }
        action = actions.get(page_key)
        if action is None:
            self.show_welcome_page()
        else:
            action()

    def _history_as_dicts(self):
        return [entry.to_dict() for entry in self.analysis_history.entries]

    def _confirm_discard_unsaved_changes(self):
        if not self.project_modified:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(self.tr("Unsaved changes"))
        box.setText(self.tr("Save the current project before continuing?"))
        box.setStandardButtons(
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
        )
        answer = box.exec()
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            return self.save_project()
        return True

    def new_project(self):
        if not self._confirm_discard_unsaved_changes():
            return
        self.dataframes.clear()
        self.nombres_archivos.clear()
        self.analysis_history._entries = []
        self.analysis_history.changed.emit()
        self.fitted_models.clear()
        self.project_path = None
        self.project_name = self.tr("Untitled project")
        self.project_modified = False
        self.show_welcome_page()
        self.update_dashboard_stats()
        self.update_project_title()

    def save_project(self):
        if self.project_path is None:
            return self.save_project_as()
        return self._write_project(self.project_path)

    def save_project_as(self):
        default_name = str(self.project_name or self.tr("Untitled project"))
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save project as"),
            default_name + PROJECT_EXTENSION,
            self.tr("EspectroApp project (*.espectroapp)"),
        )
        if not path:
            return False
        return self._write_project(path)

    def _write_project(self, path):
        try:
            self._ensure_dataset_metadata()
            saved_path = save_project_file(
                path,
                dataframes=self.dataframes,
                dataset_names=self.nombres_archivos,
                history_entries=self._history_as_dicts(),
                language=self.current_language,
                active_page=self._active_page_key(),
                project_name=Path(path).stem,
                fitted_models=self.fitted_models.to_dicts(),
                fitted_model_artifacts=self.fitted_models.artifacts_dict(),
            )
        except Exception as error:
            QMessageBox.critical(
                self,
                self.tr("Save error"),
                self.tr("The project could not be saved:\n{error}", error=error),
            )
            return False
        self.project_path = saved_path
        self.project_name = saved_path.stem
        self.project_modified = False
        self.update_project_title()
        QMessageBox.information(
            self,
            self.tr("Project saved"),
            self.tr("The complete project was saved successfully."),
        )
        return True

    def open_project(self):
        if not self._confirm_discard_unsaved_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open project"),
            "",
            self.tr("EspectroApp project (*.espectroapp)"),
        )
        if not path:
            return
        try:
            payload = load_project_file(path)
        except Exception as error:
            QMessageBox.critical(
                self,
                self.tr("Open error"),
                self.tr("The project could not be opened:\n{error}", error=error),
            )
            return

        self.dataframes = [df.copy() for df in payload["dataframes"]]
        self.nombres_archivos = list(payload["dataset_names"])
        self._ensure_dataset_metadata()
        self.analysis_history._entries = [
            AnalysisHistoryEntry.from_dict(item)
            for item in payload["history_entries"]
            if isinstance(item, dict)
        ]
        self.analysis_history.changed.emit()
        self.fitted_models.replace_from_dicts(payload.get("fitted_models", []))
        self.fitted_models.replace_artifacts(payload.get("fitted_model_artifacts", {}))
        self.project_path = Path(path)
        self.project_name = payload["project_name"]
        self.project_modified = False
        self.update_language(payload.get("language", "en"))
        self.update_dashboard_stats()
        self.update_project_title()
        self._restore_active_page(payload.get("active_page", "welcome"))

    def show_settings_menu(self):
        menu = QMenu(self)
        menu.setObjectName("settingsMenu")

        project_menu = menu.addMenu(self.tr("Project"))
        project_menu.setObjectName("projectMenu")

        project_actions = (
            (self.tr("New project"), self.new_project),
            (self.tr("Open project..."), self.open_project),
            (self.tr("Save project"), self.save_project),
            (self.tr("Save project as..."), self.save_project_as),
        )
        for label, callback in project_actions:
            action = QAction(label, project_menu)
            action.triggered.connect(callback)
            project_menu.addAction(action)

        menu.addSeparator()
        language_menu = menu.addMenu(self.tr("language"))
        language_menu.setObjectName("languageMenu")

        actions = []
        for code, key in (("en", "english"), ("es", "spanish"), ("pt", "portuguese")):
            action = QAction(self.tr(key), language_menu)
            action.setCheckable(True)
            action.setChecked(self.current_language == code)
            action.triggered.connect(
                lambda checked=False, language_code=code: self.update_language(
                    language_code
                )
            )
            language_menu.addAction(action)
            actions.append(action)

        menu.exec(
            self.settings_button.mapToGlobal(self.settings_button.rect().topRight())
        )

    def update_language(self, language_code):
        if language_code not in TRANSLATIONS:
            language_code = "en"

        self.current_language = set_language(language_code)
        self.app_settings.setValue("language", self.current_language)
        self.app_settings.sync()

        self.app_title_label.setText(
            '<span style="font-size:22px; font-weight:700;">EspectroApp</span>'
            '<br><span style="font-size:11px; font-weight:500; color:#72DCC5;">'
            + self.tr("suite")
            + "</span>"
        )

        for key, label in self.menu_section_labels.items():
            label.setText(self.tr(key))
        for key, button in self.menu_buttons.items():
            button.setText(self.tr(key))

        self.settings_button.setToolTip(self.tr("settings"))
        self.workspace_title.setText(self.tr("welcome"))
        self.workspace_subtitle.setText(self.tr("subtitle"))

        if hasattr(self, "datasets_title_label"):
            self.datasets_title_label.setText(self.tr("datasets"))
            self.operations_title_label.setText(self.tr("operations"))
            self.models_title_label.setText(self.tr("models"))
            self.history_title_label.setText(self.tr("history"))
            self.export_history_button.setText(self.tr("export"))
            self.clear_history_button.setText(self.tr("clear"))
            self.history_description_label.setText(self.tr("history_desc"))
            self.refresh_history_view()

        # Retranslate every page that is already open. This preserves the
        # current selections, entered parameters, plots and page state.
        for index in range(self.workspace_stack.count()):
            page = self.workspace_stack.widget(index)
            retranslate_widget_tree(page, language_code)

        # Also update detached result/dialog windows that may still be open.
        app = QApplication.instance()
        if app is not None:
            for window in app.topLevelWidgets():
                if window is not self:
                    retranslate_widget_tree(window, language_code)

        # Some pages build dynamic summaries, validation reports and result
        # sections from formatted strings. Translating only the existing widget
        # tree cannot reliably regenerate those texts. Reopen the currently
        # selected module once, after the language state has been updated. This
        # produces the same complete refresh as leaving the page and entering it
        # again, but happens automatically for the user.
        QTimer.singleShot(0, self._refresh_active_page_after_language_change)

    def _refresh_active_page_after_language_change(self):
        """Rebuild the active module so all dynamic text uses the new language."""
        refresh_actions = (
            ("prepare", self.open_data_preparation_assistant),
            ("view", self.view_dataframe),
            ("display", self.open_spectra_window),
            ("preprocess", self.open_preprocessing_window),
            ("pca", self.open_dimensionality_reduction_window),
            ("hca", self.open_hca_window),
            ("models_page", self.open_fitted_models_page),
            ("fusion", self.open_data_fusion_window),
        )

        for key, callback in refresh_actions:
            button = self.menu_buttons.get(key)
            if button is not None and button.isChecked():
                callback()
                return

    def create_stat_card(self, title, value, object_name):
        """Create one compact dashboard statistic card."""
        card = QFrame()
        card.setObjectName(object_name)
        card.setMinimumHeight(92)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 15, 18, 14)
        card_layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setObjectName("statCardTitle")

        value_label = QLabel(str(value))
        value_label.setObjectName("statCardValue")

        card_layout.addWidget(title_label)
        card_layout.addWidget(value_label)
        card_layout.addStretch()

        return card, title_label, value_label

    def update_dashboard_stats(self):
        """Refresh dashboard counters from the current session state."""
        if not hasattr(self, "datasets_value_label"):
            return

        entries = list(getattr(self.analysis_history, "entries", []) or [])
        fitted_model_records = list(getattr(self.fitted_models, "records", ()) or ())

        self.datasets_value_label.setText(str(len(self.dataframes)))
        self.operations_value_label.setText(str(len(entries)))
        self.models_value_label.setText(str(len(fitted_model_records)))

    def create_welcome_page(self):
        """Create the dashboard and session-history page shown on startup."""
        page = QFrame()
        page.setObjectName("workspacePage")

        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 8, 0, 0)
        page_layout.setSpacing(18)

        # Session summary cards.
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(14)

        datasets_card, self.datasets_title_label, self.datasets_value_label = (
            self.create_stat_card(self.tr("datasets"), 0, "datasetsStatCard")
        )
        operations_card, self.operations_title_label, self.operations_value_label = (
            self.create_stat_card(self.tr("operations"), 0, "operationsStatCard")
        )
        models_card, self.models_title_label, self.models_value_label = (
            self.create_stat_card(self.tr("models"), 0, "modelsStatCard")
        )

        stats_layout.addWidget(datasets_card, 1)
        stats_layout.addWidget(operations_card, 1)
        stats_layout.addWidget(models_card, 1)
        page_layout.addLayout(stats_layout)

        # History card.
        history_card = QFrame()
        history_card.setObjectName("historyCard")
        history_card_layout = QVBoxLayout(history_card)
        history_card_layout.setContentsMargins(24, 22, 24, 22)
        history_card_layout.setSpacing(12)

        self.history_title_label = QLabel(self.tr("history"))
        self.history_title_label.setObjectName("welcomeTitle")
        history_card_layout.addWidget(self.history_title_label)

        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(10)

        self.export_history_button = QPushButton(self.tr("export"))
        self.export_history_button.setObjectName("acceptButton")
        self.export_history_button.setMinimumWidth(145)
        self.export_history_button.setIcon(
            QIcon(resource_path("icom/sidebar/upload.svg"))
        )
        self.export_history_button.setIconSize(QSize(17, 17))
        self.export_history_button.clicked.connect(self.export_analysis_history)

        self.clear_history_button = QPushButton(self.tr("clear"))
        self.clear_history_button.setObjectName("deleteButton")
        self.clear_history_button.setMinimumWidth(140)
        self.clear_history_button.clicked.connect(self.confirm_clear_analysis_history)

        actions_layout.addWidget(self.export_history_button)
        actions_layout.addWidget(self.clear_history_button)
        actions_layout.addStretch()
        history_card_layout.addLayout(actions_layout)

        self.history_description_label = QLabel(self.tr("history_desc"))
        self.history_description_label.setObjectName("welcomeDescription")
        self.history_description_label.setWordWrap(True)
        history_card_layout.addWidget(self.history_description_label)

        self.history_scroll = QScrollArea()
        self.history_scroll.setObjectName("historyScroll")
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setFrameShape(QFrame.NoFrame)
        self.history_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.history_container = QWidget()
        self.history_container.setObjectName("historyContainer")
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.setSpacing(12)
        self.history_layout.setAlignment(Qt.AlignTop)

        self.history_scroll.setWidget(self.history_container)
        history_card_layout.addWidget(self.history_scroll, 1)
        page_layout.addWidget(history_card, 1)

        self.refresh_history_view()
        self.update_dashboard_stats()
        return page

    def _clear_layout(self, layout):
        """Remove every widget and nested layout from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()

            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _translate_history_fragment(self, value):
        """Translate stored history text without modifying the persisted entry."""
        if value is None:
            return ""

        text = str(value)

        # Exact translation first.
        translated = self.tr(text)
        if translated != text:
            return translated

        # Older entries may contain several English labels inside one value.
        # Replace only interface/history phrases; dataset and class names remain
        # untouched.
        replacement_keys = (
            "Dataset loaded",
            "Dataset prepared",
            "PCA analysis",
            "t-SNE analysis",
            "HCA analysis",
            "Low-level fusion",
            "Mid-level fusion",
            "First derivative",
            "Second derivative",
            "Mean normalization",
            "Area normalization",
            "Savitzky-Golay smoothing",
            "Gaussian smoothing",
            "Moving-average smoothing",
            "Linear baseline correction",
            "Shirley baseline correction",
            "Stacked spectra visualization",
            "Source datasets",
            "Range relationship",
            "Common spectral range available",
            "No common spectral range",
            "Concatenation",
            "Vertical",
            "Horizontal",
            "Interpolation",
            "Enabled",
            "Disabled",
            "Fusion range",
            "Original axes",
            "Common range",
            "PCA components",
            "Confidence interval",
            "2D axes",
            "3D axes",
            "Output dimensions",
            "Perplexity",
            "Iterations",
            "Distance metric",
            "Linkage method",
            "Number of clusters",
            "Offset mode",
            "Offset value",
            "Labels",
            "Shown",
            "Hidden",
            "Maximum spectra",
            "Sample type",
            "All",
            "Automatic",
            "Manual",
        )

        # Longer phrases first to avoid partial replacements.
        for key in sorted(replacement_keys, key=len, reverse=True):
            localized = self.tr(key)
            if localized != key:
                text = text.replace(key, localized)

        return text

    def _format_history_parameter(self, key, value):
        """Return one localized history parameter pair."""
        localized_key = self._translate_history_fragment(key)

        if isinstance(value, bool):
            localized_value = self.tr("Enabled" if value else "Disabled")
        elif isinstance(value, (list, tuple)):
            localized_value = ", ".join(
                self._translate_history_fragment(item) for item in value
            )
        else:
            localized_value = self._translate_history_fragment(value)

        return f"{localized_key}: {localized_value}"

    def refresh_history_view(self):
        """Refresh the visible session history grouped by dataset."""
        self.update_dashboard_stats()
        if not hasattr(self, "history_layout"):
            return

        self._clear_layout(self.history_layout)
        grouped_entries = self.analysis_history.grouped_by_dataset()

        if not grouped_entries:
            empty_card = QFrame()
            empty_card.setObjectName("historyEmptyState")
            empty_card.setMinimumHeight(220)
            empty_card.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Expanding,
            )

            empty_layout = QVBoxLayout(empty_card)
            empty_layout.setContentsMargins(28, 28, 28, 28)
            empty_layout.setSpacing(10)
            empty_layout.setAlignment(Qt.AlignCenter)

            empty_icon = QLabel("⌁")
            empty_icon.setObjectName("historyEmptyIcon")
            empty_icon.setAlignment(Qt.AlignCenter)

            empty_title = QLabel(self.tr("empty_title"))
            empty_title.setObjectName("historyEmptyTitle")
            empty_title.setAlignment(Qt.AlignCenter)

            empty_text = QLabel(self.tr("empty_text"))
            empty_text.setObjectName("historyEmptyText")
            empty_text.setWordWrap(True)
            empty_text.setAlignment(Qt.AlignCenter)
            empty_text.setMaximumWidth(650)

            empty_layout.addStretch()
            empty_layout.addWidget(empty_icon)
            empty_layout.addWidget(empty_title)
            empty_layout.addWidget(empty_text, alignment=Qt.AlignHCenter)
            empty_layout.addStretch()

            self.history_layout.addWidget(empty_card)
            self.clear_history_button.setEnabled(False)
            self.export_history_button.setEnabled(False)
            return

        self.clear_history_button.setEnabled(True)
        self.export_history_button.setEnabled(True)

        for dataset_name, entries in grouped_entries.items():
            card = QFrame()
            card.setObjectName("quickStartCard")
            card.setMaximumWidth(920)
            card.setMinimumWidth(760)
            card.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Minimum,
            )

            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(22, 18, 22, 18)
            card_layout.setSpacing(12)

            dataset_label = QLabel(dataset_name)
            dataset_label.setObjectName("quickStartTitle")
            card_layout.addWidget(dataset_label)

            for entry in entries:
                row = QFrame()
                row.setObjectName("historyRow")
                row.setStyleSheet("""
                    QFrame#historyRow {
                        background-color: transparent;
                        border: none;
                    }

                    QFrame#historyRow QLabel {
                        background-color: transparent;
                    }
                """)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 8, 0, 8)
                row_layout.setSpacing(12)

                timestamp = QLabel(entry.timestamp_text)
                timestamp.setObjectName("helpLabel")
                timestamp.setMinimumWidth(145)

                operation_text = self._translate_history_fragment(entry.operation)
                if entry.output_dataset:
                    operation_text += f"  →  {entry.output_dataset}"

                if entry.parameters:
                    parameter_text = " · ".join(
                        self._format_history_parameter(key, value)
                        for key, value in entry.parameters.items()
                    )
                    operation_text += f"\n{parameter_text}"

                operation = QLabel(operation_text)
                operation.setWordWrap(True)
                operation.setMinimumHeight(operation.sizeHint().height())
                operation.setSizePolicy(
                    QSizePolicy.Expanding,
                    QSizePolicy.Minimum,
                )
                operation.setAlignment(Qt.AlignLeft | Qt.AlignTop)

                row_layout.addWidget(timestamp)
                row_layout.addWidget(operation, 1)
                card_layout.addWidget(row)

            self.history_layout.addWidget(
                card,
                alignment=Qt.AlignHCenter,
            )

    def export_analysis_history(self):
        """Export the complete history as CSV or JSON."""
        if not self.analysis_history.entries:
            QMessageBox.information(
                self,
                self.tr("Empty history"),
                self.tr("There are no history entries to export."),
            )
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            self.tr("Export analysis history"),
            "analysis_history.csv",
            ("CSV file (*.csv);;" "JSON file (*.json)"),
        )

        if not path:
            return

        try:
            if selected_filter.startswith("JSON") or path.lower().endswith(".json"):
                if not path.lower().endswith(".json"):
                    path += ".json"

                self.analysis_history.export_json(path)

            else:
                if not path.lower().endswith(".csv"):
                    path += ".csv"

                self.analysis_history.export_csv(path)

            QMessageBox.information(
                self,
                self.tr("History exported"),
                (
                    self.tr(
                        "The analysis history was exported successfully:\n{path}",
                        path=path,
                    )
                ),
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                self.tr("Export error"),
                (
                    self.tr(
                        "The analysis history could not be exported:\n{error}",
                        error=error,
                    )
                ),
            )

    def confirm_clear_analysis_history(self):
        """Ask for confirmation and clear only the analysis history."""
        confirmation_box = QMessageBox(self)
        confirmation_box.setIcon(QMessageBox.Question)
        confirmation_box.setWindowTitle(self.tr("Clear history"))
        confirmation_box.setText(
            self.tr(
                "Clear the complete saved analysis history?\n\n"
                "Loaded datasets and generated results will not be deleted."
            )
        )
        confirmation_box.setStandardButtons(
            QMessageBox.Yes | QMessageBox.No
        )
        confirmation_box.setDefaultButton(QMessageBox.No)

        yes_button = confirmation_box.button(QMessageBox.Yes)
        no_button = confirmation_box.button(QMessageBox.No)

        if yes_button is not None:
            yes_button.setText(self.tr("Yes"))
        if no_button is not None:
            no_button.setText(self.tr("No"))

        answer = confirmation_box.exec()

        if answer == QMessageBox.Yes:
            self.analysis_history.clear()

    def record_analysis_step(
        self,
        dataset,
        operation,
        output_dataset=None,
        parameters=None,
        source_datasets=None,
    ):
        """Register one operation in the current session history."""
        self.analysis_history.add(
            dataset=dataset,
            operation=operation,
            output_dataset=output_dataset,
            parameters=parameters,
            source_datasets=source_datasets,
        )
        self.mark_project_modified()

    def _ensure_dataset_metadata(self):
        """Assign stable identifiers and retain dataset metadata centrally."""
        if not hasattr(self, "dataset_metadata"):
            self.dataset_metadata = {}

        for index, dataframe in enumerate(self.dataframes):
            attrs = dict(getattr(dataframe, "attrs", {}) or {})
            if not attrs.get("dataset_id"):
                attrs["dataset_id"] = str(uuid4())
            if index < len(self.nombres_archivos):
                attrs.setdefault("display_name", str(self.nombres_archivos[index]))
            dataframe.attrs = attrs
            self.dataset_metadata[str(attrs["dataset_id"])] = dict(attrs)

    def _unique_dataset_name(self, requested_name):
        """Return a visible dataset name that does not duplicate an existing one."""
        base = str(requested_name).strip() or self.tr("Generated dataset")
        existing = {str(name) for name in self.nombres_archivos}
        if base not in existing:
            return base
        suffix = 2
        while f"{base} ({suffix})" in existing:
            suffix += 1
        return f"{base} ({suffix})"

    def record_fitted_model(
        self,
        *,
        method_id,
        dataset,
        name=None,
        parameters=None,
        metrics=None,
        artifact_path=None,
        artifact=None,
    ):
        """Register one reusable fitted model using the central method registry."""
        definition = self.method_registry.get(method_id)
        if not definition.produces_model:
            raise ValueError(
                f"The registered method '{definition.method_id}' does not produce a fitted model."
            )
        record = self.fitted_models.create(
            method_id=definition.method_id,
            name=name or definition.name,
            dataset=dataset,
            parameters=parameters,
            metrics=metrics,
            artifact_path=(artifact_path or ("embedded" if artifact is not None else None)),
            artifact=artifact,
        )
        self.mark_project_modified()
        return record

    def get_export_x_column_name(self):
        """
        Determines a standardized X-axis label for exported CSV files based on the current plot label.
        The method inspects the stored X-axis text and maps it to a canonical name such as 'Raman Shift', 'Wavenumber', or a generic 'X Axis'.

        Returns
        -------
        str
            Normalized X-axis label to use as the first column name when exporting data.
        """
        etiqueta = str(self.x_label).strip().lower()

        if "raman" in etiqueta:
            return "Raman Shift"

        if (
            "wavenumber" in etiqueta
            or "wave number" in etiqueta
            or "numero de onda" in etiqueta
            or "número de onda" in etiqueta
        ):
            return "Wavenumber"

        return "X Axis"

    def detect_labels_from_df(self, df):
        """
        Infers human-readable axis labels from the first cell of an internal-format DataFrame.
        The method inspects common keywords to decide whether the X-axis represents Raman shift, wavenumber, or a generic axis and pairs it with a default intensity label.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame in the application's internal format, where the first cell may contain an axis description.

        Returns
        -------
        tuple of str
            A pair of strings (x_label, y_label) suitable for labeling plots derived from the given DataFrame.
        """
        try:
            primera_celda = str(df.iloc[0, 0]).strip().lower()
        except Exception:
            return "X Axis", "Intensity"

        if "raman shift" in primera_celda:
            return "Raman Shift (cm⁻¹)", "Intensity"

        if (
            "wavenumber" in primera_celda
            or "wave number" in primera_celda
            or "numero de onda" in primera_celda
            or "número de onda" in primera_celda
        ):
            return "Wavenumber (cm⁻¹)", "Intensity"

        if "x axis" in primera_celda:
            return "X Axis", "Intensity"

        return "X Axis", "Intensity"

    def create_section(self, title, description=None):
        """
        Builds a styled group box section used to organize related controls in the main menu.
        The section displays an optional description label at the top and returns both the container widget and its layout for further customization.

        Parameters
        ----------
        title : str
            Text to display as the section title on the group box frame.
        description : str, optional
            Additional descriptive text shown below the title, by default None.

        Returns
        -------
        tuple
            A tuple (section, section_layout) containing the QGroupBox widget and its QVBoxLayout.
        """
        section = QGroupBox(title)
        section.setObjectName("sectionGroup")
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(16, 24, 16, 16)
        section_layout.setSpacing(10)

        if description:
            description_label = QLabel(description)
            description_label.setObjectName("sectionDescription")
            description_label.setWordWrap(True)
            section_layout.addWidget(description_label)

        return section, section_layout

    def create_menu_label(self, text):
        label = QLabel(text)
        label.setObjectName("menuSectionLabel")
        label.setAlignment(Qt.AlignLeft)
        return label

    def create_button(self, texto, icon_path=None, funcion_click=None):
        """
        Creates a styled push button for use in the main menu sections.
        The button can optionally display an icon and execute a callback when clicked, while maintaining a consistent visual appearance.

        Parameters
        ----------
        texto : str
            Caption text displayed on the button.
        icon_path : str, optional
            Path to an icon file to show alongside the text, by default None.
        funcion_click : callable, optional
            Slot or function to connect to the button's clicked signal, by default None.

        Returns
        -------
        QPushButton
            Configured button widget ready to be added to a layout.
        """
        boton = QPushButton(texto)
        boton.setObjectName("menuButton")
        boton.setMinimumHeight(44)
        boton.setCursor(Qt.PointingHandCursor)

        if icon_path:
            boton.setIcon(QIcon(icon_path))
            boton.setIconSize(QSize(20, 20))

        if funcion_click:
            boton.clicked.connect(funcion_click)

        boton.setStyleSheet(MENU_BUTTON_STYLE)
        boton.setCheckable(True)
        boton.setAutoExclusive(True)

        return boton

    def open_dimensionality_reduction_window(self):
        self.workspace_title.setAlignment(Qt.AlignLeft)
        self.workspace_subtitle.setAlignment(Qt.AlignLeft)
        if not self.dataframes:
            QMessageBox.warning(
                self,
                self.tr("No data"),
                self.tr("No spectral dataset has been loaded yet."),
            )
            return

        self.workspace_title.setText("PCA and t-SNE analysis")
        self.workspace_subtitle.setText(
            "Select a spectral matrix and configure "
            "the multivariate analysis methods."
        )

        if hasattr(self, "dimensionality_page"):
            old_index = self.workspace_stack.indexOf(self.dimensionality_page)

            if old_index != -1:
                self.workspace_stack.removeWidget(self.dimensionality_page)

            self.dimensionality_page.deleteLater()

        self.dimensionality_page = DimensionalityReductionWindow(
            self.dataframes,
            self.nombres_archivos,
            self,
            embedded=True,
        )

        self.workspace_stack.addWidget(self.dimensionality_page)
        self.workspace_stack.setCurrentWidget(self.dimensionality_page)

    def open_data_fusion_window(self):
        self.workspace_title.setAlignment(Qt.AlignLeft)
        self.workspace_subtitle.setAlignment(Qt.AlignLeft)
        if len(self.dataframes) < 2:
            QMessageBox.warning(
                self,
                self.tr("Insufficient datasets"),
                self.tr("Data fusion requires at least two loaded data matrices."),
            )
            return

        self.workspace_title.setText("Data fusion")

        self.workspace_subtitle.setText(
            "Select the input datasets and configure " "the fusion strategy."
        )

        if hasattr(self, "data_fusion_page"):
            old_index = self.workspace_stack.indexOf(self.data_fusion_page)

            if old_index != -1:
                self.workspace_stack.removeWidget(self.data_fusion_page)

            self.data_fusion_page.deleteLater()

        self.data_fusion_page = DataFusionSelectionWindow(
            self.dataframes,
            self.nombres_archivos,
            self,
            embedded=True,
        )

        self.workspace_stack.addWidget(self.data_fusion_page)

        self.workspace_stack.setCurrentWidget(self.data_fusion_page)

    def create_separator(self, titulo):
        """
        Creates a styled label that visually separates groups of options in the main menu.
        The separator displays centered bold text and can be inserted between sections to improve readability.

        Parameters
        ----------
        titulo : str
            Text to display inside the separator label.

        Returns
        -------
        QLabel
            Configured label widget that can be added to a layout as a visual divider.
        """
        label = QLabel(titulo)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #AAB7C4; font-size: 16px; font-weight: bold;")
        return label

    def register_exported_dataframe(self, df_exportado, nombre_archivo):
        """
        Registers an exported CSV-ready DataFrame in the
        current EspectroApp session.
        """

        nombre_base = os.path.basename(nombre_archivo)
        nombre_visible = nombre_base
        contador = 2

        while nombre_visible in self.nombres_archivos:
            raiz, extension = os.path.splitext(nombre_base)

            nombre_visible = f"{raiz}_{contador}{extension}"
            contador += 1

        if df_exportado is None or df_exportado.empty:
            QMessageBox.warning(
                self,
                self.tr("Empty DataFrame"),
                self.tr(
                    "The exported DataFrame is empty and could not be added "
                    "to the current session."
                ),
            )
            return

        df_interno = pd.concat(
            [
                pd.DataFrame(
                    [list(df_exportado.columns)],
                    columns=df_exportado.columns,
                ),
                df_exportado.reset_index(drop=True),
            ],
            ignore_index=True,
        )

        df_interno.columns = range(df_interno.shape[1])

        df_interno.attrs = df_exportado.attrs.copy()

        self.dataframes.append(df_interno)
        self.nombres_archivos.append(nombre_visible)
        self.mark_project_modified()

    def view_dataframe(self):
        self.workspace_title.setAlignment(Qt.AlignLeft)
        self.workspace_subtitle.setAlignment(Qt.AlignLeft)
        if not self.dataframes:
            QMessageBox.warning(
                self,
                self.tr("No data"),
                self.tr("No file has been loaded yet."),
            )
            return

        self.workspace_title.setText(self.tr("Loaded data matrices"))

        self.workspace_subtitle.setText(
            self.tr(
                "Review, inspect or remove the datasets loaded in the current session."
            )
        )

        def eliminar_callback(idx):
            if idx < 0 or idx >= len(self.dataframes):
                return

            del self.dataframes[idx]
            del self.nombres_archivos[idx]
            self.mark_project_modified()

            if self.dataframes:
                self.view_dataframe()
            else:
                self.show_welcome_page()

        def visualizar_callback(idx):
            if idx < 0 or idx >= len(self.dataframes):
                return

            df_a_mostrar = self.dataframes[idx]

            self.ventana_tabla = VerDf(df_a_mostrar)
            self.ventana_tabla.show()

        def informacion_callback(idx):
            if idx < 0 or idx >= len(self.dataframes):
                return

            df_seleccionado = self.dataframes[idx]
            nombre_seleccionado = self.nombres_archivos[idx]

            if hasattr(
                self,
                "dataframe_info_page",
            ):
                old_index = self.workspace_stack.indexOf(self.dataframe_info_page)

                if old_index != -1:
                    self.workspace_stack.removeWidget(self.dataframe_info_page)

                self.dataframe_info_page.deleteLater()

            self.dataframe_info_page = DataFrameInformationPage(
                df=df_seleccionado,
                file_name=nombre_seleccionado,
                back_callback=self.view_dataframe,
            )

            self.workspace_title.setText("Dataset information")

            self.workspace_subtitle.setText(
                "Review the sample types and the "
                "number of spectra associated with "
                "each type."
            )

            self.workspace_stack.addWidget(self.dataframe_info_page)

            self.workspace_stack.setCurrentWidget(self.dataframe_info_page)

        if hasattr(
            self,
            "dataframe_selection_page",
        ):
            old_index = self.workspace_stack.indexOf(self.dataframe_selection_page)

            if old_index != -1:
                self.workspace_stack.removeWidget(self.dataframe_selection_page)

            self.dataframe_selection_page.deleteLater()

        self.dataframe_selection_page = DataFrameSelectionWindow(
            self.dataframes,
            self.nombres_archivos,
            eliminar_callback,
            visualizar_callback,
            informacion_callback,
            embedded=True,
        )

        self.workspace_stack.addWidget(self.dataframe_selection_page)

        self.workspace_stack.setCurrentWidget(self.dataframe_selection_page)

    def open_preprocessing_window(self):
        self.workspace_title.setAlignment(Qt.AlignLeft)
        self.workspace_subtitle.setAlignment(Qt.AlignLeft)
        if not self.dataframes:
            QMessageBox.warning(
                self,
                self.tr("No data"),
                self.tr("No spectral dataset has been loaded yet."),
            )
            return

        self.workspace_title.setText(self.tr("Spectral preprocessing"))
        self.workspace_subtitle.setText(
            self.tr(
                "Select a spectral matrix and configure "
                "the preprocessing methods."
            )
        )

        # Eliminar la página anterior, si ya fue creada
        if hasattr(self, "preprocessing_page"):
            old_index = self.workspace_stack.indexOf(self.preprocessing_page)

            if old_index != -1:
                self.workspace_stack.removeWidget(self.preprocessing_page)

            self.preprocessing_page.deleteLater()

        # Crear el formulario como página integrada
        self.preprocessing_page = PreprocessingWindow(
            self.dataframes,
            self.nombres_archivos,
            self,
            embedded=True,
        )

        self.workspace_stack.addWidget(self.preprocessing_page)

        self.workspace_stack.setCurrentWidget(self.preprocessing_page)

    def _add_model_output_dataset(self, dataframe, name):
        """Add a model output with a stable identifier and a unique visible name."""
        visible_name = self._unique_dataset_name(name)
        dataframe = dataframe.copy()
        dataframe.attrs = dict(getattr(dataframe, "attrs", {}))
        dataframe.attrs["dataset_id"] = str(uuid4())
        dataframe.attrs["display_name"] = visible_name
        self.dataframes.append(dataframe)
        self.nombres_archivos.append(visible_name)
        self.record_analysis_step(
            dataset=visible_name,
            operation="Apply fitted model",
            output_dataset=visible_name,
        )
        self.update_dashboard_stats()

    def open_fitted_models_page(self):
        """Open the project-level fitted-model manager."""
        self._ensure_dataset_metadata()
        if hasattr(self, "fitted_models_page") and self.fitted_models_page is not None:
            self.workspace_stack.removeWidget(self.fitted_models_page)
            self.fitted_models_page.deleteLater()

        self.fitted_models_page = FittedModelsPage(
            model_manager=self.fitted_models,
            method_registry=self.method_registry,
            translator=self.tr,
            on_project_modified=self.mark_project_modified,
            on_back=self.show_welcome_page,
            datasets_provider=lambda: (self.dataframes, self.nombres_archivos),
            on_dataset_created=self._add_model_output_dataset,
            parent=self,
        )
        self.workspace_stack.addWidget(self.fitted_models_page)
        self.workspace_stack.setCurrentWidget(self.fitted_models_page)
        self.workspace_title.setAlignment(Qt.AlignLeft)
        self.workspace_title.setText(self.tr("Reference PCA models"))
        self.workspace_subtitle.setText(self.tr("Manage saved PCA reference models and project new samples into them."))
        button = self.menu_buttons.get("models_page")
        if button is not None:
            button.setChecked(True)

    def open_hca_window(self):
        self.workspace_title.setAlignment(Qt.AlignLeft)
        self.workspace_subtitle.setAlignment(Qt.AlignLeft)
        if not self.dataframes:
            QMessageBox.warning(
                self,
                self.tr("No data"),
                self.tr("No spectral dataset has been loaded yet."),
            )
            return

        self.workspace_title.setText("Hierarchical cluster analysis")

        self.workspace_subtitle.setText(
            "Select a spectral matrix, distance metric, "
            "linkage method and number of clusters."
        )

        if hasattr(self, "hca_result_page"):
            old_result_index = self.workspace_stack.indexOf(self.hca_result_page)

            if old_result_index != -1:
                self.workspace_stack.removeWidget(self.hca_result_page)

            self.hca_result_page.deleteLater()
            del self.hca_result_page

        self.hca_page = VentanaHca(
            self.dataframes,
            self.nombres_archivos,
            self,
            embedded=True,
        )

        self.workspace_stack.addWidget(self.hca_page)

        self.workspace_stack.setCurrentWidget(self.hca_page)

    def open_file_dialog(self):
        """
        Opens a file selection dialog to load one or more spectral data files into the application.
        The method starts a background loading thread for the chosen paths and updates the internal list of dataset names, or shows a warning if nothing is selected.

        Returns
        -------
        None
        """
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            self.tr("Select spectral files"),
            "",
            "Supported files (*.csv *.xlsx *.xls *.spa *.SPA);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;SPA Files (*.spa *.SPA)",
        )

        if file_paths:
            # Dataset names are registered only after each file has actually
            # been loaded. This prevents names and DataFrames from becoming
            # misaligned when several files are loaded asynchronously.
            thread = FileLoaderThread(file_paths, self)
            thread.file_loaded.connect(self.process_loaded_files)

            thread.finished.connect(
                lambda: self.threads.remove(thread) if thread in self.threads else None
            )
            thread.finished.connect(thread.deleteLater)

            self.threads.append(thread)
            thread.start()

        else:
            QMessageBox.warning(
                self,
                self.tr("No selection"),
                self.tr("No files were selected."),
            )

    def open_data_preparation_assistant(self):
        if not self.dataframes:
            QMessageBox.warning(
                self,
                self.tr("No data"),
                self.tr("Load at least one CSV, Excel or SPA file first."),
            )
            return
        self.workspace_title.setText(self.tr("Data Preparation Assistant"))
        self.workspace_subtitle.setText(
            self.tr(
                "Convert raw tabular datasets to EspectroApp's validated internal format."
            )
        )
        if hasattr(self, "data_preparation_page"):
            old_index = self.workspace_stack.indexOf(self.data_preparation_page)
            if old_index != -1:
                self.workspace_stack.removeWidget(self.data_preparation_page)
            self.data_preparation_page.deleteLater()
        self.data_preparation_page = DataPreparationAssistant(
            self.dataframes, self.nombres_archivos, self
        )
        self.data_preparation_page.prepared_data.connect(self.register_prepared_dataset)
        self.data_preparation_page.back_requested.connect(self.show_welcome_page)
        self.workspace_stack.addWidget(self.data_preparation_page)
        self.workspace_stack.setCurrentWidget(self.data_preparation_page)

    def register_prepared_dataset(self, dataframe, name):
        visible_name = (
            str(name).strip() or f"Prepared dataset {len(self.dataframes) + 1}"
        )
        base = visible_name
        counter = 2
        while visible_name in self.nombres_archivos:
            visible_name = f"{base}_{counter}"
            counter += 1
        dataframe = dataframe.copy()
        dataframe.attrs["data_status"] = "ready"
        self.dataframes.append(dataframe)
        self.nombres_archivos.append(visible_name)

        # Assign a stable dataset ID immediately and retain all preparation
        # metadata (sample names, classes, axis and pipeline information).
        self._ensure_dataset_metadata()

        self.record_analysis_step(dataset=visible_name, operation="Dataset prepared")
        self.mark_project_modified()

    def show_welcome_page(self):
        self.workspace_title.setText(self.tr("welcome"))
        self.workspace_subtitle.setText(self.tr("subtitle"))
        self.workspace_stack.setCurrentWidget(self.welcome_page)

    def closeEvent(self, event):
        if not self._confirm_discard_unsaved_changes():
            event.ignore()
            return
        try:
            self.analysis_history.save()
        except OSError as error:
            QMessageBox.warning(
                self,
                self.tr("History warning"),
                (
                    self.tr(
                        "The analysis history could not be saved:\n{error}",
                        error=error,
                    )
                ),
            )

        for thread in getattr(self, "threads", []):
            if thread.isRunning():
                thread.quit()
                thread.wait(3000)

        event.accept()

    def process_loaded_files(self, df, source_name):
        """
        Integrates a newly loaded spectral DataFrame into the current application session.
        The method stores original and working copies, triggers a row-fix dialog when needed, and makes the data available for plotting and further analysis.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame loaded from one or more spectral files, expected to follow the application's internal format.

        Returns
        -------
        None
        """
        self.df_original = df.copy()
        self.df = df
        self.df_final = df.copy()
        self.dataframe = self.df_final
        self.dataframes.append(df)
        self.nombres_archivos.append(str(source_name))
        self.index_actual = len(self.dataframes) - 1

        dataset_name = self.nombres_archivos[self.index_actual]
        self.record_analysis_step(
            dataset=os.path.basename(str(dataset_name)),
            operation="Dataset loaded",
        )
        status = str(df.attrs.get("data_status", "ready")).lower()
        if status == "raw":
            print(
                "RAW dataset loaded. Use the Data Preparation Assistant before analysis."
            )
            return

        col, fil = get_column_with_fewest_rows(df)
        if len(df) != fil:
            self.eliminar_filas = DataFrameFixWindow(df.copy())
            self.eliminar_filas.df_modificado.connect(self.receive_modified_dataframe)
            self.eliminar_filas.show()
        else:
            print("No preprocessing is required; the spectra can be plotted directly.")

    def receive_modified_dataframe(self, df_nuevo):
        """
        Updates the current dataset with a cleaned or edited DataFrame received from a child window.
        The method replaces both the working copy and the corresponding entry in the list of loaded dataframes so subsequent actions use the corrected data.

        Parameters
        ----------
        df_nuevo : pandas.DataFrame
            Modified DataFrame that should replace the previously stored version of the current dataset.

        Returns
        -------
        None
        """
        self.df = df_nuevo
        self.df_final = df_nuevo
        self.dataframe = df_nuevo
        if hasattr(self, "index_actual") and self.index_actual is not None:
            self.dataframes[self.index_actual] = df_nuevo
        self.mark_project_modified()

    def handle_spectra_action(
        self,
        dataset_index,
        configuration,
    ):
        try:
            if dataset_index < 0 or dataset_index >= len(self.dataframes):
                raise IndexError("The selected dataset index is invalid.")

            df = self.dataframes[dataset_index]

            self.complete_df = df.copy()
            self.df_original = df.copy()
            self.df_final = df.copy()

            self.x_label, self.y_label = self.detect_labels_from_df(df)

            self.raman_shift = self.complete_df.iloc[1:, 0].reset_index(drop=True)

            sample_types = self.complete_df.iloc[
                0,
                1:,
            ]

            self.color_mapping = assign_type_colors(sample_types)

            self.current_spectra_dataset_index = dataset_index

            self.process_spectra_configuration(configuration)

        except Exception as error:
            print(
                "Error in handle_spectra_action:",
                error,
            )

            QMessageBox.critical(
                self,
                self.tr("Processing error"),
                self.tr("An error occurred:\n{error}", error=str(error)),
            )

    def open_spectra_window(self, df=None):
        self.workspace_title.setAlignment(Qt.AlignLeft)
        self.workspace_subtitle.setAlignment(Qt.AlignLeft)
        if not self.dataframes:
            QMessageBox.warning(
                self,
                self.tr("No data"),
                self.tr("No spectral dataset has been loaded yet."),
            )
            return

        self.workspace_title.setText(self.tr("Display spectra and export CSV"))
        self.workspace_subtitle.setText(
            self.tr("Select the input dataset and the operation to perform.")
        )

        if hasattr(self, "spectra_options_page"):
            index = self.workspace_stack.indexOf(self.spectra_options_page)

            if index != -1:
                self.workspace_stack.removeWidget(self.spectra_options_page)

            self.spectra_options_page.deleteLater()

        self.spectra_options_page = SpectraExportOptionsWindow(
            self.dataframes,
            self.nombres_archivos,
        )

        self.spectra_options_page.seleccion_confirmada.connect(
            self.handle_spectra_action
        )

        self.spectra_options_page.cancel_requested.connect(self.show_welcome_page)

        self.workspace_stack.addWidget(self.spectra_options_page)
        self.workspace_stack.setCurrentWidget(self.spectra_options_page)

    def show_welcome_page(self):
        self.workspace_title.setAlignment(Qt.AlignCenter)
        self.workspace_subtitle.setAlignment(Qt.AlignCenter)
        self.workspace_title.setText(self.tr("welcome"))
        self.workspace_subtitle.setText(self.tr("subtitle"))

        self.refresh_history_view()
        self.workspace_stack.setCurrentWidget(self.welcome_page)

    def return_to_spectra_options(self):
        if hasattr(self, "spectra_options_page"):
            self.workspace_title.setText(self.tr("Display spectra and export CSV"))
            self.workspace_subtitle.setText(
                self.tr("Select the input dataset and the operation to perform.")
            )

            self.workspace_stack.setCurrentWidget(self.spectra_options_page)

    def prepare_spectra_results_page(self):
        """
        Creates a clean tabbed page for spectral plots.
        """

        if hasattr(self, "spectra_results_page"):
            old_index = self.workspace_stack.indexOf(self.spectra_results_page)

            if old_index != -1:
                self.workspace_stack.removeWidget(self.spectra_results_page)

            self.spectra_results_page.deleteLater()

        self.spectra_results_page = SpectraResultsPage(
            back_callback=self.return_to_spectra_options
        )

        self.workspace_stack.addWidget(self.spectra_results_page)

        self.workspace_title.setText("Spectral visualization results")

        self.workspace_subtitle.setText(
            "Use the tabs to inspect the selected " "spectral plots."
        )

        self.workspace_stack.setCurrentWidget(self.spectra_results_page)

    def process_spectra_configuration(
        self,
        configuration,
    ):
        plots = configuration.get("plots", {})

        self.min_val = configuration.get("range_min")

        self.max_val = configuration.get("range_max")

        self.selected_type = configuration.get("sample_type")

        export_action = configuration.get("export_action")

        has_plots = any(plots.values())

        if has_plots:
            self.prepare_spectra_results_page()

            df_to_plot = self.complete_df.reset_index(drop=True)

            if plots.get("full"):
                full_widget = SpectraPlotWindow(
                    df_to_plot,
                    self.raman_shift,
                    self.color_mapping,
                    x_label=self.x_label,
                    y_label=self.y_label,
                )

                self.spectra_results_page.add_plot(
                    full_widget,
                    "Full spectra",
                )

            if plots.get("limited"):
                limited_widget = LimitedRangeSpectraPlotWindow(
                    df_to_plot,
                    self.raman_shift,
                    self.color_mapping,
                    self.min_val,
                    self.max_val,
                    x_label=self.x_label,
                    y_label=self.y_label,
                )

                self.spectra_results_page.add_plot(
                    limited_widget,
                    (f"Range " f"{self.min_val:g}–" f"{self.max_val:g}"),
                )

            if plots.get("type"):
                type_widget = SpectraByTypePlotWindow(
                    df_to_plot,
                    self.raman_shift,
                    self.color_mapping,
                    self.selected_type,
                    x_label=self.x_label,
                    y_label=self.y_label,
                )

                self.spectra_results_page.add_plot(
                    type_widget,
                    str(self.selected_type),
                )

            if plots.get("limited_type"):
                limited_type_widget = LimitedRangeSpectraByTypePlotWindow(
                    df_to_plot,
                    self.raman_shift,
                    self.color_mapping,
                    self.selected_type,
                    self.min_val,
                    self.max_val,
                    x_label=self.x_label,
                    y_label=self.y_label,
                )

                self.spectra_results_page.add_plot(
                    limited_type_widget,
                    (
                        f"{self.selected_type} — "
                        f"{self.min_val:g}–"
                        f"{self.max_val:g}"
                    ),
                )

        if plots.get("stacked"):
            stacked_options = configuration.get("stacked_options") or {}

            try:
                stacked_widget = StackedSpectraPlotWindow(
                    df_to_plot,
                    self.raman_shift,
                    self.color_mapping,
                    x_label=self.x_label,
                    y_label=self.y_label,
                    offset_mode=stacked_options.get(
                        "offset_mode",
                        "automatic",
                    ),
                    offset_value=stacked_options.get(
                        "offset_value",
                        1.15,
                    ),
                    show_labels=stacked_options.get(
                        "show_labels",
                        True,
                    ),
                    maximum_spectra=stacked_options.get(
                        "maximum_spectra",
                        10,
                    ),
                    sample_type=stacked_options.get("sample_type"),
                    range_min=stacked_options.get("range_min"),
                    range_max=stacked_options.get("range_max"),
                )

                self.spectra_results_page.add_plot(
                    stacked_widget,
                    "Stacked spectra",
                )

                dataset_name = (
                    os.path.basename(
                        str(self.nombres_archivos[self.current_spectra_dataset_index])
                    )
                    if self.current_spectra_dataset_index < len(self.nombres_archivos)
                    else ("Dataset " f"{self.current_spectra_dataset_index + 1}")
                )

                self.record_analysis_step(
                    dataset=dataset_name,
                    operation="Stacked spectra visualization",
                    parameters={
                        "Offset mode": stacked_options.get(
                            "offset_mode",
                            "automatic",
                        ),
                        "Offset value": stacked_options.get(
                            "offset_value",
                            1.15,
                        ),
                        "Labels": (
                            "Shown"
                            if stacked_options.get(
                                "show_labels",
                                True,
                            )
                            else "Hidden"
                        ),
                        "Maximum spectra": stacked_options.get(
                            "maximum_spectra",
                            10,
                        ),
                        "Sample type": (stacked_options.get("sample_type") or "All"),
                        "Range": (
                            (
                                f"{stacked_options.get('range_min')}–"
                                f"{stacked_options.get('range_max')}"
                            )
                            if stacked_options.get("range_min") is not None
                            else "Full"
                        ),
                    },
                )

            except Exception as error:
                QMessageBox.critical(
                    self,
                    self.tr("Stacked spectra error"),
                    ("The stacked spectra plot could not " f"be generated:\n{error}"),
                )

        # Mantener las exportaciones existentes
        if export_action is not None:
            self.process_spectra_export(
                export_action,
                configuration,
            )

    def process_spectra_export(
        self,
        export_action,
        configuration,
    ):
        self.min_val = configuration.get(
            "export_range_min",
            configuration.get("range_min"),
        )

        self.max_val = configuration.get(
            "export_range_max",
            configuration.get("range_max"),
        )

        self.selected_type = configuration.get(
            "export_sample_type",
            configuration.get("sample_type"),
        )

        file_name = str(
            configuration.get(
                "export_file_name",
                "",
            )
        ).strip()

        if not file_name:
            QMessageBox.warning(
                self,
                self.tr("Invalid file name"),
                self.tr("Enter a name for the CSV file."),
            )
            return

        if not file_name.lower().endswith(".csv"):
            file_name += ".csv"

        try:
            if export_action in {
                "full",
                "Export full matrix as .csv",
            }:
                df_export = normalize_visual_dataframe(self.df_original)

            elif export_action in {
                "limited",
                "Export limited-range matrix as .csv",
            }:
                self.raman = self.complete_df.iloc[:, 0].reset_index(drop=True)

                df_export = self.export_limited_range_csv(
                    self.complete_df,
                    self.raman,
                    self.min_val,
                    self.max_val,
                    self.df_final,
                    nombre_eje_x=(self.get_export_x_column_name()),
                )

            elif export_action in {
                "type",
                "Export matrix by sample type as .csv",
            }:
                self.raman = self.complete_df.iloc[:, 0].reset_index(drop=True)

                df_export = self.export_type_csv(
                    self.complete_df,
                    self.raman,
                    self.df_final,
                    self.selected_type,
                    nombre_eje_x=(self.get_export_x_column_name()),
                )

            elif export_action in {
                "limited_type",
                ("Export limited-range matrix " "by sample type as .csv"),
            }:
                self.raman = self.complete_df.iloc[:, 0].reset_index(drop=True)

                df_export = self.export_limited_type_csv(
                    self.complete_df,
                    self.raman,
                    self.df_final,
                    self.selected_type,
                    self.min_val,
                    self.max_val,
                    nombre_eje_x=(self.get_export_x_column_name()),
                )

            else:
                QMessageBox.warning(
                    self,
                    self.tr("Unsupported export"),
                    self.tr("The selected CSV export operation is not supported."),
                )
                return

            df_export.to_csv(
                file_name,
                index=False,
                header=True,
            )

            self.register_exported_dataframe(
                df_export.copy(),
                file_name,
            )

            QMessageBox.information(
                self,
                self.tr("CSV exported"),
                self.tr(
                    "The CSV file was saved successfully:\n{path}",
                    path=os.path.abspath(file_name),
                ),
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                self.tr("Export error"),
                self.tr(
                    "The DataFrame could not be exported:\n{error}",
                    error=error,
                ),
            )

    def export_limited_range_csv(
        self, datos, raman, val_min, val_max, df_final, nombre_eje_x="X Axis"
    ):
        """
        Builds a CSV-ready DataFrame containing only the portion of each spectrum within a selected X-axis range.
        The function normalizes the internal-format matrix, filters rows between the given limits, and returns a clean numeric subset labeled with a standardized X-axis name.

        Parameters
        ----------
        datos : pandas.DataFrame
            Internal-format DataFrame where the first column is the X-axis and subsequent columns are spectra grouped by sample type.
        raman : array-like
            Original X-axis values associated with the spectra; this argument is kept for compatibility but recomputed from `datos`.
        val_min : float
            Lower bound of the X-axis range to retain.
        val_max : float
            Upper bound of the X-axis range to retain.
        df_final : pandas.DataFrame
            Processed or visual DataFrame corresponding to `datos`; included for API consistency even though it is not modified here.
        nombre_eje_x : str, optional
            Column name to use for the X-axis in the exported DataFrame, by default "X Axis".

        Returns
        -------
        pandas.DataFrame
            New DataFrame containing only rows within the specified X-axis range and all corresponding spectral columns.
        """
        datos = normalize_visual_dataframe(datos)

        raman = pd.to_numeric(datos.iloc[:, 0], errors="coerce").to_numpy()
        intensidades = (
            datos.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()
        )
        cabecera_np = list(datos.columns[1:])

        limited_indices = (raman >= val_min) & (raman <= val_max)
        raman_acotado = raman[limited_indices]
        intensidades_acotadas = intensidades[limited_indices, :]

        df_acotado = pd.DataFrame(
            data=np.column_stack([raman_acotado, intensidades_acotadas]),
            columns=[nombre_eje_x] + cabecera_np,
        )

        return df_acotado

    def export_type_csv(
        self, datos, raman, df_final, selected_type, nombre_eje_x="X Axis"
    ):
        """
        Builds a CSV-ready DataFrame containing only spectra that belong to a given sample type.
        The function normalizes the internal-format matrix, selects columns matching the requested type, and returns a clean subset labeled with a standardized X-axis name.

        Parameters
        ----------
        datos : pandas.DataFrame
            Internal-format DataFrame where the first column is the X-axis and subsequent columns are spectra grouped by sample type.
        raman : array-like
            Original X-axis values associated with the spectra; this argument is kept for compatibility but recomputed from `datos`.
        df_final : pandas.DataFrame
            Processed or visual DataFrame corresponding to `datos`; included for API consistency even though it is not modified here.
        selected_type : str
            Sample type or class name used to filter spectral columns.
        nombre_eje_x : str, optional
            Column name to use for the X-axis in the exported DataFrame, by default "X Axis".

        Returns
        -------
        pandas.DataFrame
            New DataFrame containing all rows of the X-axis and only the spectral columns matching the requested sample type.
        """
        datos = normalize_visual_dataframe(datos)

        raman = pd.to_numeric(datos.iloc[:, 0], errors="coerce").to_numpy()
        data_without_x = datos.iloc[:, 1:].copy()

        tipo_buscado = str(selected_type).strip().lower()

        indices_conservar = [
            i
            for i, col in enumerate(data_without_x.columns)
            if str(col).strip().lower() == tipo_buscado
        ]

        if not indices_conservar:
            QMessageBox.warning(
                self,
                self.tr("Sample type not found"),
                self.tr(
                    "No columns were found for sample type: {sample_type}",
                    sample_type=selected_type,
                ),
            )
            return pd.DataFrame({nombre_eje_x: raman})

        filtered_data = data_without_x.iloc[:, indices_conservar].copy()
        filtered_data.insert(0, nombre_eje_x, raman)

        return filtered_data

    def export_limited_type_csv(
        self,
        datos,
        raman,
        df_final,
        selected_type,
        min_val,
        max_val,
        nombre_eje_x="X Axis",
    ):
        """
        Builds a CSV-ready DataFrame containing only spectra of a given sample type within a selected X-axis range.
        The function normalizes the internal-format matrix, filters both by type and range, and returns a clean numeric subset labeled with a standardized X-axis name.

        Parameters
        ----------
        datos : pandas.DataFrame
            Internal-format DataFrame where the first column is the X-axis and subsequent columns are spectra grouped by sample type.
        raman : array-like
            Original X-axis values associated with the spectra; this argument is kept for compatibility but recomputed from `datos`.
        df_final : pandas.DataFrame
            Processed or visual DataFrame corresponding to `datos`; included for API consistency even though it is not modified here.
        selected_type : str
            Sample type or class name used to filter spectral columns.
        min_val : float
            Lower bound of the X-axis range to retain.
        max_val : float
            Upper bound of the X-axis range to retain.
        nombre_eje_x : str, optional
            Column name to use for the X-axis in the exported DataFrame, by default "X Axis".

        Returns
        -------
        pandas.DataFrame
            New DataFrame containing rows within the specified X-axis range and only the spectral columns matching the requested sample type. If no columns match, an empty DataFrame with only the X-axis column is returned.
        """
        datos = normalize_visual_dataframe(datos)

        raman = pd.to_numeric(datos.iloc[:, 0], errors="coerce").to_numpy()
        data_without_x = datos.iloc[:, 1:].copy()

        tipo_buscado = str(selected_type).strip().lower()

        indices_conservar = [
            i
            for i, col in enumerate(data_without_x.columns)
            if str(col).strip().lower() == tipo_buscado
        ]

        if not indices_conservar:
            QMessageBox.warning(
                self,
                self.tr("Sample type not found"),
                self.tr(
                    "No columns were found for sample type: {sample_type}",
                    sample_type=selected_type,
                ),
            )
            return pd.DataFrame({nombre_eje_x: []})

        filtered_data = data_without_x.iloc[:, indices_conservar].copy()
        intensidades = filtered_data.apply(pd.to_numeric, errors="coerce").to_numpy()

        limited_indices = (raman >= min_val) & (raman <= max_val)

        raman_acotado = raman[limited_indices]
        intensidades_acotadas = intensidades[limited_indices, :]

        datos_acotado_tipo = pd.DataFrame(
            data=np.column_stack([raman_acotado, intensidades_acotadas]),
            columns=[nombre_eje_x] + list(filtered_data.columns),
        )

        return datos_acotado_tipo

    def execute_option(self, texto):
        if texto == "Salir":
            self.close()
        else:
            QMessageBox.information(
                self,
                self.tr("Selected option"),
                self.tr("You selected: {text}", text=texto),
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = MainMenu()
    ventana.show()
    sys.exit(app.exec())