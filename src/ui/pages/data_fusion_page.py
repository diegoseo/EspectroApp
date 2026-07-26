import os

import pandas as pd

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from thread import (
    DataFusionThread,
    LowLevelDataFusionThread,
    LowLevelDataFusionNoCommonRangeThread,
    MidLevelDataFusionThread,
    MidLevelDataFusionNoCommonRangeThread,
    MidLevelPlotThread,
)
from ui.pages.dimensionality_page import (
    VentanaGraficoPCA2D,
    VentanaGraficoPCA3D,
)


from core.translations import translate, get_language, retranslate_widget_tree


def tr(text, **values):
    return translate(text, get_language(), **values)


class FusionPreviewWindow(QDialog):
    def __init__(
        self,
        dataframe,
        title=tr("Fusion preview"),
        parent=None,
    ):
        super().__init__(parent)

        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr(title))
        self.resize(1000, 650)
        self.setMinimumSize(800, 520)

        self.setStyleSheet("""
            QDialog {
                background-color: #F8F7F3;
                color: #17231D;
            }

            QLabel#previewTitle {
                color: #17231D;
                font-size: 20px;
                font-weight: 700;
            }

            QLabel#previewInfo {
                color: #607067;
                font-size: 13px;
            }

            QPushButton {
                background-color: #FFFFFF;
                color: #26332D;
                border: 1px solid #C8CECA;
                border-radius: 7px;
                padding: 7px 16px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #F0F3F1;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        title_label = QLabel(tr(title))
        title_label.setObjectName("previewTitle")

        info_label = QLabel(
            tr(
                "{rows} rows · {columns} columns",
                rows=f"{dataframe.shape[0]:,}",
                columns=f"{dataframe.shape[1]:,}",
            )
        )
        info_label.setObjectName("previewInfo")

        layout.addWidget(title_label)
        layout.addWidget(info_label)

        figure = Figure(figsize=(10, 6))
        canvas = FigureCanvas(figure)
        axes = figure.add_subplot(111)

        self.plot_dataframe_preview(
            dataframe,
            axes,
        )

        figure.tight_layout()

        layout.addWidget(canvas, 1)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_button = QPushButton(tr("Close"))
        close_button.clicked.connect(self.accept)

        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

    def plot_dataframe_preview(
        self,
        dataframe,
        axes,
    ):
        df = dataframe.copy()

        if df.empty:
            axes.text(
                0.5,
                0.5,
                tr("The fused matrix is empty."),
                ha="center",
                va="center",
                transform=axes.transAxes,
            )
            return

        # Formato interno de EspectroApp:
        # primera fila = nombres/tipos
        # primera columna = eje espectral
        x_values = pd.to_numeric(
            df.iloc[1:, 0],
            errors="coerce",
        )

        numeric_data = df.iloc[1:, 1:].apply(
            pd.to_numeric,
            errors="coerce",
        )

        valid_x = x_values.notna()

        x_values = x_values[valid_x]
        numeric_data = numeric_data.loc[valid_x]

        if x_values.empty or numeric_data.empty:
            # Alternativa para matrices de componentes
            numeric_df = df.apply(
                pd.to_numeric,
                errors="coerce",
            )

            numeric_df = numeric_df.dropna(
                axis=0,
                how="all",
            ).dropna(
                axis=1,
                how="all",
            )

            if numeric_df.empty:
                axes.text(
                    0.5,
                    0.5,
                    tr("No numeric data are available for plotting."),
                    ha="center",
                    va="center",
                    transform=axes.transAxes,
                )
                return

            numeric_df.plot(
                ax=axes,
                legend=False,
            )

            axes.set_xlabel(tr("Observation"))
            axes.set_ylabel(tr("Value"))
            axes.set_title(tr("Fused data preview"))
            axes.grid(
                True,
                alpha=0.25,
            )
            return

        maximum_spectra = min(
            numeric_data.shape[1],
            30,
        )

        for column_index in range(maximum_spectra):
            y_values = numeric_data.iloc[
                :,
                column_index,
            ]

            axes.plot(
                x_values,
                y_values,
                linewidth=0.8,
                alpha=0.75,
            )

        axes.set_xlabel(tr("Spectral axis"))
        axes.set_ylabel(tr("Intensity"))

        axes.set_title(
            tr(
                "Fusion preview — displaying {shown} of {total} spectra",
                shown=maximum_spectra,
                total=numeric_data.shape[1],
            )
        )

        axes.grid(
            True,
            alpha=0.25,
        )


class ComponentSelectionDialog(QDialog):
    """Select two or three fused components without manual comma input."""

    def __init__(
        self,
        variance_list,
        dataset_names,
        parent=None,
    ):
        super().__init__(parent)
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("Select principal components"))
        self.resize(680, 280)

        layout = QVBoxLayout(self)
        instruction = QLabel(
            tr(
                "Choose the fused components to plot. Each option identifies "
                "the source dataset, its original PC and explained variance."
            )
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        self.component_items = []
        global_component = 1

        for block_index, variance in enumerate(variance_list):
            dataset_name = (
                os.path.basename(str(dataset_names[block_index]))
                if block_index < len(dataset_names)
                else f"Dataset {block_index + 1}"
            )

            for original_component, value in enumerate(variance, start=1):
                label = (
                    f"CP{global_component} — {dataset_name} "
                    f"PC{original_component} — {float(value):.2f}%"
                )
                self.component_items.append((global_component, label))
                global_component += 1

        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.z_combo = QComboBox()
        self.z_combo.addItem(tr("No Z axis (2D plot)"), None)

        for component, label in self.component_items:
            self.x_combo.addItem(label, component)
            self.y_combo.addItem(label, component)
            self.z_combo.addItem(label, component)

        if self.y_combo.count() > 1:
            self.y_combo.setCurrentIndex(1)

        for label, combo in (
            (tr("X axis"), self.x_combo),
            (tr("Y axis"), self.y_combo),
            (tr("Optional Z axis"), self.z_combo),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label + ":"))
            row.addWidget(combo, 1)
            layout.addLayout(row)

        buttons = QHBoxLayout()
        cancel_button = QPushButton(tr("Cancel"))
        plot_button = QPushButton(tr("Plot"))
        plot_button.setObjectName("acceptButton")
        cancel_button.clicked.connect(self.reject)
        plot_button.clicked.connect(self._validate_and_accept)
        buttons.addStretch()
        buttons.addWidget(cancel_button)
        buttons.addWidget(plot_button)
        layout.addLayout(buttons)

    def _validate_and_accept(self):
        components = self.selected_components()
        if len(set(components)) != len(components):
            QMessageBox.warning(
                self,
                tr("Invalid selection"),
                tr("Select different components for each axis."),
            )
            return
        self.accept()

    def selected_components(self):
        components = [
            int(self.x_combo.currentData()),
            int(self.y_combo.currentData()),
        ]
        z_value = self.z_combo.currentData()
        if z_value is not None:
            components.append(int(z_value))
        return components


class MidLevelResultWindow(QDialog):
    """Inspect fused scores and explained variance before choosing a plot."""

    def __init__(
        self,
        dataframe,
        variance_list,
        dataset_names,
        plot_callback,
        parent=None,
    ):
        super().__init__(parent)
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("Mid-level fusion result"))
        self.resize(980, 680)
        self.setMinimumSize(800, 560)

        layout = QVBoxLayout(self)
        title = QLabel(tr("Mid-level fusion result"))
        title.setObjectName("previewTitle")
        layout.addWidget(title)

        total_components = sum(len(values) for values in variance_list)
        info = QLabel(
            tr(
                "{samples} samples · {datasets} datasets · "
                "{components} fused components",
                samples=f"{dataframe.shape[0]:,}",
                datasets=len(variance_list),
                components=total_components,
            )
        )
        info.setObjectName("previewInfo")
        layout.addWidget(info)

        table = QTableWidget(total_components, 5)
        table.setHorizontalHeaderLabels(
            [
                tr("Fused CP"),
                tr("Source dataset"),
                tr("Original PC"),
                tr("Explained variance"),
                tr("Cumulative variance"),
            ]
        )
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)

        row = 0
        global_component = 1
        for block_index, variance in enumerate(variance_list):
            dataset_name = (
                os.path.basename(str(dataset_names[block_index]))
                if block_index < len(dataset_names)
                else f"Dataset {block_index + 1}"
            )
            cumulative = 0.0
            for original_component, value in enumerate(variance, start=1):
                cumulative += float(value)
                values = (
                    f"CP{global_component}",
                    dataset_name,
                    f"PC{original_component}",
                    f"{float(value):.2f}%",
                    f"{cumulative:.2f}%",
                )
                for column, text in enumerate(values):
                    table.setItem(row, column, QTableWidgetItem(text))
                row += 1
                global_component += 1

        layout.addWidget(table, 1)

        matrix_info = QLabel(
            tr(
                "The table above shows each PCA block before plotting. "
                "The fused score matrix remains available in View DataFrame."
            )
        )
        matrix_info.setWordWrap(True)
        layout.addWidget(matrix_info)

        buttons = QHBoxLayout()
        plot_button = QPushButton(tr("Select components to plot"))
        plot_button.setObjectName("acceptButton")
        close_button = QPushButton(tr("Close"))
        plot_button.clicked.connect(plot_callback)
        close_button.clicked.connect(self.close)
        buttons.addStretch()
        buttons.addWidget(plot_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)


class DataFusionSelectionWindow(QWidget):
    """
    Lets the user choose which spectral data matrices will participate in a data fusion workflow.
    The window lists all loaded datasets with basic statistics, allows multi-selection via checkboxes, and launches the fusion configuration dialog for the chosen subset.

    Parameters
    ----------
    lista_df : list of pandas.DataFrame
        List of spectral DataFrames currently available for fusion.
    file_names : list of str
        File paths or human-readable names corresponding to each DataFrame in `lista_df`.
    menu_principal : QWidget
        Reference to the main application window used to coordinate subsequent fusion results.
    """

    def __init__(
        self,
        lista_df,
        file_names,
        menu_principal,
        embedded=False,
    ):
        super().__init__()

        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.embedded = embedded
        self.menu_principal = menu_principal
        if not self.embedded:
            self.setWindowTitle(tr("Data Fusion"))
            self.setMinimumSize(580, 470)
            self.resize(580, 470)

        self.lista_df = lista_df.copy()
        self.nombres_archivos = list(file_names)

        self.seleccionados = []
        self.nombres_seleccionados = []
        self.lista_rangos = []
        self.interseccion = False
        self.rang_comun = None
        self.tipos_orden = []
        self.configuration_widget = None
        self.df = None

        self.setStyleSheet("""
            QWidget {
                background-color: #F8F7F3;
                color: #17231D;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 14px;
            }

            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #D8DDD9;
                border-radius: 10px;
                margin-top: 12px;
                padding: 12px;
                color: #26332D;
                font-weight: 700;
            }

            QPushButton#backButton {
                background-color: #FFFFFF;
                color: #26332D;
                border: 1px solid #C8CECA;
            }

            QPushButton#backButton:hover {
                background-color: #F0F3F1;
                border: 1px solid #9DAAA3;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                color: #34463D;
                background-color: #F8F7F3;
            }

            QGroupBox#datasetCard {
                background-color: #FFFFFF;
                border: 1px solid #D8DDD9;
                border-radius: 9px;
                margin-top: 0px;
                padding: 5px;
            }

            QGroupBox#datasetCard:hover {
                background-color: #F6FAF8;
                border: 1px solid #8BB7A5;
            }

            QLabel#fileName {
                background-color: transparent;
                color: #17231D;
                font-size: 15px;
                font-weight: 700;
            }

            QLabel#fileInfo {
                background-color: transparent;
                color: #607067;
                font-size: 12px;
                font-weight: 400;
            }

            QLabel#commonRangeLabel {
                background-color: #E6F1FF;
                color: #285F9A;
                border: 1px solid #BED7F3;
                border-radius: 7px;
                padding: 7px 10px;
                font-size: 12px;
                font-weight: 600;
            }

            QScrollArea {
                background-color: transparent;
                border: none;
            }

            QScrollArea QWidget#qt_scrollarea_viewport {
                background-color: transparent;
            }

            QCheckBox {
                background-color: transparent;
                color: #26332D;
                spacing: 8px;
            }

            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #91A39A;
                border-radius: 4px;
                background-color: #FFFFFF;
            }

            QCheckBox::indicator:checked {
                background-color: #13876E;
                border: 1px solid #13876E;
            }

            QPushButton {
                min-height: 34px;
                border-radius: 7px;
                padding: 6px 15px;
                font-size: 13px;
                font-weight: 600;
            }

            QPushButton#acceptButton {
                background-color: #13876E;
                color: #FFFFFF;
                border: 1px solid #13876E;
            }

            QPushButton#acceptButton:hover {
                background-color: #0E725D;
            }

            QPushButton#acceptButton:disabled {
                background-color: #AFCAC0;
                color: #F5F7F6;
                border: 1px solid #AFCAC0;
            }

            QPushButton#cancelButton {
                background-color: #FFFFFF;
                color: #A13F49;
                border: 1px solid #D8AEB3;
            }

            QPushButton#cancelButton:hover {
                background-color: #FFF1F2;
            }

            QPushButton#previewButton {
                background-color: #FFFFFF;
                color: #26332D;
                border: 1px solid #C8CECA;
            }

            QPushButton#previewButton:hover {
                background-color: #F0F3F1;
                border: 1px solid #9DAAA3;
            }

            QPushButton#previewButton:disabled {
                background-color: #F2F2EF;
                color: #9A9F9C;
                border: 1px solid #D7D9D7;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            16,
            8,
            16,
            10,
        )
        main_layout.setSpacing(8)

        if not self.embedded:
            title = QLabel(tr("🧩 Data fusion"))
            title.setObjectName("windowTitle")
            title.setAlignment(Qt.AlignCenter)

            subtitle = QLabel(tr("Select spectral matrices for data fusion."))
            subtitle.setObjectName("windowSubtitle")
            subtitle.setAlignment(Qt.AlignCenter)

            main_layout.addWidget(title)
            main_layout.addWidget(subtitle)

        datasets_group = QGroupBox(tr("Input datasets"))
        datasets_layout = QVBoxLayout(datasets_group)
        datasets_layout.setSpacing(10)

        self.checkboxes = []

        scroll_widget = QWidget()
        layout_checkboxes = QVBoxLayout(scroll_widget)
        layout_checkboxes.setContentsMargins(4, 4, 4, 4)
        layout_checkboxes.setSpacing(7)

        for i, nombre in enumerate(self.nombres_archivos):
            df = self.lista_df[i]
            nombre_visible = os.path.basename(nombre)
            n_filas, n_columnas = df.shape
            n_nulos = df.isnull().sum().sum()
            x_values = pd.to_numeric(
                df.iloc[1:, 0],
                errors="coerce",
            ).dropna()

            if x_values.empty:
                range_min = None
                range_max = None
                range_text = "range unavailable"
            else:
                range_min = float(x_values.min())
                range_max = float(x_values.max())

                range_text = f"range {range_min:.2f} – " f"{range_max:.2f}"

            card = QGroupBox()
            card.setObjectName("datasetCard")
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(
                12,
                7,
                12,
                7,
            )
            card_layout.setSpacing(10)

            card.setMinimumHeight(52)
            card.setMaximumHeight(60)

            checkbox = QCheckBox()
            checkbox.setChecked(False)

            checkbox.stateChanged.connect(self.update_fusion_selection)

            checkbox.setToolTip("Select this matrix for data fusion")

            info_layout = QVBoxLayout()
            info_layout.setContentsMargins(0, 0, 0, 0)
            info_layout.setSpacing(4)

            label_nombre = QLabel(nombre_visible)
            label_nombre.setObjectName("fileName")

            label_info = QLabel(
                f"{n_filas:,} rows × "
                f"{n_columnas:,} columns · "
                f"{range_text} · "
                f"nulls {n_nulos:,}"
            )

            label_info.setObjectName("fileInfo")

            info_layout.addWidget(label_nombre)
            info_layout.addWidget(label_info)

            card_layout.addWidget(checkbox)
            card_layout.addLayout(info_layout)
            card_layout.addStretch()

            layout_checkboxes.addWidget(card)

            self.checkboxes.append(
                (
                    checkbox,
                    self.lista_df[i],
                    self.nombres_archivos[i],
                    range_min,
                    range_max,
                )
            )

        layout_checkboxes.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setMinimumHeight(130)
        scroll_area.setMaximumHeight(170)

        datasets_layout.addWidget(scroll_area)
        self.common_range_label = QLabel()
        self.common_range_label.setObjectName("commonRangeLabel")
        self.common_range_label.setWordWrap(True)

        datasets_layout.addWidget(self.common_range_label)
        main_layout.addWidget(datasets_group)
        self.configuration_container = QFrame()
        self.configuration_container.setObjectName("configurationContainer")

        self.configuration_layout = QVBoxLayout(self.configuration_container)

        self.configuration_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.configuration_layout.setSpacing(0)

        main_layout.addWidget(self.configuration_container)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        btn_back = QPushButton(tr("Back"))
        btn_back.setObjectName("backButton")

        if self.embedded:
            btn_back.clicked.connect(self.return_to_main_page)
        else:
            btn_back.clicked.connect(self.close)

        self.btn_plot_preview = QPushButton(tr("Plot preview"))
        self.btn_plot_preview.setObjectName("previewButton")
        self.btn_plot_preview.clicked.connect(self.plot_preview)

        self.btn_plot_mid_result = QPushButton(tr("View mid-level result"))
        self.btn_plot_mid_result.setObjectName("previewButton")
        self.btn_plot_mid_result.setEnabled(False)

        self.btn_plot_mid_result.clicked.connect(self.plot_cached_mid_level_result)

        self.btn_accept = QPushButton(tr("Accept"))
        self.btn_accept.setObjectName("acceptButton")
        self.btn_accept.clicked.connect(self.accept_current_fusion)

        buttons_layout.addStretch()
        buttons_layout.addWidget(btn_back)
        buttons_layout.addWidget(self.btn_plot_preview)

        buttons_layout.addWidget(self.btn_plot_mid_result)

        buttons_layout.addWidget(self.btn_accept)

        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

        self.update_fusion_selection()

    def plot_cached_mid_level_result(self):
        config = self.configuration_widget

        if config is None:
            QMessageBox.warning(
                self, "No fusion result", "No mid-level fusion result is available."
            )
            return

        if not hasattr(config, "df_concat_midfusion"):
            QMessageBox.warning(
                self, "No fusion result", "Run mid-level fusion before plotting."
            )
            return

        if config.df_concat_midfusion is None:
            QMessageBox.warning(
                self, "No fusion result", "Run mid-level fusion before plotting."
            )
            return

        config.show_mid_level_result()

    def accept_current_fusion(self):
        if len(self.seleccionados) < 2:
            QMessageBox.warning(
                self,
                "Insufficient datasets",
                "Select at least two datasets " "for data fusion.",
            )
            return

        if self.configuration_widget is None:
            QMessageBox.warning(
                self,
                "Missing configuration",
                "The fusion configuration is " "not available.",
            )
            return

        if self.configuration_widget.lowfusion.isChecked():
            self.configuration_widget.aplicar_fusion()

        elif self.configuration_widget.midfusion.isChecked():
            self.configuration_widget.aplicar_fusion_mid()

        else:
            QMessageBox.warning(
                self,
                "No fusion strategy",
                "Select Low-level fusion or " "Mid-level fusion.",
            )

    def return_to_main_page(self):
        if hasattr(self.menu_principal, "show_welcome_page"):
            self.menu_principal.show_welcome_page()
            return

        self.menu_principal.workspace_stack.setCurrentWidget(
            self.menu_principal.welcome_page
        )

    def plot_preview(self):
        if len(self.seleccionados) < 2:
            QMessageBox.warning(
                self,
                "Insufficient datasets",
                "Select at least two datasets " "to generate a fusion preview.",
            )
            return

        if self.configuration_widget is None:
            QMessageBox.warning(
                self,
                "Missing configuration",
                "Wait until the fusion configuration " "has finished loading.",
            )
            return

        config = self.configuration_widget
        config.preview_mode = True
        self.btn_plot_preview.setEnabled(False)
        self.btn_accept.setEnabled(False)

        if config.lowfusion.isChecked():
            config.aplicar_fusion()

        elif config.midfusion.isChecked():
            config.aplicar_fusion_mid()

        else:
            config.preview_mode = False

            self.btn_plot_preview.setEnabled(True)
            self.btn_accept.setEnabled(True)

            QMessageBox.warning(
                self,
                "No fusion strategy",
                "Select Low-level fusion or " "Mid-level fusion.",
            )

    def update_fusion_selection(self):
        self.seleccionados = []
        self.nombres_seleccionados = []
        self.lista_rangos = []

        minimum_values = []
        maximum_values = []

        for (
            checkbox,
            df,
            nombre,
            range_min,
            range_max,
        ) in self.checkboxes:

            if not checkbox.isChecked():
                continue

            self.seleccionados.append(df)
            self.nombres_seleccionados.append(nombre)

            self.lista_rangos.append(
                (
                    range_min,
                    range_max,
                )
            )

            if range_min is not None and range_max is not None:
                minimum_values.append(range_min)
                maximum_values.append(range_max)

        if len(self.seleccionados) < 2:
            self.interseccion = False
            self.rang_comun = None

            self.common_range_label.setText(
                tr("Select at least two datasets to calculate a common range.")
            )

            self.clear_configuration_widget()
            self._update_action_buttons(preparing=False)
            return

        if len(minimum_values) != len(self.seleccionados):
            self.interseccion = False
            self.rang_comun = None

            self.common_range_label.setText(
                tr(
                    "The common range could not be calculated for all selected datasets."
                )
            )

            self.start_fusion_preparation()
            return

        common_min = max(minimum_values)
        common_max = min(maximum_values)

        self.interseccion = common_min <= common_max

        if self.interseccion:
            self.rang_comun = (
                common_min,
                common_max,
            )

            self.common_range_label.setText(
                tr(
                    "✓ Common spectral-axis intersection found: {minimum} – {maximum}",
                    minimum=f"{common_min:.2f}",
                    maximum=f"{common_max:.2f}",
                )
            )

        else:
            self.rang_comun = None

            self.common_range_label.setText(
                tr("No common spectral-axis intersection was found.")
            )

        self.start_fusion_preparation()

    def _update_action_buttons(self, preparing=False):
        """Keep fusion action buttons synchronized with the current selection."""
        has_selection = len(self.seleccionados) >= 2
        has_configuration = self.configuration_widget is not None

        enabled = has_selection and has_configuration and not preparing
        self.btn_accept.setEnabled(enabled)
        self.btn_plot_preview.setEnabled(enabled)

        has_mid_result = (
            has_configuration
            and getattr(
                self.configuration_widget,
                "df_concat_midfusion",
                None,
            )
            is not None
        )
        self.btn_plot_mid_result.setEnabled(enabled and has_mid_result)

    def _restore_after_preparation(self):
        """Restore controls even when the preparation worker ends unexpectedly."""
        for checkbox, *_ in self.checkboxes:
            checkbox.setEnabled(True)

        if len(self.seleccionados) >= 2 and self.configuration_widget is None:
            try:
                self.refresh_configuration_widget()
            except Exception as error:
                QMessageBox.critical(
                    self,
                    tr("Fusion configuration error"),
                    tr(
                        "The fusion configuration could not be created:\n{error}",
                        error=error,
                    ),
                )

        self._update_action_buttons(preparing=False)

    def start_fusion_preparation(self):
        if len(self.seleccionados) < 2:
            self.clear_configuration_widget()
            return

        current_thread = getattr(
            self,
            "preparation_thread",
            None,
        )

        if current_thread is not None and current_thread.isRunning():
            return

        self._update_action_buttons(preparing=True)

        for checkbox, *_ in self.checkboxes:
            checkbox.setEnabled(False)

        self.common_range_label.setText("Preparing the fusion configuration...")

        self.preparation_thread = DataFusionThread(self.seleccionados)

        self.preparation_thread.signal_datafusion.connect(
            self.finish_fusion_preparation
        )

        self.preparation_thread.finished.connect(self.cleanup_preparation_thread)
        self.preparation_thread.finished.connect(self._restore_after_preparation)

        self.preparation_thread.start()

    def cleanup_preparation_thread(self):
        thread = self.sender()

        if thread is not None:
            thread.deleteLater()

        if (
            getattr(
                self,
                "preparation_thread",
                None,
            )
            is thread
        ):
            self.preparation_thread = None

    def finish_fusion_preparation(
        self,
        lista_rangos,
        interseccion,
        rang_comun,
        tipos_orden,
    ):
        try:
            self.lista_rangos = lista_rangos
            self.interseccion = interseccion
            self.rang_comun = rang_comun
            self.tipos_orden = tipos_orden

            if interseccion and rang_comun:
                self.common_range_label.setText(
                    tr(
                        "✓ Common spectral-axis intersection found: {minimum} – {maximum}",
                        minimum=f"{rang_comun[0]:.2f}",
                        maximum=f"{rang_comun[1]:.2f}",
                    )
                )
            else:
                self.common_range_label.setText(
                    tr("No common spectral-axis intersection was found.")
                )

            self.refresh_configuration_widget()

        except Exception as error:
            QMessageBox.critical(
                self,
                tr("Fusion configuration error"),
                tr(
                    "The fusion configuration could not be created:\n{error}",
                    error=error,
                ),
            )

        finally:
            for checkbox, *_ in self.checkboxes:
                checkbox.setEnabled(True)

            self._update_action_buttons(preparing=False)
            QTimer.singleShot(
                0,
                lambda: self._update_action_buttons(preparing=False),
            )

    def clear_configuration_widget(self):
        if self.configuration_widget is None:
            return

        self.configuration_layout.removeWidget(self.configuration_widget)

        self.configuration_widget.deleteLater()
        self.configuration_widget = None

    def refresh_configuration_widget(self):
        self.clear_configuration_widget()

        if len(self.seleccionados) < 2:
            return

        self.configuration_widget = DataFusionConfigurationWindow(
            self.lista_df,
            self.seleccionados,
            self.nombres_seleccionados,
            self.lista_rangos,
            self.interseccion,
            self.rang_comun,
            self.tipos_orden,
            self.menu_principal,
            embedded=True,
            show_buttons=False,
            show_summary=False,
        )

        self.configuration_layout.addWidget(self.configuration_widget)


class DataFusionConfigurationWindow(QWidget):
    """
    Configures and executes low-level and mid-level data fusion workflows for multiple spectral datasets.
    The window summarizes selected matrices, lets the user choose interpolation, concatenation, and PCA options, and then runs the corresponding fusion and plotting threads.

    Parameters
    ----------
    lista_df : list of pandas.DataFrame
        Full list of spectral DataFrames loaded in the application, including those not selected for fusion.
    seleccionado : list of pandas.DataFrame
        Subset of DataFrames chosen for fusion operations.
    nombres_seleccionados : list of str
        Names or file paths corresponding to the DataFrames in `seleccionado`.
    lista_rangos : list of tuple
        Per-dataset X-axis ranges (min, max) for the selected matrices, used to determine intersections and fusion grids.
    interseccion : bool
        Indicates whether all selected datasets share a non-empty common X-axis range.
    rang_comun : tuple or None
        Common X-axis range shared by all selected datasets when `interseccion` is True, otherwise None.
    tipos_orden : list
        Ordered list of sample types or classes used to align columns or rows across fused datasets.
    menu_principal : QWidget
        Reference to the main application window so newly fused matrices and plots can be registered and reused.
    parent : QWidget, optional
        Parent widget that will own this configuration window, by default None.
    """

    def __init__(
        self,
        lista_df,
        seleccionado,
        nombres_seleccionados,
        lista_rangos,
        interseccion,
        rang_comun,
        tipos_orden,
        menu_principal,
        parent=None,
        embedded=False,
        show_buttons=True,
        show_summary=True,
    ):
        super().__init__(parent)

        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.embedded = embedded
        self.show_buttons = show_buttons
        self.show_summary = show_summary
        self.menu_principal = menu_principal
        if not self.embedded:
            self.setWindowTitle(tr("Data Fusion Configuration"))
            self.setMinimumSize(760, 680)
            self.resize(760, 680)

        self.seleccionados = seleccionado
        self.nombres_seleccionados = nombres_seleccionados
        self.lista_rangos = lista_rangos
        self.interseccion = interseccion
        self.rang_comun = rang_comun
        self.tipos_orden = tipos_orden
        self.lista_df = lista_df
        self.preview_mode = False
        self.df_concat_midfusion = None
        self.lista_varianza = None

        self.setStyleSheet("""
            QWidget {
                background-color: #F8F7F3;
                color: #17231D;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 13px;
            }

            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #D8DDD9;
                border-radius: 9px;
                margin-top: 11px;
                padding: 10px;
                color: #26332D;
                font-weight: 700;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                background-color: #F8F7F3;
                color: #34463D;
            }

            QGroupBox#gb_concat {
                background-color: #FFFFFF;
                border: 1px solid #D8DDD9;
                border-radius: 9px;
                margin-top: 11px;
                padding: 10px;
            }

            QLabel {
                background-color: transparent;
                color: #35453D;
            }

            QLabel#fieldLabel {
                color: #35453D;
                font-weight: 600;
            }

            QLabel#rangeLabel {
                color: #607067;
                font-size: 12px;
            }

            QRadioButton,
            QCheckBox {
                background-color: transparent;
                color: #26332D;
                padding: 5px 4px;
                spacing: 8px;
                font-weight: 600;
            }

            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #91A39A;
                border-radius: 9px;
                background-color: #FFFFFF;
            }

            QRadioButton::indicator:checked {
                background-color: #13876E;
                border: 4px solid #DDF1EB;
            }

            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #91A39A;
                border-radius: 4px;
                background-color: #FFFFFF;
            }

            QCheckBox::indicator:checked {
                background-color: #13876E;
                border: 1px solid #13876E;
            }

            QLineEdit {
                background-color: #FFFFFF;
                color: #17231D;
                border: 1px solid #C9D0CC;
                border-radius: 7px;
                padding: 6px 8px;
                min-height: 27px;
            }

            QLineEdit:focus {
                border: 1px solid #13876E;
            }

            QScrollArea {
                background-color: transparent;
                border: none;
            }

            QScrollArea QWidget#qt_scrollarea_viewport {
                background-color: transparent;
            }

            QTableWidget {
                background-color: #FFFFFF;
                color: #26332D;
                gridline-color: #D8DDD9;
                border: 1px solid #D8DDD9;
                border-radius: 7px;
            }

            QHeaderView::section {
                background-color: #EAF3EF;
                color: #26332D;
                border: none;
                padding: 6px;
                font-weight: 700;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            4,
            2,
            4,
            2,
        )
        main_layout.setSpacing(6)

        if not self.embedded:
            title = QLabel(tr("🧩 Data fusion configuration"))
            title.setObjectName("windowTitle")
            title.setAlignment(Qt.AlignCenter)

            subtitle = QLabel(
                "Review the selected spectral "
                "matrices and configure low-level "
                "or mid-level fusion."
            )
            subtitle.setObjectName("windowSubtitle")
            subtitle.setAlignment(Qt.AlignCenter)

            main_layout.addWidget(title)
            main_layout.addWidget(subtitle)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(
            2,
            2,
            2,
            2,
        )
        content_layout.setSpacing(7)

        if self.show_summary:
            summary_group = QGroupBox(tr("Selected data matrices"))
            summary_layout = QVBoxLayout(summary_group)
            summary_layout.setSpacing(8)

            tabla = QTableWidget(len(nombres_seleccionados), 3)
            tabla.setHorizontalHeaderLabels([tr("File"), tr("Minimum range"), tr("Maximum range")])
            tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tabla.verticalHeader().setVisible(False)
            tabla.setEditTriggers(QTableWidget.NoEditTriggers)
            tabla.setSelectionBehavior(QTableWidget.SelectRows)
            tabla.setMaximumHeight(150)

            for i, nombre in enumerate(nombres_seleccionados):
                min_val, max_val = lista_rangos[i]
                tabla.setItem(i, 0, QTableWidgetItem(os.path.basename(nombre)))
                tabla.setItem(i, 1, QTableWidgetItem(f"{min_val:.2f}"))
                tabla.setItem(i, 2, QTableWidgetItem(f"{max_val:.2f}"))

            summary_layout.addWidget(tabla)

            interseccion_label = QLabel(
                f"Common spectral-axis intersection: {'Yes' if interseccion else 'No'}"
            )
            interseccion_label.setObjectName("statusLabel")
            summary_layout.addWidget(interseccion_label)

            if interseccion:
                rango_label = QLabel(
                    f"Common range: {rang_comun[0]:.2f} – {rang_comun[1]:.2f}"
                )
                rango_label.setObjectName("rangeLabel")
                summary_layout.addWidget(rango_label)

            content_layout.addWidget(summary_group)

        # ------------------------------------------------------------------
        # Step 1 — choose the fusion strategy
        # ------------------------------------------------------------------
        fusion_group = QGroupBox(tr("1. Choose the fusion strategy"))
        fusion_layout = QVBoxLayout(fusion_group)
        fusion_layout.setSpacing(8)

        strategy_help = QLabel(
            tr(
                "Low-level joins the original spectral variables. Mid-level runs an independent PCA for each dataset and joins the scores."
            )
        )
        strategy_help.setObjectName("rangeLabel")
        strategy_help.setWordWrap(True)
        fusion_layout.addWidget(strategy_help)

        self.lowfusion = QRadioButton(
            tr("Low-level fusion — combine the original spectral blocks")
        )
        self.midfusion = QRadioButton(
            tr("Mid-level fusion — combine PCA scores from each dataset")
        )

        self.fusion_strategy_group = QButtonGroup(self)
        self.fusion_strategy_group.setExclusive(True)
        self.fusion_strategy_group.addButton(self.lowfusion)
        self.fusion_strategy_group.addButton(self.midfusion)

        fusion_layout.addWidget(self.lowfusion)
        fusion_layout.addWidget(self.midfusion)
        self.lowfusion.setChecked(True)
        content_layout.addWidget(fusion_group)

        # ------------------------------------------------------------------
        # Step 2A — low-level settings
        # ------------------------------------------------------------------
        self.contenedor_lowf = QGroupBox(tr("2. Configure low-level fusion"))
        self.layout_lowf = QVBoxLayout(self.contenedor_lowf)
        self.layout_lowf.setSpacing(8)

        low_help = QLabel(
            tr(
                "For complementary techniques such as FTIR and Raman, stack the spectral blocks and preserve their original axes. Interpolation is only needed when the datasets must share the same spectral grid."
            )
        )
        low_help.setObjectName("rangeLabel")
        low_help.setWordWrap(True)
        self.layout_lowf.addWidget(low_help)

        self.gb_concat = QGroupBox(tr("Block arrangement"))
        self.gb_concat.setObjectName("gb_concat")
        concat_layout = QVBoxLayout(self.gb_concat)

        self.rb_concat_v = QRadioButton(
            tr("Stack spectral blocks (recommended for FTIR + Raman)")
        )
        self.rb_concat_v.setToolTip(
            "Places one spectral block below the other in EspectroApp's internal "
            "format, preserving the paired sample columns."
        )
        self.rb_concat_h = QRadioButton(
            tr("Merge columns on a shared spectral axis (advanced)")
        )
        self.rb_concat_h.setToolTip(
            "Use only when the datasets represent compatible variables on the "
            "same aligned spectral axis."
        )

        self.concat_group = QButtonGroup(self)
        self.concat_group.setExclusive(True)
        self.concat_group.addButton(self.rb_concat_v)
        self.concat_group.addButton(self.rb_concat_h)
        self.rb_concat_v.setChecked(True)
        concat_layout.addWidget(self.rb_concat_v)
        concat_layout.addWidget(self.rb_concat_h)
        self.layout_lowf.addWidget(self.gb_concat)

        low_axis_group = QGroupBox(tr("Spectral-axis treatment"))
        low_axis_layout = QVBoxLayout(low_axis_group)
        self.sin_interpolacion = QRadioButton(
            tr("Keep each block on its original spectral axis (recommended)")
        )
        self.interpolarsi = QRadioButton(
            tr("Align datasets by interpolation (advanced)")
        )
        self.low_axis_group = QButtonGroup(self)
        self.low_axis_group.setExclusive(True)
        self.low_axis_group.addButton(self.sin_interpolacion)
        self.low_axis_group.addButton(self.interpolarsi)
        self.sin_interpolacion.setChecked(True)
        low_axis_layout.addWidget(self.sin_interpolacion)
        low_axis_layout.addWidget(self.interpolarsi)
        self.layout_lowf.addWidget(low_axis_group)

        self.contenedor_interpolacion_low = QWidget()
        self.layout_interpolacion_low = QVBoxLayout(self.contenedor_interpolacion_low)
        self.layout_interpolacion_low.setContentsMargins(18, 0, 0, 0)
        self.layout_interpolacion_low.setSpacing(6)

        self.rango_comun = QRadioButton(tr("Use only the common spectral range"))
        self.rango_completo = QRadioButton(tr("Use the full combined spectral range"))
        self.grp_rangos_low = QButtonGroup(self)
        self.grp_rangos_low.setExclusive(True)
        self.grp_rangos_low.addButton(self.rango_comun)
        self.grp_rangos_low.addButton(self.rango_completo)
        if interseccion:
            self.rango_comun.setChecked(True)
        else:
            self.rango_completo.setChecked(True)
            self.rango_comun.setEnabled(False)
            self.rango_comun.setToolTip(
                "Unavailable because the selected datasets have no common range."
            )
        self.layout_interpolacion_low.addWidget(self.rango_comun)
        self.layout_interpolacion_low.addWidget(self.rango_completo)

        self.label_metodo_interpolacion = QLabel(tr("Interpolation method"))
        self.label_metodo_interpolacion.setObjectName("fieldLabel")
        self.layout_interpolacion_low.addWidget(self.label_metodo_interpolacion)
        self.lineal = QRadioButton("Linear")
        self.cubica = QRadioButton("Cubic")
        self.polinomica = QRadioButton("Second-order polynomial")
        self.nearest = QRadioButton("Nearest")
        self.low_method_group = QButtonGroup(self)
        self.low_method_group.setExclusive(True)
        for button in (self.lineal, self.cubica, self.polinomica, self.nearest):
            self.low_method_group.addButton(button)
            self.layout_interpolacion_low.addWidget(button)
        self.lineal.setChecked(True)

        step_label = QLabel(tr("Interpolation grid"))
        step_label.setObjectName("fieldLabel")
        self.layout_interpolacion_low.addWidget(step_label)
        self.valor = QRadioButton("Enter a step value")
        self.promedio = QRadioButton("Use the average step of the files")
        self.numero = QRadioButton("Define a fixed number of points")
        self.low_step_group = QButtonGroup(self)
        self.low_step_group.setExclusive(True)
        for button in (self.valor, self.promedio, self.numero):
            self.low_step_group.addButton(button)
            self.layout_interpolacion_low.addWidget(button)
        self.promedio.setChecked(True)
        self.input_paso = QLineEdit()
        self.input_paso.setPlaceholderText(tr("Step value"))
        self.input_n_puntos = QLineEdit()
        self.input_n_puntos.setPlaceholderText(tr("Number of points"))
        self.layout_interpolacion_low.addWidget(self.input_paso)
        self.layout_interpolacion_low.addWidget(self.input_n_puntos)
        self.contenedor_interpolacion_low.hide()
        self.layout_lowf.addWidget(self.contenedor_interpolacion_low)
        content_layout.addWidget(self.contenedor_lowf)

        # ------------------------------------------------------------------
        # Step 2B — mid-level settings
        # ------------------------------------------------------------------
        self.contenedor_midf = QGroupBox(tr("2. Configure mid-level fusion"))
        layout_mf = QVBoxLayout(self.contenedor_midf)
        layout_mf.setSpacing(8)

        mid_help = QLabel(
            "Each dataset is reduced independently by PCA. The resulting scores "
            "are concatenated sample by sample. Different spectral ranges and "
            "different numbers of variables are allowed."
        )
        mid_help.setObjectName("rangeLabel")
        mid_help.setWordWrap(True)
        layout_mf.addWidget(mid_help)

        self.n_componentes_label = QLabel(
            "Principal components retained for each dataset"
        )
        self.n_componentes_label.setObjectName("fieldLabel")
        layout_mf.addWidget(self.n_componentes_label)

        self.components_table = QTableWidget(
            len(self.seleccionados),
            3,
        )
        self.components_table.setHorizontalHeaderLabels(
            [
                "Dataset",
                "Spectral variables",
                "Components retained",
            ]
        )
        self.components_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.components_table.verticalHeader().setVisible(False)
        self.components_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.components_table.setMaximumHeight(
            min(230, 56 + 38 * len(self.seleccionados))
        )

        self.component_spinboxes = []

        for row, (df, name) in enumerate(
            zip(self.seleccionados, self.nombres_seleccionados)
        ):
            dataset_name = os.path.basename(str(name))
            number_variables = max(int(df.shape[0] - 1), 1)
            number_samples = max(int(df.shape[1] - 1), 1)
            maximum_components = max(
                2,
                min(number_variables, number_samples),
            )

            spinbox = QSpinBox()
            spinbox.setMinimum(2)
            spinbox.setMaximum(maximum_components)
            spinbox.setValue(min(5, maximum_components))
            spinbox.setToolTip(
                f"Maximum available for {dataset_name}: " f"{maximum_components}"
            )

            self.components_table.setItem(
                row,
                0,
                QTableWidgetItem(dataset_name),
            )
            self.components_table.setItem(
                row,
                1,
                QTableWidgetItem(str(number_variables)),
            )
            self.components_table.setCellWidget(
                row,
                2,
                spinbox,
            )
            self.component_spinboxes.append(spinbox)

        layout_mf.addWidget(self.components_table)

        self.intervalo_confianza_label = QLabel(tr("Confidence interval (%)"))
        self.intervalo_confianza_label.setObjectName("fieldLabel")
        self.intervalo_confianza = QLineEdit("95")
        self.intervalo_confianza.setPlaceholderText(tr("E.g.: 95"))
        layout_mf.addWidget(self.intervalo_confianza_label)
        layout_mf.addWidget(self.intervalo_confianza)

        mid_axis_group = QGroupBox(tr("Spectral-axis treatment"))
        mid_axis_layout = QVBoxLayout(mid_axis_group)
        self.sin_interpolacion_mid = QRadioButton(
            "Keep each dataset on its original axis (recommended)"
        )
        self.interpolar_mid = QRadioButton(
            "Resample each dataset before PCA (advanced)"
        )
        self.mid_axis_group = QButtonGroup(self)
        self.mid_axis_group.setExclusive(True)
        self.mid_axis_group.addButton(self.sin_interpolacion_mid)
        self.mid_axis_group.addButton(self.interpolar_mid)
        self.sin_interpolacion_mid.setChecked(True)
        mid_axis_layout.addWidget(self.sin_interpolacion_mid)
        mid_axis_layout.addWidget(self.interpolar_mid)
        layout_mf.addWidget(mid_axis_group)

        self.contenedor_opciones_dinamicas_mid = QWidget()
        mid_adv_layout = QVBoxLayout(self.contenedor_opciones_dinamicas_mid)
        mid_adv_layout.setContentsMargins(18, 0, 0, 0)
        mid_adv_layout.setSpacing(6)

        self.rango_comun_mid = QRadioButton("Use only the common spectral range")
        self.rango_completo_mid = QRadioButton("Use the full range of each dataset")
        self.grp_rangos_mid = QButtonGroup(self)
        self.grp_rangos_mid.setExclusive(True)
        self.grp_rangos_mid.addButton(self.rango_comun_mid)
        self.grp_rangos_mid.addButton(self.rango_completo_mid)
        if interseccion:
            self.rango_comun_mid.setChecked(True)
        else:
            self.rango_completo_mid.setChecked(True)
            self.rango_comun_mid.setEnabled(False)
        mid_adv_layout.addWidget(self.rango_comun_mid)
        mid_adv_layout.addWidget(self.rango_completo_mid)

        self.label_metodo_interpolacion_mid = QLabel(tr("Interpolation method"))
        self.label_metodo_interpolacion_mid.setObjectName("fieldLabel")
        mid_adv_layout.addWidget(self.label_metodo_interpolacion_mid)
        self.lineal_mid = QRadioButton("Linear")
        self.cubica_mid = QRadioButton("Cubic")
        self.polinomica_mid = QRadioButton("Second-order polynomial")
        self.nearest_mid = QRadioButton("Nearest")
        self.mid_method_group = QButtonGroup(self)
        self.mid_method_group.setExclusive(True)
        for button in (
            self.lineal_mid,
            self.cubica_mid,
            self.polinomica_mid,
            self.nearest_mid,
        ):
            self.mid_method_group.addButton(button)
            mid_adv_layout.addWidget(button)
        self.lineal_mid.setChecked(True)

        self.valor_mid = QRadioButton("Enter a step value")
        self.promedio_mid = QRadioButton("Use the average step of the files")
        self.numero_mid = QRadioButton("Define a fixed number of points")
        self.mid_step_group = QButtonGroup(self)
        self.mid_step_group.setExclusive(True)
        for button in (self.valor_mid, self.promedio_mid, self.numero_mid):
            self.mid_step_group.addButton(button)
            mid_adv_layout.addWidget(button)
        self.promedio_mid.setChecked(True)
        self.input_paso_mid = QLineEdit()
        self.input_paso_mid.setPlaceholderText(tr("Step value"))
        self.input_n_puntos_mid = QLineEdit()
        self.input_n_puntos_mid.setPlaceholderText(tr("Number of points"))
        mid_adv_layout.addWidget(self.input_paso_mid)
        mid_adv_layout.addWidget(self.input_n_puntos_mid)
        self.contenedor_opciones_dinamicas_mid.hide()
        layout_mf.addWidget(self.contenedor_opciones_dinamicas_mid)
        self.contenedor_midf.hide()
        content_layout.addWidget(self.contenedor_midf)

        # Connections are made after all dependent widgets exist.
        self.lowfusion.toggled.connect(self.toggle_lowfusion)
        self.midfusion.toggled.connect(self.toggle_midfusion)
        self.rb_concat_h.toggled.connect(self.update_low_level_axis_options)
        self.rb_concat_v.toggled.connect(self.update_low_level_axis_options)
        self.interpolarsi.toggled.connect(self.toggle_interpolarsi)
        self.sin_interpolacion.toggled.connect(self.toggle_interpolarsi)
        self.interpolar_mid.toggled.connect(self.mostrar_opciones_interpolacion_mid)
        self.sin_interpolacion_mid.toggled.connect(
            self.mostrar_opciones_interpolacion_mid
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)

        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        if self.embedded:
            scroll.setMinimumHeight(190)
            scroll.setMaximumHeight(360)

        main_layout.addWidget(scroll)

        if self.show_buttons:
            botones_layout = QHBoxLayout()
            botones_layout.setSpacing(10)

            btn_aceptar = QPushButton(tr("Accept"))
            btn_aceptar.setObjectName("acceptButton")

            btn_cancelar = QPushButton(tr("Cancel"))
            btn_cancelar.setObjectName("cancelButton")
            btn_cancelar.clicked.connect(self.close)

            btn_graficar_low = QPushButton(tr("Plot low-level"))
            btn_graficar_low.setObjectName("plotLowButton")

            btn_graficar = QPushButton(tr("Plot mid-level"))
            btn_graficar.setObjectName("plotMidButton")

            def ejecutar_fusion():
                if self.lowfusion.isChecked():
                    self.aplicar_fusion()
                elif self.midfusion.isChecked():
                    self.aplicar_fusion_mid()
                else:
                    QMessageBox.warning(
                        self, "Warning", "You must select at least one fusion option."
                    )

            btn_aceptar.clicked.connect(ejecutar_fusion)
            btn_graficar.clicked.connect(self.pedir_pc_para_graficar)
            btn_graficar_low.clicked.connect(
                self.menu_principal.open_dimensionality_reduction_window
            )

            botones_layout.addWidget(btn_aceptar)
            botones_layout.addWidget(btn_cancelar)
            botones_layout.addWidget(btn_graficar_low)
            botones_layout.addWidget(btn_graficar)

            main_layout.addLayout(botones_layout)

        self.toggle_lowfusion(self.lowfusion.isChecked())

        self.toggle_midfusion(self.midfusion.isChecked())

        self.setLayout(main_layout)

    def pedir_pc_para_graficar(self):
        """Open a graphical component selector for 2D or 3D plotting."""
        if self.df_concat_midfusion is None or self.lista_varianza is None:
            QMessageBox.warning(
                self,
                "No mid-level result",
                "Run mid-level fusion before plotting.",
            )
            return

        dialog = ComponentSelectionDialog(
            variance_list=self.lista_varianza,
            dataset_names=self.nombres_seleccionados,
            parent=self,
        )

        if dialog.exec() == QDialog.Accepted:
            self.graficar_componentes_principales(dialog.selected_components())

    def mostrar_dialogo_pc(self):
        self.pedir_pc_para_graficar()

    def update_low_level_axis_options(self, checked=False):
        """Keep spectral-axis options compatible with the block arrangement."""
        merge_columns = self.rb_concat_h.isChecked()

        if merge_columns:
            # Merging columns requires a common aligned spectral axis.
            self.interpolarsi.setChecked(True)
            self.sin_interpolacion.setEnabled(False)
            self.sin_interpolacion.setToolTip(
                tr(
                    "Unavailable because merging columns requires "
                    "alignment by interpolation."
                )
            )
        else:
            # Stacked blocks may preserve their original axes or be interpolated.
            self.sin_interpolacion.setEnabled(True)
            self.sin_interpolacion.setToolTip("")

        self.toggle_interpolarsi()

    def toggle_interpolarsi(self, state=None):
        """Display low-level resampling controls only when interpolation is selected."""
        enabled = hasattr(self, "interpolarsi") and self.interpolarsi.isChecked()
        if hasattr(self, "contenedor_interpolacion_low"):
            self.contenedor_interpolacion_low.setVisible(enabled)

    def _component_counts(self):
        """Return the PCA component count selected for every dataset."""
        counts = [
            int(spinbox.value()) for spinbox in getattr(self, "component_spinboxes", [])
        ]
        if len(counts) != len(self.seleccionados):
            raise ValueError("Select a component count for every dataset.")
        return counts

    def _confidence_value(self):
        """Return a validated confidence interval percentage."""
        text = self.intervalo_confianza.text().strip()
        try:
            value = float(text)
        except ValueError as error:
            raise ValueError("The confidence interval must be numeric.") from error
        if not 0 < value < 100:
            raise ValueError("The confidence interval must be between 0 and 100.")
        return value

    def show_mid_level_result(self):
        """Open the reusable variance/result inspector."""
        if self.df_concat_midfusion is None or self.lista_varianza is None:
            QMessageBox.warning(
                self,
                "No mid-level result",
                "Run mid-level fusion before viewing the result.",
            )
            return

        self.mid_result_window = MidLevelResultWindow(
            dataframe=self.df_concat_midfusion,
            variance_list=self.lista_varianza,
            dataset_names=self.nombres_seleccionados,
            plot_callback=self.pedir_pc_para_graficar,
            parent=self,
        )
        self.mid_result_window.show()

    def _fusion_source_names(self):
        """Return the visible names of all selected source datasets."""
        return [os.path.basename(str(name)) for name in self.nombres_seleccionados]

    def _checked_interpolation_method(self, suffix=""):
        """Return the selected interpolation method."""
        candidates = (
            ("lineal", "Linear"),
            ("cubica", "Cubic"),
            ("polinomica", "Second-order polynomial"),
            ("nearest", "Nearest"),
        )

        for attribute, label in candidates:
            widget = getattr(
                self,
                f"{attribute}{suffix}",
                None,
            )

            if widget is not None and widget.isChecked():
                return label

        return "None"

    def _fusion_history_parameters(self, strategy):
        """Collect the active fusion configuration for the history."""
        is_mid = strategy == "Mid-level fusion"
        suffix = "_mid" if is_mid else ""

        parameters = {
            "Source datasets": ", ".join(self._fusion_source_names()),
            "Range relationship": (
                "Common spectral range available"
                if self.interseccion
                else "No common spectral range"
            ),
        }

        if not is_mid:
            parameters["Concatenation"] = (
                "Horizontal" if self.rb_concat_h.isChecked() else "Vertical"
            )

        common_range_widget = getattr(
            self,
            f"rango_comun{suffix}",
            None,
        )
        full_range_widget = getattr(
            self,
            f"rango_completo{suffix}",
            None,
        )
        no_interpolation_widget = getattr(
            self,
            f"sin_interpolacion{suffix}",
            None,
        )

        if no_interpolation_widget is not None and no_interpolation_widget.isChecked():
            parameters["Interpolation"] = "Disabled"
            parameters["Fusion range"] = "Original axes"

        else:
            interpolation_enabled = (
                is_mid
                or getattr(
                    self,
                    "interpolarsi",
                    None,
                )
                is not None
                and self.interpolarsi.isChecked()
                or not self.interseccion
            )

            parameters["Interpolation"] = (
                "Enabled" if interpolation_enabled else "Disabled"
            )

            if common_range_widget is not None and common_range_widget.isChecked():
                parameters["Fusion range"] = "Common range"

            elif full_range_widget is not None and full_range_widget.isChecked():
                parameters["Fusion range"] = "Full combined range"

            elif not self.interseccion:
                parameters["Fusion range"] = "Artificial common axis"

            parameters["Interpolation method"] = self._checked_interpolation_method(
                suffix
            )

        points_widget = getattr(
            self,
            f"input_n_puntos{suffix}",
            None,
        )

        if points_widget is not None and points_widget.text().strip():
            parameters["Interpolation points"] = points_widget.text().strip()

        if is_mid:
            try:
                counts = self._component_counts()
                parameters["Principal components by dataset"] = ", ".join(
                    f"{os.path.basename(str(name))}: {count}"
                    for name, count in zip(
                        self.nombres_seleccionados,
                        counts,
                    )
                )
            except ValueError:
                pass

            confidence = getattr(
                self,
                "intervalo_confianza",
                None,
            )
            if confidence is not None and confidence.text().strip():
                parameters["Confidence interval"] = f"{confidence.text().strip()}%"

        return parameters

    def _record_fusion_history(
        self,
        output_name,
        strategy,
    ):
        """Register a completed fusion under the resulting DataFrame."""
        if not hasattr(
            self.menu_principal,
            "record_analysis_step",
        ):
            return

        sources = self._fusion_source_names()

        self.menu_principal.record_analysis_step(
            dataset=output_name,
            operation=strategy,
            output_dataset=output_name,
            parameters=self._fusion_history_parameters(strategy),
            source_datasets=sources,
        )

    def aplicar_fusion_mid(self, estado=None):
        """
        Validates the current mid-level data fusion selection and dispatches the appropriate interpolation workflow.
        The method checks that mid-level fusion is enabled, then calls the intersection or non-intersection handler depending on whether the datasets share a common spectral range.

        Parameters
        ----------
        estado : any, optional
            Optional state value from Qt signals, ignored by the method logic but kept for signature compatibility.

        Returns
        -------
        None
        """
        if not self.midfusion.isChecked():
            QMessageBox.warning(
                self, "Notice", "You must enable 'Mid-Level Fusion' to continue."
            )
            return

        if (
            hasattr(self, "sin_interpolacion_mid")
            and self.sin_interpolacion_mid.isChecked()
        ):
            self.mostrar_opciones_interpolacionconinterseccion_mid()
        elif self.interseccion:
            self.mostrar_opciones_interpolacionconinterseccion_mid()
        else:
            self.mostrar_opciones_interpolacionsinintersecctar_mid()

    def aplicar_fusion(self, estado=None):
        """
        Validates the current low-level data fusion selection and dispatches the appropriate interpolation workflow.
        The method checks that low-level fusion is enabled, then calls the intersection or non-intersection handler depending on whether the datasets share a common spectral range.

        Parameters
        ----------
        estado : any, optional
            Optional state value from Qt signals, ignored by the method logic but kept for signature compatibility.

        Returns
        -------
        None
        """
        if not self.lowfusion.isChecked():
            QMessageBox.warning(
                self, "Notice", "You must enable 'Low-Level Fusion' to continue."
            )
            return

        if hasattr(self, "sin_interpolacion") and self.sin_interpolacion.isChecked():
            self.mostrar_opciones_interpolacionconinterseccion()
        elif self.interseccion:
            self.mostrar_opciones_interpolacionconinterseccion()
        else:
            self.mostrar_opciones_interpolacionsinintersecctar()

    def toggle_lowfusion(self, state):
        """Show only the controls that belong to low-level fusion."""
        visible = bool(state)
        if hasattr(self, "contenedor_lowf"):
            self.contenedor_lowf.setVisible(visible)
        if visible and hasattr(self, "contenedor_midf"):
            self.contenedor_midf.setVisible(False)

    def toggle_midfusion(self, state):
        """Show only the controls that belong to mid-level fusion."""
        visible = bool(state)
        if hasattr(self, "contenedor_midf"):
            self.contenedor_midf.setVisible(visible)
        if visible and hasattr(self, "contenedor_lowf"):
            self.contenedor_lowf.setVisible(False)

    def mostrar_opciones_interpolacion(self, estado):
        """
        Shows or hides the dynamic low-level interpolation configuration panel based on the selected range option.
        The panel is created lazily the first time a valid range checkbox is checked and is only hidden again when all related range options are unchecked.

        Parameters
        ----------
        estado : int
            State value from the associated QCheckBox (for example Qt.Checked or Qt.Unchecked) that determines whether the panel should be visible.

        Returns
        -------
        None
        """
        if estado in (Qt.Checked, 2):
            if not hasattr(self, "panel_dinamico_low"):
                self.panel_dinamico_low = QWidget()
                lay = QVBoxLayout(self.panel_dinamico_low)

                self.label_metodo_interpolacion = QLabel(
                    "1-Which interpolation method would you like to use?"
                )
                self.lineal = QCheckBox("Linear")
                self.cubica = QCheckBox("Cubic")
                self.polinomica = QCheckBox("Second-order polynomial")
                self.nearest = QCheckBox("Nearest")

                self.label_forma_paso = QLabel(
                    "2-How would you like to determine the step?"
                )
                self.valor = QCheckBox("Enter step value")
                self.input_paso = QLineEdit()
                self.input_paso.setPlaceholderText(tr("Enter the step value"))
                self.promedio = QCheckBox("Average of the files")
                self.numero = QCheckBox("Define a fixed number of points")
                self.input_n_puntos = QLineEdit()
                self.input_n_puntos.setPlaceholderText(tr("Enter the number of points"))

                for w in (
                    self.label_metodo_interpolacion,
                    self.lineal,
                    self.cubica,
                    self.polinomica,
                    self.nearest,
                    self.label_forma_paso,
                    self.valor,
                    self.input_paso,
                    self.promedio,
                    self.numero,
                    self.input_n_puntos,
                ):
                    lay.addWidget(w)

                self.layout_interpolacion_low.addWidget(self.panel_dinamico_low)

            self.panel_dinamico_low.show()
        else:
            if hasattr(self, "panel_dinamico_low"):
                if (
                    hasattr(self, "rango_comun") and not self.rango_comun.isChecked()
                ) and (
                    hasattr(self, "rango_completo")
                    and not self.rango_completo.isChecked()
                ):
                    self.panel_dinamico_low.hide()

    def mostrar_opciones_interpolacion_mid(self, estado=None):
        """Display advanced mid-level resampling controls only when requested."""
        enabled = hasattr(self, "interpolar_mid") and self.interpolar_mid.isChecked()
        if hasattr(self, "contenedor_opciones_dinamicas_mid"):
            self.contenedor_opciones_dinamicas_mid.setVisible(enabled)

    def mostrar_opciones_interpolacionconinterseccion(self):
        """
        Gathers low-level fusion options when the selected datasets share a common spectral range and starts the fusion thread.
        The method reads interpolation choices, range settings, and concatenation orientation from the UI, builds the configuration dictionaries, and launches a LowLevelDataFusionThread with those parameters.

        Returns
        -------
        None
        """
        sin_interp = (
            hasattr(self, "sin_interpolacion") and self.sin_interpolacion.isChecked()
        )

        if sin_interp:
            self.hilo = LowLevelDataFusionThread(
                self.seleccionados,
                self.nombres_seleccionados,
                self.lista_rangos,
                self.interseccion,
                self.rang_comun,
                False,
                True,
                {},
                {},
                "",
                "",
                self.tipos_orden,
                "horizontal" if self.rb_concat_h.isChecked() else "vertical",
                False,
            )
            self.hilo.signal_datalowfusion.connect(self.lowfusionfinal)
            self.hilo.start()
            return

        if self.interpolarsi.isChecked():
            opcion_rango_completo = self.rango_completo.isChecked()
            opcion_rango_comun = self.rango_comun.isChecked()
            valor_paso = self.input_paso.text().strip()
            n_puntos = self.input_n_puntos.text().strip()
            opciones_metodo = {}

            if self.lineal.isChecked():
                opciones_metodo["Lineal"] = True
            if self.cubica.isChecked():
                opciones_metodo["Cubica"] = True
            if self.polinomica.isChecked():
                opciones_metodo["Polinomica de segundo orden"] = True
            if self.nearest.isChecked():
                opciones_metodo["Nearest"] = True

            opciones_paso = {}

            if self.valor.isChecked():
                opciones_paso["Ingrese el valor del paso"] = True
            if self.numero.isChecked():
                opciones_paso["Ingrese cantidad de puntos:"] = True
            if self.promedio.isChecked():
                opciones_paso["Calcular el promedio de los archivos"] = True

            print("self.seleccionados inside main")
            print(self.seleccionados)
            interpolar = True
        else:
            opcion_rango_completo = False
            opcion_rango_comun = False
            valor_paso = ""
            n_puntos = ""
            opciones_metodo = {}
            opciones_paso = {}
            interpolar = False

        if self.rb_concat_h.isChecked():
            modo_concat = "horizontal"
        elif self.rb_concat_v.isChecked():
            modo_concat = "vertical"
        else:
            modo_concat = None

        print("CONCATENATION ORIENTATION: ", modo_concat)

        if self.interpolarsi.isChecked():
            print("✅ The user selected 'Interpolate'")
        else:
            print("❌ The user did NOT select 'Interpolate'")

        self.hilo = LowLevelDataFusionThread(
            self.seleccionados,
            self.nombres_seleccionados,
            self.lista_rangos,
            self.interseccion,
            self.rang_comun,
            opcion_rango_completo,
            opcion_rango_comun,
            opciones_metodo,
            opciones_paso,
            valor_paso,
            n_puntos,
            self.tipos_orden,
            modo_concat,
            interpolar,
        )
        self.hilo.signal_datalowfusion.connect(self.lowfusionfinal)
        self.hilo.start()

    def mostrar_opciones_interpolacionconinterseccion_mid(self):
        """Validate mid-level settings and start the fusion worker."""
        try:
            component_counts = self._component_counts()
            confidence = self._confidence_value()
        except ValueError as error:
            QMessageBox.warning(self, tr(tr("Invalid mid-level settings")), str(error))
            return

        # Recommended path: preserve each block on its original axis.
        if self.sin_interpolacion_mid.isChecked():
            self.hilo = MidLevelDataFusionThread(
                self.seleccionados,
                self.nombres_seleccionados,
                self.lista_rangos,
                self.interseccion,
                self.rang_comun,
                False,
                False,
                {},
                {},
                "",
                "",
                self.tipos_orden,
                component_counts,
                confidence,
            )
            self.hilo.signal_datamidfusion.connect(self.midfusionfinal)
            self.hilo.start()
            return

        option_full = self.rango_completo_mid.isChecked()
        option_common = self.rango_comun_mid.isChecked()

        if not (option_full or option_common):
            QMessageBox.warning(
                self,
                "Missing range option",
                "Select a spectral range for resampling.",
            )
            return

        method_options = {}
        for widget, key in (
            (self.lineal_mid, "Lineal"),
            (self.cubica_mid, "Cubica"),
            (self.polinomica_mid, "Polinomica de segundo orden"),
            (self.nearest_mid, "Nearest"),
        ):
            if widget.isChecked():
                method_options[key] = True

        step_options = {}
        if self.valor_mid.isChecked():
            step_options["Ingrese el valor del paso"] = True
        elif self.promedio_mid.isChecked():
            step_options["Calcular el promedio de los archivos"] = True
        elif self.numero_mid.isChecked():
            step_options["Ingrese cantidad de puntos:"] = True

        if not method_options or not step_options:
            QMessageBox.warning(
                self,
                "Missing interpolation settings",
                "Select one interpolation method and one grid definition.",
            )
            return

        self.hilo = MidLevelDataFusionThread(
            self.seleccionados,
            self.nombres_seleccionados,
            self.lista_rangos,
            self.interseccion,
            self.rang_comun,
            option_full,
            option_common,
            method_options,
            step_options,
            self.input_paso_mid.text().strip(),
            self.input_n_puntos_mid.text().strip(),
            self.tipos_orden,
            component_counts,
            confidence,
        )
        self.hilo.signal_datamidfusion.connect(self.midfusionfinal)
        self.hilo.start()

    def mostrar_opciones_interpolacionsinintersecctar(self):
        """
        Collects low-level fusion options when the selected datasets do not share a common spectral range and starts the corresponding fusion thread.
        The method reads the desired number of interpolation points and interpolation methods from the UI, builds a configuration dictionary, and launches a LowLevelDataFusionNoCommonRangeThread with those values.

        Returns
        -------
        None
        """
        n_puntos = self.input_n_puntos.text().strip()

        opciones_metodo = {}

        if self.lineal.isChecked():
            opciones_metodo["Lineal"] = True
        if self.cubica.isChecked():
            opciones_metodo["Cubica"] = True
        if self.polinomica.isChecked():
            opciones_metodo["Polinomica de segundo orden"] = True
        if self.nearest.isChecked():
            opciones_metodo["Nearest"] = True

        self.hilo = LowLevelDataFusionNoCommonRangeThread(
            self.seleccionados,
            self.nombres_seleccionados,
            self.lista_rangos,
            n_puntos,
            opciones_metodo,
            self.tipos_orden,
        )
        self.hilo.signal_datalowfusionsininterseccion.connect(
            self.lowfusionfinalsininterseccion
        )
        self.hilo.start()

    def mostrar_opciones_interpolacionsinintersecctar_mid(self):
        """Run mid-level fusion for non-overlapping blocks."""
        try:
            component_counts = self._component_counts()
            confidence = self._confidence_value()
        except ValueError as error:
            QMessageBox.warning(self, tr(tr("Invalid mid-level settings")), str(error))
            return

        # Original axes are valid for mid-level fusion even without intersection.
        if self.sin_interpolacion_mid.isChecked():
            self.hilo = MidLevelDataFusionThread(
                self.seleccionados,
                self.nombres_seleccionados,
                self.lista_rangos,
                self.interseccion,
                self.rang_comun,
                False,
                False,
                {},
                {},
                "",
                "",
                self.tipos_orden,
                component_counts,
                confidence,
            )
            self.hilo.signal_datamidfusion.connect(self.midfusionfinal)
            self.hilo.start()
            return

        points = self.input_n_puntos_mid.text().strip()
        if not points:
            QMessageBox.warning(
                self,
                "Missing interpolation points",
                "Enter the number of points for resampling.",
            )
            return

        method_options = {}
        for widget, key in (
            (self.lineal_mid, "Lineal"),
            (self.cubica_mid, "Cubica"),
            (self.polinomica_mid, "Polinomica de segundo orden"),
            (self.nearest_mid, "Nearest"),
        ):
            if widget.isChecked():
                method_options[key] = True

        self.hilo = MidLevelDataFusionNoCommonRangeThread(
            self.seleccionados,
            self.nombres_seleccionados,
            self.lista_rangos,
            points,
            method_options,
            self.tipos_orden,
            component_counts,
            confidence,
        )
        self.hilo.signal_datamidfusionsininterseccion.connect(
            self.midfusionfinalsininterseccion
        )
        self.hilo.start()

    def lowfusionfinal(self, df_concat):
        """
        Finalizes a low-level data fusion run by registering and exporting the fused DataFrame.
        The method asks the user for a name, stores the matrix in the main application's list,
        writes it to a CSV file, and shows a confirmation message.

        Parameters
        ----------
        df_concat : pandas.DataFrame
            Concatenated DataFrame produced by the low-level fusion thread.

        Returns
        -------
        None
        """
        self.df_concat_midfusion = df_concat

        if self.preview_mode:
            self.preview_mode = False

            self.preview_window = FusionPreviewWindow(
                dataframe=df_concat,
                title=tr("Low-level fusion preview"),
                parent=self,
            )

            self.preview_window.show()

            selection_page = getattr(
                self.menu_principal,
                "data_fusion_page",
                None,
            )

            if selection_page is not None:
                selection_page.btn_plot_preview.setEnabled(True)

                selection_page.btn_accept.setEnabled(True)
                selection_page.btn_plot_mid_result.setEnabled(True)

            self.show_mid_level_result()
            return

        df_concat.attrs = getattr(df_concat, "attrs", {}).copy()

        nombre_df, ok = QInputDialog.getText(
            self,
            tr(tr("Save DataFrame")),
            tr(tr("Enter a name for the transformed DataFrame:")),
        )

        if ok and nombre_df.strip():
            nombre_limpio = nombre_df.strip()

            self.menu_principal.dataframes.append(df_concat)
            self.menu_principal.nombres_archivos.append(nombre_limpio)

            self._record_fusion_history(
                nombre_limpio,
                "Low-level fusion",
            )

            ruta = os.path.join("archivos_guardados", f"{nombre_limpio}.csv")
            os.makedirs("archivos_guardados", exist_ok=True)

            df_concat.to_csv(
                ruta,
                index=False,
                header=False,
            )

            QMessageBox.information(
                self,
                tr("Success"),
                tr(
                    "Transformed DataFrame saved as '{name}' and exported to CSV.",
                    name=nombre_limpio,
                ),
            )

    def midfusionfinal(self, df_concat, lista_varianza):
        """
        Finalizes a mid-level data fusion run by registering the fused DataFrame and its
        explained-variance information, then exporting the matrix to CSV.

        Parameters
        ----------
        df_concat : pandas.DataFrame
            Concatenated DataFrame produced by the mid-level fusion thread.
        lista_varianza : list
            List of explained-variance values associated with the fused components.

        Returns
        -------
        None
        """
        self.df_concat_midfusion = df_concat
        self.lista_varianza = lista_varianza

        selection_page = getattr(
            self.menu_principal,
            "data_fusion_page",
            None,
        )

        if selection_page is not None:
            selection_page.btn_plot_mid_result.setEnabled(True)

        if self.preview_mode:
            self.preview_mode = False

            self.df_concat_midfusion = df_concat
            self.lista_varianza = lista_varianza

            selection_page = getattr(
                self.menu_principal,
                "data_fusion_page",
                None,
            )

            if selection_page is not None:
                selection_page.btn_plot_preview.setEnabled(True)

                selection_page.btn_accept.setEnabled(True)
                selection_page.btn_plot_mid_result.setEnabled(True)

            self.show_mid_level_result()
            return

        df_concat.attrs = getattr(
            df_concat,
            "attrs",
            {},
        ).copy()

        nombre_df, ok = QInputDialog.getText(
            self,
            tr(tr("Save DataFrame")),
            tr(tr("Enter a name for the transformed DataFrame:")),
        )

        if ok and nombre_df.strip():
            nombre_limpio = nombre_df.strip()

            self.menu_principal.dataframes.append(df_concat)

            self.menu_principal.nombres_archivos.append(nombre_limpio)

            self._record_fusion_history(
                nombre_limpio,
                "Mid-level fusion",
            )

            ruta = os.path.join(
                "archivos_guardados",
                f"{nombre_limpio}.csv",
            )

            os.makedirs(
                "archivos_guardados",
                exist_ok=True,
            )

            df_concat.to_csv(
                ruta,
                index=False,
            )

            QMessageBox.information(
                self,
                tr("Success"),
                tr(
                    "Transformed DataFrame saved as '{name}' and exported to CSV.",
                    name=nombre_limpio,
                ),
            )

        self.show_mid_level_result()

    def lowfusionfinalsininterseccion(self, df_concat):
        """
        Finalizes a low-level data fusion run without a common spectral range by registering and exporting the fused DataFrame.
        The method prompts the user for a name, adds the matrix to the main application's list, writes it to a CSV file, and shows a confirmation message.

        Parameters
        ----------
        df_concat : pandas.DataFrame
            Concatenated DataFrame produced by the low-level fusion thread when no intersection exists between X-axis ranges.

        Returns
        -------
        None
        """
        self.df_concat_midfusion = df_concat

        if self.preview_mode:
            self.preview_mode = False

            self.preview_window = FusionPreviewWindow(
                dataframe=df_concat,
                title=tr("Low-level fusion preview"),
                parent=self,
            )

            self.preview_window.show()

            selection_page = getattr(
                self.menu_principal,
                "data_fusion_page",
                None,
            )

            if selection_page is not None:
                selection_page.btn_plot_preview.setEnabled(True)

                selection_page.btn_accept.setEnabled(True)

            return

        nombre_df, ok = QInputDialog.getText(
            self,
            tr(tr("Save DataFrame")),
            tr(tr("Enter a name for the transformed DataFrame:")),
        )
        if ok and nombre_df.strip():
            nombre_limpio = nombre_df.strip()
            self.menu_principal.dataframes.append(df_concat)
            self.menu_principal.nombres_archivos.append(nombre_limpio)

            self._record_fusion_history(
                nombre_limpio,
                "Low-level fusion",
            )

            ruta = os.path.join("archivos_guardados", f"{nombre_limpio}.csv")
            os.makedirs("archivos_guardados", exist_ok=True)
            df_concat.to_csv(ruta, index=False)
            QMessageBox.information(
                self,
                tr("Success"),
                tr(
                    "Transformed DataFrame saved as '{name}' and exported to CSV.",
                    name=nombre_limpio,
                ),
            )

    def midfusionfinalsininterseccion(self, df_concat, lista_varianza):
        """
        Finalizes a mid-level data fusion run without a common spectral range by registering the fused DataFrame and its variance information, then exporting the matrix to CSV.
        The method prompts the user for a name, stores the matrix and variance list in the main application, writes the DataFrame to disk, and displays a confirmation message.

        Parameters
        ----------
        df_concat : pandas.DataFrame
            Concatenated DataFrame produced by the mid-level fusion thread when no X-axis intersection exists.
        lista_varianza : list
            List of explained-variance values associated with the fused components, kept for subsequent visualization or reporting.

        Returns
        -------
        None
        """
        self.df_concat_midfusion = df_concat
        self.lista_varianza = lista_varianza

        selection_page = getattr(
            self.menu_principal,
            "data_fusion_page",
            None,
        )

        if selection_page is not None:
            selection_page.btn_plot_mid_result.setEnabled(True)

        if self.preview_mode:
            self.preview_mode = False

            self.df_concat_midfusion = df_concat
            self.lista_varianza = lista_varianza

            selection_page = getattr(
                self.menu_principal,
                "data_fusion_page",
                None,
            )

            if selection_page is not None:
                selection_page.btn_plot_preview.setEnabled(True)

                selection_page.btn_accept.setEnabled(True)

            return

        nombre_df, ok = QInputDialog.getText(
            self,
            tr(tr("Save DataFrame")),
            tr(tr("Enter a name for the transformed DataFrame:")),
        )
        if ok and nombre_df.strip():
            nombre_limpio = nombre_df.strip()
            self.menu_principal.dataframes.append(df_concat)
            self.menu_principal.nombres_archivos.append(nombre_limpio)

            self._record_fusion_history(
                nombre_limpio,
                "Mid-level fusion",
            )

            ruta = os.path.join("archivos_guardados", f"{nombre_limpio}.csv")
            os.makedirs("archivos_guardados", exist_ok=True)
            df_concat.to_csv(ruta, index=False)
            QMessageBox.information(
                self,
                tr("Success"),
                tr(
                    "Transformed DataFrame saved as '{name}' and exported to CSV.",
                    name=nombre_limpio,
                ),
            )

        self.show_mid_level_result()

    def graficar_componentes_principales(self, pcs):
        """
        Launches a background job to plot selected principal components from a fused mid-level data matrix.
        The method constructs a MidLevelPlotThread with all required data and options, connects its figure signals to local display callbacks, and starts the thread.

        Parameters
        ----------
        pcs : list or sequence of int
            Indices of principal components to visualize in 2D, 3D, or heatmap form.

        Returns
        -------
        None
        """
        self.hilo = MidLevelPlotThread(
            self.lista_df,
            self.seleccionados,
            self.df_concat_midfusion,
            pcs,
            self._component_counts(),
            self._confidence_value(),
            self.lista_varianza,
        )
        self.hilo.pca_2d_figure_signal.connect(self.mostrar_grafico_pca_2d_mid)
        self.hilo.pca_3d_figure_signal.connect(self.mostrar_grafico_pca_3d_mid)
        self.hilo.signal_figura_heatmap.connect(self.mostrar_grafico_mapa_calor)
        self.hilo.start()

    def mostrar_grafico_pca_2d_mid(self, fig):
        """
        Opens a window that displays a 2D PCA plot produced by mid-level data fusion.
        The method wraps the given figure in a VentanaGraficoPCA2D widget, stores the reference, and shows it to the user.

        Parameters
        ----------
        fig : object
            Plotly figure containing the 2D PCA projection to be visualized.

        Returns
        -------
        None
        """
        self.ventana_pca = VentanaGraficoPCA2D(fig)
        self.ventana_pca.show()

    def mostrar_grafico_pca_3d_mid(self, fig):
        """
        Opens a window that displays a 3D PCA plot produced by mid-level data fusion.
        The method wraps the given figure in a VentanaGraficoPCA3D widget, stores the reference, and shows it to the user.

        Parameters
        ----------
        fig : object
            Plotly figure containing the 3D PCA projection to be visualized.

        Returns
        -------
        None
        """
        self.ventana_pca = VentanaGraficoPCA3D(fig)
        self.ventana_pca.show()

    def mostrar_grafico_mapa_calor(self, fig):
        """
        Opens a window that displays a PCA heatmap produced by mid-level data fusion.
        The method wraps the given figure in a VentanaGraficoMapaCalor widget, stores the reference, and shows it to the user.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object containing the heatmap of principal components to be visualized.

        Returns
        -------
        None
        """
        self.ventana_pca = VentanaGraficoMapaCalor(fig)
        self.ventana_pca.show()


class VentanaGraficoMapaCalor(QMainWindow):
    """
    Displays a PCA heatmap figure inside a dedicated window.
    The widget embeds the given Matplotlib heatmap in a FigureCanvas and sets it as the central widget so users can inspect principal component patterns.

    Parameters
    ----------
    figura : matplotlib.figure.Figure
        Figure object containing the heatmap of principal components to display.
    parent : QWidget, optional
        Parent widget that will own this window, by default None.
    """

    def __init__(self, figura, parent=None):
        super().__init__(parent)
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("Heatmap - Principal Components"))
        self.canvas = FigureCanvas(figura)
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)